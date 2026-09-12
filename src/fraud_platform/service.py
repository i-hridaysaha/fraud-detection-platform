"""The scoring service: one endpoint in front of `fraud_platform.serving`.

    POST /score   {"transaction": {...raw columns...}, "top_factors": 5, "return_features": true}
    GET  /health  the model version behind the alias, the store backend, the request count
    GET  /model   what the registry says about the version being served

The model is loaded once, at startup, from the MLflow registry by name and alias
(`models:/fraud-detector@production` unless `FRAUD_MODEL_NAME` and `FRAUD_MODEL_ALIAS` say
otherwise), never from a file path. A service started against an empty registry refuses to
start, which is the intended failure: there is no fallback model to serve by accident.

The store is Redis when `FRAUD_REDIS_URL` is set and in-process otherwise; both run the same
two-phase contract, `get_features` before the row is prepared and `commit` after the score is
out, so a feature vector never contains the transaction it describes (ADR 0033).

A payload is refused with 422 when it carries a column the file does not have, a value that is
not what the stage 1 dtype map says it is, or a prepared row that breaks the stage 3 schema; the
response lists every violation rather than the first. A transaction older than the latest one
committed on any of its keys is refused with 409, because the store's definitions are "strictly
before t" and an older row cannot be folded in without rewriting history.

The endpoint is a plain `def`, so Starlette runs it on its thread pool and requests overlap;
each request's inference is pinned to one thread inside the bundle (`serving.INFERENCE_THREADS`,
ADR 0035), so N concurrent requests occupy N threads and not N times the core count.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from fraud_platform import online_store, schema, serving
from fraud_platform.online_store import MemoryBackend, OnlineFeatureStore, RedisBackend

# --- settings, from the environment --------------------------------------------------------------

ENV_MODEL_NAME = "FRAUD_MODEL_NAME"
ENV_MODEL_ALIAS = "FRAUD_MODEL_ALIAS"
ENV_REDIS_URL = "FRAUD_REDIS_URL"
ENV_REDIS_PREFIX = "FRAUD_REDIS_PREFIX"
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"

MAX_FACTORS = 50


class Settings(BaseModel):
    model_name: str = serving.MODEL_NAME
    model_alias: str = serving.MODEL_ALIAS
    redis_url: str | None = None
    redis_prefix: str = online_store.REDIS_PREFIX

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_name=os.environ.get(ENV_MODEL_NAME, serving.MODEL_NAME),
            model_alias=os.environ.get(ENV_MODEL_ALIAS, serving.MODEL_ALIAS),
            redis_url=os.environ.get(ENV_REDIS_URL) or None,
            redis_prefix=os.environ.get(ENV_REDIS_PREFIX, online_store.REDIS_PREFIX),
        )


# --- the contract on the wire ------------------------------------------------------------------------


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction: dict[str, Any] = Field(
        description="the raw columns of one transaction, as the file names them; omitted is null"
    )
    top_factors: int = Field(0, ge=0, le=MAX_FACTORS, description="SHAP factors to return")
    return_features: bool = Field(False, description="return the store's feature vector")


class Factor(BaseModel):
    feature: str
    value: float | None
    contribution: float


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: int
    run_id: str
    uri: str


class ScoreResponse(BaseModel):
    TransactionID: int
    score: float = Field(description="the booster's raw probability")
    probability: float = Field(description="the stage 6 calibrated probability")
    band: Literal["approve", "review", "block"]
    alert: bool = Field(description="score at or above the stage 6 tuned threshold")
    factors: list[Factor] | None = None
    base_value: float | None = None
    features: dict[str, float | int | None] | None = None
    model: ModelInfo
    latency_ms: float


class Health(BaseModel):
    status: str
    model: ModelInfo
    store: str
    inference_threads: int
    requests: int


# --- the app ------------------------------------------------------------------------------------------


class State:
    """What one process holds: the bundle, the store, and a request counter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bundle, about = serving.load_from_registry(settings.model_name, settings.model_alias)
        self.model = ModelInfo(**about)
        self.store, self.store_name = make_store(settings, self.bundle)
        self.requests = 0
        self._lock = threading.Lock()

    def count(self) -> None:
        with self._lock:
            self.requests += 1


def make_store(settings: Settings, bundle: serving.ServingBundle) -> tuple[OnlineFeatureStore, str]:
    if settings.redis_url:
        import redis

        client = redis.Redis.from_url(settings.redis_url)
        client.ping()
        backend: online_store.StateBackend = RedisBackend(client, prefix=settings.redis_prefix)
        name = f"redis ({settings.redis_url})"
    else:
        backend = MemoryBackend()
        name = "memory"
    # The stage 4 running-state features are always held; the stage 5 counters only when the
    # served stack reads them, because their keys are hub values (a device string is shared by a
    # large share of the stream) and holding them costs two keys per request for nothing.
    graph_cfg = online_store.GRAPH_ON.switched(bool(bundle.stack.get("graph", False)))
    return OnlineFeatureStore(backend, bundle.dist_buckets, graph_cfg=graph_cfg), name


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    holder: dict[str, State] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        holder["state"] = State(settings)
        yield
        holder.clear()

    app = FastAPI(title="fraud-detection-platform scoring service", lifespan=lifespan)

    def state() -> State:
        if "state" not in holder:
            raise HTTPException(status_code=503, detail="the model is not loaded")
        return holder["state"]

    @app.get("/health", response_model=Health)
    def health() -> Health:
        current = state()
        return Health(
            status="ok",
            model=current.model,
            store=current.store_name,
            inference_threads=serving.INFERENCE_THREADS,
            requests=current.requests,
        )

    @app.get("/model", response_model=ModelInfo)
    def model() -> ModelInfo:
        return state().model

    @app.post("/score", response_model=ScoreResponse, response_model_exclude_none=True)
    def score(request: ScoreRequest) -> ScoreResponse:
        current = state()
        started = time.perf_counter()
        try:
            out = serving.score_transaction(
                current.bundle,
                request.transaction,
                current.store,
                top_factors=request.top_factors,
            )
        except serving.PayloadError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except schema.SchemaError as error:
            raise HTTPException(status_code=422, detail=error.violations) from error
        except online_store.OutOfOrderError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        current.count()
        if not request.return_features:
            out.pop("features", None)
        return ScoreResponse(
            **out, model=current.model, latency_ms=(time.perf_counter() - started) * 1000.0
        )

    return app


app = create_app()
