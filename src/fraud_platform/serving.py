"""The scoring path: one raw transaction in, a score, a band and its reasons out.

Everything the endpoint does to a row is here, with no HTTP in it, so the same code runs under
the service, under the latency benchmark, under the parity checks and under a plain test. The
order is fixed and every step is one of the earlier stages' own functions:

1. `raw_frame`: the payload becomes a one-row frame with every raw column the file has, cast
   with the stage 1 dtype map, plus the ADR 0001 entity key from `data_loader.add_entity_key`.
2. `prepared_row`: the stage 3 pipeline, `prepare.apply_preparation`, with the objects that
   were fitted on the training split when the bundle was registered. Then the stage 3 schema,
   `schema.validate`, with the label removed from the contract because a transaction being
   scored does not have one. A row that fails here is refused before anything reads it.
3. `matrix`: `modelling.tree_matrix` over the bundle's column list, the stage 6 model's own view
   of the row. A column the prepared frame does not carry is looked up in the store's feature
   vector, which is what a stack with the causal block switched on would read.
4. `score`, `calibrate`, `band`: the booster, then the isotonic map fitted on validation in
   stage 6 as its breakpoints, then the cost-derived edges of ADR 0029.
5. `explain`: the booster's own TreeSHAP contributions, the same numbers `shap.TreeExplainer`
   produced for `reports/shap/`, checked against that artifact by a test.

**The bundle is what the registry holds.** `ServingBundle` is the set of files
`scripts/register_model.py` logs to MLflow as one model version: the booster, the pickled stage
3 fit, and a JSON with the column list, the schema, the calibrator breakpoints, the band edges
and the dist1 bucket edges. The service loads it by name and alias through the registry, never
by path, so what is served is what was registered and nothing else. `FraudScorer` is the MLflow
wrapper that makes the bundle a `pyfunc` model.

**One thread for inference.** `load_bundle` pins the booster to one thread. Measured in
`reports/latency.json`: see `PIN_REASON` for the number and the comment beside it for why the
number, not the folklore, is the argument. ADR 0035.
"""

from __future__ import annotations

import json
import os
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost

from fraud_platform import config, data_loader, evaluation, modelling, prepare, schema
from fraud_platform.online_store import OnlineFeatureStore

# --- choices --------------------------------------------------------------------------------------

MODEL_NAME = "fraud-detector"
MODEL_ALIAS = "production"

BOOSTER_FILE = "model.ubj"
PREPARATION_FILE = "preparation.pkl"
SERVING_FILE = "serving.json"

BANDS: tuple[str, str, str] = ("approve", "review", "block")

# How many threads the booster predicts with. One. The reason is measured, not assumed:
# reports/latency.json, block `pinning`, regenerated with `make serving-latency`, times the same
# single-row predict_proba on the same rows in five fresh processes with the booster pinned to
# one thread and with its default of every core. The pinned call is the faster one in every
# process, by the ratio `unpinned_over_pinned_p50` records, because one row gives the thread
# pool nothing to split and the fork-join is pure cost. The same block keeps the counterpoint:
# on a batch of a thousand rows the unpinned call wins by the ratio `batch` records, so this is
# a single-row setting and not a property of the booster. What the pin does under concurrent
# requests is not isolated by the benchmark; the load test runs pinned and reports what it saw.
INFERENCE_THREADS = 1
PIN_REASON = "reports/latency.json: pinning"


class PayloadError(ValueError):
    """The payload cannot become a row: a column that is not a raw column, or a broken value."""


# --- the bundle -------------------------------------------------------------------------------------


@dataclass
class ServingBundle:
    """Everything one model version needs to score, held together and loaded once."""

    model: xgboost.XGBClassifier
    fitted: dict[str, Any]
    plan: dict[str, Any]
    serving: dict[str, Any]
    schema: schema.Schema

    @property
    def columns(self) -> list[str]:
        return list(self.serving["columns"])

    @property
    def raw_columns(self) -> list[str]:
        return list(self.serving["raw_columns"])

    @property
    def dist_buckets(self) -> dict[str, Any]:
        return dict(self.serving["dist_buckets"])

    @property
    def bands(self) -> dict[str, float]:
        return {
            "low": float(self.serving["bands"]["low"]),
            "high": float(self.serving["bands"]["high"]),
        }

    @property
    def tuned_threshold(self) -> float:
        return float(self.serving["tuned_threshold"]["threshold"])

    @property
    def stack(self) -> dict[str, Any]:
        return dict(self.serving["stack"])


def serving_schema(declared: schema.Schema) -> schema.Schema:
    """The stage 3 contract for a row that has no label yet.

    Everything else stands: the column set, the families, the nulls, the bounds. Only the target
    leaves, because the service scores rows whose label arrives weeks later, and a contract that
    demanded it would refuse every real request.
    """
    return schema.Schema(
        columns=tuple(spec for spec in declared.columns if spec.name != config.TARGET),
        allow_extra_columns=declared.allow_extra_columns,
        name=f"{declared.name}, serving",
    )


def write_bundle(
    directory: Path,
    model: xgboost.XGBClassifier,
    fitted: Mapping[str, Any],
    plan: Mapping[str, Any],
    serving: Mapping[str, Any],
) -> dict[str, Path]:
    """Lay the bundle out as files, ready for `mlflow.pyfunc.log_model(artifacts=...)`."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "booster": directory / BOOSTER_FILE,
        "preparation": directory / PREPARATION_FILE,
        "serving": directory / SERVING_FILE,
    }
    model.save_model(paths["booster"])
    paths["preparation"].write_bytes(pickle.dumps({"fitted": dict(fitted), "plan": dict(plan)}))
    paths["serving"].write_text(json.dumps(serving, indent=2, allow_nan=False) + "\n")
    return paths


def load_bundle(paths: Mapping[str, str | Path]) -> ServingBundle:
    """The bundle from its three files, with the booster pinned to `INFERENCE_THREADS`."""
    model = xgboost.XGBClassifier()
    model.load_model(str(paths["booster"]))
    model.set_params(n_jobs=INFERENCE_THREADS)
    model.get_booster().set_param({"nthread": INFERENCE_THREADS})
    preparation = pickle.loads(Path(paths["preparation"]).read_bytes())
    serving = json.loads(Path(paths["serving"]).read_text())
    return ServingBundle(
        model=model,
        fitted=preparation["fitted"],
        plan=preparation["plan"],
        serving=serving,
        schema=serving_schema(schema.Schema.from_dict(serving["schema"])),
    )


# --- the row ------------------------------------------------------------------------------------------


def raw_frame(payload: Mapping[str, Any], raw_columns: Sequence[str]) -> pd.DataFrame:
    """A one-row frame shaped like the joined file, with the entity key attached.

    A column the payload does not carry is null. A column the file does not have is refused
    here rather than silently dropped, because the stage 3 contract has no room for it and
    a producer sending one has a bug worth hearing about. The dtype map is stage 1's: a measured
    null-free integer column is cast to its integer type only when the value is present, so a
    null there reaches the schema as a float and is refused as the family mismatch it is.
    """
    unknown = sorted(set(payload) - set(raw_columns))
    if unknown:
        raise PayloadError(f"not raw columns of the file: {unknown}")
    if config.TARGET in payload:
        raise PayloadError(f"{config.TARGET} is not accepted: a scored transaction has no label")
    dtypes = data_loader.build_dtype_map(raw_columns)
    columns: dict[str, Any] = {}
    for column in raw_columns:
        if column == config.TARGET:
            continue
        value = payload.get(column)
        dtype = dtypes[column]
        if value is None or (isinstance(value, float) and np.isnan(value)):
            columns[column] = (
                pd.Categorical([None])
                if dtype == "category"
                else np.array([np.nan], dtype="float64")
            )
        elif dtype == "category":
            columns[column] = pd.Categorical([str(value)])
        else:
            try:
                columns[column] = np.array([value]).astype(np.dtype(dtype))
            except (TypeError, ValueError) as error:
                raise PayloadError(f"{column}: {value!r} is not a {dtype}") from error
    return data_loader.add_entity_key(pd.DataFrame(columns, copy=False))


def prepared_row(bundle: ServingBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """The stage 3 pipeline on one row, then the stage 3 contract. Raises `schema.SchemaError`."""
    prepared = prepare.apply_preparation(frame, bundle.fitted, bundle.plan)
    schema.validate(prepared, bundle.schema)
    return prepared


def matrix(
    bundle: ServingBundle, prepared: pd.DataFrame, store_features: Mapping[str, float | int] | None
) -> npt.NDArray[np.float32]:
    """The model's view of the row: the bundle's columns, prepared first, store second."""
    frame = prepared
    missing = [column for column in bundle.columns if column not in prepared.columns]
    if missing:
        if store_features is None:
            raise KeyError(f"the model reads {missing}, which need the feature store")
        extra = pd.DataFrame(
            {name: [store_features[name]] for name in missing}, index=prepared.index
        )
        frame = pd.concat([prepared, extra], axis=1)
    return modelling.tree_matrix(frame, bundle.columns).to_numpy(dtype="float32")


# --- the score --------------------------------------------------------------------------------------


def score(bundle: ServingBundle, x: npt.NDArray[np.float32]) -> float:
    return float(modelling.score(bundle.model, x)[0])


def calibrate(bundle: ServingBundle, raw: float) -> float:
    """The stage 6 isotonic map from its breakpoints: linear between them, clipped outside."""
    x = np.asarray(bundle.serving["calibrator"]["x"], dtype="float64")
    y = np.asarray(bundle.serving["calibrator"]["y"], dtype="float64")
    return float(np.interp(raw, x, y))


def band(bundle: ServingBundle, probability: float) -> str:
    edges = bundle.bands
    index = int(evaluation.apply_bands(np.array([probability]), edges["low"], edges["high"])[0])
    return BANDS[index]


def explain(
    bundle: ServingBundle, x: npt.NDArray[np.float32], top: int
) -> tuple[list[dict[str, Any]], float]:
    """The `top` largest contributions by absolute value, and the base value, in log-odds.

    `pred_contribs` is XGBoost's TreeSHAP with the tree-path-dependent weighting, which is what
    `shap.TreeExplainer` computes for a booster; the last column is the bias.
    """
    booster = bundle.model.get_booster()
    contributions = booster.predict(
        xgboost.DMatrix(x, feature_names=bundle.columns, nthread=INFERENCE_THREADS),
        pred_contribs=True,
    )[0]
    base = float(contributions[-1])
    values = contributions[:-1]
    order = np.argsort(-np.abs(values), kind="stable")[:top]
    factors = [
        {
            "feature": bundle.columns[int(i)],
            "value": None if np.isnan(x[0, int(i)]) else float(x[0, int(i)]),
            "contribution": float(values[int(i)]),
        }
        for i in order
    ]
    return factors, base


# --- the whole path ---------------------------------------------------------------------------------


def score_transaction(
    bundle: ServingBundle,
    payload: Mapping[str, Any],
    store: OnlineFeatureStore | None,
    top_factors: int = 0,
) -> dict[str, Any]:
    """The endpoint's work, without the endpoint: get_features, prepare, score, band, commit.

    The store is read before the row is prepared and written after the score is out, so the
    feature vector describes the state strictly before this transaction whatever the model
    then does with it.
    """
    frame = raw_frame(payload, bundle.raw_columns)
    row: dict[str, Any] = {str(k): v for k, v in frame.iloc[0].to_dict().items()}
    features = store.get_features(row) if store is not None else None
    prepared = prepared_row(bundle, frame)
    x = matrix(bundle, prepared, features)
    raw = score(bundle, x)
    probability = calibrate(bundle, raw)
    decision = band(bundle, probability)
    out: dict[str, Any] = {
        config.ID_COLUMN: int(frame[config.ID_COLUMN].iloc[0]),
        "score": raw,
        "probability": probability,
        "band": decision,
        "alert": raw >= bundle.tuned_threshold,
    }
    if top_factors > 0:
        factors, base = explain(bundle, x, top_factors)
        out["factors"] = factors
        out["base_value"] = base
    if features is not None:
        out["features"] = {k: (None if _is_nan(v) else v) for k, v in features.items()}
    if store is not None:
        store.commit(row)
    return out


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and np.isnan(value)


# --- the MLflow wrapper -------------------------------------------------------------------------------


class FraudScorer:
    """The `mlflow.pyfunc` model: the bundle, loaded from the version's own artifacts.

    Kept free of an MLflow base class at import time so the module imports without MLflow;
    `as_pyfunc` builds the subclass when the registration script needs it.
    """

    def __init__(self) -> None:
        self.bundle: ServingBundle | None = None

    def load_context(self, context: Any) -> None:
        self.bundle = load_bundle(context.artifacts)

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        """Score a frame of raw rows, no store: the registry's own view of the model."""
        assert self.bundle is not None
        rows = []
        for record in model_input.to_dict(orient="records"):
            payload = {str(k): v for k, v in record.items() if not _is_nan(v)}
            rows.append(score_transaction(self.bundle, payload, None))
        return pd.DataFrame(rows)


def as_pyfunc() -> Any:
    import mlflow.pyfunc

    base: Any = mlflow.pyfunc.PythonModel

    # FraudScorer first: PythonModel defines no-op load_context and predict of its own, and the
    # method resolution order has to reach ours before those.
    class FraudScorerModel(FraudScorer, base):  # type: ignore[misc]
        def __init__(self) -> None:
            FraudScorer.__init__(self)

    return FraudScorerModel()


def registry_uri(name: str = MODEL_NAME, alias: str = MODEL_ALIAS) -> str:
    """`models:/<name>@<alias>`. A name and an alias, never a path.

    MLflow's registry stages are deprecated in the installed 3.16 (`transition_model_version_stage`
    carries `deprecated(since="2.9.0")`, checked this session) in favour of aliases, so the
    alias is the stage: `production` is what the service loads.
    """
    return f"models:/{name}@{alias}"


MODEL_CODE = Path(__file__).with_name("serving_model.py")
EXPERIMENT = "serving"
MLRUNS_DIR = config.ROOT / "mlruns"
ENV_TRACKING_URI = "MLFLOW_TRACKING_URI"


def tracking_uri() -> str:
    """Where the registry is: `MLFLOW_TRACKING_URI`, or a SQLite file under mlruns/.

    The installed MLflow (3.16) refuses its plain filesystem backend unless opted into, so the
    local registry is a database file with the artifacts beside it. Everything that touches the
    registry, the registration script, the service and the benchmarks, resolves the location
    here and nowhere else.
    """
    configured = os.environ.get(ENV_TRACKING_URI)
    if configured:
        return configured
    return f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}"


def register_bundle(
    paths: Mapping[str, Path],
    params: Mapping[str, Any],
    tracking_uri: str,
    artifact_root: Path,
    name: str = MODEL_NAME,
    alias: str = MODEL_ALIAS,
    experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    """Log the bundle as one model version, register it, and point the alias at it.

    The model's code is `serving_model.py`, logged as a file, so nothing is pickled but the
    stage 3 fit inside the bundle. Returns what the registry now says.
    """
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    found = client.get_experiment_by_name(experiment)
    if found is None:
        artifact_root.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            experiment, artifact_location=f"file://{artifact_root.resolve()}"
        )
    else:
        experiment_id = found.experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        info = mlflow.pyfunc.log_model(
            name="fraud_scorer",
            python_model=str(MODEL_CODE),
            artifacts={label: str(path) for label, path in paths.items()},
            registered_model_name=name,
        )
    version = str(info.registered_model_version)
    client.set_registered_model_alias(name, alias, version)
    for key, value in params.items():
        client.set_model_version_tag(name, version, key, str(value)[:250])
    return {
        "tracking_uri": tracking_uri,
        "experiment": experiment,
        "run_id": run.info.run_id,
        "model_name": name,
        "version": int(version),
        "alias": alias,
        "uri": registry_uri(name, alias),
        "model_uri": info.model_uri,
        "python_model": MODEL_CODE.name,
        "artifacts": sorted(paths),
    }


def load_from_registry(
    name: str = MODEL_NAME, alias: str = MODEL_ALIAS
) -> tuple[ServingBundle, dict[str, Any]]:
    """The bundle behind a registered alias, and what the registry says about it."""
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri())
    version = MlflowClient().get_model_version_by_alias(name, alias)
    loaded = mlflow.pyfunc.load_model(registry_uri(name, alias))
    scorer = loaded.unwrap_python_model()
    assert scorer.bundle is not None
    about = {
        "name": name,
        "alias": alias,
        "version": int(version.version),
        "run_id": version.run_id,
        "uri": registry_uri(name, alias),
    }
    return scorer.bundle, about
