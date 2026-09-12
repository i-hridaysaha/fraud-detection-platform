"""The scoring service end to end, on a bundle registered into a throwaway registry.

The fixtures build a small synthetic stream shaped like the file (every raw column, the stage
1 dtypes, the cardinalities the stage 3 encoders need), fit the real stage 3 pipeline on it,
train a small booster, and register the bundle through `serving.register_bundle` into a SQLite
registry under the test's temporary directory. The service then starts against that registry
by name and alias, exactly as it would against the real one. Nothing here reads the data files,
so the whole module runs in CI.

What is asserted: the model loads from the registry and only from it; a scored row comes back
with a band from the stage 6 edges and factors that sum to the log-odds; the two-phase store
contract shows in the responses; and the four refusals (a column the file lacks, a label in the
payload, a broken schema, an older transaction) come back as the status codes the module
promises, with every violation listed.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xgboost
from fastapi.testclient import TestClient

from fraud_platform import (
    cleaning,
    config,
    data_loader,
    evaluation,
    features,
    modelling,
    online_store,
    prepare,
    schema,
    service,
    serving,
)

DAY = config.SECONDS_PER_DAY
N_ROWS = 1_500
# The frequency encoder skips a column narrower than this; the synthetic identifiers clear it.
N_LEVELS = features.N_DIST_BUCKETS * 8


# --- a synthetic file --------------------------------------------------------------------------------


def synthetic_stream(n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Every raw column of the joined file, with the stage 1 dtypes, over a short window."""
    rng = np.random.default_rng(seed)
    plan = prepare.read_column_plan()
    raw_columns = [row["column"] for row in plan["columns"]]
    dtypes = data_loader.build_dtype_map(raw_columns)
    ts = np.sort(rng.integers(DAY, 40 * DAY, n)).astype("int64")
    # A few tied timestamps on one card, so the store's tie branch is exercised through HTTP.
    ts[100:103] = ts[100]
    columns: dict[str, Any] = {}
    for column in raw_columns:
        dtype = dtypes[column]
        if column == config.ID_COLUMN:
            values: Any = np.arange(2_987_000, 2_987_000 + n)
        elif column == config.TARGET:
            values = (rng.random(n) < 0.05).astype("int8")
        elif column == config.TIME_COLUMN:
            values = ts
        elif column == config.AMOUNT_COLUMN:
            values = np.round(rng.uniform(1.0, 500.0, n), 2)
        elif column == "card1":
            values = rng.integers(1_000, 1_000 + N_LEVELS, n)
        elif column in config.COUNT_COLUMNS:
            values = rng.integers(0, 30, n)
        elif dtype == "category":
            levels = [f"level{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(N_LEVELS)]
            if column in ("ProductCD",):
                levels = ["W", "C", "H", "R", "S"]
            if column.startswith("M"):
                levels = ["T", "F"]
            picked = rng.choice(levels, n).astype(object)
            picked[rng.random(n) < 0.3] = None
            values = picked
        else:
            base = rng.choice(np.arange(N_LEVELS, dtype="float64") * 3.0, n)
            base[rng.random(n) < 0.3] = np.nan
            values = base
        columns[column] = values
    frame = pd.DataFrame(columns)
    frame.loc[100:102, "card1"] = frame.loc[100, "card1"]
    frame.loc[100:102, "addr1"] = frame.loc[100, "addr1"]
    frame.loc[100:102, "D1"] = 3.0
    for column, dtype in dtypes.items():
        if dtype == "category":
            frame[column] = frame[column].astype("category")
        elif column in config.INTEGER_COLUMN_DTYPES:
            frame[column] = frame[column].astype(dtype)
        elif dtype == "float64":
            frame[column] = frame[column].astype("float64")
        else:
            frame[column] = frame[column].astype("float32")
    return data_loader.add_entity_key(frame)


@pytest.fixture(scope="module")
def stream() -> pd.DataFrame:
    return synthetic_stream(N_ROWS)


@pytest.fixture(scope="module")
def bundle_paths(stream: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The real stage 3 fit and a small booster on the synthetic stream, written as a bundle."""
    plan = prepare.read_column_plan()
    facts = cleaning.EdaFacts.load()
    decisions = prepare.read_decisions()
    cut = int(stream[config.TIME_COLUMN].quantile(0.7))
    train = stream[stream[config.TIME_COLUMN] < cut]
    val = stream[stream[config.TIME_COLUMN] >= cut]
    fitted = prepare.fit_preparation(
        train,
        plan,
        facts,
        d_origin_columns=decisions["d_origin_columns"],
        v_strategy=decisions["v_strategy"],
        lag_days=decisions["lag_days"],
        smoothing=decisions["smoothing"],
    )
    prepared_train = prepare.apply_preparation(train, fitted, plan)
    prepared_val = prepare.apply_preparation(val, fitted, plan)
    columns = modelling.prepared_feature_columns(prepared_train)
    x_train = modelling.tree_matrix(prepared_train, columns).to_numpy(dtype="float32")
    x_val = modelling.tree_matrix(prepared_val, columns).to_numpy(dtype="float32")
    y_train = train[config.TARGET].to_numpy(dtype="int64")
    y_val = val[config.TARGET].to_numpy(dtype="int64")
    model = xgboost.XGBClassifier(
        n_estimators=20, max_depth=3, learning_rate=0.3, random_state=config.SEED, n_jobs=1
    )
    model.fit(x_train, y_train)
    s_val = modelling.score(model, x_val)
    calibrator = evaluation.fit_calibrator(s_val, y_val)
    bands = evaluation.cost_bands()
    declared = schema.declare(prepared_train, facts.negative_value_columns, facts.all_columns())
    serving_json = {
        "model_name": serving.MODEL_NAME,
        "alias": serving.MODEL_ALIAS,
        "family": "xgboost",
        "stack": {"name": "synthetic", "causal": False, "graph": False, "native_groups": []},
        "columns": columns,
        "raw_columns": [row["column"] for row in plan["columns"]],
        "dist_buckets": features.fit_dist_buckets(train),
        "calibrator": {
            "x": [float(v) for v in calibrator.X_thresholds_],
            "y": [float(v) for v in calibrator.y_thresholds_],
        },
        "bands": {"low": bands["low"], "high": bands["high"]},
        "tuned_threshold": {"threshold": 0.5},
        "schema": declared.to_dict(),
    }
    directory = tmp_path_factory.mktemp("bundle")
    return serving.write_bundle(directory, model, fitted, plan, serving_json)


@pytest.fixture(scope="module")
def registry(
    bundle_paths: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = uri
    about = serving.register_bundle(
        bundle_paths, {"family": "xgboost", "n_columns": 0}, uri, root / "artifacts"
    )
    yield about
    if previous is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = previous


@pytest.fixture(scope="module")
def client(registry: dict[str, Any]) -> Iterator[TestClient]:
    with TestClient(service.create_app(service.Settings())) as running:
        yield running


def payload_of(stream: pd.DataFrame, position: int) -> dict[str, Any]:
    record = stream.iloc[position]
    return {
        str(k): (v.item() if isinstance(v, np.generic) else v)
        for k, v in record.items()
        if str(k) not in (config.TARGET, config.ENTITY_ID_COLUMN, config.CARD_START_DAY_COLUMN)
        and not pd.isna(v)
    }


# --- the registry ---------------------------------------------------------------------------------------


def test_the_service_loads_the_model_from_the_registry_by_alias(
    client: TestClient, registry: dict[str, Any]
) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["model"]["version"] == registry["version"] == 1
    assert health["model"]["uri"] == "models:/fraud-detector@production"
    assert health["model"]["run_id"] == registry["run_id"]
    assert health["store"] == "memory"
    assert health["inference_threads"] == serving.INFERENCE_THREADS == 1


def test_an_alias_nobody_registered_refuses_to_start(registry: dict[str, Any]) -> None:
    app = service.create_app(service.Settings(model_alias="nobody-set-this"))
    with pytest.raises(Exception, match="nobody-set-this"), TestClient(app):
        pass


def test_the_bundle_is_pinned_to_one_thread(registry: dict[str, Any]) -> None:
    bundle, _ = serving.load_from_registry()
    assert bundle.model.get_params()["n_jobs"] == 1
    assert bundle.model.get_booster().attributes() is not None


# --- scoring ---------------------------------------------------------------------------------------------


def test_a_row_scores_with_a_band_and_factors(client: TestClient, stream: pd.DataFrame) -> None:
    body = {"transaction": payload_of(stream, 200), "top_factors": 5}
    response = client.post("/score", json=body)
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["TransactionID"] == body["transaction"][config.ID_COLUMN]
    assert 0.0 <= out["score"] <= 1.0 and 0.0 <= out["probability"] <= 1.0
    assert out["band"] in serving.BANDS
    assert len(out["factors"]) == 5
    magnitudes = [abs(f["contribution"]) for f in out["factors"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert "features" not in out
    assert out["latency_ms"] > 0


def test_the_factors_sum_to_the_log_odds(client: TestClient, stream: pd.DataFrame) -> None:
    """Every contribution plus the base value is the logit of the score: TreeSHAP's identity."""
    body = {"transaction": payload_of(stream, 201), "top_factors": service.MAX_FACTORS}
    out = client.post("/score", json=body).json()
    bundle, _ = serving.load_from_registry()
    assert len(out["factors"]) == min(service.MAX_FACTORS, len(bundle.columns))
    frame = serving.raw_frame(body["transaction"], bundle.raw_columns)
    x = serving.matrix(bundle, serving.prepared_row(bundle, frame), None)
    factors, base = serving.explain(bundle, x, len(bundle.columns))
    total = base + sum(f["contribution"] for f in factors)
    assert math.isclose(1.0 / (1.0 + math.exp(-total)), out["score"], rel_tol=1e-5)


def test_the_band_is_the_stage_6_rule_on_the_calibrated_probability(
    client: TestClient, stream: pd.DataFrame
) -> None:
    bundle, _ = serving.load_from_registry()
    for position in range(210, 230):
        out = client.post("/score", json={"transaction": payload_of(stream, position)}).json()
        assert out["band"] == serving.band(bundle, out["probability"])
        assert out["alert"] == (out["score"] >= bundle.tuned_threshold)


def test_the_two_phase_contract_shows_in_the_features(
    client: TestClient, stream: pd.DataFrame
) -> None:
    """The first row on a card sees no history; the next one on the same card sees exactly it."""
    first = payload_of(stream, 300)
    second = dict(first)
    second[config.ID_COLUMN] = first[config.ID_COLUMN] + 100_000
    second[config.TIME_COLUMN] = first[config.TIME_COLUMN] + 60
    one = client.post("/score", json={"transaction": first, "return_features": True}).json()
    two = client.post("/score", json={"transaction": second, "return_features": True}).json()
    assert one["features"]["rec_prior_count"] == 0
    assert one["features"]["vel_count_1h_card"] == 0
    assert two["features"]["rec_prior_count"] == 1
    assert two["features"]["vel_count_1h_card"] == 1
    assert two["features"]["rec_seconds_since_prev"] == 60.0
    assert two["features"]["amt_repeat_within_24h"] == 1.0
    assert set(one["features"]) == set(features.feature_names())


def test_the_service_matches_the_scoring_module_and_the_registry_view(
    client: TestClient, stream: pd.DataFrame
) -> None:
    import mlflow

    payload = payload_of(stream, 400)
    served = client.post("/score", json={"transaction": payload}).json()
    bundle, _ = serving.load_from_registry()
    direct = serving.score_transaction(bundle, payload, None)
    assert served["score"] == direct["score"]
    pyfunc = mlflow.pyfunc.load_model(serving.registry_uri())
    frame = pd.DataFrame([payload])
    assert float(pyfunc.predict(frame)["score"].iloc[0]) == direct["score"]


# --- refusals ------------------------------------------------------------------------------------------------


def test_a_column_the_file_does_not_have_is_refused(
    client: TestClient, stream: pd.DataFrame
) -> None:
    payload = payload_of(stream, 500)
    payload["merchant_id"] = "m-1"
    response = client.post("/score", json={"transaction": payload})
    assert response.status_code == 422
    assert "merchant_id" in response.json()["detail"]


def test_a_label_in_the_payload_is_refused(client: TestClient, stream: pd.DataFrame) -> None:
    payload = payload_of(stream, 501)
    payload[config.TARGET] = 0
    response = client.post("/score", json={"transaction": payload})
    assert response.status_code == 422 and config.TARGET in response.json()["detail"]


def test_a_row_that_breaks_the_schema_is_refused_with_every_violation(
    client: TestClient, stream: pd.DataFrame
) -> None:
    payload = payload_of(stream, 502)
    del payload["C1"]
    payload[config.AMOUNT_COLUMN] = -5.0
    response = client.post("/score", json={"transaction": payload})
    assert response.status_code == 422
    problems = {(v["column"], v["problem"].split(",")[0]) for v in response.json()["detail"]}
    assert ("C1", "dtype family is float") in problems
    assert any(column == config.AMOUNT_COLUMN for column, _ in problems)
    assert len(problems) >= 2


def test_a_value_of_the_wrong_type_is_refused(client: TestClient, stream: pd.DataFrame) -> None:
    payload = payload_of(stream, 503)
    payload["TransactionDT"] = "yesterday"
    response = client.post("/score", json={"transaction": payload})
    assert response.status_code == 422 and "TransactionDT" in response.json()["detail"]


def test_an_older_transaction_on_a_seen_key_is_refused(
    client: TestClient, stream: pd.DataFrame
) -> None:
    payload = payload_of(stream, 600)
    assert client.post("/score", json={"transaction": payload}).status_code == 200
    older = dict(payload)
    older[config.ID_COLUMN] += 1
    older[config.TIME_COLUMN] -= 1
    response = client.post("/score", json={"transaction": older})
    assert response.status_code == 409


def test_an_unknown_request_field_is_refused(client: TestClient, stream: pd.DataFrame) -> None:
    response = client.post("/score", json={"transaction": payload_of(stream, 601), "explain": 1})
    assert response.status_code == 422


# --- the Redis store through the service -----------------------------------------------------------------


def test_the_service_runs_on_a_redis_store(
    registry: dict[str, Any], stream: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fakeredis
    import redis

    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, url, **kw: fake))
    settings = service.Settings(redis_url="redis://fake:6379/0", redis_prefix="svc-test")
    with TestClient(service.create_app(settings)) as running:
        assert running.get("/health").json()["store"].startswith("redis")
        first = payload_of(stream, 700)
        running.post("/score", json={"transaction": first})
        second = dict(first)
        second[config.ID_COLUMN] += 1
        second[config.TIME_COLUMN] += 30
        out = running.post("/score", json={"transaction": second, "return_features": True}).json()
        assert out["features"]["rec_prior_count"] == 1
    assert fake.keys("svc-test:*")
    online_store.RedisBackend(fake, prefix="svc-test").flush()
    assert not fake.keys("svc-test:*")
