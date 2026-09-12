"""Stage 8: register the shipped model as a serving bundle in the MLflow registry.

One model version, under one alias, holding everything the service needs and nothing the
service can reach any other way: the booster stage 6 shipped, the stage 3 pipeline fitted on
the training split, the stage 3 schema, the stage 6 calibrator as its breakpoints, the cost-band
edges, the tuned threshold and the dist1 bucket edges for the online store.

The script refits what it needs rather than reading the stage 6 cache, and then checks itself
against the committed stage 6 artifacts: the calibrator has to reproduce the raw-score band
edges `reports/operating_points.json` recorded, the serving path has to reproduce the cached
test scores where the cache exists, and the SHAP factors have to reproduce
`reports/shap/shap_example.json`. Every check's result goes into `reports/serving_bundle.json`,
which is the committed record of what was registered and how far it agrees with stage 6.

    .venv/bin/python scripts/register_model.py \\
        --transactions data/train_transaction.csv --identity data/train_identity.csv

Writes reports/serving_bundle.json. The registry itself lives under mlruns/, gitignored: the
model binary is 3.7 MB and the fitted pipeline holds the training split's target tables.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import sklearn
import xgboost

from fraud_platform import (
    cleaning,
    config,
    data_loader,
    encoders,
    evaluation,
    features,
    modelling,
    prepare,
    serving,
)

MODELS_DIR = config.ROOT / "models"
REPORT_PATH = config.REPORTS_DIR / "serving_bundle.json"
OPERATING_POINTS_PATH = config.REPORTS_DIR / "operating_points.json"
SCHEMA_PATH = config.REPORTS_DIR / "prep" / "schema.json"
SHAP_EXAMPLE_PATH = config.REPORTS_DIR / "shap" / "shap_example.json"
CACHE_DIR = config.REPORTS_DIR / "cache"
MLRUNS_DIR = config.ROOT / "mlruns"
EXPERIMENT = "serving"

# Rows of the test split scored through the serving path against the stage 6 cache.
N_SMOKE_ROWS = 500
N_SHAP_FACTORS = 10


def tracking_uri() -> str:
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return serving.tracking_uri()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


# --- the fit -----------------------------------------------------------------------------------------


def fit_pipeline(transactions: str, identity: str) -> dict[str, Any]:
    """The stage 3 fit on train, the dist1 edges, and the three split frames."""
    started = time.perf_counter()
    print("loading the joined frame")
    frame = data_loader.add_entity_key(
        data_loader.load_raw(transactions_path=transactions, identity_path=identity)
    )
    train, val, test = data_loader.time_based_split(frame)
    data_loader.assert_train_only(train, what="stage 8 serving bundle fit")
    plan = prepare.read_column_plan()
    facts = cleaning.EdaFacts.load()
    decisions = prepare.read_decisions()
    print("fitting the stage 3 pipeline on train")
    fitted = prepare.fit_preparation(
        train,
        plan,
        facts,
        d_origin_columns=decisions["d_origin_columns"],
        v_strategy=decisions["v_strategy"],
        lag_days=decisions["lag_days"],
        smoothing=decisions["smoothing"],
    )
    buckets = features.fit_dist_buckets(encoders.add_normalised_free_text(train))
    raw_columns = [
        str(column)
        for column in frame.columns
        if column
        not in (
            config.ENTITY_ID_COLUMN,
            config.CARD_START_DAY_COLUMN,
        )
    ]
    return {
        "plan": plan,
        "decisions": decisions,
        "fitted": fitted,
        "buckets": buckets,
        "raw_columns": raw_columns,
        "splits": {"train": train, "val": val, "test": test},
        "seconds": time.perf_counter() - started,
    }


def load_shipped() -> tuple[xgboost.XGBClassifier, dict[str, Any]]:
    description = read(MODELS_DIR / "shipped_model.json")
    if description["family"] != "xgboost":
        raise SystemExit("the serving bundle is written for the xgboost family stage 6 shipped")
    model = xgboost.XGBClassifier()
    model.load_model(MODELS_DIR / description["model_file"])
    return model, description


# --- the calibrator ---------------------------------------------------------------------------------------


def batch_scores(
    model: xgboost.XGBClassifier,
    fitted: dict[str, Any],
    plan: dict[str, Any],
    columns: list[str],
    part: pd.DataFrame,
) -> np.ndarray:
    prepared = prepare.apply_preparation(part, fitted, plan)
    x = modelling.tree_matrix(prepared, columns).to_numpy(dtype="float32")
    return modelling.score(model, x)


def calibrator_breakpoints(s_val: np.ndarray, y_val: np.ndarray) -> dict[str, Any]:
    """The isotonic fit as its breakpoints, and the check that interpolation reproduces it."""
    calibrator = evaluation.fit_calibrator(s_val, y_val)
    x = np.asarray(calibrator.X_thresholds_, dtype="float64")
    y = np.asarray(calibrator.y_thresholds_, dtype="float64")
    via_sklearn = evaluation.calibrate(calibrator, s_val)
    via_interp = np.interp(s_val, x, y)
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "n_breakpoints": int(x.size),
        "fitted_on": "val",
        "method": "sklearn IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip'), stored "
        "as its breakpoints and applied with np.interp",
        "max_abs_difference_interp_vs_sklearn_on_val": float(
            np.max(np.abs(via_sklearn - via_interp))
        ),
    }


def edges_as_raw_score(
    s_val: np.ndarray, p_val: np.ndarray, bands: dict[str, Any]
) -> dict[str, float]:
    return {name: float(np.min(s_val[p_val >= bands[name]])) for name in ("low", "high")}


# --- the registration --------------------------------------------------------------------------------------


def register(bundle_dir: Path, paths: dict[str, Path], params: dict[str, Any]) -> dict[str, Any]:
    out = serving.register_bundle(paths, params, tracking_uri(), MLRUNS_DIR / "artifacts")
    return {**out, "bundle_dir_at_registration": str(bundle_dir)}


# --- the checks -------------------------------------------------------------------------------------------


def smoke_against_the_cache(
    loaded: serving.ServingBundle, test: pd.DataFrame, s_test: np.ndarray
) -> dict[str, Any]:
    """Score test rows one at a time through the serving path and compare with the batch scores.

    `s_test` is this script's own batch scoring of the test split; the cache from stage 6, where
    it exists, is compared too so the chain back to the committed metrics is closed.
    """
    rng = np.random.default_rng(config.SEED)
    positions = np.sort(rng.choice(len(test), size=N_SMOKE_ROWS, replace=False))
    raw_columns = set(loaded.raw_columns)
    largest = 0.0
    seconds: list[float] = []
    for position in positions:
        record = test.iloc[int(position)]
        payload = {
            str(k): (v.item() if isinstance(v, np.generic) else v)
            for k, v in record.items()
            if str(k) in raw_columns and str(k) != config.TARGET and not pd.isna(v)
        }
        started = time.perf_counter()
        out = serving.score_transaction(loaded, payload, None)
        seconds.append(time.perf_counter() - started)
        largest = max(largest, abs(out["score"] - float(s_test[position])))
    result: dict[str, Any] = {
        "n_rows": int(positions.size),
        "max_abs_score_difference_vs_this_scripts_batch": float(largest),
        "seconds_per_row_median": float(np.median(seconds)),
    }
    cache = CACHE_DIR / "scores_shipped.parquet"
    if cache.exists():
        cached = pd.read_parquet(cache)
        cached_test = cached.loc[cached["split"] == "test", "score"].to_numpy(dtype="float64")
        if cached_test.size == s_test.size:
            result["max_abs_score_difference_batch_vs_stage_6_cache"] = float(
                np.max(np.abs(cached_test - s_test))
            )
            result["stage_6_cache"] = str(cache.relative_to(config.ROOT))
    return result


def shap_against_the_artifact(loaded: serving.ServingBundle, test: pd.DataFrame) -> dict[str, Any]:
    example = read(SHAP_EXAMPLE_PATH)
    txn_id = int(example["transaction"][config.ID_COLUMN])
    row = test[test[config.ID_COLUMN] == txn_id]
    if row.empty:
        return {"transaction_id": txn_id, "found": False}
    record = row.iloc[0]
    payload = {
        str(k): (v.item() if isinstance(v, np.generic) else v)
        for k, v in record.items()
        if str(k) in set(loaded.raw_columns) and str(k) != config.TARGET and not pd.isna(v)
    }
    out = serving.score_transaction(
        loaded, payload, None, top_factors=len(example["contributions"])
    )
    expected = {c["column"]: float(c["shap"]) for c in example["contributions"]}
    got = {f["feature"]: f["contribution"] for f in out["factors"]}
    shared = [name for name in expected if name in got]
    largest = max(abs(expected[name] - got[name]) for name in shared) if shared else None
    return {
        "transaction_id": txn_id,
        "found": True,
        "score_serving": out["score"],
        "score_stage_6": float(example["transaction"]["score"]),
        "base_value_serving": out["base_value"],
        "expected_value_stage_6": float(example["expected_value"]),
        "n_factors_compared": len(shared),
        "n_factors_in_artifact": len(expected),
        "max_abs_contribution_difference": largest,
        "top_factor_serving": out["factors"][0]["feature"],
        "top_factor_stage_6": example["contributions"][0]["column"],
    }


# --- main --------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    args = parser.parse_args()
    started = time.perf_counter()

    pipeline = fit_pipeline(args.transactions, args.identity)
    model, description = load_shipped()
    columns = list(description["columns"])
    fitted, plan = pipeline["fitted"], pipeline["plan"]
    splits = pipeline["splits"]

    print("scoring val and test through the batch path")
    s_val = batch_scores(model, fitted, plan, columns, splits["val"])
    s_test = batch_scores(model, fitted, plan, columns, splits["test"])
    y_val = splits["val"][config.TARGET].to_numpy(dtype="int64")
    calibrator = calibrator_breakpoints(s_val, y_val)
    p_val = np.interp(s_val, calibrator["x"], calibrator["y"])
    bands = evaluation.cost_bands()
    points = read(OPERATING_POINTS_PATH)
    raw_edges = edges_as_raw_score(s_val, p_val, bands)
    recorded_edges = points["bands"]["edges_as_raw_score"]

    schema_report = read(SCHEMA_PATH)
    serving_json = {
        "model_name": serving.MODEL_NAME,
        "alias": serving.MODEL_ALIAS,
        "family": description["family"],
        "stack": description["stack"],
        "hyperparameters": description["hyperparameters"],
        "columns": columns,
        "n_columns": len(columns),
        "raw_columns": pipeline["raw_columns"],
        "dist_buckets": pipeline["buckets"],
        "decisions": pipeline["decisions"],
        "calibrator": calibrator,
        "bands": {
            "low": bands["low"],
            "high": bands["high"],
            "on": "calibrated probability",
            "costs": bands["costs"],
            "source": "evaluation.cost_bands() from config, as reports/operating_points.json",
        },
        "tuned_threshold": {
            "threshold": points["tuned_vertex"]["val"]["threshold"],
            "on": "raw score",
            "rule": points["tuned_vertex"]["val"]["rule"],
            "source": "reports/operating_points.json, tuned_vertex.val",
        },
        "schema": schema_report["schema"],
        "schema_source": str(SCHEMA_PATH.relative_to(config.ROOT)),
        "inference_threads": serving.INFERENCE_THREADS,
        "stage_6_git_commit": description["git_commit"],
        "registered_git_commit": git_commit(),
        "registered_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    params = {
        "family": description["family"],
        "stack": description["stack"]["name"],
        "n_columns": len(columns),
        "stage_6_git_commit": description["git_commit"],
        "git_commit": git_commit(),
        "inference_threads": serving.INFERENCE_THREADS,
        "calibrator_breakpoints": calibrator["n_breakpoints"],
        "band_low": bands["low"],
        "band_high": bands["high"],
    }

    bundle_dir = Path(tempfile.mkdtemp(prefix="serving-bundle-"))
    paths = serving.write_bundle(bundle_dir, model, fitted, plan, _clean(serving_json))
    print("registering")
    registry = register(bundle_dir, paths, params)
    print(f"  {registry['uri']} is version {registry['version']} (run {registry['run_id']})")

    print("loading back through the registry and checking")
    loaded, about = serving.load_from_registry()
    smoke = smoke_against_the_cache(loaded, splits["test"], s_test)
    shap_check = shap_against_the_artifact(loaded, splits["test"])
    report = {
        "section": "serving_bundle",
        "stage": 8,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make register  # or: .venv/bin/python scripts/register_model.py",
        "command": " ".join(sys.argv),
        "n_train_rows": len(splits["train"]),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "mlflow": __import__("mlflow").__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
        "registry": {**registry, "loaded_back": about},
        "bundle": {
            "files": {name: path.name for name, path in paths.items()},
            "bytes": {name: path.stat().st_size for name, path in paths.items()},
            "columns": len(columns),
            "raw_columns": len(pipeline["raw_columns"]),
            "schema_columns_declared": len(schema_report["schema"]["columns"]),
            "schema_columns_serving": len(loaded.schema.columns),
            "stack": description["stack"],
            "fit_seconds": pipeline["seconds"],
        },
        "calibrator": {k: v for k, v in calibrator.items() if k not in ("x", "y")},
        "band_edges": {
            "on_probability": {"low": bands["low"], "high": bands["high"]},
            "as_raw_score_this_fit": raw_edges,
            "as_raw_score_stage_6": recorded_edges,
            "max_abs_difference": max(
                abs(raw_edges[name] - float(recorded_edges[name])) for name in ("low", "high")
            ),
        },
        "smoke": smoke,
        "shap": shap_check,
        "seconds": time.perf_counter() - started,
    }
    REPORT_PATH.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")
    print(f"wrote {REPORT_PATH.relative_to(config.ROOT)} in {report['seconds']:.0f} s")
    print(
        json.dumps(
            _clean({"band_edges": report["band_edges"], "smoke": smoke, "shap": shap_check}),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
