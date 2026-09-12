"""Stage 9, the drift job: build the reference, calibrate its edges, run it over the accepted windows.

Reads no labels. The reference is the shipped model's 186 input columns and its scores on the
training split, as decile histograms; the edges are calibrated on the weekly batches of the val
and test windows, which are the windows the model was accepted on and therefore what normal
drift looks like at the monitoring grain. Then the job itself runs over every one of those
batches, and the artifact records what it would have said each week, what the textbook 0.20
band would have said, and the leave-one-out false alert rate of the calibrated rule.

    .venv/bin/python scripts/monitor_drift.py            # the reference and the sweep
    .venv/bin/python scripts/monitor_drift.py --batch 6  # one batch, the cron-shaped call

Writes reports/monitoring/drift_reference.json and reports/monitoring/drift.json. Reads the
stage 6 cache, so `make train` runs first.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import xgboost

from fraud_platform import config, modelling, monitoring

CACHE_DIR = config.REPORTS_DIR / "cache"
MODELS_DIR = config.ROOT / "models"
OUT_DIR = config.REPORTS_DIR / "monitoring"
REFERENCE_PATH = OUT_DIR / "drift_reference.json"
REPORT_PATH = OUT_DIR / "drift.json"
TEMPORAL_PATH = config.REPORTS_DIR / "eda" / "temporal.json"

# The in-control windows: val and test are the windows the shipped model was accepted on.
CALIBRATION_WINDOWS: tuple[str, ...] = ("val", "test")
N_TOP = 10


# --- scaffolding -----------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def envelope() -> dict[str, Any]:
    return {
        "section": "drift",
        "stage": 9,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make monitor-drift  # or: .venv/bin/python scripts/monitor_drift.py",
        "command": ".venv/bin/python scripts/monitor_drift.py",
        "reads_labels": False,
        "split": {
            "train_end_dt": config.TRAIN_END_DT,
            "val_end_dt": config.VAL_END_DT,
            "rule": "dt < boundary, ADR 0002",
        },
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def write(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")
    print(f"  wrote {path.relative_to(config.ROOT)}")


def read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path.relative_to(config.ROOT)} does not exist; run `make train` first")
    return dict(json.loads(path.read_text()))


# --- inputs -----------------------------------------------------------------------------------


def load_cache() -> dict[str, pd.DataFrame]:
    frames = {}
    for split in config.SPLIT_NAMES:
        path = CACHE_DIR / f"prepared_{split}.parquet"
        if not path.exists():
            raise SystemExit("reports/cache is missing; run `make train` first")
        frames[split] = pd.read_parquet(path)
    return frames


def load_shipped() -> tuple[xgboost.XGBClassifier, dict[str, Any]]:
    description = read(MODELS_DIR / "shipped_model.json")
    model = xgboost.XGBClassifier()
    model.load_model(MODELS_DIR / description["model_file"])
    return model, description


def score_all(
    model: xgboost.XGBClassifier, frames: Mapping[str, pd.DataFrame], columns: list[str]
) -> dict[str, np.ndarray]:
    return {
        split: modelling.score(model, modelling.tree_matrix(frame, columns).to_numpy("float32"))
        for split, frame in frames.items()
    }


def against_the_cache(scores: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """The val and test scores recomputed here against the stage 6 cache, the chain back."""
    path = CACHE_DIR / "scores_shipped.parquet"
    if not path.exists():
        return {"cache": None}
    cached = pd.read_parquet(path)
    out: dict[str, Any] = {"cache": str(path.relative_to(config.ROOT))}
    for split in ("val", "test"):
        got = cached.loc[cached["split"] == split, "score"].to_numpy(dtype="float64")
        out[f"max_abs_difference_{split}"] = (
            float(np.max(np.abs(got - scores[split]))) if got.size == scores[split].size else None
        )
    return out


# --- the stage 2 cross-check ----------------------------------------------------------------------


def against_stage_2(reference: monitoring.DriftReference) -> dict[str, Any]:
    """Whole-test-window PSI here against reports/eda/temporal.json for the columns both hold.

    Stage 2 binned the raw columns; the model reads some of them unchanged, and on those the
    two numbers have to agree. A column the pipeline transforms is not compared.
    """
    if not TEMPORAL_PATH.exists():
        return {"artifact": None}
    temporal = read(TEMPORAL_PATH)
    stage_2 = {row["column"]: row for row in temporal["psi"]["per_feature"]}
    whole = reference.calibration["whole_window_psi"]
    rows = []
    for column in reference.columns:
        if column not in stage_2:
            continue
        rows.append(
            {
                "column": column,
                "psi_test_here": float(whole["test"][column]),
                "psi_test_stage_2": float(stage_2[column]["psi_test"]),
                "psi_val_here": float(whole["val"][column]),
                "psi_val_stage_2": float(stage_2[column]["psi_val"]),
            }
        )
    differences = [abs(r["psi_test_here"] - r["psi_test_stage_2"]) for r in rows]
    exact = sum(1 for d in differences if d < 1e-9)
    return {
        "artifact": str(TEMPORAL_PATH.relative_to(config.ROOT)),
        "n_shared_columns": len(rows),
        "n_exact_to_1e_9": exact,
        "max_abs_difference_test": max(differences) if differences else None,
        "not_exact": [r for r, d in zip(rows, differences, strict=True) if d >= 1e-9],
        "stage_2_summary": temporal["psi"]["summary"],
    }


# --- the sweep -------------------------------------------------------------------------------------


def one_batch(
    reference: monitoring.DriftReference,
    frame: pd.DataFrame,
    scores: np.ndarray,
    bounds: Mapping[str, Any],
    window: str,
) -> dict[str, Any]:
    ts = frame[config.TIME_COLUMN].to_numpy(dtype="int64")
    mask = monitoring.batch_mask(ts, bounds)
    part = frame.loc[mask, [*reference.columns, config.TIME_COLUMN]]
    report = monitoring.drift_report(reference, part, scores[mask], {"window": window, **bounds})
    top = sorted(report["per_feature"], key=lambda f: -f["psi"])[:N_TOP]
    return {
        "window": window,
        **bounds,
        "score": report["score"],
        "aggregate": report["aggregate"],
        "drift_alert": report["drift_alert"],
        "top_psi": [
            {k: f[k] for k in ("column", "psi", "band", "alert_edge", "alert")} for f in top
        ],
    }


def sweep(
    reference: monitoring.DriftReference,
    frames: Mapping[str, pd.DataFrame],
    scores: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    out = []
    for window in CALIBRATION_WINDOWS:
        ts = frames[window][config.TIME_COLUMN].to_numpy(dtype="int64")
        for bounds in monitoring.batch_bounds(ts, reference.batch_days):
            out.append(one_batch(reference, frames[window], scores[window], bounds, window))
    return out


def longest_run(flags: list[bool]) -> int:
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


# --- main -------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=None, help="score one batch of the sweep")
    args = parser.parse_args()
    started = time.perf_counter()

    print("loading the stage 6 cache and the shipped model")
    frames = load_cache()
    model, description = load_shipped()
    columns = list(description["columns"])
    scores = score_all(model, frames, columns)
    cache_check = against_the_cache(scores)

    print("building the reference on train and calibrating on val and test")
    reference = monitoring.build_reference(
        frames["train"],
        scores["train"],
        columns,
        {name: (frames[name], scores[name]) for name in CALIBRATION_WINDOWS},
        built_from={
            "model": str((MODELS_DIR / description["model_file"]).relative_to(config.ROOT)),
            "columns_from": "models/shipped_model.json",
            "prepared_from": "reports/cache/prepared_train.parquet",
            "n_train_rows": len(frames["train"]),
            "git_commit": git_commit(),
        },
    )
    if args.batch is not None:
        batches = sweep(reference, frames, scores)
        print(json.dumps(_clean(batches[args.batch]), indent=2))
        return
    write(reference.to_dict(), REFERENCE_PATH)

    print("running the job over every in-control batch")
    batches = sweep(reference, frames, scores)
    loo = monitoring.leave_one_out_false_alerts(reference)
    edges = reference.alert_edges
    raised = {c: e for c, e in edges.items() if e > monitoring.PSI_ALERT}
    worst = reference.calibration["worst_batch_psi"]
    calibration_summary = {
        "n_columns": len(columns),
        "n_batches": len(batches),
        "batch_days": reference.batch_days,
        "n_alertable_columns": len(reference.alertable_columns),
        "not_alertable": [c for c in columns if not monitoring.alertable(c)],
        "n_edges_raised_above_textbook": len(raised),
        "n_alertable_edges_raised": sum(1 for c in raised if monitoring.alertable(c)),
        "edges_raised": dict(sorted(raised.items(), key=lambda kv: -kv[1])),
        "feature_count_edge": reference.feature_count_edge,
        "leave_one_out_counts": reference.calibration["leave_one_out_counts"],
        "score_alert_edge": reference.score_alert_edge,
        "worst_batch_score_psi": reference.calibration["worst_batch_score_psi"],
        "whole_window_score_psi": reference.calibration["whole_window_score_psi"],
        "median_worst_batch_psi": float(np.median(list(worst.values()))),
        "n_columns_worst_batch_above_0_10": sum(1 for v in worst.values() if v >= 0.10),
        "n_columns_worst_batch_above_0_20": sum(1 for v in worst.values() if v >= 0.20),
        "n_columns_whole_test_above_0_10": sum(
            1 for v in reference.calibration["whole_window_psi"]["test"].values() if v >= 0.10
        ),
        "n_columns_whole_test_above_0_20": sum(
            1 for v in reference.calibration["whole_window_psi"]["test"].values() if v >= 0.20
        ),
        "n_columns_whole_test_above_0_25": sum(
            1 for v in reference.calibration["whole_window_psi"]["test"].values() if v >= 0.25
        ),
        "median_whole_test_psi": float(
            np.median(list(reference.calibration["whole_window_psi"]["test"].values()))
        ),
    }
    in_control = {
        "textbook_alert_band_per_batch": [b["aggregate"]["n_alert_band"] for b in batches],
        "calibrated_alerts_per_batch": [b["aggregate"]["n_alert"] for b in batches],
        "feature_alert_per_batch": [b["aggregate"]["feature_alert"] for b in batches],
        "score_alert_per_batch": [b["score"]["alert"] for b in batches],
        "score_psi_per_batch": [b["score"]["psi"] for b in batches],
        "drift_alert_per_batch": [b["drift_alert"] for b in batches],
        "longest_run_of_drift_alerts": longest_run([b["drift_alert"] for b in batches]),
        "leave_one_out": {k: v for k, v in loo.items() if k != "batches"},
        "leave_one_out_longest_run": longest_run([b["any_alert"] for b in loo["batches"]]),
        "leave_one_out_batches": loo["batches"],
    }
    report = {
        **envelope(),
        "reference": str(REFERENCE_PATH.relative_to(config.ROOT)),
        "n_train_rows": len(frames["train"]),
        "scores_against_stage_6_cache": cache_check,
        "against_stage_2": against_stage_2(reference),
        "calibration": calibration_summary,
        "in_control": in_control,
        "batches": batches,
        "seconds": time.perf_counter() - started,
    }
    write(report, REPORT_PATH)
    print(
        json.dumps(
            _clean(
                {
                    "calibration": {
                        k: v for k, v in calibration_summary.items() if k != "edges_raised"
                    },
                    "in_control": {
                        k: v for k, v in in_control.items() if k not in ("leave_one_out_batches",)
                    },
                    "against_stage_2": {
                        k: v
                        for k, v in report["against_stage_2"].items()
                        if k not in ("not_exact", "stage_2_summary")
                    },
                }
            ),
            indent=2,
        )
    )
    print(f"done in {report['seconds']:.0f} s")


if __name__ == "__main__":
    main()
