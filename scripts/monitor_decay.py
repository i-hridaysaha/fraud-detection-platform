"""Stage 9, the decay job: precision, recall and PR-AUC on every batch whose labels have matured.

Reads labels, and only mature ones. The reference is the shipped model on the test window, the
window it was accepted at in stage 6, bootstrapped on the same seed and resamples as
`reports/metric_variance.json` so the interval here is that artifact's. Every weekly batch of
val and test is scored against it once the clock says its labels are final; a batch the clock
refuses is listed as refused with the day it matures, which is the job's answer to it.

    .venv/bin/python scripts/monitor_decay.py                 # as of the file's end plus maturity
    .venv/bin/python scripts/monitor_decay.py --as-of-day 170 # the cron-shaped call, some refused

Writes reports/monitoring/decay.json. Reads the stage 6 cache, so `make train` runs first.
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

from fraud_platform import config, lifecycle, modelling, monitoring

CACHE_DIR = config.REPORTS_DIR / "cache"
MODELS_DIR = config.ROOT / "models"
OUT_DIR = config.REPORTS_DIR / "monitoring"
REPORT_PATH = OUT_DIR / "decay.json"
OPERATING_POINTS_PATH = config.REPORTS_DIR / "operating_points.json"
VARIANCE_PATH = config.REPORTS_DIR / "metric_variance.json"

WINDOWS: tuple[str, ...] = ("val", "test")
N_BOOTSTRAP = 1000


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
        "section": "decay",
        "stage": 9,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make monitor-decay  # or: .venv/bin/python scripts/monitor_decay.py",
        "command": ".venv/bin/python scripts/monitor_decay.py",
        "reads_labels": True,
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
    for split in WINDOWS:
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


# --- main -------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of-day",
        type=float,
        default=None,
        help="the clock, in days of TransactionDT; default is the file's end plus the maturity",
    )
    parser.add_argument("--maturity-days", type=int, default=lifecycle.MATURITY_DAYS)
    args = parser.parse_args()
    started = time.perf_counter()

    print("loading the stage 6 cache and the shipped model")
    frames = load_cache()
    model, description = load_shipped()
    columns = list(description["columns"])
    scores = {
        split: modelling.score(model, modelling.tree_matrix(frame, columns).to_numpy("float32"))
        for split, frame in frames.items()
    }
    labels = {
        split: frame[config.TARGET].to_numpy(dtype="int64") for split, frame in frames.items()
    }
    points = read(OPERATING_POINTS_PATH)
    threshold = float(points["tuned_vertex"]["val"]["threshold"])

    print("bootstrapping the reference window")
    reference = monitoring.reference_draws(labels["test"], scores["test"], N_BOOTSTRAP, config.SEED)
    variance = read(VARIANCE_PATH)
    recorded = variance["by_row"]["per_model"]["shipped"]
    interval = {
        "point": float(reference["points"]["reference"]),
        "low": float(np.quantile(reference["draws"][:, 0], 0.025)),
        "high": float(np.quantile(reference["draws"][:, 0], 0.975)),
    }
    reference_block = {
        "window": "test",
        "what": "the shipped model on the window it was accepted at, stage 6",
        "n_rows": int(labels["test"].size),
        "n_fraud": int(labels["test"].sum()),
        "pr_auc": interval,
        "stage_6_recorded": {k: recorded[k] for k in ("point", "low", "high", "half_width")},
        "max_abs_difference_vs_stage_6": max(
            abs(interval[k] - float(recorded[k])) for k in ("point", "low", "high")
        ),
        "n_boot": N_BOOTSTRAP,
        "seed": config.SEED,
        "threshold": threshold,
        "threshold_source": "reports/operating_points.json, tuned_vertex.val",
    }

    bounds_by_window = {
        window: monitoring.batch_bounds(
            frames[window][config.TIME_COLUMN].to_numpy(dtype="int64"), monitoring.BATCH_DAYS
        )
        for window in WINDOWS
    }
    # The default clock: the last batch, partial or not, has closed and its labels matured.
    last_close = max(b["end_dt"] for bounds in bounds_by_window.values() for b in bounds)
    as_of = (
        int(args.as_of_day * config.SECONDS_PER_DAY)
        if args.as_of_day is not None
        else monitoring.matures_at(last_close, args.maturity_days)
    )

    print(f"running the job as of day {monitoring.day_index(as_of):.1f}")
    batches: list[dict[str, Any]] = []
    for window in WINDOWS:
        ts = frames[window][config.TIME_COLUMN].to_numpy(dtype="int64")
        for bounds in bounds_by_window[window]:
            mask = monitoring.batch_mask(ts, bounds)
            record: dict[str, Any] = {
                "window": window,
                "position": "before the reference window"
                if window == "val"
                else "inside the reference window",
                **bounds,
                "matures_at_day": monitoring.day_index(
                    monitoring.matures_at(bounds["end_dt"], args.maturity_days)
                ),
            }
            try:
                report = monitoring.performance_report(
                    labels[window][mask],
                    scores[window][mask],
                    threshold,
                    reference,
                    bounds,
                    as_of,
                    args.maturity_days,
                    N_BOOTSTRAP,
                    config.SEED,
                )
            except monitoring.MonitoringError as refused:
                record.update(status="refused", reason=str(refused))
            else:
                record.update(
                    status="evaluated",
                    **{
                        k: report[k]
                        for k in (
                            "n_fraud",
                            "fraud_rate",
                            "precision",
                            "recall",
                            "alerts_per_day",
                            "pr_auc",
                            "roc_auc",
                            "drop",
                        )
                    },
                    decay=lifecycle.decay_flag(report),
                )
            batches.append(record)
            flag = record.get("decay")
            shown = f"pr_auc {record['pr_auc']['point']:.4f}" if flag is not None else "refused"
            print(f"  {window} batch {bounds['batch']}: {record['status']}, {shown}")

    evaluated = [b for b in batches if b["status"] == "evaluated"]
    drops = [float(b["drop"]["point"]) for b in evaluated]
    in_control = {
        "n_batches": len(batches),
        "n_evaluated": len(evaluated),
        "n_refused": len(batches) - len(evaluated),
        "pr_auc_per_batch": [b["pr_auc"]["point"] for b in evaluated],
        "pr_auc_min": min(b["pr_auc"]["point"] for b in evaluated) if evaluated else None,
        "pr_auc_max": max(b["pr_auc"]["point"] for b in evaluated) if evaluated else None,
        "drop_per_batch": drops,
        "largest_drop": max(drops) if drops else None,
        "largest_drop_batch": (
            {k: evaluated[int(np.argmax(drops))][k] for k in ("window", "batch")} if drops else None
        ),
        "n_drops_excluding_zero": sum(1 for b in evaluated if b["drop"]["excludes_zero"]),
        "n_decay": sum(1 for b in evaluated if b["decay"]),
        "decay_tolerance": lifecycle.DECAY_TOLERANCE,
        "headroom": (lifecycle.DECAY_TOLERANCE - max(drops)) if drops else None,
        "precision_range": [
            min(b["precision"] for b in evaluated),
            max(b["precision"] for b in evaluated),
        ]
        if evaluated
        else None,
        "recall_range": [min(b["recall"] for b in evaluated), max(b["recall"] for b in evaluated)]
        if evaluated
        else None,
    }
    report_out = {
        **envelope(),
        "as_of_dt": as_of,
        "as_of_day": monitoring.day_index(as_of),
        "maturity_days": args.maturity_days,
        "maturity_source": "lifecycle.MATURITY_DAYS, an assumed input; ADR 0032 established no delay",
        "batch_days": monitoring.BATCH_DAYS,
        "reference": reference_block,
        "in_control": in_control,
        "batches": batches,
        "seconds": time.perf_counter() - started,
    }
    write(report_out, REPORT_PATH)
    print(json.dumps(_clean({"reference": reference_block, "in_control": in_control}), indent=2))
    print(f"done in {report_out['seconds']:.0f} s")


if __name__ == "__main__":
    main()
