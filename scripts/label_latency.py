"""Stage 7, experiment 2: simulated label latency.

A chargeback arrives weeks to months after the transaction, so the most recent rows of any
training set have labels that are not yet final: a transaction that has not been charged back
yet is not confidently legitimate. This file carries no chargeback dates, so the latency is
simulated, and the simulation is the simplest one that has the failure in it: at training
time, every fraud in the final N days of the training window reads as legitimate.

Two ways to train at a latency of N days, against the reference that this file happens to make
available because its labels are mature:

  immature   fit on every train row with the immature labels, treating them as final
  excluded   fit on the train rows more than N days before the training boundary

The target encoder is refitted on the same labels the model sees in both cases, so nothing at
training time reads a label that has not arrived. N is swept over 15, 30, 45, 60 and 90 days.
What is measured is test PR-AUC, with a paired bootstrap interval against the reference, and the
direction and size of the bias in the fraud rate the model implies. What is not measured, and
cannot be from this file, is the real chargeback delay: the finding is the shape of the bias and
its sensitivity to N, not a correction.

Reads the stage 6 cache under reports/cache/ (`make train` builds it). One artifact,
reports/label_latency.json, from `make latency`.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
import xgboost

from fraud_platform import config, encoders, evaluation, modelling, prepare

CACHE_DIR = config.REPORTS_DIR / "cache"
OUT_PATH = config.REPORTS_DIR / "label_latency.json"
BUNDLE_PATH = config.ROOT / "models" / "shipped_model.json"
OPERATING_POINTS_PATH = config.REPORTS_DIR / "operating_points.json"
VARIANCE_PATH = config.REPORTS_DIR / "metric_variance.json"

N_BOOTSTRAP = 1_000
FAMILY = "xgboost"
IMBALANCE = "none"

# The latencies swept, in days before the training boundary. Chosen to bracket the weeks to
# months a chargeback takes; nothing in this repo measures where in that range the truth sits.
LATENCY_DAYS: tuple[int, ...] = (15, 30, 45, 60, 90)

TREATMENTS: tuple[str, ...] = ("immature", "excluded")

# The latency at which the regime diagnostic refits the immature model with a block removed,
# and the blocks it removes. 60 is the first swept value at which the immature model's implied
# rate collapses; the blocks are the two a reader would name first as carrying a clock.
REGIME_ABLATION_LATENCY_DAYS = 60
REGIME_ABLATION_BLOCKS: dict[str, str] = {
    "target_encodings": "every column ending in encoders.TARGET_SUFFIX",
    "timedelta_columns": "every raw D column",
    "both": "the two blocks together",
}


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
        "section": "label_latency",
        "stage": 7,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make latency  # or: .venv/bin/python scripts/label_latency.py",
        "command": ".venv/bin/python scripts/label_latency.py",
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


def target_encoded_sources(columns: Sequence[str]) -> list[str]:
    """The raw columns behind the `_te` columns the shipped model reads."""
    suffix = encoders.TARGET_SUFFIX
    return sorted(c[: -len(suffix)] for c in columns if c.endswith(suffix))


def refit_target_encoding(
    frames: Mapping[str, pd.DataFrame],
    fit_rows: npt.NDArray[np.bool_],
    labels: npt.NDArray[np.int64],
    sources: Sequence[str],
    decisions: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """The three frames with their `_te` columns recomputed from one label vector.

    The encoder is stage 3's, fitted through the guard on the train rows `fit_rows` selects with
    `labels` in place of the file's, at the recorded lag and smoothing. Every other column is
    left as the cache holds it: nothing else in the shipped stack reads a label.
    """
    train = frames["train"].loc[fit_rows].copy()
    train[config.TARGET] = labels[fit_rows].astype("int8")
    fitted = encoders.fit_target_encoding(
        train,
        list(sources),
        lag_days=int(decisions["lag_days"]),
        smoothing=float(decisions["smoothing"]),
    )
    return {split: encoders.apply_target_encoding(frame, fitted) for split, frame in frames.items()}


# --- one fit ----------------------------------------------------------------------------------


def fit_and_score(
    frames: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    fit_rows: npt.NDArray[np.bool_],
    labels: npt.NDArray[np.int64],
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """One xgboost fit on the selected train rows with the given labels, scored on val and test."""
    matrices = {split: modelling.tree_matrix(frame, columns) for split, frame in frames.items()}
    x_fit = matrices["train"].to_numpy(dtype="float32")[fit_rows]
    y_fit = labels[fit_rows]
    model = modelling.make_model(FAMILY, IMBALANCE, y_fit, config.SEED, hyperparameters)
    started = time.perf_counter()
    model.fit(x_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    scores = {
        split: modelling.score(model, matrices[split].to_numpy(dtype="float32"))
        for split in ("train", "val", "test")
    }
    importance = gain_by_column(model, columns)
    del model, matrices, x_fit
    gc.collect()
    return {
        "n_fit_rows": int(y_fit.size),
        "n_fit_positive": int(y_fit.sum()),
        "fit_positive_rate": float(y_fit.mean()),
        "fit_seconds": fit_seconds,
        "importance": importance,
        "scores": scores,
    }


def gain_by_column(model: Any, columns: Sequence[str]) -> dict[str, Any]:
    """The ten largest xgboost gains and the share the target encodings take between them.

    A target encoding read at the row's own lagged day is also a clock: its value for a common
    level moves with the cumulative count, so a model that has learned that recent rows are
    never fraud can find the recent rows through it. The share is recorded per fit so the
    mechanism is measured rather than argued.
    """
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    names = booster.feature_names or [f"f{i}" for i in range(len(columns))]
    per_column = {
        column: float(gain.get(name, 0.0)) for column, name in zip(columns, names, strict=True)
    }
    total = sum(per_column.values()) or 1.0
    ranked = sorted(per_column.items(), key=lambda kv: -kv[1])
    te = [c for c in per_column if c.endswith(encoders.TARGET_SUFFIX)]
    return {
        "kind": "xgboost gain, mean loss reduction over the splits on the column",
        "top_10": [{"column": c, "gain": g, "gain_share": g / total} for c, g in ranked[:10]],
        "target_encoding_gain_share": sum(per_column[c] for c in te) / total,
        "n_target_encoding_columns": len(te),
    }


def metrics(
    frames: Mapping[str, pd.DataFrame],
    scores: Mapping[str, npt.NDArray[np.float64]],
    threshold: float,
    n_days: Mapping[str, float],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        y = frames[split][config.TARGET].to_numpy(dtype="int64")
        s = scores[split]
        observed = float(y.mean())
        implied = evaluation.mean_score(y, s)
        out[split] = {
            "pr_auc": evaluation.pr_auc(y, s),
            "roc_auc": evaluation.roc_auc(y, s),
            "observed_fraud_rate": observed,
            "implied_fraud_rate": implied,
            "implied_minus_observed": implied - observed,
            "implied_over_observed": implied / observed,
            "at_shipped_threshold": evaluation.at_threshold(y, s, threshold, n_days[split]),
        }
    return out


def regime_summary(
    scores: Mapping[str, npt.NDArray[np.float64]], recent: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    """The mean score a fit gives the older train rows, the recent train rows, and the later
    windows. A model that has learned the recent window as a regime of its own scores the recent
    rows and the test rows alike and the older rows differently; the three numbers say whether
    it has."""
    train = scores["train"]
    return {
        "train_older_mean_score": float(train[~recent].mean()) if (~recent).any() else None,
        "train_recent_mean_score": float(train[recent].mean()) if recent.any() else None,
        "val_mean_score": float(scores["val"].mean()),
        "test_mean_score": float(scores["test"].mean()),
    }


def block_columns(columns: Sequence[str], block: str) -> list[str]:
    te = {c for c in columns if c.endswith(encoders.TARGET_SUFFIX)}
    d = {c for c in columns if c in config.TIMEDELTA_COLUMNS}
    dropped = {"target_encodings": te, "timedelta_columns": d, "both": te | d}[block]
    return [c for c in columns if c not in dropped]


# --- the experiment ---------------------------------------------------------------------------


def run(n_boot: int) -> None:
    started = time.perf_counter()
    bundle = read(BUNDLE_PATH)
    columns: list[str] = list(bundle["columns"])
    hyperparameters: dict[str, Any] = dict(bundle["hyperparameters"])
    if bundle["family"] != FAMILY or bundle["imbalance"] != IMBALANCE:
        raise SystemExit("this script is written for the shipped xgboost model without reweighting")
    points = read(OPERATING_POINTS_PATH)
    shipped_threshold = float(points["tuned_vertex"]["val"]["threshold"])
    decisions = prepare.read_decisions()

    print("loading the stage 6 cache")
    frames = load_cache()
    for split, frame in frames.items():
        print(f"  {split}: {len(frame):,} rows by {frame.shape[1]}")
    ts = frames["train"][config.TIME_COLUMN].to_numpy(dtype="int64")
    y_true = frames["train"][config.TARGET].to_numpy(dtype="int64")
    n_days = {
        split: evaluation.span_days(frame[config.TIME_COLUMN].to_numpy(dtype="int64"))
        for split, frame in frames.items()
    }
    sources = target_encoded_sources(columns)
    missing = [c for c in sources if c not in frames["train"].columns]
    if missing:
        raise SystemExit(f"the cache does not carry the raw columns {missing}")
    all_rows = np.ones(len(ts), dtype=bool)

    # The reference goes through the same refit as every treatment, on the file's labels, and
    # the refit must reproduce the cache's own encodings before anything else is trusted.
    print("reference: the target encoder refitted on the file's labels must match the cache")
    reference_frames = refit_target_encoding(frames, all_rows, y_true, sources, decisions)
    te_columns = [encoders.target_name(c) for c in sources]
    refit_gap = max(
        float(
            np.nanmax(
                np.abs(
                    reference_frames[split][te_columns].to_numpy(dtype="float64")
                    - frames[split][te_columns].to_numpy(dtype="float64")
                )
            )
        )
        for split in config.SPLIT_NAMES
    )
    print(f"  largest difference against the cached encodings: {refit_gap:.2e}")
    if refit_gap > 1e-9:
        raise SystemExit("the refit does not reproduce the cached target encodings")

    runs: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
    regime_ablations: list[dict[str, Any]] = []

    print("fitting: reference")
    fitted = fit_and_score(reference_frames, columns, all_rows, y_true, hyperparameters)
    scores["reference"] = fitted.pop("scores")
    runs["reference"] = {
        "id": "reference",
        "treatment": "reference",
        "latency_days": 0,
        "n_recent_rows": 0,
        "n_recent_positive": 0,
        **fitted,
        **metrics(reference_frames, scores["reference"], shipped_threshold, n_days),
        "regime": regime_summary(scores["reference"], np.zeros(len(ts), dtype=bool)),
    }
    print(f"  test PR-AUC {runs['reference']['test']['pr_auc']:.4f}")
    del reference_frames
    gc.collect()

    for latency in LATENCY_DAYS:
        recent = ts >= config.TRAIN_END_DT - latency * config.SECONDS_PER_DAY
        n_recent = int(recent.sum())
        n_recent_positive = int(y_true[recent].sum())
        for treatment in TREATMENTS:
            run_id = f"{treatment}_{latency:02d}"
            print(
                f"fitting: {run_id} ({n_recent:,} recent rows, {n_recent_positive:,} of them fraud)"
            )
            if treatment == "immature":
                labels = y_true.copy()
                labels[recent] = 0
                fit_rows = all_rows
            else:
                labels = y_true
                fit_rows = ~recent
            treated = refit_target_encoding(frames, fit_rows, labels, sources, decisions)
            fitted = fit_and_score(treated, columns, fit_rows, labels, hyperparameters)
            scores[run_id] = fitted.pop("scores")
            runs[run_id] = {
                "id": run_id,
                "treatment": treatment,
                "latency_days": latency,
                "recent_rule": f"TransactionDT >= {config.TRAIN_END_DT} - {latency} * {config.SECONDS_PER_DAY}",
                "n_recent_rows": n_recent,
                "n_recent_positive": n_recent_positive,
                "recent_positive_share_of_train": n_recent_positive / int(y_true.sum()),
                **fitted,
                **metrics(treated, scores[run_id], shipped_threshold, n_days),
                "regime": regime_summary(scores[run_id], recent),
            }
            entry = runs[run_id]
            print(
                f"  test PR-AUC {entry['test']['pr_auc']:.4f}, implied fraud rate "
                f"{entry['test']['implied_fraud_rate']:.4f} against {entry['test']['observed_fraud_rate']:.4f}"
            )
            if treatment == "immature" and latency == REGIME_ABLATION_LATENCY_DAYS:
                ablations = []
                for block, description in REGIME_ABLATION_BLOCKS.items():
                    kept = block_columns(columns, block)
                    print(f"  regime ablation: {block} dropped ({len(kept)} columns)")
                    refit = fit_and_score(treated, kept, fit_rows, labels, hyperparameters)
                    ablations.append(
                        {
                            "dropped": block,
                            "description": description,
                            "n_columns": len(kept),
                            "test_pr_auc": evaluation.pr_auc(
                                frames["test"][config.TARGET].to_numpy(dtype="int64"),
                                refit["scores"]["test"],
                            ),
                            **regime_summary(refit["scores"], recent),
                        }
                    )
                regime_ablations = ablations
            del treated
            gc.collect()

    print("bootstrapping on the test rows")
    y_test = frames["test"][config.TARGET].to_numpy(dtype="int64")
    test_scores = {name: s["test"] for name, s in scores.items()}
    bootstrap: dict[str, Any] = {}
    for metric in ("pr_auc", "roc_auc", "mean_score"):
        print(f"  {metric} ({len(test_scores)} models)")
        bootstrap[metric] = evaluation.paired_bootstrap(
            y_test, test_scores, n_boot, config.SEED, metric=metric
        )

    def against_reference(run_id: str, metric: str) -> dict[str, Any]:
        pair = next(
            p
            for p in bootstrap[metric]["pairs"]
            if {p["first"], p["second"]} == {run_id, "reference"}
        )
        difference = dict(pair["difference"])
        if pair["first"] == "reference":
            difference = {
                "point": -difference["point"],
                "low": -difference["high"],
                "high": -difference["low"],
                "half_width": difference["half_width"],
                "bootstrap_mean": -difference["bootstrap_mean"],
            }
        return {
            "run": bootstrap[metric]["per_model"][run_id],
            "reference": bootstrap[metric]["per_model"]["reference"],
            "difference": difference,
            "sign_agreement": pair["sign_agreement"],
            "excludes_zero": pair["excludes_zero"],
        }

    sweep = []
    for latency in LATENCY_DAYS:
        row: dict[str, Any] = {"latency_days": latency}
        for treatment in TREATMENTS:
            run_id = f"{treatment}_{latency:02d}"
            row[treatment] = {
                "id": run_id,
                "n_fit_rows": runs[run_id]["n_fit_rows"],
                "n_fit_positive": runs[run_id]["n_fit_positive"],
                "test_pr_auc": against_reference(run_id, "pr_auc"),
                "test_roc_auc": against_reference(run_id, "roc_auc"),
                "test_implied_fraud_rate": against_reference(run_id, "mean_score"),
                "test_implied_minus_observed": runs[run_id]["test"]["implied_minus_observed"],
                "test_recall_at_shipped_threshold": runs[run_id]["test"]["at_shipped_threshold"][
                    "recall"
                ],
            }
        immature_pair = next(
            p
            for p in bootstrap["pr_auc"]["pairs"]
            if {p["first"], p["second"]} == {f"immature_{latency:02d}", f"excluded_{latency:02d}"}
        )
        sign = 1.0 if immature_pair["first"].startswith("immature") else -1.0
        row["immature_minus_excluded_test_pr_auc"] = {
            "point": sign * immature_pair["difference"]["point"],
            "low": min(
                sign * immature_pair["difference"]["low"],
                sign * immature_pair["difference"]["high"],
            ),
            "high": max(
                sign * immature_pair["difference"]["low"],
                sign * immature_pair["difference"]["high"],
            ),
            "excludes_zero": immature_pair["excludes_zero"],
        }
        sweep.append(row)

    report = envelope()
    report["simulation"] = {
        "statement": "the label latency is simulated. This file carries no chargeback dates and "
        "the true chargeback delay distribution is not known here; the labels in the file are "
        "treated as mature, which is what makes the reference available. The simulation is a "
        "deterministic latency of N days: at training time every fraud in the final N days of "
        "the training window reads as legitimate",
        "latency_days": list(LATENCY_DAYS),
        "treatments": {
            "immature": "fit on every train row, the final N days' fraud labels read as 0",
            "excluded": "fit on the train rows more than N days before the training boundary",
            "reference": "fit on every train row with the file's labels: the shipped configuration",
        },
        "target_encoding": "refitted on the same rows and labels the model fits on, at the "
        "stage 3 lag and smoothing, so nothing at training time reads a label the simulation "
        "says has not arrived. Every other column is the stage 6 cache's",
        "finding_is": "the shape of the bias and its sensitivity to N, not a correction",
    }
    report["model"] = {
        "family": FAMILY,
        "imbalance": IMBALANCE,
        "stack": bundle["stack"],
        "n_columns": len(columns),
        "hyperparameters": hyperparameters,
        "hyperparameters_are_module_defaults": hyperparameters
        == modelling.DEFAULT_HYPERPARAMETERS[FAMILY],
        "seed": config.SEED,
        "target_encoded_sources": sources,
        "lag_days": int(decisions["lag_days"]),
        "smoothing": float(decisions["smoothing"]),
        "refit_reproduces_cache_within": refit_gap,
    }
    report["shipped_model_test_pr_auc"] = points["primary_metric"]["test"]["pr_auc"]
    report["shipped_threshold"] = {
        "value": shipped_threshold,
        "source": "reports/operating_points.json, tuned_vertex.val.threshold, on the raw score",
    }
    report["stage_6_noise_band"] = read(VARIANCE_PATH)["noise_band"]
    report["train_window"] = {
        "n_rows": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "span_days": n_days["train"],
        "min_dt": int(ts.min()),
        "max_dt": int(ts.max()),
    }
    report["n_days"] = dict(n_days)
    report["runs"] = list(runs.values())
    report["sweep"] = sweep
    report["regime_diagnostic"] = {
        "statement": "the mean score per fit on the older train rows, the recent train rows "
        "and the later windows is in each run's `regime` block. At the latency below, the "
        "immature model is refitted with a block of columns removed, to ask whether the recent "
        "window is readable without it",
        "ablation_latency_days": REGIME_ABLATION_LATENCY_DAYS,
        "ablations": regime_ablations,
    }
    report["bootstrap"] = bootstrap
    report["seconds"] = time.perf_counter() - started
    write(report, OUT_PATH)
    print(f"done in {report['seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 experiment 2: simulated label latency.")
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()
    run(args.n_boot)


if __name__ == "__main__":
    main()
