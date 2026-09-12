"""Stage 9: adversarial validation, train against test, on the columns the shipped model reads.

A classifier is fitted to tell a training row from a test row on the model's 186 input columns,
out of fold, and its gain ranking says what separates the windows. If the top of the ranking is
identifiers, their encodings and clocks, the windows differ in who is transacting and when, which
is population turnover; if it is the behavioural columns, they differ in what the transactions
look like, which is closer to concept drift. The script then refits without the entity and clock
columns to measure how much of the separation they carry, and reads the result against stage 2:
the PSI ranking on the raw columns both hold, and the 59 columns whose single-feature AUC
inverted inside train.

    .venv/bin/python scripts/adversarial_validation.py

Writes reports/adversarial_validation.json. Reads the stage 6 cache, so `make train` runs first.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fraud_platform import cleaning, config, eda, encoders, modelling, prepare, reduction

CACHE_DIR = config.REPORTS_DIR / "cache"
MODELS_DIR = config.ROOT / "models"
REPORT_PATH = config.REPORTS_DIR / "adversarial_validation.json"
TEMPORAL_PATH = config.REPORTS_DIR / "eda" / "temporal.json"
REFERENCE_PATH = config.REPORTS_DIR / "monitoring" / "drift_reference.json"

N_FOLDS = 5
N_TOP = 30
# A small model: the question is which columns separate the windows, not how well.
LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.1,
    "min_child_samples": 200,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "verbose": -1,
    "n_jobs": -1,
}

# How each column of the stack is read. An identifier is a column that names a card, an
# address, a domain, a product, a device or a browser (encoders.IDENTIFIER_COLUMNS); its
# encodings carry the same fact through stage 3; a timedelta counts days since something and
# moves with the calendar; the rest describe the transaction.
ENCODING_SUFFIXES: tuple[str, ...] = (
    encoders.FREQUENCY_SUFFIX,
    encoders.VOCABULARY_SUFFIX,
)
GROUPS: tuple[str, ...] = (
    "identifier",
    "identifier_encoding",
    "target_encoding",
    "timedelta",
    "count",
    "vesta_block_mean",
    "amount",
    "identity_numeric",
    "missing_indicator",
    "other",
)
ENTITY_AND_CLOCK_GROUPS: tuple[str, ...] = (
    "identifier",
    "identifier_encoding",
    "target_encoding",
    "timedelta",
)

# The three fits. The target encodings are read separately because a training row is encoded
# at its own lagged day and a test row at the frozen end of the window (ADR 0014), so those
# columns tell the windows apart by construction and say nothing about who is transacting.
FITS: dict[str, tuple[str, ...]] = {
    "full": (),
    "without_target_encodings": ("target_encoding",),
    "without_entity_and_clock": ENTITY_AND_CLOCK_GROUPS,
}


def group_of(column: str) -> str:
    identifiers = set(encoders.IDENTIFIER_COLUMNS)
    if column in identifiers:
        return "identifier"
    if column.endswith(encoders.TARGET_SUFFIX):
        return "target_encoding"
    for suffix in ENCODING_SUFFIXES:
        if column.endswith(suffix):
            source = column[: -len(suffix)]
            if source.endswith(encoders.NORMALISED_SUFFIX):
                source = source[: -len(encoders.NORMALISED_SUFFIX)]
            if source in identifiers:
                return "identifier_encoding"
    if column in config.TIMEDELTA_COLUMNS:
        return "timedelta"
    if column in config.COUNT_COLUMNS:
        return "count"
    if column.startswith("vblockmean_"):
        return "vesta_block_mean"
    if column.startswith(config.AMOUNT_COLUMN):
        return "amount"
    if column in config.IDENTITY_NUMBERED_COLUMNS:
        return "identity_numeric"
    if column.startswith("missing_block_"):
        return "missing_indicator"
    return "other"


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
        "section": "adversarial_validation",
        "stage": 9,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": (
            "make adversarial  # or: .venv/bin/python scripts/adversarial_validation.py"
        ),
        "command": ".venv/bin/python scripts/adversarial_validation.py",
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
            "lightgbm": lgb.__version__,
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
        raise SystemExit(f"{path.relative_to(config.ROOT)} does not exist; run the stage first")
    return dict(json.loads(path.read_text()))


# --- the fit ----------------------------------------------------------------------------------


def adversarial_fit(
    x: pd.DataFrame, is_test: np.ndarray, seed: int
) -> tuple[float, list[float], dict[str, float]]:
    """Out-of-fold AUC of train-versus-test, and the gain share of every column."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(x), dtype="float64")
    fold_aucs: list[float] = []
    gain = np.zeros(x.shape[1], dtype="float64")
    for k, (fit_index, hold_index) in enumerate(folds.split(x, is_test)):
        model = lgb.LGBMClassifier(random_state=seed + k, **LGB_PARAMS)
        model.fit(x.iloc[fit_index], is_test[fit_index])
        oof[hold_index] = model.predict_proba(x.iloc[hold_index])[:, 1]
        fold_aucs.append(float(roc_auc_score(is_test[hold_index], oof[hold_index])))
        gain += model.booster_.feature_importance(importance_type="gain")
    total = float(gain.sum())
    shares = {str(c): float(g / total) for c, g in zip(x.columns, gain, strict=True)}
    return float(roc_auc_score(is_test, oof)), fold_aucs, shares


def single_column_auc(values: pd.Series, is_test: np.ndarray) -> float | None:
    """AUC of the column's own order for telling test from train, nulls ranked lowest.

    A model-free check on the gain ranking: a column the classifier leans on should separate
    the windows on its own order, and one that does not is being used in interaction.
    """
    v = values.to_numpy(dtype="float64")
    if np.isnan(v).all():
        return None
    filled = np.where(np.isnan(v), np.nanmin(v) - 1.0, v)
    if np.unique(filled).size < 2:
        return None
    auc = float(roc_auc_score(is_test, filled))
    return auc


def source_columns(column: str, blocks: Mapping[str, Sequence[str]]) -> list[str]:
    """The raw columns a stack column is computed from: a block mean's members, an encoding's
    source, otherwise the column itself."""
    if column.startswith("vblockmean_"):
        return list(blocks.get(column[len("vblockmean_") :], []))
    for suffix in (*ENCODING_SUFFIXES, encoders.TARGET_SUFFIX):
        if column.endswith(suffix):
            source = column[: -len(suffix)]
            if source.endswith(encoders.NORMALISED_SUFFIX):
                source = source[: -len(encoders.NORMALISED_SUFFIX)]
            return [source]
    if column.startswith("missing_block_"):
        return [column[len("missing_block_") :]]
    return [column]


def v_block_members() -> dict[str, list[str]]:
    plan = prepare.read_column_plan()
    facts = cleaning.EdaFacts.load()
    blocks = reduction.v_blocks(facts.missingness_blocks, cleaning.v_reduction_pool(plan))
    return {str(b["block"]): [str(c) for c in b["columns"]] for b in blocks}


# --- one fit ------------------------------------------------------------------------------------


def one_fit(
    name: str,
    x: pd.DataFrame,
    columns: Sequence[str],
    is_test: np.ndarray,
    psi_test: Mapping[str, float],
    n_top: int,
) -> dict[str, Any]:
    print(f"fitting {name}: train against test on {len(columns)} columns, {N_FOLDS} folds")
    auc, fold_aucs, shares = adversarial_fit(x[list(columns)], is_test, config.SEED)
    print(f"  out-of-fold AUC {auc:.4f}")
    ranked = sorted(shares.items(), key=lambda kv: -kv[1])
    top = [
        {
            "rank": rank,
            "column": column,
            "group": group_of(column),
            "gain_share": share,
            "single_column_auc": single_column_auc(x[column], is_test),
            "psi_test_whole_window": psi_test.get(column),
        }
        for rank, (column, share) in enumerate(ranked[:n_top], start=1)
    ]
    by_group: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        members = [c for c in columns if group_of(c) == group]
        by_group[group] = {
            "n_columns": len(members),
            "gain_share": float(sum(shares[c] for c in members)),
            "n_in_top_10": sum(1 for t in top[:10] if t["group"] == group),
            "n_in_top_30": sum(1 for t in top if t["group"] == group),
        }
    top_10_gain = float(sum(t["gain_share"] for t in top[:10]))
    return {
        "n_columns": len(columns),
        "dropped_groups": [g for g in GROUPS if not any(group_of(c) == g for c in columns)],
        "auc": {"out_of_fold": auc, "per_fold": fold_aucs},
        "top": top,
        "by_group": by_group,
        "entity_and_clock_share_of_top_10_gain": (
            float(sum(t["gain_share"] for t in top[:10] if t["group"] in ENTITY_AND_CLOCK_GROUPS))
            / top_10_gain
        ),
        "gain_share": shares,
    }


# --- main -------------------------------------------------------------------------------------------


def main() -> None:
    started = time.perf_counter()
    print("loading the stage 6 cache")
    train = pd.read_parquet(CACHE_DIR / "prepared_train.parquet")
    test = pd.read_parquet(CACHE_DIR / "prepared_test.parquet")
    columns = list(read(MODELS_DIR / "shipped_model.json")["columns"])
    both = pd.concat([train, test], ignore_index=True)
    is_test = np.concatenate(
        [np.zeros(len(train), dtype="int64"), np.ones(len(test), dtype="int64")]
    )
    x = modelling.tree_matrix(both, columns)
    del both

    reference = read(REFERENCE_PATH) if REFERENCE_PATH.exists() else None
    psi_test = reference["calibration"]["whole_window_psi"]["test"] if reference else {}

    fits: dict[str, dict[str, Any]] = {}
    for name, dropped in FITS.items():
        kept = [c for c in columns if group_of(c) not in dropped]
        fits[name] = one_fit(name, x, kept, is_test, psi_test, N_TOP)

    # Against stage 2, on the fit whose ranking is about the population: the PSI ranking on the
    # raw columns both hold, and the columns whose single-feature AUC inverted inside train.
    reading = fits["without_target_encodings"]
    shares = reading["gain_share"]
    temporal = read(TEMPORAL_PATH)
    stage_2_psi = {r["column"]: float(r["psi_test"]) for r in temporal["psi"]["per_feature"]}
    shared = [c for c in columns if c in stage_2_psi]
    rho = spearmanr([shares[c] for c in shared], [stage_2_psi[c] for c in shared])
    flagged = set(temporal["time_consistency"]["flagged_columns"])
    top_30_columns = [t["column"] for t in reading["top"]]
    blocks = v_block_members()
    through_sources = [
        {
            "rank": t["rank"],
            "column": t["column"],
            "flagged_sources": sorted(set(source_columns(t["column"], blocks)) & flagged),
        }
        for t in reading["top"]
        if set(source_columns(t["column"], blocks)) & flagged
    ]
    stage_2_top = [c for c, _ in sorted(stage_2_psi.items(), key=lambda kv: -kv[1])[:10]]
    against_stage_2 = {
        "artifact": str(TEMPORAL_PATH.relative_to(config.ROOT)),
        "read_on_fit": "without_target_encodings",
        "n_shared_raw_columns": len(shared),
        "spearman_gain_vs_stage_2_psi": {
            "rho": float(rho.statistic),
            "p_value": float(rho.pvalue),
        },
        "stage_2_top_10_by_psi": stage_2_top,
        "stage_2_top_10_in_stack": [c for c in stage_2_top if c in columns],
        "stage_2_top_10_in_adversarial_top_30": [c for c in stage_2_top if c in top_30_columns],
        "n_time_inconsistent_columns_stage_2": len(flagged),
        "time_inconsistent_columns_in_stack": sorted(c for c in columns if c in flagged),
        "time_inconsistent_columns_in_adversarial_top_30": [
            c for c in top_30_columns if c in flagged
        ],
        "time_inconsistent_sources_in_adversarial_top_30": through_sources,
        "entity_identifying_columns_stage_2": list(eda.ENTITY_IDENTIFYING_COLUMNS),
        "stage_2_psi_summary": temporal["psi"]["summary"],
    }

    report = {
        **envelope(),
        "what": (
            "one classifier told to separate train rows (0) from test rows (1) on the shipped "
            "model's input columns, out of fold; the gain ranking is what separates the windows"
        ),
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_columns": len(columns),
        "model": {"estimator": "lightgbm.LGBMClassifier", **LGB_PARAMS, "n_folds": N_FOLDS},
        "groups": {g: sum(1 for c in columns if group_of(c) == g) for g in GROUPS},
        "fits": {
            name: {k: v for k, v in fit.items() if k != "gain_share"} for name, fit in fits.items()
        },
        "against_stage_2": against_stage_2,
        "drift_reference": str(REFERENCE_PATH.relative_to(config.ROOT)) if reference else None,
        "seconds": time.perf_counter() - started,
    }
    write(report, REPORT_PATH)
    print(
        json.dumps(
            _clean(
                {
                    name: {
                        "n_columns": fit["n_columns"],
                        "auc": fit["auc"]["out_of_fold"],
                        "top_10": [
                            (t["column"], t["group"], round(t["gain_share"], 4))
                            for t in fit["top"][:10]
                        ],
                        "by_group": {
                            g: round(v["gain_share"], 4) for g, v in fit["by_group"].items()
                        },
                        "entity_and_clock_share_of_top_10_gain": fit[
                            "entity_and_clock_share_of_top_10_gain"
                        ],
                    }
                    for name, fit in fits.items()
                }
                | {
                    "against_stage_2": {
                        k: v
                        for k, v in against_stage_2.items()
                        if k not in ("time_inconsistent_columns_in_stack", "stage_2_psi_summary")
                    }
                }
            ),
            indent=2,
        )
    )
    print(f"done in {report['seconds']:.0f} s")


if __name__ == "__main__":
    main()
