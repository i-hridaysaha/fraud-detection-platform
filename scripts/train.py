"""Stage 6: train, compare, ablate, search, bootstrap, threshold, band, segment and explain.

One driver, seven sections, each writing its own artifact under reports/. The sections run in
order because each reads what the one before decided, and every decision a section makes is
made by a rule written into the code before the numbers were read:

- `matrices`: rebuild the stage 3 prepared frame from the raw files through
  `fraud_platform.prepare`, add stage 4's features and stage 5's candidate set over the whole
  stream, and cache the three splits under reports/cache/ (gitignored, rebuilt on demand).
- `comparison`: three families by three imbalance strategies on the default feature stack, each
  family through its stage 3 path, one seed, fixed hyperparameters. The shipped family and
  strategy are whichever pair has the largest validation PR-AUC. reports/model_comparison.json.
- `ablations`: the shipped family and strategy on twelve feature stacks, three seeds each.
  Which blocks ship is decided by a rule stated in the artifact. reports/ablations.json.
- `search`: hyperparameters for the shipped family on the shipped stack, under an expanding
  window cross-validation that never fits on a row later than its validation rows. The final
  model is refit on the whole train split and saved under models/. reports/hyperparameter_search.json.
- `variance`: 1,000 bootstrap resamples of the test split, every model scored on the identical
  resample, by row and by card. reports/metric_variance.json.
- `thresholds`: the shipped model calibrated on validation, the threshold sweep, the tuned
  vertices, the cost-derived bands and the per-segment thresholds. reports/threshold_curve.json
  and reports/operating_points.json, from this one command and nowhere else.
- `shap`: global attributions and one transaction, with the card-level caveat measured rather
  than stated. reports/shap_global.json and reports/shap_example.json.

Every artifact carries the git commit, the seed, the split boundaries and the exact stack and
hyperparameters that produced it.
"""

from __future__ import annotations

import argparse
import gc
import itertools
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

from fraud_platform import (
    cleaning,
    config,
    data_loader,
    encoders,
    evaluation,
    features,
    graph_features,
    missing_policy,
    modelling,
    prepare,
)
from fraud_platform.graph_features import GraphConfig

SECTIONS = ("matrices", "comparison", "ablations", "search", "variance", "thresholds", "shap")

CACHE_DIR = config.REPORTS_DIR / "cache"
MODELS_DIR = config.ROOT / "models"
SHAP_DIR = config.REPORTS_DIR / "shap"

COMPARISON_PATH = config.REPORTS_DIR / "model_comparison.json"
ABLATIONS_PATH = config.REPORTS_DIR / "ablations.json"
SEARCH_PATH = config.REPORTS_DIR / "hyperparameter_search.json"
VARIANCE_PATH = config.REPORTS_DIR / "metric_variance.json"
THRESHOLD_CURVE_PATH = config.REPORTS_DIR / "threshold_curve.json"
OPERATING_POINTS_PATH = config.REPORTS_DIR / "operating_points.json"
SHAP_GLOBAL_PATH = SHAP_DIR / "shap_global.json"
SHAP_EXAMPLE_PATH = SHAP_DIR / "shap_example.json"

# Columns carried beside the features so a row can be traced back and grouped.
META_COLUMNS: tuple[str, ...] = (
    config.ID_COLUMN,
    config.TARGET,
    config.TIME_COLUMN,
    config.ENTITY_ID_COLUMN,
    config.SEGMENT_COLUMN,
)

# Seeds for the ablations. The first is the repo seed; the others are the two stage 3 used for
# its own paired probe (scripts/prepare.py PROBE_SEEDS), so the three are one set across stages.
ABLATION_SEEDS: tuple[int, ...] = (config.SEED, 7, 1_337)

N_BOOTSTRAP = 1_000
N_CV_FOLDS = 3
SHAP_SAMPLE_ROWS = 5_000

# The search space per family. Random draws from a grid, the default included as draw zero so
# the search cannot lose to the point it started from without the artifact showing it.
SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "xgboost": {
        "n_estimators": [300, 600, 1000],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 5, 20],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.4, 0.6, 0.8],
        "reg_lambda": [1.0, 5.0, 10.0],
    },
    "random_forest": {
        "n_estimators": [200, 400],
        "min_samples_leaf": [1, 5, 20],
        "max_features": ["sqrt", 0.2, 0.4],
        "max_depth": [None, 12, 24],
    },
    "logistic_regression": {
        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
    },
}


# --- scaffolding -----------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def envelope(section: str, n_train_rows: int) -> dict[str, Any]:
    return {
        "section": section,
        "stage": 6,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": f"make train  # or: .venv/bin/python scripts/train.py --sections {section}",
        "command": f".venv/bin/python scripts/train.py --sections {section}",
        "split": {
            "train_end_dt": config.TRAIN_END_DT,
            "val_end_dt": config.VAL_END_DT,
            "rule": "dt < boundary, ADR 0002",
        },
        "n_train_rows": n_train_rows,
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
        raise SystemExit(
            f"{path.relative_to(config.ROOT)} does not exist; run the earlier section first"
        )
    return dict(json.loads(path.read_text()))


# --- matrices ---------------------------------------------------------------------------------


def cache_path(split: str) -> Path:
    return CACHE_DIR / f"prepared_{split}.parquet"


def build_matrices(transactions: str, identity: str) -> dict[str, Any]:
    """Rebuild the prepared frame and the two feature blocks, and cache the three splits."""
    started = time.perf_counter()
    print("loading the joined frame")
    frame = data_loader.add_entity_key(
        data_loader.load_raw(transactions_path=transactions, identity_path=identity)
    )
    print(f"  {len(frame):,} rows by {frame.shape[1]} columns")
    train, val, test = data_loader.time_based_split(frame)
    data_loader.assert_train_only(train, what="stage 6 preparation fit")

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
    prepared = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        prepared[name] = prepare.apply_preparation(part, fitted, plan)
        print(f"  prepared {name}: {prepared[name].shape[0]:,} rows by {prepared[name].shape[1]}")
    del train, val, test
    gc.collect()

    print("building stage 4 features over the whole stream")
    stream = encoders.add_normalised_free_text(frame)
    buckets = features.fit_dist_buckets(stream[data_loader.split_masks(stream)["train"]])
    causal = features.build_features(stream, buckets, features.DEFAULT_CONFIG)
    print(f"  {causal.shape[1]} causal features")

    print("building stage 5 candidate graph features over the whole stream")
    graph_cfg = GraphConfig(enabled=True)
    graph = graph_features.build_graph_features(stream, None, graph_cfg)
    print(f"  {graph.shape[1]} graph features")
    del stream, frame
    gc.collect()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in prepared.items():
        joined = pd.concat([part, causal.loc[part.index], graph.loc[part.index]], axis=1)
        joined.to_parquet(cache_path(name), index=False)
        print(f"  cached {name}: {joined.shape[0]:,} rows by {joined.shape[1]} columns")

    prepared_columns = modelling.prepared_feature_columns(prepared["train"])
    columns = {
        "prepared": prepared_columns,
        "causal": list(causal.columns),
        "graph": list(graph.columns),
        "graph_config": graph_cfg.to_dict(),
        "feature_config": features.DEFAULT_CONFIG.to_dict(),
        "dist_buckets": buckets,
        "decisions": decisions,
        "n_rows": {name: len(part) for name, part in prepared.items()},
        "seconds": time.perf_counter() - started,
    }
    (CACHE_DIR / "columns.json").write_text(json.dumps(_clean(columns), indent=2) + "\n")
    return columns


class Inputs:
    """The three cached splits and the column lists, loaded once per section."""

    def __init__(self) -> None:
        for split in config.SPLIT_NAMES:
            if not cache_path(split).exists():
                raise SystemExit("reports/cache is missing; run --sections matrices first")
        self.columns = dict(json.loads((CACHE_DIR / "columns.json").read_text()))
        self.frames = {split: pd.read_parquet(cache_path(split)) for split in config.SPLIT_NAMES}
        # The prepared column list is the module's rule applied now, not the cached copy, so a
        # change to the rule takes effect without a rebuild of the cache.
        later = [*self.columns["causal"], *self.columns["graph"]]
        self.columns["prepared"] = modelling.prepared_feature_columns(
            self.frames["train"].drop(columns=later)
        )
        self.y = {
            split: frame[config.TARGET].to_numpy(dtype="int64")
            for split, frame in self.frames.items()
        }
        self.ts = {
            split: frame[config.TIME_COLUMN].to_numpy(dtype="int64")
            for split, frame in self.frames.items()
        }
        self.n_days = {split: evaluation.span_days(ts) for split, ts in self.ts.items()}
        self._tree: dict[tuple[str, ...], dict[str, pd.DataFrame]] = {}
        self._complete: dict[tuple[str, ...], dict[str, Any]] = {}

    def n_rows_train(self) -> int:
        return int(self.y["train"].size)

    def stack_columns(self, stack: modelling.FeatureStack) -> list[str]:
        return modelling.stack_columns(
            self.columns["prepared"], self.columns["causal"], self.columns["graph"], stack
        )

    def tree(self, columns: Sequence[str]) -> dict[str, pd.DataFrame]:
        key = tuple(columns)
        if key not in self._tree:
            self._tree[key] = {
                split: modelling.tree_matrix(frame, columns) for split, frame in self.frames.items()
            }
        return self._tree[key]

    def complete(self, columns: Sequence[str]) -> dict[str, Any]:
        key = tuple(columns)
        if key not in self._complete:
            fitted = modelling.fit_complete_data_path(self.frames["train"], list(columns))
            matrices = {}
            names: list[str] = []
            mask = np.zeros(0, dtype=bool)
            for split, frame in self.frames.items():
                matrices[split], names, mask = modelling.apply_complete_data_path(frame, fitted)
            self._complete[key] = {
                "fitted": fitted,
                "matrices": matrices,
                "names": names,
                "interpolable": mask,
            }
        return self._complete[key]

    def smote_plan(self, columns: Sequence[str], seed: int) -> dict[str, Any]:
        """Neighbours found in the complete-data numeric space, one plan per seed."""
        complete = self.complete(columns)
        space = complete["matrices"]["train"][:, complete["interpolable"]]
        return modelling.smote_plan(space, self.y["train"], seed)

    def keys(self, split: str) -> npt.NDArray[Any]:
        return self.frames[split][config.ENTITY_ID_COLUMN].to_numpy()

    def segments(self, split: str) -> npt.NDArray[Any]:
        return self.frames[split][config.SEGMENT_COLUMN].astype("string").to_numpy()


def fit_and_score(
    inputs: Inputs,
    family: str,
    imbalance: str,
    columns: Sequence[str],
    seed: int,
    hyperparameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One fit on the train split, scored on validation and test. Everything else derives."""
    path = modelling.PATH_BY_FAMILY[family]
    y_fit = inputs.y["train"]
    if path == "tree":
        matrices = inputs.tree(columns)
        x_fit: Any = matrices["train"].to_numpy(dtype="float32")
        x_val: Any = matrices["val"].to_numpy(dtype="float32")
        x_test: Any = matrices["test"].to_numpy(dtype="float32")
        interpolable = modelling.interpolable_mask(matrices["train"])
    else:
        complete = inputs.complete(columns)
        x_fit = complete["matrices"]["train"]
        x_val = complete["matrices"]["val"]
        x_test = complete["matrices"]["test"]
        interpolable = complete["interpolable"]

    n_synthetic = 0
    if imbalance == "smote":
        plan = inputs.smote_plan(columns, seed)
        x_fit, y_fit = modelling.materialise_smote(x_fit, y_fit, plan, interpolable)
        n_synthetic = int(plan["n_synthetic"])

    model = modelling.make_model(family, imbalance, y_fit, seed, hyperparameters)
    started = time.perf_counter()
    model.fit(x_fit, y_fit)
    fit_seconds = time.perf_counter() - started
    s_val = modelling.score(model, x_val)
    s_test = modelling.score(model, x_test)
    return {
        "model": model,
        "family": family,
        "imbalance": imbalance,
        "path": path,
        "seed": seed,
        "n_columns": int(x_fit.shape[1]),
        "n_fit_rows": int(x_fit.shape[0]),
        "n_synthetic_rows": n_synthetic,
        "fit_seconds": fit_seconds,
        "convergence": modelling.convergence(model),
        "scores": {"val": s_val, "test": s_test},
    }


def metrics_block(inputs: Inputs, scores: Mapping[str, npt.NDArray[np.float64]]) -> dict[str, Any]:
    """PR-AUC and ROC-AUC on both later splits, at transaction and card level, plus the
    validation-tuned vertex applied to test."""
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        y, s = inputs.y[split], scores[split]
        y_card, s_card = evaluation.card_level(y, s, inputs.keys(split))
        out[split] = {
            "pr_auc": evaluation.pr_auc(y, s),
            "roc_auc": evaluation.roc_auc(y, s),
            "n_rows": int(y.size),
            "n_positive": int(y.sum()),
            "card_level": {
                "pr_auc": evaluation.pr_auc(y_card, s_card),
                "roc_auc": evaluation.roc_auc(y_card, s_card),
                "n_cards": int(y_card.size),
                "n_positive_cards": int(y_card.sum()),
                "key": config.ENTITY_ID_COLUMN,
                "aggregation": "maximum score per key, any fraud per key",
            },
        }
    vertex = evaluation.best_f1_vertex(inputs.y["val"], scores["val"], inputs.n_days["val"])
    out["tuned_threshold"] = {
        "rule": vertex["rule"],
        "fitted_on": "val",
        "threshold": vertex["threshold"],
        "val": vertex,
        "test": evaluation.at_threshold(
            inputs.y["test"], scores["test"], vertex["threshold"], inputs.n_days["test"]
        ),
    }
    return out


def score_cache_path(name: str) -> Path:
    return CACHE_DIR / f"scores_{name}.parquet"


def save_scores(name: str, scores: Mapping[str, npt.NDArray[np.float64]]) -> None:
    frame = pd.DataFrame(
        {
            "split": np.concatenate([np.repeat(s, scores[s].size) for s in ("val", "test")]),
            "score": np.concatenate([scores["val"], scores["test"]]),
        }
    )
    frame.to_parquet(score_cache_path(name), index=False)


def load_scores(name: str) -> dict[str, npt.NDArray[np.float64]]:
    path = score_cache_path(name)
    if not path.exists():
        raise SystemExit(
            f"{path.relative_to(config.ROOT)} is missing; rerun the section that fits it"
        )
    frame = pd.read_parquet(path)
    return {
        split: frame.loc[frame["split"] == split, "score"].to_numpy(dtype="float64")
        for split in ("val", "test")
    }


def model_key(family: str, imbalance: str) -> str:
    return f"{family}__{imbalance}"


# --- comparison ---------------------------------------------------------------------------------


def build_comparison(inputs: Inputs) -> dict[str, Any]:
    stack = modelling.DEFAULT_STACK
    columns = inputs.stack_columns(stack)
    report = envelope("comparison", inputs.n_rows_train())
    report["stack"] = stack.to_dict()
    report["n_columns"] = len(columns)
    report["hyperparameters"] = {
        "status": "fixed defaults, chosen before any number was read; the search section moves "
        "the shipped family's values and records both",
        **modelling.DEFAULT_HYPERPARAMETERS,
    }
    report["paths"] = dict(modelling.PATH_BY_FAMILY)
    report["imbalance"] = {
        "strategies": list(modelling.IMBALANCE_STRATEGIES),
        "class_weight": "balanced: negatives per positive on the fit rows, as scale_pos_weight for xgboost",
        "smote": (
            f"synthetic minority rows on the segment between a positive row and one of its "
            f"{modelling.SMOTE_NEIGHBOURS} nearest positive neighbours, neighbours found in the "
            "complete-data standardised numeric space, enough rows to balance the classes 1:1, "
            "fitted on the train split only and applied to no other split"
        ),
        "none": "the fit rows as they are",
    }
    report["n_days"] = dict(inputs.n_days)

    models: list[dict[str, Any]] = []
    for family in modelling.FAMILIES:
        for imbalance in modelling.IMBALANCE_STRATEGIES:
            print(f"  fitting {family} with {imbalance}")
            result = fit_and_score(inputs, family, imbalance, columns, config.SEED)
            name = model_key(family, imbalance)
            save_scores(name, result["scores"])
            entry = {
                "name": name,
                **{k: v for k, v in result.items() if k not in ("model", "scores")},
                **metrics_block(inputs, result["scores"]),
            }
            print(
                f"    val PR-AUC {entry['val']['pr_auc']:.4f}, test PR-AUC {entry['test']['pr_auc']:.4f}, "
                f"{result['fit_seconds']:.0f}s"
            )
            models.append(entry)
            del result
            gc.collect()

    ranked = sorted(models, key=lambda m: -m["val"]["pr_auc"])
    winner = ranked[0]
    by_family = {}
    for family in modelling.FAMILIES:
        rows = [m for m in models if m["family"] == family]
        best = max(rows, key=lambda m: m["val"]["pr_auc"])
        by_family[family] = {
            "best_imbalance": best["imbalance"],
            "val_pr_auc_by_imbalance": {m["imbalance"]: m["val"]["pr_auc"] for m in rows},
            "test_pr_auc_by_imbalance": {m["imbalance"]: m["test"]["pr_auc"] for m in rows},
        }
    report["models"] = models
    report["selection"] = {
        "rule": "the family and strategy with the largest validation PR-AUC; test is not consulted",
        "shipped_family": winner["family"],
        "shipped_imbalance": winner["imbalance"],
        "shipped_name": winner["name"],
        "ranking_by_val_pr_auc": [
            {
                "name": m["name"],
                "val_pr_auc": m["val"]["pr_auc"],
                "test_pr_auc": m["test"]["pr_auc"],
            }
            for m in ranked
        ],
        "by_family": by_family,
    }
    report["logistic_regression_convergence"] = [
        {"name": m["name"], **m["convergence"]} for m in models if m["convergence"] is not None
    ]
    report["metric_agreement"] = metric_agreement(models)
    return report


def metric_agreement(models: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Where ROC-AUC and PR-AUC order the same nine models differently, on validation.

    ADR 0026 rests on this block: the two metrics are computed on the same scores, so a pair of
    models they order differently is a pair on which the choice of metric is the decision.
    """
    by_roc = sorted(models, key=lambda m: -m["val"]["roc_auc"])
    by_pr = sorted(models, key=lambda m: -m["val"]["pr_auc"])
    discordant = []
    for first, second in itertools.combinations(models, 2):
        roc_sign = np.sign(first["val"]["roc_auc"] - second["val"]["roc_auc"])
        pr_sign = np.sign(first["val"]["pr_auc"] - second["val"]["pr_auc"])
        if roc_sign != pr_sign:
            discordant.append(
                {
                    "first": first["name"],
                    "second": second["name"],
                    "val_roc_auc": [first["val"]["roc_auc"], second["val"]["roc_auc"]],
                    "val_pr_auc": [first["val"]["pr_auc"], second["val"]["pr_auc"]],
                }
            )
    roc = [m["val"]["roc_auc"] for m in models]
    pr = [m["val"]["pr_auc"] for m in models]
    return {
        "on": "val",
        "rank_by_roc_auc": [m["name"] for m in by_roc],
        "rank_by_pr_auc": [m["name"] for m in by_pr],
        "n_pairs": len(list(itertools.combinations(models, 2))),
        "n_discordant_pairs": len(discordant),
        "discordant_pairs": discordant,
        "spread": {
            "roc_auc": max(roc) - min(roc),
            "pr_auc": max(pr) - min(pr),
        },
        "false_positive_rate_at_tuned_threshold_test": {
            m["name"]: m["tuned_threshold"]["test"]["false_positive_rate"] for m in models
        },
    }


# --- ablations ----------------------------------------------------------------------------------


def ablation_stacks() -> list[dict[str, Any]]:
    """The twelve stacks, each with the comparison that gives it meaning."""
    all_on = features.NATIVE_GROUPS
    base = modelling.FeatureStack(causal=False, graph=False, native_groups=all_on)
    full = modelling.FeatureStack(causal=True, graph=True, native_groups=all_on)
    stacks = [
        {"id": "prepared", "stack": base, "reference": None},
        {
            "id": "prepared+causal",
            "stack": modelling.FeatureStack(True, False, all_on),
            "reference": "prepared",
        },
        {"id": "prepared+causal+graph", "stack": full, "reference": "prepared+causal"},
        {
            "id": "prepared|native=none",
            "stack": modelling.FeatureStack(False, False, ()),
            "reference": "prepared",
        },
        {
            "id": "prepared+causal|native=none",
            "stack": modelling.FeatureStack(True, False, ()),
            "reference": "prepared+causal",
        },
        {
            "id": "prepared+causal+graph|native=none",
            "stack": modelling.FeatureStack(True, True, ()),
            "reference": "prepared+causal+graph",
        },
    ]
    for group in features.NATIVE_GROUPS:
        stacks.append(
            {
                "id": f"prepared+causal+graph|without={group}",
                "stack": full.without_group(group),
                "reference": "prepared+causal+graph",
            }
        )
    # The 2 by 2 ADR 0021 asked for: own velocity against the native counters.
    stacks.append(
        {
            "id": "prepared|without=count",
            "stack": base.without_group("count"),
            "reference": "prepared",
        }
    )
    stacks.append(
        {
            "id": "prepared+causal|without=count",
            "stack": modelling.FeatureStack(True, False, all_on).without_group("count"),
            "reference": "prepared+causal",
        }
    )
    return stacks


def importance_by_block(
    model: Any, columns: Sequence[str], family: str, inputs: Inputs
) -> dict[str, Any]:
    """Model importances summed per block: own velocity, native counters, the rest."""
    if family == "xgboost":
        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        weight = booster.get_score(importance_type="weight")
        names = booster.feature_names or [f"f{i}" for i in range(len(columns))]
        per_column = {
            column: {"gain": float(gain.get(name, 0.0)), "weight": float(weight.get(name, 0.0))}
            for column, name in zip(columns, names, strict=True)
        }
        kind = "xgboost gain (mean loss reduction over the splits on the feature) and weight (split count)"
    elif family == "random_forest":
        per_column = {
            column: {"gain": float(v), "weight": 0.0}
            for column, v in zip(columns, model.feature_importances_, strict=True)
        }
        kind = "sklearn impurity importance"
    else:
        coefficients = np.abs(model.coef_[0])
        names = inputs.complete(columns)["names"]
        per_column = {
            column: {"gain": float(v), "weight": 0.0}
            for column, v in zip(names, coefficients, strict=True)
        }
        kind = "absolute standardised coefficient"

    velocity = [c for c in per_column if c.startswith("vel_")]
    counters = modelling.native_group_members(list(per_column), "count")
    graph = [c for c in per_column if c in graph_features.ALL_FEATURES]
    causal = [c for c in per_column if c in set(features.feature_names())]

    def block(cols: Sequence[str]) -> dict[str, Any]:
        gains = sorted(((per_column[c]["gain"], c) for c in cols), reverse=True)
        total_gain = float(sum(per_column[c]["gain"] for c in per_column)) or 1.0
        return {
            "n_columns": len(cols),
            "gain_sum": float(sum(g for g, _ in gains)),
            "gain_share": float(sum(g for g, _ in gains)) / total_gain,
            "gain_mean_per_column": (float(sum(g for g, _ in gains)) / len(cols)) if cols else None,
            "top": [{"column": c, "gain": g} for g, c in gains[:10]],
        }

    ranked = sorted(per_column.items(), key=lambda kv: -kv[1]["gain"])
    return {
        "kind": kind,
        "blocks": {
            "own_velocity": block(velocity),
            "native_count": block(counters),
            "stage_4_causal_all": block(causal),
            "stage_5_graph": block(graph),
        },
        "top_30": [{"column": c, **v} for c, v in ranked[:30]],
    }


def build_ablations(inputs: Inputs, comparison: Mapping[str, Any]) -> dict[str, Any]:
    family = comparison["selection"]["shipped_family"]
    imbalance = comparison["selection"]["shipped_imbalance"]
    report = envelope("ablations", inputs.n_rows_train())
    report["family"] = family
    report["imbalance"] = imbalance
    report["hyperparameters"] = modelling.DEFAULT_HYPERPARAMETERS[family]
    report["seeds"] = list(ABLATION_SEEDS)
    report["rule"] = (
        "a block ships when switching it on raises validation PR-AUC on every seed and the paired "
        "bootstrap interval of the difference excludes zero on every seed, against the reference "
        "stack fitted at the same seed. Test is reported and not consulted. The same rule "
        "stage 4 wrote for its counts (ADR 0022), applied to a model instead of a marginal."
    )

    stacks = ablation_stacks()
    results: dict[str, dict[str, Any]] = {}
    for entry in stacks:
        stack = entry["stack"]
        columns = inputs.stack_columns(stack)
        per_seed = []
        for seed in ABLATION_SEEDS:
            print(f"  {entry['id']} seed {seed} ({len(columns)} columns)")
            fitted = fit_and_score(inputs, family, imbalance, columns, seed)
            row = {
                "seed": seed,
                "fit_seconds": fitted["fit_seconds"],
                "val_pr_auc": evaluation.pr_auc(inputs.y["val"], fitted["scores"]["val"]),
                "test_pr_auc": evaluation.pr_auc(inputs.y["test"], fitted["scores"]["test"]),
                "val_roc_auc": evaluation.roc_auc(inputs.y["val"], fitted["scores"]["val"]),
                "test_roc_auc": evaluation.roc_auc(inputs.y["test"], fitted["scores"]["test"]),
                "scores": fitted["scores"],
            }
            if entry["id"] in ("prepared+causal", "prepared+causal+graph") and seed == config.SEED:
                row["importance"] = importance_by_block(fitted["model"], columns, family, inputs)
            per_seed.append(row)
            print(f"    val PR-AUC {row['val_pr_auc']:.4f}, test PR-AUC {row['test_pr_auc']:.4f}")
            del fitted
            gc.collect()
        results[entry["id"]] = {
            "id": entry["id"],
            "stack": stack.to_dict(),
            "reference": entry["reference"],
            "n_columns": len(columns),
            "per_seed": per_seed,
        }

    # Paired differences against the reference, one per seed, with a bootstrap interval each.
    for entry in results.values():
        reference = entry["reference"]
        summary: dict[str, Any] = {}
        for metric in ("val_pr_auc", "test_pr_auc", "val_roc_auc", "test_roc_auc"):
            values = [row[metric] for row in entry["per_seed"]]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "min": min(values),
                "max": max(values),
            }
        entry["summary"] = summary
        if reference is None:
            continue
        differences = []
        for row, ref_row in zip(entry["per_seed"], results[reference]["per_seed"], strict=True):
            boot = evaluation.paired_bootstrap(
                inputs.y["val"],
                {"stack": row["scores"]["val"], "reference": ref_row["scores"]["val"]},
                n_boot=200,
                seed=row["seed"],
            )
            pair = boot["pairs"][0]
            differences.append(
                {
                    "seed": row["seed"],
                    "val_pr_auc_difference": pair["difference"],
                    "excludes_zero": pair["excludes_zero"],
                    "test_pr_auc_difference": row["test_pr_auc"] - ref_row["test_pr_auc"],
                }
            )
        entry["against_reference"] = {
            "reference": reference,
            "per_seed": differences,
            "val_gain_positive_on_every_seed": all(
                d["val_pr_auc_difference"]["point"] > 0 for d in differences
            ),
            "val_interval_excludes_zero_on_every_seed": all(
                d["excludes_zero"] for d in differences
            ),
            "mean_val_gain": float(
                np.mean([d["val_pr_auc_difference"]["point"] for d in differences])
            ),
            "mean_test_gain": float(np.mean([d["test_pr_auc_difference"] for d in differences])),
        }

    for entry in results.values():
        for row in entry["per_seed"]:
            del row["scores"]

    def ships(stack_id: str) -> bool:
        against = results[stack_id]["against_reference"]
        return bool(
            against["val_gain_positive_on_every_seed"]
            and against["val_interval_excludes_zero_on_every_seed"]
        )

    def removal_is_harmless(stack_id: str) -> bool:
        """For a stack that removes a provided block: no seed shows a loss outside its interval."""
        against = results[stack_id]["against_reference"]
        return not any(
            d["val_pr_auc_difference"]["point"] < 0 and d["excludes_zero"]
            for d in against["per_seed"]
        )

    def removal_hurts_on_every_seed(stack_id: str) -> bool:
        against = results[stack_id]["against_reference"]
        return bool(
            all(d["val_pr_auc_difference"]["point"] < 0 for d in against["per_seed"])
            and against["val_interval_excludes_zero_on_every_seed"]
        )

    causal_ships = ships("prepared+causal")
    graph_ships = ships("prepared+causal+graph")
    native_decisions = {
        group: not removal_is_harmless(f"prepared+causal+graph|without={group}")
        for group in features.NATIVE_GROUPS
    }
    # The rule as first written, kept in the artifact because it was replaced after the numbers
    # were read: it retired a provided block unless every seed showed a loss outside its
    # interval, which composed three single-block removals into a stack no ablation had
    # measured. docs/notes/stage-06.md records the change.
    first_draft = {
        group: removal_hurts_on_every_seed(f"prepared+causal+graph|without={group}")
        for group in features.NATIVE_GROUPS
    }
    shipped = modelling.FeatureStack(
        causal=causal_ships,
        graph=graph_ships,
        native_groups=tuple(g for g in features.NATIVE_GROUPS if native_decisions[g]),
    )
    report["stacks"] = list(results.values())
    report["decision"] = {
        "causal_block_ships": causal_ships,
        "graph_block_ships": graph_ships,
        "native_group_ships": native_decisions,
        "native_rule": (
            "a provided block (the native C, D, M and V groups) is retired only when removing it "
            "from the full stack lowers validation PR-AUC on no seed with the paired interval "
            "excluding zero; one seed with a loss outside its interval keeps the block. The burden "
            "is on the removal for a block the frame already carries, and on the addition for a "
            "block this repo builds (the causal and graph rules above), because a wrong drop costs "
            "detection and a wrong keep costs width"
        ),
        "superseded_first_draft": {
            "rule": (
                "a native group ships only when removing it lowers validation PR-AUC on every seed "
                "with the interval excluding zero on every seed. Replaced after the numbers were "
                "read: it composed three single-block removals into a stack no ablation measured"
            ),
            "native_group_ships": first_draft,
            "stack_it_would_have_shipped": modelling.FeatureStack(
                causal=causal_ships,
                graph=graph_ships,
                native_groups=tuple(g for g in features.NATIVE_GROUPS if first_draft[g]),
            ).to_dict(),
        },
        "shipped_stack": shipped.to_dict(),
        "n_shipped_columns": len(inputs.stack_columns(shipped)),
    }
    report["velocity_against_native_c"] = {
        "adr": "0021",
        "two_by_two_val_pr_auc": {
            stack_id: results[stack_id]["summary"]["val_pr_auc"]
            for stack_id in (
                "prepared",
                "prepared|without=count",
                "prepared+causal",
                "prepared+causal|without=count",
            )
        },
        "importance": {
            stack_id: results[stack_id]["per_seed"][0].get("importance")
            for stack_id in ("prepared+causal", "prepared+causal+graph")
        },
    }
    return report


# --- search -------------------------------------------------------------------------------------


def draw_configurations(family: str, n_draws: int, seed: int) -> list[dict[str, Any]]:
    space = SEARCH_SPACES[family]
    rng = np.random.default_rng(seed)
    default = {k: modelling.DEFAULT_HYPERPARAMETERS[family].get(k) for k in space}
    seen = {json.dumps(default, sort_keys=True)}
    draws = [default]
    attempts = 0
    while len(draws) < n_draws + 1 and attempts < n_draws * 50:
        attempts += 1
        candidate = {k: values[int(rng.integers(0, len(values)))] for k, values in space.items()}
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        draws.append(candidate)
    return draws


def build_search(
    inputs: Inputs, comparison: Mapping[str, Any], ablations: Mapping[str, Any], n_draws: int
) -> dict[str, Any]:
    family = comparison["selection"]["shipped_family"]
    imbalance = comparison["selection"]["shipped_imbalance"]
    shipped_stack = ablations["decision"]["shipped_stack"]
    stack = modelling.FeatureStack(
        causal=bool(shipped_stack["causal"]),
        graph=bool(shipped_stack["graph"]),
        native_groups=tuple(shipped_stack["native_groups"]),
    )
    columns = inputs.stack_columns(stack)
    report = envelope("search", inputs.n_rows_train())
    report["family"] = family
    report["imbalance"] = imbalance
    report["stack"] = stack.to_dict()
    report["n_columns"] = len(columns)

    folds = modelling.expanding_window_folds(inputs.ts["train"], N_CV_FOLDS)
    report["cross_validation"] = {
        "scheme": "expanding window on the train split: the rows are cut into four contiguous "
        "blocks at timestamp quantiles; fold k fits on blocks 0 to k and validates on block k+1, "
        "so no fold fits on a row later than its validation rows",
        "n_folds": N_CV_FOLDS,
        "folds": [
            {k: v for k, v in fold.items() if k not in ("fit", "validate")}
            | {
                "n_fit": int(fold["fit"].size),
                "n_validate": int(fold["validate"].size),
                "n_validate_positive": int(inputs.y["train"][fold["validate"]].sum()),
            }
            for fold in folds
        ],
    }

    path = modelling.PATH_BY_FAMILY[family]
    if path == "tree":
        x_train = inputs.tree(columns)["train"].to_numpy(dtype="float32")
        interpolable = modelling.interpolable_mask(inputs.tree(columns)["train"])
    else:
        complete = inputs.complete(columns)
        x_train = complete["matrices"]["train"]
        interpolable = complete["interpolable"]
    y_train = inputs.y["train"]
    # The space SMOTE finds neighbours in, whichever path the family is on; a fold reads only
    # its own fit rows of it.
    neighbour_space = None
    if imbalance == "smote":
        complete = inputs.complete(columns)
        neighbour_space = complete["matrices"]["train"][:, complete["interpolable"]]

    configurations = draw_configurations(family, n_draws, config.SEED)
    report["space"] = SEARCH_SPACES[family]
    report["n_configurations"] = len(configurations)
    trials = []
    for index, candidate in enumerate(configurations):
        fold_scores = []
        seconds = 0.0
        for fold in folds:
            fit_index, validate_index = fold["fit"], fold["validate"]
            x_fit, y_fit = x_train[fit_index], y_train[fit_index]
            if neighbour_space is not None:
                plan = modelling.smote_plan(neighbour_space[fit_index], y_fit, config.SEED)
                x_fit, y_fit = modelling.materialise_smote(x_fit, y_fit, plan, interpolable)
            model = modelling.make_model(family, imbalance, y_fit, config.SEED, candidate)
            started = time.perf_counter()
            model.fit(x_fit, y_fit)
            seconds += time.perf_counter() - started
            s = modelling.score(model, x_train[validate_index])
            fold_scores.append(evaluation.pr_auc(y_train[validate_index], s))
            del model
            gc.collect()
        trial = {
            "index": index,
            "is_default": index == 0,
            "hyperparameters": candidate,
            "fold_pr_auc": fold_scores,
            "mean_pr_auc": float(np.mean(fold_scores)),
            "min_pr_auc": float(np.min(fold_scores)),
            "fit_seconds": seconds,
        }
        trials.append(trial)
        print(
            f"  trial {index}: mean CV PR-AUC {trial['mean_pr_auc']:.4f} ({seconds:.0f}s) {candidate}"
        )

    best = max(trials, key=lambda t: t["mean_pr_auc"])
    default_trial = trials[0]
    report["trials"] = trials
    report["selection"] = {
        "rule": "the configuration with the largest mean validation-fold PR-AUC across the folds",
        "best_index": best["index"],
        "best_hyperparameters": best["hyperparameters"],
        "best_mean_pr_auc": best["mean_pr_auc"],
        "default_mean_pr_auc": default_trial["mean_pr_auc"],
        "gain_over_default": best["mean_pr_auc"] - default_trial["mean_pr_auc"],
    }

    print("  refitting the default and the best on the whole train split")
    final = {}
    for label, params in (
        ("default", default_trial["hyperparameters"]),
        ("tuned", best["hyperparameters"]),
    ):
        fitted = fit_and_score(inputs, family, imbalance, columns, config.SEED, params)
        final[label] = {
            **{k: v for k, v in fitted.items() if k not in ("model", "scores")},
            "hyperparameters": {**modelling.DEFAULT_HYPERPARAMETERS[family], **params},
            **metrics_block(inputs, fitted["scores"]),
        }
        save_scores(f"final_{label}", fitted["scores"])
        if label == "tuned":
            tuned_model = fitted["model"]
        del fitted
        gc.collect()
    shipped_label = (
        "tuned"
        if final["tuned"]["val"]["pr_auc"] >= final["default"]["val"]["pr_auc"]
        else "default"
    )
    report["final"] = final
    report["shipped"] = {
        "rule": "the tuned configuration ships when its validation PR-AUC on the full train fit "
        "is at least the default's; otherwise the default ships and the search is recorded as "
        "having found nothing",
        "label": shipped_label,
        "hyperparameters": final[shipped_label]["hyperparameters"],
        "val_pr_auc": final[shipped_label]["val"]["pr_auc"],
        "test_pr_auc": final[shipped_label]["test"]["pr_auc"],
    }
    if shipped_label == "default":
        # Refit so the saved model is the one the artifact describes.
        fitted = fit_and_score(
            inputs, family, imbalance, columns, config.SEED, default_trial["hyperparameters"]
        )
        tuned_model = fitted["model"]
    save_scores("shipped", load_scores(f"final_{shipped_label}"))
    save_shipped_model(
        tuned_model, family, imbalance, stack, columns, final[shipped_label]["hyperparameters"]
    )
    return report


def save_shipped_model(
    model: Any,
    family: str,
    imbalance: str,
    stack: modelling.FeatureStack,
    columns: Sequence[str],
    hyperparameters: Mapping[str, Any],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if family == "xgboost":
        model_path = MODELS_DIR / "shipped_model.ubj"
        model.save_model(model_path)
    else:
        import pickle

        model_path = MODELS_DIR / "shipped_model.pkl"
        model_path.write_bytes(pickle.dumps(model))
    bundle = {
        "family": family,
        "imbalance": imbalance,
        "path": modelling.PATH_BY_FAMILY[family],
        "stack": stack.to_dict(),
        "columns": list(columns),
        "hyperparameters": dict(hyperparameters),
        "model_file": model_path.name,
        "git_commit": git_commit(),
        "seed": config.SEED,
    }
    (MODELS_DIR / "shipped_model.json").write_text(json.dumps(_clean(bundle), indent=2) + "\n")
    print(
        f"  saved {model_path.relative_to(config.ROOT)} ({model_path.stat().st_size / 1e6:.1f} MB)"
    )


def load_shipped_model() -> tuple[Any, dict[str, Any]]:
    bundle = read(MODELS_DIR / "shipped_model.json")
    model_path = MODELS_DIR / bundle["model_file"]
    if bundle["family"] == "xgboost":
        model = xgboost.XGBClassifier()
        model.load_model(model_path)
    else:
        import pickle

        model = pickle.loads(model_path.read_bytes())
    return model, bundle


# --- variance -----------------------------------------------------------------------------------


def build_variance(inputs: Inputs, comparison: Mapping[str, Any]) -> dict[str, Any]:
    report = envelope("variance", inputs.n_rows_train())
    names = [m["name"] for m in comparison["models"]] + ["shipped"]
    scores = {name: load_scores(name)["test"] for name in names}
    y = inputs.y["test"]
    report["models"] = names
    report["n_boot"] = N_BOOTSTRAP
    print("  bootstrapping by row")
    report["by_row"] = evaluation.paired_bootstrap(y, scores, N_BOOTSTRAP, config.SEED)
    print("  bootstrapping by card")
    report["by_card"] = evaluation.paired_bootstrap(
        y, scores, N_BOOTSTRAP, config.SEED, groups=inputs.keys("test")
    )
    print("  bootstrapping the card-level metric by card")
    card_scores = {}
    y_card = None
    for name, s in scores.items():
        y_card, card_scores[name] = evaluation.card_level(y, s, inputs.keys("test"))
    assert y_card is not None
    report["card_level_by_card"] = evaluation.paired_bootstrap(
        y_card, card_scores, N_BOOTSTRAP, config.SEED
    )
    widest = max(
        (report[k]["widest_paired_half_width"]["half_width"], k) for k in ("by_row", "by_card")
    )
    report["noise_band"] = {
        "definition": "the widest paired 95 percent half-width on transaction-level PR-AUC across "
        "the row and card resamples. A promotion margin inside it promotes on noise; stage 9 sets "
        "its margin above it",
        "half_width": widest[0],
        "from": widest[1],
        "shipped_pr_auc_half_width_by_row": report["by_row"]["per_model"]["shipped"]["half_width"],
        "shipped_pr_auc_half_width_by_card": report["by_card"]["per_model"]["shipped"][
            "half_width"
        ],
    }
    lead = [p for p in report["by_row"]["pairs"] if "shipped" in (p["first"], p["second"])]
    report["shipped_against_each"] = lead
    return report


# --- thresholds ---------------------------------------------------------------------------------


def build_thresholds(
    inputs: Inputs, search: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    scores = load_scores("shipped")
    y_val, y_test = inputs.y["val"], inputs.y["test"]
    s_val, s_test = scores["val"], scores["test"]
    n_days = inputs.n_days

    calibrator = evaluation.fit_calibrator(s_val, y_val)
    p_val = evaluation.calibrate(calibrator, s_val)
    p_test = evaluation.calibrate(calibrator, s_test)
    bands = evaluation.cost_bands()

    common = {
        "model": {
            "family": search["family"],
            "imbalance": search["imbalance"],
            "stack": search["stack"],
            "hyperparameters": search["shipped"]["hyperparameters"],
            "model_file": "models/shipped_model.json",
        },
        "n_days": dict(n_days),
        "band_edges": {"low": bands["low"], "high": bands["high"], "on": "calibrated probability"},
        "calibration": {
            "method": "isotonic regression from the model score to a probability, fitted on the "
            "validation split, monotone so it moves no ranking",
            "fitted_on": "val",
            "raw_score": {
                "val": evaluation.reliability(y_val, s_val),
                "test": evaluation.reliability(y_test, s_test),
            },
            "calibrated": {
                "val": evaluation.reliability(y_val, p_val),
                "test": evaluation.reliability(y_test, p_test),
            },
        },
    }

    curve = envelope("thresholds", inputs.n_rows_train())
    curve.update(common)
    curve["thresholds"] = list(evaluation.SWEEP_THRESHOLDS)
    curve["on"] = "the calibrated probability; the raw-score sweep is beside it because the "
    curve["on"] += "tuned vertices in operating_points.json are on the raw score"
    curve["calibrated"] = {
        "val": evaluation.threshold_sweep(y_val, p_val, n_days["val"]),
        "test": evaluation.threshold_sweep(y_test, p_test, n_days["test"]),
    }
    curve["raw_score"] = {
        "val": evaluation.threshold_sweep(y_val, s_val, n_days["val"]),
        "test": evaluation.threshold_sweep(y_test, s_test, n_days["test"]),
    }

    points = envelope("thresholds", inputs.n_rows_train())
    points.update(common)
    points["primary_metric"] = {
        "val": {
            "pr_auc": evaluation.pr_auc(y_val, s_val),
            "roc_auc": evaluation.roc_auc(y_val, s_val),
        },
        "test": {
            "pr_auc": evaluation.pr_auc(y_test, s_test),
            "roc_auc": evaluation.roc_auc(y_test, s_test),
        },
    }
    vertex = evaluation.best_f1_vertex(y_val, s_val, n_days["val"])
    points["tuned_vertex"] = {
        "on": "raw score",
        "fitted_on": "val",
        "val": vertex,
        "test": evaluation.at_threshold(y_test, s_test, vertex["threshold"], n_days["test"]),
    }
    points["bands"] = {
        **bands,
        "on": "calibrated probability",
        "val": evaluation.band_report(y_val, p_val, bands, n_days["val"]),
        "test": evaluation.band_report(y_test, p_test, bands, n_days["test"]),
        "edges_as_raw_score": {
            "low": float(np.min(s_val[p_val >= bands["low"]]))
            if np.any(p_val >= bands["low"])
            else None,
            "high": float(np.min(s_val[p_val >= bands["high"]]))
            if np.any(p_val >= bands["high"])
            else None,
            "note": "the smallest validation raw score the calibrator maps at or above each edge",
        },
    }
    points["edges_at_threshold"] = {
        name: {
            "val": evaluation.at_threshold(y_val, p_val, bands[name], n_days["val"]),
            "test": evaluation.at_threshold(y_test, p_test, bands[name], n_days["test"]),
        }
        for name in ("low", "high", "single_threshold_without_review")
    }

    segment_values = inputs.segments("val")
    segment_values_test = inputs.segments("test")
    levels = sorted(set(segment_values) | set(segment_values_test))
    segments = {
        level: {
            "y_val": y_val[segment_values == level],
            "s_val": s_val[segment_values == level],
            "y_test": y_test[segment_values_test == level],
            "s_test": s_test[segment_values_test == level],
        }
        for level in levels
    }
    points["segments"] = {
        "key": config.SEGMENT_COLUMN,
        "why": (
            "stage 2 measured addr2 at a top-value share of 0.9901 on train, so a per-market "
            "calibration has one market. ProductCD is the partition with five levels none of "
            "which stage 2 put in the rare tail; the same mechanic runs on it. ADR 0029"
        ),
        "per_segment": evaluation.segment_report(segments, n_days, bands),
        "global_bands_applied_per_segment": {
            level: {
                "val": evaluation.band_report(
                    parts["y_val"], p_val[segment_values == level], bands, n_days["val"]
                ),
                "test": evaluation.band_report(
                    parts["y_test"], p_test[segment_values_test == level], bands, n_days["test"]
                ),
            }
            for level, parts in segments.items()
        },
    }
    return curve, points


# --- shap ---------------------------------------------------------------------------------------


def build_shap(inputs: Inputs, search: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import shap

    model, bundle = load_shipped_model()
    columns = list(bundle["columns"])
    if bundle["path"] != "tree":
        raise SystemExit("SHAP here is written for the tree path; the shipped model is not on it")
    matrix = inputs.tree(columns)["test"]
    y_test = inputs.y["test"]
    scores = load_scores("shipped")["test"]

    rng = np.random.default_rng(config.SEED)
    sample = np.sort(
        rng.choice(len(matrix), size=min(SHAP_SAMPLE_ROWS, len(matrix)), replace=False)
    )
    explainer = shap.TreeExplainer(model)
    started = time.perf_counter()
    values = np.asarray(explainer.shap_values(matrix.iloc[sample].to_numpy(dtype="float32")))
    seconds = time.perf_counter() - started
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(-mean_abs)

    # Blocks the model can be read by. The native groups and the stage 3 encodings partition the
    # prepared columns; the stage 4 and 5 blocks are empty when the ablations retired them.
    native = {
        group: modelling.native_group_members(columns, group) for group in features.NATIVE_GROUPS
    }
    in_native = {c for cols in native.values() for c in cols}
    prepared_only = [
        c for c in columns if c in set(inputs.columns["prepared"]) and c not in in_native
    ]
    blocks = {
        **{f"native_{group}": cols for group, cols in native.items()},
        "stage_3_target_encodings": [
            c for c in prepared_only if c.endswith(encoders.TARGET_SUFFIX)
        ],
        "stage_3_frequency_encodings": [
            c for c in prepared_only if c.endswith(encoders.FREQUENCY_SUFFIX)
        ],
        "stage_3_vocabulary_codes": [
            c for c in prepared_only if c.endswith(encoders.VOCABULARY_SUFFIX)
        ],
        "stage_3_missing_indicators": [
            c
            for c in prepared_only
            if c.startswith(missing_policy.INDICATOR_PREFIX)
            or c == missing_policy.IDENTITY_INDICATOR
        ],
        "stage_3_amount": [c for c in prepared_only if c.startswith(config.AMOUNT_COLUMN)],
        "stage_4_causal": [c for c in columns if c in set(inputs.columns["causal"])],
        "stage_5_graph": [c for c in columns if c in set(inputs.columns["graph"])],
        "own_velocity": [c for c in columns if c.startswith("vel_")],
    }
    named = {c for cols in blocks.values() for c in cols}
    blocks["other_raw_columns"] = [c for c in columns if c not in named]
    index = {c: i for i, c in enumerate(columns)}

    global_report = envelope("shap", inputs.n_rows_train())
    global_report["model"] = {
        k: bundle[k] for k in ("family", "imbalance", "stack", "hyperparameters")
    }
    global_report["method"] = {
        "explainer": "shap.TreeExplainer on the shipped booster, tree_path_dependent",
        "n_rows": int(sample.size),
        "rows": "a seeded uniform sample of the test split",
        "output": "log-odds contributions; the expected value is the explainer's base value",
        "seconds": seconds,
        "shap_version": shap.__version__,
    }
    global_report["expected_value"] = float(np.ravel(explainer.expected_value)[-1])
    global_report["mean_abs_shap"] = [
        {"column": columns[i], "mean_abs_shap": float(mean_abs[i]), "rank": rank + 1}
        for rank, i in enumerate(order)
    ]
    global_report["by_block"] = {
        name: {
            "n_columns": len(cols),
            "mean_abs_shap_sum": float(sum(mean_abs[index[c]] for c in cols)),
            "mean_abs_shap_share": float(sum(mean_abs[index[c]] for c in cols) / mean_abs.sum()),
            "mean_abs_shap_per_column": float(np.mean([mean_abs[index[c]] for c in cols]))
            if cols
            else None,
        }
        for name, cols in blocks.items()
    }
    global_report["caveat"] = {
        "labels": "card-level: ADR 0001 measured the entity's label purity, so a transaction's "
        "attribution partly explains which card it is on rather than what the transaction did",
        "made_visible_by": "the transaction-level and card-level PR-AUC reported side by side in "
        "reports/model_comparison.json and reports/metric_variance.json",
    }

    # The example: the highest-scoring fraud transaction in the sample, deterministic.
    fraud_in_sample = sample[y_test[sample] == 1]
    example_position = int(np.argmax(scores[fraud_in_sample]))
    example_row = int(fraud_in_sample[example_position])
    local = int(np.flatnonzero(sample == example_row)[0])
    contributions = values[local]
    row_values = matrix.iloc[example_row]
    top = np.argsort(-np.abs(contributions))[:20]
    example = envelope("shap", inputs.n_rows_train())
    example["selection"] = (
        "the fraud transaction with the largest shipped score inside the SHAP sample"
    )
    example["transaction"] = {
        config.ID_COLUMN: int(inputs.frames["test"][config.ID_COLUMN].iloc[example_row]),
        "label": int(y_test[example_row]),
        "score": float(scores[example_row]),
        "entity_id": str(inputs.frames["test"][config.ENTITY_ID_COLUMN].iloc[example_row]),
        "n_rows_on_entity_in_test": int(
            (
                inputs.frames["test"][config.ENTITY_ID_COLUMN]
                == inputs.frames["test"][config.ENTITY_ID_COLUMN].iloc[example_row]
            ).sum()
        ),
    }
    example["expected_value"] = global_report["expected_value"]
    example["sum_of_contributions"] = float(contributions.sum())
    example["contributions"] = [
        {
            "column": columns[i],
            "value": _clean(row_values.iloc[i]),
            "shap": float(contributions[i]),
        }
        for i in top
    ]
    example["caveat"] = global_report["caveat"]
    return global_report, example


# --- entry point --------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 training and evaluation.")
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    parser.add_argument("--sections", nargs="*", default=list(SECTIONS), choices=list(SECTIONS))
    parser.add_argument("--search-draws", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    inputs: Inputs | None = None
    for section in args.sections:
        print(f"section {section}")
        section_started = time.perf_counter()
        if section == "matrices":
            build_matrices(args.transactions, args.identity)
            inputs = None
            continue
        if inputs is None:
            inputs = Inputs()
        if section == "comparison":
            report = build_comparison(inputs)
            report["seconds"] = time.perf_counter() - section_started
            write(report, COMPARISON_PATH)
        elif section == "ablations":
            report = build_ablations(inputs, read(COMPARISON_PATH))
            report["seconds"] = time.perf_counter() - section_started
            write(report, ABLATIONS_PATH)
        elif section == "search":
            report = build_search(
                inputs, read(COMPARISON_PATH), read(ABLATIONS_PATH), args.search_draws
            )
            report["seconds"] = time.perf_counter() - section_started
            write(report, SEARCH_PATH)
        elif section == "variance":
            report = build_variance(inputs, read(COMPARISON_PATH))
            report["seconds"] = time.perf_counter() - section_started
            write(report, VARIANCE_PATH)
        elif section == "thresholds":
            curve, points = build_thresholds(inputs, read(SEARCH_PATH))
            curve["seconds"] = points["seconds"] = time.perf_counter() - section_started
            write(curve, THRESHOLD_CURVE_PATH)
            write(points, OPERATING_POINTS_PATH)
        else:
            global_report, example = build_shap(inputs, read(SEARCH_PATH))
            global_report["seconds"] = example["seconds"] = time.perf_counter() - section_started
            write(global_report, SHAP_GLOBAL_PATH)
            write(example, SHAP_EXAMPLE_PATH)
        gc.collect()

    print(f"\ndone in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
