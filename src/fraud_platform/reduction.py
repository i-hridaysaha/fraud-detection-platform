"""V-column reduction: three strategies, measured against each other rather than chosen.

Stage 2 established the two structures the V block has. It goes missing in blocks: 14 multi-
column blocks with nulls, columns inside one null on identical rows. And inside those blocks it
is redundant: at Spearman 0.95, 339 columns across 14 blocks reduce to 207 representatives, so a
representative-per-group reduction removes 132.

Three ways to spend that structure, and they are not the same trade.

    representative   keep one column per correlated group and drop the rest. The kept column is
                     a real column, so a later explanation names something a reviewer can look
                     up. It throws away whatever the discarded members held that the
                     representative does not.
    pca              fit a PCA inside each missingness block and keep the components that reach
                     an explained-variance target. Keeps more of the variance per column kept.
                     The components are not columns anyone can name, which stage 7 will feel.
    block_mean       one standardised mean per block. The smallest frame by a wide margin and
                     the crudest: it assumes the block is one thing measured several ways.

PCA and the block mean both run inside a missingness block rather than across the frame, and
that is not a convenience. Columns in a block are null on identical rows, so within a block the
present rows are the same rows for every column and there is a complete matrix to decompose.
Across blocks there is not, and filling one to make one would put invented values into the
decomposition.

Every fit is on train rows, through the guard. The comparison is a single small gradient boosting
probe per strategy, fitted on train and scored on validation, and it is a probe and not a model:
stage 5 is where models are built. ADR 0016 records what came back and what was chosen.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config
from fraud_platform.data_loader import train_only

# --- choices ------------------------------------------------------------------------------

# Components are kept until they explain this share of the block's variance, so a block whose
# columns are copies of each other keeps one component and a block that is genuinely wide keeps
# several. A fixed component count per block would have hidden that difference.
PCA_EXPLAINED_VARIANCE = 0.95

# No block keeps more components than this, whatever the variance target says. It bounds the
# worst case on a block that does not compress at all.
PCA_MAX_COMPONENTS = 12

PCA_PREFIX = "vpca_"
BLOCK_MEAN_PREFIX = "vblockmean_"

STRATEGIES: tuple[str, ...] = ("all", "representative", "pca", "block_mean")

# The probe. Small, fixed, and identical across strategies, because the comparison is between
# column sets and anything that varied between them would confound it.
PROBE_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 31,
    "min_child_samples": 200,
    "learning_rate": 0.1,
}


def v_blocks(
    all_blocks: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    """The missingness blocks, restricted to the V columns still in play.

    A block that keeps fewer than two V columns is not a block for this purpose: there is
    nothing to decompose and nothing to average.
    """
    pool = set(columns)
    blocks: list[dict[str, Any]] = []
    for block in all_blocks:
        members = sorted(c for c in block["columns"] if c in pool)
        if len(members) < 2:
            continue
        blocks.append(
            {
                "block": members[0],
                "n_columns": len(members),
                "columns": members,
                "missing_rate_train": float(block["missing_rate"]),
            }
        )
    blocks.sort(key=lambda b: str(b["block"]))
    return blocks


@train_only
def fit_block_pca(
    frame: pd.DataFrame,
    blocks: Sequence[Mapping[str, Any]],
    explained_variance: float = PCA_EXPLAINED_VARIANCE,
    max_components: int = PCA_MAX_COMPONENTS,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """One PCA per missingness block, fitted on the rows where the block is present.

    Standardisation comes first and is not optional. Stage 2 measured 212 columns with an
    absolute skew above 10 and the V columns run over wildly different scales, so a PCA on raw
    values would return the largest column with a small rotation applied to it.
    """
    from sklearn.decomposition import PCA

    fitted: dict[str, Any] = {}
    for block in blocks:
        columns = list(block["columns"])
        present = frame[columns].notna().all(axis=1)
        matrix = frame.loc[present, columns].astype("float64")
        if len(matrix) < len(columns) + 1:
            continue
        mean = matrix.mean()
        std = matrix.std(ddof=0).replace(0.0, 1.0)
        scaled = ((matrix - mean) / std).to_numpy()
        n_components = min(len(columns), max_components, len(matrix))
        pca = PCA(n_components=n_components, random_state=seed)
        pca.fit(scaled)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cumulative, explained_variance) + 1)
        keep = max(1, min(keep, n_components))
        fitted[str(block["block"])] = {
            "columns": columns,
            "n_components": keep,
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_[:keep]],
            "cumulative_explained_variance": float(cumulative[keep - 1]),
            "mean": {c: float(v) for c, v in mean.items()},
            "std": {c: float(v) for c, v in std.items()},
            "components": pca.components_[:keep].tolist(),
            "n_rows_fitted": len(matrix),
        }
    return {
        "explained_variance_target": explained_variance,
        "max_components": max_components,
        "n_blocks": len(fitted),
        "n_components_total": sum(int(b["n_components"]) for b in fitted.values()),
        "blocks": fitted,
    }


def apply_block_pca(frame: pd.DataFrame, fitted: Mapping[str, Any]) -> pd.DataFrame:
    """Project each block onto its components. Rows where the block is missing stay missing."""
    out: dict[str, Any] = {}
    for name, block in fitted["blocks"].items():
        columns = list(block["columns"])
        available = [c for c in columns if c in frame.columns]
        if len(available) != len(columns):
            continue
        matrix = frame[columns].astype("float64")
        present = matrix.notna().all(axis=1).to_numpy()
        mean = np.array([block["mean"][c] for c in columns], dtype="float64")
        std = np.array([block["std"][c] for c in columns], dtype="float64")
        components = np.array(block["components"], dtype="float64")
        scaled = (matrix.to_numpy(dtype="float64") - mean) / std
        scaled = np.nan_to_num(scaled, nan=0.0)
        projected = scaled @ components.T
        for index in range(block["n_components"]):
            column = f"{PCA_PREFIX}{name}_{index}"
            out[column] = np.where(present, projected[:, index], np.nan)
    return pd.DataFrame(out, index=frame.index)


@train_only
def fit_block_mean(frame: pd.DataFrame, blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-block standardisation parameters for the block-mean strategy.

    The mean is taken over standardised columns for the same reason the PCA standardises: an
    unstandardised mean over a block is the block's largest-scale column plus noise.
    """
    fitted: dict[str, Any] = {}
    for block in blocks:
        columns = list(block["columns"])
        matrix = frame[columns].astype("float64")
        mean = matrix.mean()
        std = matrix.std(ddof=0).replace(0.0, 1.0)
        fitted[str(block["block"])] = {
            "columns": columns,
            "mean": {c: float(v) for c, v in mean.items()},
            "std": {c: float(v) for c, v in std.items()},
        }
    return {"n_blocks": len(fitted), "blocks": fitted}


def apply_block_mean(frame: pd.DataFrame, fitted: Mapping[str, Any]) -> pd.DataFrame:
    """One column per block: the mean of that block's standardised columns."""
    out: dict[str, Any] = {}
    for name, block in fitted["blocks"].items():
        columns = [c for c in block["columns"] if c in frame.columns]
        if len(columns) != len(block["columns"]):
            continue
        mean = np.array([block["mean"][c] for c in block["columns"]], dtype="float64")
        std = np.array([block["std"][c] for c in block["columns"]], dtype="float64")
        scaled = (frame[block["columns"]].to_numpy(dtype="float64") - mean) / std
        # A row where the whole block is null has nothing to average and the mean is null, which
        # is the answer this pipeline wants. numpy warns about it; the warning is the expected
        # case here and not a problem to fix.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice", category=RuntimeWarning
            )
            out[f"{BLOCK_MEAN_PREFIX}{name}"] = np.nanmean(scaled, axis=1)
    return pd.DataFrame(out, index=frame.index)


def probe(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    val_x: pd.DataFrame,
    val_y: np.ndarray,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """One small gradient boosting fit, train in and validation out. A probe, not a model."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    if train_x.shape[1] == 0:
        return {"n_columns": 0, "train_auc": None, "validation_auc": None}
    model = lgb.LGBMClassifier(**PROBE_PARAMS, random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(train_x, train_y)
    train_score = np.asarray(model.predict_proba(train_x))[:, 1]
    val_score = np.asarray(model.predict_proba(val_x))[:, 1]
    return {
        "n_columns": int(train_x.shape[1]),
        "train_auc": float(roc_auc_score(train_y, train_score)),
        "validation_auc": float(roc_auc_score(val_y, val_score)),
    }


def compare_strategies(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    pool: Sequence[str],
    representatives: Sequence[str],
    blocks: Sequence[Mapping[str, Any]],
    target: str = config.TARGET,
    seed: int = config.SEED,
    seeds: Sequence[int] = (),
) -> dict[str, Any]:
    """Column count and validation AUC for each of the four column sets.

    `all` is the baseline: every V column that survived the other three drop rules, unreduced.
    It is here because a reduction that costs validation AUC against the unreduced frame is a
    reduction that is buying width with signal, and without the baseline that trade is invisible.
    """
    y_train = train[target].to_numpy(dtype="int64")
    y_val = validation[target].to_numpy(dtype="int64")
    columns = [c for c in pool if c in train.columns]

    pca_fitted = fit_block_pca(train, blocks, seed=seed)
    mean_fitted = fit_block_mean(train, blocks)

    sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "all": (train[columns], validation[columns]),
        "representative": (
            train[[c for c in representatives if c in train.columns]],
            validation[[c for c in representatives if c in validation.columns]],
        ),
        "pca": (apply_block_pca(train, pca_fitted), apply_block_pca(validation, pca_fitted)),
        "block_mean": (
            apply_block_mean(train, mean_fitted),
            apply_block_mean(validation, mean_fitted),
        ),
    }

    results: list[dict[str, Any]] = []
    for name in STRATEGIES:
        train_x, val_x = sets[name]
        result = probe(train_x, y_train, val_x, y_val, seed=seed)
        result["strategy"] = name
        results.append(result)

    # Run-to-run variance, measured rather than assumed. Stage 2 left this open on its own
    # single-feature screen and said so; a comparison between four strategies is worthless
    # without knowing how much of the spread between them one probe would produce on its own.
    variance: list[dict[str, Any]] = []
    for name in STRATEGIES:
        train_x, val_x = sets[name]
        aucs = [probe(train_x, y_train, val_x, y_val, seed=s)["validation_auc"] for s in seeds]
        clean = [a for a in aucs if a is not None]
        variance.append(
            {
                "strategy": name,
                "seeds": list(seeds),
                "validation_aucs": clean,
                "min": min(clean) if clean else None,
                "max": max(clean) if clean else None,
                "spread": (max(clean) - min(clean)) if len(clean) > 1 else None,
            }
        )

    return {
        "probe": {
            "estimator": "lightgbm.LGBMClassifier",
            "params": {**PROBE_PARAMS, "random_state": seed},
            "note": (
                "one fit per strategy on the V columns alone. It compares column sets, not "
                "models, and no number here is a performance estimate."
            ),
        },
        "n_pool_columns": len(columns),
        "pca": {k: v for k, v in pca_fitted.items() if k != "blocks"},
        "block_mean": {"n_blocks": mean_fitted["n_blocks"]},
        "blocks": [
            {
                "block": b["block"],
                "n_columns": b["n_columns"],
                "missing_rate_train": b["missing_rate_train"],
                "n_components": pca_fitted["blocks"].get(str(b["block"]), {}).get("n_components"),
            }
            for b in blocks
        ],
        "results": results,
        "seed_variance": {
            "note": (
                "the same column set, the same rows, different LightGBM seeds. The spread here "
                "is the floor on what a difference between two strategies has to clear before it "
                "means anything."
            ),
            "per_strategy": variance,
            "largest_spread": max(
                (v["spread"] for v in variance if v["spread"] is not None), default=None
            ),
        },
    }
