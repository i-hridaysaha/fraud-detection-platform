"""Feature stacks, the two preparation paths, the model families and the imbalance strategies.

Stage 6 crosses four things: a feature stack (stage 3 prepared columns, plus stage 4's causal
features, plus stage 5's graph candidates, with the native C, D, M and V groups switchable), a
preparation path, a model family and an imbalance strategy. Each is a named object here so the
driver can cross them without a second definition of any of them, and so a test can assert that
the artifact lists exactly what the module offers.

The two paths are stage 3's, not new ones. The tree path receives the prepared frame as it is:
nulls left missing (ADR 0012), no clipping (ADR 0013), every categorical as the integer code of
its vocabulary level. The complete-data path receives what a model that cannot read a null
needs: stage 3's imputer (a train median, or the sentinel below the train minimum for the D
columns), stage 3's clip at the 0.999 train quantile, one column per vocabulary level, and a
train mean and standard deviation per numeric column. Everything fitted here is fitted on the
train split through the guard, and the fitted objects are plain dictionaries so the artifact can
record them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

from fraud_platform import config, encoders, features, graph_features, missing_policy, transforms
from fraud_platform.data_loader import train_only

FAMILIES: tuple[str, ...] = ("logistic_regression", "random_forest", "xgboost")

# Which stage 3 path each family goes through. A linear model reads a value, so it gets the
# complete-data path; a tree cuts on rank and reads a null natively, so it gets the tree path.
# Stage 3 measured both tree libraries learning a direction for a null (reports/prep/
# missing_policy.json, library_verification), which is what makes the tree path a path at all.
PATH_BY_FAMILY: dict[str, str] = {
    "logistic_regression": "complete_data",
    "random_forest": "tree",
    "xgboost": "tree",
}

IMBALANCE_STRATEGIES: tuple[str, ...] = ("none", "class_weight", "smote")

# Nearest minority neighbours a synthetic row is drawn between. Chosen, not measured.
SMOTE_NEIGHBOURS = 5

# Columns of the prepared frame no model reads. The label and the identifier for the obvious
# reason; the clock columns because a model that reads the day index learns the weekly fraud
# rate and nothing else (stage 5 measured the component size doing exactly that); the entity key
# components because ADR 0015 excluded the entity from every encoding.
EXCLUDED_COLUMNS: tuple[str, ...] = (
    config.ID_COLUMN,
    config.TARGET,
    config.TIME_COLUMN,
    transforms.DAY_INDEX_COLUMN,
    transforms.WEEK_INDEX_COLUMN,
    transforms.DOW_POSITION_COLUMN,
    config.ENTITY_ID_COLUMN,
    config.CARD_START_DAY_COLUMN,
)

# Suffixes stage 3 derives from a raw column. When a native group is switched off, the derived
# columns go with the raw ones, or the switch would switch off nothing for a categorical block
# whose raw column no model reads in the first place.
_DERIVED_SUFFIXES: tuple[str, ...] = (
    encoders.VOCABULARY_SUFFIX,
    encoders.FREQUENCY_SUFFIX,
    encoders.TARGET_SUFFIX,
)

# The encodings of a raw free-text column are not read either: stage 3 normalised `DeviceInfo`
# and `id_31` so that the normalised column replaces the raw one rather than supplementing it
# (`fraud_platform.prepare`), and the raw string stays in the frame for a reviewer, not a model.
# Measured on the cached validation frame: the raw `DeviceInfo` vocabulary holds 913 levels
# against 67 for the normalised one, so reading both would hand a linear model 913 one-hot
# columns of build numbers the normaliser exists to remove.
REPLACED_BY_NORMALISED: tuple[str, ...] = tuple(
    f"{column}{suffix}" for column in encoders.FREE_TEXT_NORMALISERS for suffix in _DERIVED_SUFFIXES
)

# Hyperparameters per family. Chosen once, before any number was read, and held fixed across the
# comparison and the ablations so that the only thing changing between two fits is the thing being
# measured. The search in the driver moves the shipped family's values, and the artifact records
# both sets side by side.
DEFAULT_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "solver": "lbfgs", "max_iter": 5000, "tol": 1e-4},
    "random_forest": {
        "n_estimators": 200,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 600,
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "n_jobs": -1,
    },
}


# --- feature stacks ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureStack:
    """Which blocks a model sees. Four switches, every combination nameable.

    `causal` is stage 4's feature block, `graph` is stage 5's candidate set (off by default,
    as `graph_features.DEFAULT_ENABLED` records), and `native_groups` is the switch ADR 0021
    asked for, passed through to `features.FeatureConfig` so the two stages agree on what a
    group is.
    """

    causal: bool = True
    graph: bool = graph_features.DEFAULT_ENABLED
    native_groups: tuple[str, ...] = features.NATIVE_GROUPS

    @property
    def name(self) -> str:
        blocks = ["prepared"]
        if self.causal:
            blocks.append("causal")
        if self.graph:
            blocks.append("graph")
        native = ",".join(self.native_groups) if self.native_groups else "none"
        return "+".join(blocks) + f"|native={native}"

    def without_group(self, group: str) -> FeatureStack:
        if group not in features.NATIVE_GROUPS:
            raise KeyError(f"unknown native group {group!r}")
        return replace(self, native_groups=tuple(g for g in self.native_groups if g != group))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "causal": self.causal,
            "graph": self.graph,
            "native_groups": list(self.native_groups),
        }


DEFAULT_STACK = FeatureStack()


def prepared_feature_columns(prepared: pd.DataFrame) -> list[str]:
    """Columns of a stage 3 prepared frame a model may read, sorted.

    A raw categorical column is not read: its vocabulary column (`<name>_cat`) carries the same
    levels with the rare tail collapsed and unseen levels tokenised, and reading both would be
    reading one column twice. The encodings of a raw free-text column are not read either, its
    normalised column's are (`REPLACED_BY_NORMALISED`). Everything numeric that is not excluded
    is read.
    """
    excluded = set(EXCLUDED_COLUMNS) | set(REPLACED_BY_NORMALISED)
    out: list[str] = []
    for column in prepared.columns:
        name = str(column)
        if name in excluded:
            continue
        series = prepared[column]
        if encoders.is_categorical_like(series) and not name.endswith(encoders.VOCABULARY_SUFFIX):
            continue
        out.append(name)
    return sorted(out)


def native_group_members(columns: Sequence[str], group: str) -> list[str]:
    """Every column in `columns` that belongs to a native group, raw and derived alike.

    Raw names from `features.NATIVE_GROUP_COLUMNS`, the stage 3 suffix columns built from them,
    the missing-block indicator whose representative is one of them, and for the V block the
    reduced columns that replaced it. An indicator whose block spans two groups is named after
    its representative and follows that column's group; the artifact lists what was dropped.
    """
    if group not in features.NATIVE_GROUP_COLUMNS:
        raise KeyError(
            f"unknown native group {group!r}, expected one of {list(features.NATIVE_GROUP_COLUMNS)}"
        )
    raw = set(features.NATIVE_GROUP_COLUMNS[group])
    derived = {f"{name}{suffix}" for name in raw for suffix in _DERIVED_SUFFIXES}
    indicators = {missing_policy.indicator_name(name) for name in raw}
    named = raw | derived | indicators
    prefixes = features._DERIVED_GROUP_PREFIXES.get(group, ())
    return [c for c in columns if c in named or (bool(prefixes) and str(c).startswith(prefixes))]


def stack_columns(
    prepared: Sequence[str],
    causal: Sequence[str],
    graph: Sequence[str],
    stack: FeatureStack = DEFAULT_STACK,
) -> list[str]:
    """The column list one stack reads, in a fixed order: prepared, then causal, then graph."""
    off: set[str] = set()
    for group in features.NATIVE_GROUPS:
        if group not in stack.native_groups:
            off.update(native_group_members(prepared, group))
    out = [c for c in prepared if c not in off]
    if stack.causal:
        out.extend(c for c in causal if c not in out)
    if stack.graph:
        out.extend(c for c in graph if c not in out)
    return out


# --- the tree path ----------------------------------------------------------------------------


def tree_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """The frame as a tree reads it: floats with their nulls, categoricals as level codes.

    The code is the position of the level in the vocabulary the encoder fixed on train, so it is
    the same integer for the same level on every split. A null in a vocabulary column cannot
    occur: the encoder maps it to its own token. Booleans and indicators become small integers.
    """
    out: dict[str, Any] = {}
    for column in columns:
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            out[column] = series.cat.codes.to_numpy(dtype="int16")
        elif series.dtype == np.dtype("bool"):
            out[column] = series.to_numpy(dtype="int8")
        elif pd.api.types.is_integer_dtype(series.dtype):
            out[column] = series.to_numpy()
        else:
            out[column] = series.to_numpy(dtype="float32")
    return pd.DataFrame(out, index=frame.index)


def interpolable_mask(matrix: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Which columns a synthetic row interpolates: the floats. Codes and indicators are copied."""
    return np.array(
        [pd.api.types.is_float_dtype(matrix[c].dtype) for c in matrix.columns], dtype=bool
    )


# --- the complete-data path -------------------------------------------------------------------


@train_only
def fit_complete_data_path(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """Fit stage 3's complete-data path on the train split, for one column list.

    Order: impute, clip, one-hot, standardise. The imputer and the clip are stage 3's own
    functions with stage 3's own column lists, so the path here and the one
    reports/prep/missing_policy.json and reports/prep/transforms.json describe are one path.
    A feature from stage 4 or 5 that is null on train gets a median and an indicator of its own,
    which is the same policy stage 3 gave the raw columns: an indicator beside the fill, never
    a fill alone.
    """
    numeric = [c for c in columns if not isinstance(frame[c].dtype, pd.CategoricalDtype)]
    categorical = [c for c in columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]

    sentinel_columns = [c for c in numeric if c in config.TIMEDELTA_COLUMNS]
    median_columns = [c for c in numeric if c not in config.TIMEDELTA_COLUMNS]
    imputer = missing_policy.fit_imputer(frame, median_columns, sentinel_columns)

    # Indicators for the columns stage 3 did not already give one: a stage 4 or 5 feature with a
    # null on train. A stage 3 column already sits under a block indicator.
    later_stage = set(features.feature_names()) | set(graph_features.ALL_FEATURES)
    extra_indicators = sorted(
        c for c in numeric if c in later_stage and bool(frame[c].isna().any())
    )

    clip_columns = [c for c in transforms.CLIP_COLUMNS if c in numeric]
    clip = transforms.fit_clip_bounds(frame, clip_columns)

    filled = missing_policy.apply_imputer(frame[numeric], imputer)
    filled = transforms.apply_clip(filled, clip)
    values = filled.to_numpy(dtype="float64")
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    # A constant column standardises to zero rather than to a division by zero.
    std = np.where(std > 0, std, 1.0)

    levels = {c: [str(level) for level in frame[c].cat.categories] for c in categorical}
    return {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "extra_indicator_columns": extra_indicators,
        "imputer": imputer,
        "clip": clip,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "levels": levels,
        "n_output_columns": len(numeric)
        + len(extra_indicators)
        + sum(len(v) for v in levels.values()),
    }


def apply_complete_data_path(
    frame: pd.DataFrame, fitted: Mapping[str, Any]
) -> tuple[npt.NDArray[np.float32], list[str], npt.NDArray[np.bool_]]:
    """The dense matrix the linear model reads, its column names, and the interpolable mask.

    Column order: the numeric columns standardised, then the extra indicators, then one column
    per vocabulary level. The mask marks the standardised numerics as the columns a synthetic
    row interpolates; indicators and one-hot columns are copied from the base row.
    """
    numeric = list(fitted["numeric_columns"])
    extra = list(fitted["extra_indicator_columns"])
    indicators = frame[extra].isna().to_numpy(dtype="float32") if extra else None

    filled = missing_policy.apply_imputer(frame[numeric], fitted["imputer"])
    filled = transforms.apply_clip(filled, fitted["clip"])
    values = filled.to_numpy(dtype="float64")
    mean = np.asarray(fitted["mean"], dtype="float64")
    std = np.asarray(fitted["std"], dtype="float64")
    standardised = ((values - mean) / std).astype("float32")

    blocks: list[npt.NDArray[np.float32]] = [standardised]
    names: list[str] = list(numeric)
    if indicators is not None:
        blocks.append(indicators)
        names.extend(f"{c}_missing" for c in extra)
    for column in fitted["categorical_columns"]:
        levels = fitted["levels"][column]
        codes = frame[column].cat.codes.to_numpy(dtype="int64")
        one_hot = np.zeros((len(frame), len(levels)), dtype="float32")
        present = codes >= 0
        one_hot[np.flatnonzero(present), codes[present]] = 1.0
        blocks.append(one_hot)
        names.extend(f"{column}={level}" for level in levels)

    matrix = np.hstack(blocks)
    mask = np.zeros(matrix.shape[1], dtype=bool)
    mask[: len(numeric)] = True
    return matrix, names, mask


# --- imbalance strategies ---------------------------------------------------------------------


def class_weight_ratio(y: npt.NDArray[np.int_]) -> float:
    """Negatives per positive on the fit rows, which is what `balanced` weighting applies."""
    n_positive = int(y.sum())
    n_negative = int(y.size - n_positive)
    if n_positive == 0:
        raise ValueError("no positive rows to weight")
    return n_negative / n_positive


def smote_plan(
    space: npt.NDArray[np.float32],
    y: npt.NDArray[np.int_],
    seed: int,
    k: int = SMOTE_NEIGHBOURS,
) -> dict[str, Any]:
    """Which minority rows to interpolate between, and how far, to balance the classes 1:1.

    `space` is the matrix the neighbours are found in: the complete-data path's standardised
    numerics, so every distance is defined and every column is on one scale. The plan is only
    indices and weights, so both paths materialise the same synthetic rows from it, each in its
    own representation. One plan per seed, and the same plan for every family at that seed.

    Each synthetic row is a point on the segment between a minority row and one of its k nearest
    minority neighbours, at a uniform random position along it. The number of synthetic rows is
    the number that makes the two classes equal, which is the balance class weighting reaches
    by another route, so the two strategies are compared at the same effective balance.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    positive = np.flatnonzero(y == 1)
    negative_count = int(y.size - positive.size)
    n_synthetic = negative_count - int(positive.size)
    if positive.size <= k:
        raise ValueError(
            f"need more than {k} positive rows to draw neighbours, got {positive.size}"
        )
    if n_synthetic <= 0:
        return {
            "k": k,
            "n_synthetic": 0,
            "base": np.zeros(0, dtype="int64"),
            "neighbour": np.zeros(0, dtype="int64"),
            "position": np.zeros(0, dtype="float32"),
        }

    minority = np.ascontiguousarray(space[positive], dtype="float32")
    finder = NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(minority)
    _, neighbours = finder.kneighbors(minority)
    # The nearest neighbour of a row is itself, in column 0; the rest are the candidates.
    candidates = neighbours[:, 1:]

    rng = np.random.default_rng(seed)
    base_local = rng.integers(0, positive.size, n_synthetic)
    pick = rng.integers(0, k, n_synthetic)
    neighbour_local = candidates[base_local, pick]
    return {
        "k": k,
        "n_synthetic": int(n_synthetic),
        "base": positive[base_local].astype("int64"),
        "neighbour": positive[neighbour_local].astype("int64"),
        "position": rng.random(n_synthetic, dtype="float32"),
    }


def materialise_smote(
    matrix: npt.NDArray[Any],
    y: npt.NDArray[np.int_],
    plan: Mapping[str, Any],
    interpolable: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[Any], npt.NDArray[np.int_]]:
    """The fit matrix with the plan's synthetic rows appended, labelled positive.

    Interpolable columns take `base + position * (neighbour - base)`; a null in either endpoint
    stays a null on the tree path, because a value between a number and no number is no number.
    Every other column is copied from the base row, so a level code stays a level code.
    """
    base = np.asarray(plan["base"])
    if base.size == 0:
        return matrix, y
    neighbour = np.asarray(plan["neighbour"])
    position = np.asarray(plan["position"], dtype="float32")[:, None]

    synthetic = np.array(matrix[base], copy=True)
    left = matrix[base][:, interpolable].astype("float32")
    right = matrix[neighbour][:, interpolable].astype("float32")
    synthetic[:, interpolable] = left + position * (right - left)

    out_x = np.vstack([matrix, synthetic])
    out_y = np.concatenate([np.asarray(y), np.ones(base.size, dtype=np.asarray(y).dtype)])
    return out_x, out_y


# --- the families -----------------------------------------------------------------------------


def make_model(
    family: str,
    imbalance: str,
    y_fit: npt.NDArray[np.int_],
    seed: int = config.SEED,
    hyperparameters: Mapping[str, Any] | None = None,
) -> Any:
    """An unfitted estimator for one family, one strategy, one seed.

    Class weighting is `balanced` for the sklearn families and `scale_pos_weight` at the same
    negatives-per-positive ratio for xgboost, which is the same reweighting expressed the way
    each library takes it. Under SMOTE the rows are already balanced and no weight is applied.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, expected one of {list(FAMILIES)}")
    if imbalance not in IMBALANCE_STRATEGIES:
        raise ValueError(
            f"unknown imbalance strategy {imbalance!r}, expected one of {list(IMBALANCE_STRATEGIES)}"
        )
    params = dict(DEFAULT_HYPERPARAMETERS[family])
    if hyperparameters:
        params.update(hyperparameters)
    weighted = imbalance == "class_weight"

    if family == "logistic_regression":
        return LogisticRegression(
            **params, class_weight="balanced" if weighted else None, random_state=seed
        )
    if family == "random_forest":
        return RandomForestClassifier(
            **params, class_weight="balanced" if weighted else None, random_state=seed
        )
    return XGBClassifier(
        **params,
        scale_pos_weight=class_weight_ratio(y_fit) if weighted else 1.0,
        random_state=seed,
        objective="binary:logistic",
        eval_metric="aucpr",
    )


def convergence(model: Any) -> dict[str, Any] | None:
    """Whether a logistic regression stopped because it converged or because it ran out.

    `n_iter_` at `max_iter` means the solver hit the cap, and lbfgs stops there whether or not
    the gradient is small. A baseline reported without this is a baseline of unknown quality.
    """
    if not isinstance(model, LogisticRegression):
        return None
    n_iter = int(np.max(model.n_iter_))
    return {
        "n_iter": n_iter,
        "max_iter": int(model.max_iter),
        "converged": n_iter < int(model.max_iter),
        "solver": model.solver,
        "tol": float(model.tol),
    }


def score(model: Any, matrix: Any) -> npt.NDArray[np.float64]:
    """Positive-class probability, as a float64 vector."""
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype="float64")


# --- time-aware cross-validation ---------------------------------------------------------------


def expanding_window_folds(ts: npt.NDArray[np.int64], n_folds: int) -> list[dict[str, Any]]:
    """Folds that never train on a row later than their validation rows.

    The rows are cut into `n_folds + 1` contiguous blocks at quantiles of the timestamp, on the
    stage 0 rule that assignment is `ts < boundary` so a shared timestamp never straddles two
    blocks. Fold k fits on blocks 0 to k and validates on block k + 1, so every fold's fit rows
    end before its validation rows begin. Indices are positions into `ts`.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    quantiles = np.linspace(0.0, 1.0, n_folds + 2)[1:-1]
    boundaries = np.quantile(ts, quantiles)
    edges = [-np.inf, *boundaries, np.inf]
    folds: list[dict[str, Any]] = []
    for k in range(n_folds):
        fit = np.flatnonzero(ts < edges[k + 1])
        validate = np.flatnonzero((ts >= edges[k + 1]) & (ts < edges[k + 2]))
        if fit.size == 0 or validate.size == 0:
            raise ValueError(f"fold {k} has an empty side; the timestamps do not spread enough")
        folds.append(
            {
                "fold": k,
                "fit": fit,
                "validate": validate,
                "fit_end_ts": int(ts[fit].max()),
                "validate_start_ts": int(ts[validate].min()),
                "validate_end_ts": int(ts[validate].max()),
            }
        )
    return folds
