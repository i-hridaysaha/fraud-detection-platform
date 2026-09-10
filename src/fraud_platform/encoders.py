"""Four encoders for the categorical and identifier columns, every one fitted on train alone.

    frequency          value to its train frequency. Unseen values encode to zero, which is the
                       true answer: the training window saw it zero times.
    target             value to a smoothed, lagged fraud rate. The lag is the whole argument and
                       ADR 0014 is where it is made.
    vocabulary         a bounded level set for tree-native categorical handling, with the rare
                       tail collapsed on the threshold stage 2 measured and an explicit token
                       for a level the training window never saw.
    free text          DeviceInfo and id_31 normalised to a bounded token, structurally, with
                       the coverage that achieves measured on all three splits.

Every fit is decorated with `data_loader.train_only`, so the guard runs before the fit and not
as a convention around it. A test calls each of them on a validation frame and requires the
raise.

**On target encoding and time.** A target encoding is a summary of labels, and a label that would
not have been available when the transaction was scored is a leak whatever window it came from.
Chargebacks mature over weeks, so the rate for a card on day d is not known on day d. This module
encodes a row on day d from train rows on day d - LAG or earlier, and nothing later. Where no
such rows exist, the encoding is null rather than a global fallback: a null is what the pipeline
already knows how to carry, and stage 2 established that the models being aimed at read nulls
natively.

**On entities.** Stage 0 and stage 2 both measured that 0.9675 of multi-transaction entities are
label-pure at the ADR 0001 key. Target-encoding the entity key on top of that is close to
encoding the label, and no lag fixes it: the problem is not that the labels are recent, it is
that they belong to the same entity. `entity_target_encoding_value` measures it at several lags
so the decision is taken against a number. ADR 0015.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraud_platform import config, transforms
from fraud_platform.data_loader import train_only

# --- choices ------------------------------------------------------------------------------

# A level below the rare-tail line is collapsed into this token, and a level the training window
# never saw becomes the other. They are different states and merging them would hide the second.
RARE_TOKEN = "(rare)"
UNSEEN_TOKEN = "(unseen)"
MISSING_TOKEN = "(missing)"

# The rare tail: taking levels from the least frequent upwards, collapse while their rows total
# no more than this share of the column's non-null rows. 0.01 is the line stage 2 measured the
# tail against (univariate.json thresholds.rare_tail_share), so the collapse and the measurement
# use one number.
RARE_TAIL_SHARE = 0.01

# Frequency encoding is for the columns wide enough that a level set is not a practical
# representation. 50 is the width at which stage 2 flagged a column high-cardinality in
# bivariate.json, reused here rather than re-drawn.
FREQUENCY_MIN_CARDINALITY = 50

# Suffixes for the derived columns.
FREQUENCY_SUFFIX = "_freq"
TARGET_SUFFIX = "_te"
VOCABULARY_SUFFIX = "_cat"
NORMALISED_SUFFIX = "_norm"

# Default smoothing and lag. Both are swept and both are chosen against validation rather than
# picked; these are the values the sweeps in reports/encoding_spec.json settled on, and the
# script writes the sweep next to them.
DEFAULT_SMOOTHING = 20.0
DEFAULT_LAG_DAYS = 7

# The sweep grids. Powers of roughly two either side of the default, wide enough that the chosen
# value is not at an end of the range unless the data puts it there.
SMOOTHING_GRID: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
LAG_GRID: tuple[int, ...] = (0, 1, 3, 7, 14, 28)

# Never target-encoded, whatever else is done with them. The entity key recovers the label
# rather than describing behaviour: measured at a validation AUC of 0.9877 on the rows whose
# entity the training window saw and 0.4915 on the rows it did not, against a stage 2 entity
# label purity of 0.9675. card_start_day is a component of that key and carries the highest PSI
# in the frame. ADR 0015 and ADR 0011.
NEVER_TARGET_ENCODED: tuple[str, ...] = (
    config.ENTITY_ID_COLUMN,
    config.CARD_START_DAY_COLUMN,
)

# Identifier-like columns: the ones a frequency or target encoding is for. Everything else
# categorical goes through the vocabulary encoder and nothing more.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    *config.CARD_COLUMNS,
    *config.ADDRESS_COLUMNS,
    *config.EMAIL_COLUMNS,
    *config.PRODUCT_COLUMNS,
    "DeviceInfo",
    "DeviceType",
    "id_30",
    "id_31",
    "id_33",
)


# --- free text normalisation ----------------------------------------------------------------

_BUILD_SUFFIX = re.compile(r"\s+build/.*$", re.IGNORECASE)
_VERSION_TOKEN = re.compile(r"\b\d+(?:\.\d+)*\b")
_TRAILING_VERSION = re.compile(r"\d+(?:\.\d+)*")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")
_PLATFORM_SUFFIX = re.compile(r"\bfor\s+(\w+)$")


def normalise_device_info(value: object) -> str | None:
    """DeviceInfo to a structural token: no brand table, no knowledge this repo has not measured.

    Four steps, each visible in the data rather than inferred from what a device is called.
    Lowercase. Drop the `Build/...` suffix, which stage 2 found on the model strings and which
    carries a firmware identifier rather than a device. Collapse `rv:57.0` and `Trident/7.0`,
    which are engine tokens carrying a version. Otherwise take the first token before the first
    separator and strip its digits, so `SM-J700M` and `SM-G610M` land together.

    The result is a token and not a brand. `sm` is the prefix a large family of model strings
    starts with; this module does not claim to know whose. Naming it would be a fact from
    outside the repo, and the token works without one.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    low = str(value).strip().lower()
    if not low:
        return None
    low = _BUILD_SUFFIX.sub("", low).strip()
    if low.startswith("rv:"):
        return "rv"
    if low.startswith("trident"):
        return "trident"
    head = _NON_ALNUM.split(low, maxsplit=1)[0]
    head = _TRAILING_VERSION.sub("", head).strip("-_")
    return head or "other"


def normalise_browser(value: object) -> str | None:
    """id_31 to `family|platform`, with the version removed.

    Stage 2 measured id_31 at PSI 1.500 against test with 19 levels test holds that train does
    not, and said the family is stable while the version is not. This drops the version and keeps
    the `for android` style platform suffix as its own field, because those two are different
    facts about the row and the version is the half that turns over.

    `generic` is dropped as a word: it appears as a version stand-in (`safari generic`) rather
    than as part of a family name.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    low = str(value).strip().lower()
    if not low:
        return None
    platform = "none"
    match = _PLATFORM_SUFFIX.search(low)
    if match:
        platform = match.group(1)
        low = low[: match.start()].strip()
    low = _VERSION_TOKEN.sub("", low)
    low = low.replace("generic", "")
    low = _WHITESPACE.sub(" ", low).strip()
    return f"{low or 'unknown'}|{platform}"


FREE_TEXT_NORMALISERS = {
    "DeviceInfo": normalise_device_info,
    "id_31": normalise_browser,
}


def is_categorical_like(series: pd.Series) -> bool:
    """Whether a column should go through the vocabulary encoder.

    Three dtypes qualify and the third is the one that catches people out: the normalisers
    return pandas string columns, not object columns, so a predicate written as `dtype == object`
    silently skips exactly the columns this stage went to the trouble of normalising.
    """
    dtype = series.dtype
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or dtype is np.dtype("O")
        or bool(pd.api.types.is_string_dtype(dtype))
    )


def normalised_name(column: str) -> str:
    return f"{column}{NORMALISED_SUFFIX}"


def add_normalised_free_text(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach `<column>_norm` for every free-text column present. Idempotent."""
    out = frame.copy()
    for column, normalise in FREE_TEXT_NORMALISERS.items():
        if column not in frame.columns:
            continue
        out[normalised_name(column)] = frame[column].astype("object").map(normalise)
    return out


def normalisation_coverage(
    train: pd.DataFrame, later: Mapping[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    """What the normalisation buys, measured on train and on every later split.

    Two numbers per split and they answer different questions. `in_vocabulary_share` is how much
    of the split the bounded token set covers. `unseen_rows` is how many rows carry a token the
    training window never saw, before and after normalisation, which is the number stage 2's
    id_31 finding is about: a browser version string turns over between windows and a browser
    family does not.
    """
    rows: list[dict[str, Any]] = []
    for column, normalise in FREE_TEXT_NORMALISERS.items():
        if column not in train.columns:
            continue
        raw_train = train[column].astype("object").dropna()
        norm_train = raw_train.map(normalise)
        seen_raw = set(raw_train.unique())
        seen_norm = set(norm_train.unique())
        entry: dict[str, Any] = {
            "column": column,
            "normalised_column": normalised_name(column),
            "train": {
                "n_present": len(raw_train),
                "n_levels_raw": int(raw_train.nunique()),
                "n_levels_normalised": int(norm_train.nunique()),
            },
            "splits": [],
        }
        for name, part in later.items():
            raw = part[column].astype("object").dropna()
            if raw.empty:
                continue
            norm = raw.map(normalise)
            unseen_raw = ~raw.isin(seen_raw)
            unseen_norm = ~norm.isin(seen_norm)
            entry["splits"].append(
                {
                    "split": name,
                    "n_present": len(raw),
                    "n_levels_raw": int(raw.nunique()),
                    "n_levels_normalised": int(norm.nunique()),
                    "unseen_rows_raw": int(unseen_raw.sum()),
                    "unseen_levels_raw": int(raw[unseen_raw].nunique()),
                    "unseen_rows_normalised": int(unseen_norm.sum()),
                    "unseen_levels_normalised": int(norm[unseen_norm].nunique()),
                }
            )
        rows.append(entry)
    return rows


# --- vocabulary --------------------------------------------------------------------------------


def vocabulary_name(column: str) -> str:
    return f"{column}{VOCABULARY_SUFFIX}"


@train_only
def fit_category_vocabulary(
    frame: pd.DataFrame,
    columns: Sequence[str],
    rare_tail_share: float = RARE_TAIL_SHARE,
) -> dict[str, Any]:
    """A bounded level set per column, with the rare tail collapsed and unseen levels named.

    The tail is taken the way stage 2 measured it: sort the levels by frequency ascending and
    collapse while the rows they hold total no more than `rare_tail_share` of the column's
    non-null rows. That makes the collapse threshold a count read off the train distribution
    rather than a round number, and it means the columns stage 2 reported a large tail for
    (DeviceInfo at 636 of 1,546 levels, id_33 at 135 of 183) are the ones that shrink.

    Three tokens are reserved and they mean three different things: a level below the line, a
    level the training window never saw, and a null. Merging any two of them would erase
    something stage 2 measured.
    """
    per_column: dict[str, Any] = {}
    for column in sorted(set(columns)):
        if column not in frame.columns:
            continue
        present = frame[column].astype("object").dropna()
        if present.empty:
            continue
        counts: pd.Series = present.value_counts()
        ascending: pd.Series = counts.sort_values(kind="stable")
        cumulative = ascending.to_numpy(dtype="float64").cumsum()
        below_the_line = cumulative <= rare_tail_share * len(present)
        rare_levels = [str(level) for level in ascending.index[below_the_line]]
        rare_counts = ascending.to_numpy(dtype="int64")[below_the_line]
        collapsed = set(rare_levels)
        kept = [str(level) for level in counts.index if str(level) not in collapsed]
        kept_counts = [int(count) for level, count in counts.items() if str(level) not in collapsed]
        per_column[column] = {
            "n_levels_train": len(counts),
            "n_levels_kept": len(kept),
            "n_levels_collapsed": len(rare_levels),
            "n_rows_collapsed": int(rare_counts.sum()),
            "share_rows_collapsed": float(rare_counts.sum()) / len(present),
            "min_kept_count": min(kept_counts) if kept_counts else None,
            "max_collapsed_count": int(rare_counts.max()) if len(rare_counts) else None,
            "levels": sorted(kept),
            "collapsed_levels": sorted(rare_levels),
        }
    return {
        "rare_tail_share": rare_tail_share,
        "rare_token": RARE_TOKEN,
        "unseen_token": UNSEEN_TOKEN,
        "missing_token": MISSING_TOKEN,
        "n_train_rows": len(frame),
        "n_columns": len(per_column),
        "columns": per_column,
    }


def apply_category_vocabulary(frame: pd.DataFrame, fitted: Mapping[str, Any]) -> pd.DataFrame:
    """Map each column onto its vocabulary, adding `<column>_cat` as a pandas categorical.

    The category set is the same object on every frame the encoder is applied to, so the integer
    code for a level does not depend on which rows happen to be in the frame. That matters for
    the tree-native categorical path, where the code is what the model splits on.
    """
    out = frame.copy()
    for column, spec in fitted["columns"].items():
        if column not in frame.columns:
            continue
        levels = list(spec["levels"])
        categories = [
            *levels,
            fitted["rare_token"],
            fitted["unseen_token"],
            fitted["missing_token"],
        ]
        known = set(levels)
        collapsed = set(spec["collapsed_levels"])
        values = frame[column].astype("object")

        def resolve(value: object, known: set[str] = known, collapsed: set[str] = collapsed) -> str:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return str(fitted["missing_token"])
            text = str(value)
            if text in known:
                return text
            # A level the training window saw and put below the rare line is not the same thing
            # as a level it never saw at all, and the two tokens keep them apart.
            return str(fitted["rare_token"] if text in collapsed else fitted["unseen_token"])

        out[vocabulary_name(column)] = pd.Categorical(values.map(resolve), categories=categories)
    return out


def vocabulary_application_report(
    frame: pd.DataFrame, fitted: Mapping[str, Any], split: str
) -> list[dict[str, Any]]:
    """How many rows of a split land in each of the three reserved tokens."""
    encoded = apply_category_vocabulary(frame, fitted)
    rows: list[dict[str, Any]] = []
    for column in sorted(fitted["columns"]):
        name = vocabulary_name(column)
        if name not in encoded.columns:
            continue
        series = encoded[name].astype("object")
        rows.append(
            {
                "split": split,
                "column": column,
                "n_rows": len(series),
                "n_unseen": int((series == fitted["unseen_token"]).sum()),
                "n_rare": int((series == fitted["rare_token"]).sum()),
                "n_missing": int((series == fitted["missing_token"]).sum()),
                "n_known": int(
                    (
                        ~series.isin(
                            [
                                fitted["unseen_token"],
                                fitted["rare_token"],
                                fitted["missing_token"],
                            ]
                        )
                    ).sum()
                ),
            }
        )
    return rows


# --- frequency ----------------------------------------------------------------------------------


def frequency_name(column: str) -> str:
    return f"{column}{FREQUENCY_SUFFIX}"


@train_only
def fit_frequency_encoding(
    frame: pd.DataFrame,
    columns: Sequence[str],
    min_cardinality: int = FREQUENCY_MIN_CARDINALITY,
) -> dict[str, Any]:
    """Train frequency per level, for the columns wide enough to need one.

    A column narrower than `min_cardinality` is skipped rather than encoded. ProductCD takes five
    levels and every one of them is a level a model can split on directly; replacing them with
    five counts throws the identity away and buys nothing.
    """
    tables: dict[str, dict[str, int]] = {}
    summary: dict[str, Any] = {}
    for column in sorted(set(columns)):
        if column not in frame.columns:
            continue
        counts = frame[column].astype("object").dropna().value_counts()
        if len(counts) < min_cardinality:
            summary[column] = {"encoded": False, "cardinality": len(counts)}
            continue
        tables[column] = {str(level): int(count) for level, count in counts.items()}
        summary[column] = {
            "encoded": True,
            "cardinality": len(counts),
            "max_count": int(counts.max()),
            "min_count": int(counts.min()),
            "n_levels_appearing_once": int((counts == 1).sum()),
        }
    return {
        "min_cardinality": min_cardinality,
        "unseen_value": 0,
        "note": (
            "an unseen level encodes to 0, which is the measurement and not a fallback: the "
            "training window contains it zero times"
        ),
        "n_train_rows": len(frame),
        "n_columns_encoded": len(tables),
        "per_column": summary,
        "tables": tables,
    }


def apply_frequency_encoding(frame: pd.DataFrame, fitted: Mapping[str, Any]) -> pd.DataFrame:
    """Attach `<column>_freq`. A null stays null; an unseen level becomes 0."""
    out = frame.copy()
    for column, table in fitted["tables"].items():
        if column not in frame.columns:
            continue
        values = frame[column].astype("object")
        encoded = values.map(lambda v, table=table: table.get(str(v), 0))
        out[frequency_name(column)] = encoded.where(values.notna()).astype("float32")
    return out


# --- target encoding with a label lag -------------------------------------------------------

# Packing constant for the (level, day) lookup key. Day indices are offset so a boundary earlier
# than the first day stays non-negative, and the multiplier is comfortably wider than any day
# index the data reaches.
_DAY_OFFSET = 128
_DAY_STRIDE = 4096


@dataclass
class LaggedTargetTable:
    """Cumulative label counts per level per day, for one column.

    `keys` is sorted on `level_code * _DAY_STRIDE + (day + _DAY_OFFSET)`, which makes it sorted
    on the pair, so one `searchsorted` answers "how many rows of this level fell on or before
    this day" for every row at once. Storing increments rather than a dense level-by-day grid is
    what makes this affordable on a column with 164,702 levels: the memory is one array the
    length of the training split, not one grid of levels by days.
    """

    level_index: pd.Index
    keys: npt.NDArray[np.int64]
    cumulative_y: npt.NDArray[np.float64]
    block_start: npt.NDArray[np.int64]
    n_levels: int
    stats: dict[str, Any] = field(default_factory=dict)


def target_name(column: str) -> str:
    return f"{column}{TARGET_SUFFIX}"


def _level_strings(values: pd.Series) -> pd.Series:
    """Levels as strings, with nulls left as nulls so they never join a level's statistics."""
    as_object = values.astype("object")
    return as_object.where(as_object.isna(), as_object.astype("string"))


def _build_table(
    levels: pd.Series, y: npt.NDArray[np.int64], day: npt.NDArray[np.int64]
) -> LaggedTargetTable:
    present = levels.notna().to_numpy()
    codes, uniques = pd.factorize(levels[present], use_na_sentinel=True)
    order = np.lexsort((day[present], codes))
    sorted_codes = codes[order].astype("int64")
    sorted_days = day[present][order].astype("int64")
    sorted_y = y[present][order].astype("float64")

    keys = sorted_codes * _DAY_STRIDE + (sorted_days + _DAY_OFFSET)
    cumulative_y = np.cumsum(sorted_y)
    block_start = np.searchsorted(sorted_codes, np.arange(len(uniques), dtype="int64"), side="left")

    return LaggedTargetTable(
        level_index=pd.Index(uniques),
        keys=keys,
        cumulative_y=cumulative_y,
        block_start=block_start.astype("int64"),
        n_levels=len(uniques),
        stats={"n_rows": int(present.sum()), "n_missing": int((~present).sum())},
    )


def level_codes(table: LaggedTargetTable, values: pd.Series) -> npt.NDArray[np.int64]:
    """Position of each value in the fitted level set, or -1 for a null or an unseen level."""
    return np.asarray(
        table.level_index.get_indexer(pd.Index(_level_strings(values))), dtype="int64"
    )


def _counts_at(
    table: LaggedTargetTable, codes: npt.NDArray[np.int64], boundary: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rows and label sum available for each query, at that query's own day boundary."""
    n = np.zeros(len(codes), dtype="float64")
    y = np.zeros(len(codes), dtype="float64")
    known = codes >= 0
    if not known.any():
        return n, y
    query = codes[known] * _DAY_STRIDE + (boundary[known] + _DAY_OFFSET)
    index = np.searchsorted(table.keys, query, side="right")
    start = table.block_start[codes[known]]
    index = np.maximum(index, start)
    y_before = np.where(start > 0, table.cumulative_y[np.maximum(start - 1, 0)], 0.0)
    y_upto = np.where(index > 0, table.cumulative_y[np.maximum(index - 1, 0)], 0.0)
    n[known] = (index - start).astype("float64")
    y[known] = y_upto - y_before
    return n, y


def encode_column(
    table: LaggedTargetTable,
    codes: npt.NDArray[np.int64],
    boundary: npt.NDArray[np.int64],
    prior: npt.NDArray[np.float64],
    smoothing: float,
    missing: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """The smoothed rate for one column, one row at a time, at each row's lagged boundary.

    `(y + m * prior) / (n + m)`. Three cases and they get three different answers.

    A level the training window has counted zero times before this row's boundary falls to the
    prior, which is what the formula says and what the window knows: nothing specific to the
    level, so the base rate. A level the window has never seen at all is the same case.

    A null input encodes to null. There is no level to look up, and the vocabulary encoder's
    `(missing)` token is where that state is represented.

    A row whose boundary precedes every training row also encodes to null, because at that point
    the window knows nothing at all, prior included. That is the price of the lag and it is paid
    on the first `lag_days` days of the training split.
    """
    n, y = _counts_at(table, codes, boundary)
    with np.errstate(invalid="ignore", divide="ignore"):
        encoded = (y + smoothing * prior) / (n + smoothing)
    return np.where(np.isfinite(prior) & ~missing, encoded, np.nan)


@train_only
def _fit_unchecked(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str = config.TARGET,
    lag_days: int = DEFAULT_LAG_DAYS,
    smoothing: float = DEFAULT_SMOOTHING,
) -> dict[str, Any]:
    """`fit_target_encoding` without the exclusion list, for the probe that produced the list."""
    return _fit(frame, columns, target, lag_days, smoothing)


@train_only
def fit_target_encoding(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str = config.TARGET,
    lag_days: int = DEFAULT_LAG_DAYS,
    smoothing: float = DEFAULT_SMOOTHING,
) -> dict[str, Any]:
    """Fit the lagged, smoothed target encoder on the train split.

    The encoder holds cumulative label counts per level per day, plus the cumulative prior. Both
    are read at the row's own lagged boundary at transform time, so one fitted object serves
    every row without a per-row refit and without a row ever seeing a label from its own day.

    The tables do not depend on the lag or the smoothing. That is what makes the sweeps
    affordable: one fit, and every point on the grid is a different read of the same tables.
    """
    banned = set(NEVER_TARGET_ENCODED) & set(columns)
    if banned:
        raise ValueError(
            f"{sorted(banned)} cannot be target encoded. ADR 0015 excluded the entity key on a "
            "measurement, and the exclusion is enforced here rather than left to the caller."
        )
    return _fit(frame, columns, target, lag_days, smoothing)


def _fit(
    frame: pd.DataFrame,
    columns: Sequence[str],
    target: str,
    lag_days: int,
    smoothing: float,
) -> dict[str, Any]:
    """The fit itself. Both public entry points route here after their own checks."""
    y = frame[target].to_numpy(dtype="int64")
    day = transforms.day_index(frame).to_numpy(dtype="int64")

    order = np.argsort(day, kind="stable")
    prior_days = day[order]
    prior_n = np.arange(1, len(prior_days) + 1, dtype="int64")
    prior_y = np.cumsum(y[order].astype("float64"))

    tables: dict[str, LaggedTargetTable] = {}
    summary: dict[str, Any] = {}
    for column in sorted(set(columns)):
        if column not in frame.columns:
            continue
        table = _build_table(_level_strings(frame[column]), y, day)
        tables[column] = table
        summary[column] = {
            "n_levels": table.n_levels,
            "n_rows_with_level": table.stats["n_rows"],
            "n_rows_missing": table.stats["n_missing"],
        }

    return {
        "lag_days": int(lag_days),
        "smoothing": float(smoothing),
        "target": target,
        "n_train_rows": len(frame),
        "train_prior": float(y.mean()),
        "min_train_day": int(day.min()),
        "max_train_day": int(day.max()),
        "undefined_policy": (
            "a row whose lagged boundary precedes every training row encodes to null. There is "
            "no global fallback: a fallback would put a number from outside the row's own "
            "information set into the column, which is the leak this lag exists to prevent."
        ),
        "per_column": summary,
        "_tables": tables,
        "_prior_days": prior_days,
        "_prior_n": prior_n,
        "_prior_y": prior_y,
    }


def prior_at(fitted: Mapping[str, Any], boundary: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """The training window's fraud rate over every row on or before each boundary day."""
    index = np.searchsorted(fitted["_prior_days"], boundary, side="right")
    n = np.where(index > 0, fitted["_prior_n"][np.maximum(index - 1, 0)], 0)
    y = np.where(index > 0, fitted["_prior_y"][np.maximum(index - 1, 0)], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(np.where(n > 0, y / np.maximum(n, 1), np.nan), dtype="float64")


def apply_target_encoding(
    frame: pd.DataFrame,
    fitted: Mapping[str, Any],
    lag_days: int | None = None,
    smoothing: float | None = None,
) -> pd.DataFrame:
    """Attach `<column>_te`, each row read at its own day minus the lag.

    `lag_days` and `smoothing` override the fitted values without refitting. The sweeps use that;
    the pipeline does not, and passes neither.
    """
    lag = int(fitted["lag_days"] if lag_days is None else lag_days)
    m = float(fitted["smoothing"] if smoothing is None else smoothing)
    out = frame.copy()
    boundary = transforms.day_index(frame).to_numpy(dtype="int64") - lag
    prior = prior_at(fitted, boundary)
    for column, table in fitted["_tables"].items():
        if column not in frame.columns:
            continue
        codes = level_codes(table, frame[column])
        missing = frame[column].isna().to_numpy()
        encoded = encode_column(table, codes, boundary, prior, m, missing)
        out[target_name(column)] = pd.Series(encoded, index=frame.index, dtype="float64")
    return out


# --- sweeps -----------------------------------------------------------------------------------


def _auc(y_true: npt.NDArray[np.int64], score: npt.NDArray[np.float64]) -> dict[str, Any]:
    """AUC over the rows where the score is defined, and a count of the rows where it is not."""
    from sklearn.metrics import roc_auc_score

    defined = np.isfinite(score)
    n_defined = int(defined.sum())
    if n_defined == 0 or len(np.unique(y_true[defined])) < 2:
        return {"auc": None, "n_scored": n_defined, "n_undefined": int((~defined).sum())}
    return {
        "auc": float(roc_auc_score(y_true[defined], score[defined])),
        "n_scored": n_defined,
        "n_undefined": int((~defined).sum()),
    }


def target_encoding_sweep(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int],
    smoothings: Sequence[float],
    target: str = config.TARGET,
) -> list[dict[str, Any]]:
    """Every (lag, smoothing) pair, scored by the AUC of the encoded column alone on validation.

    No model is fitted. The encoded column is already a probability-like score, so its AUC
    against the label is what the encoding is worth on its own, which is what the sweep is
    choosing between. The train AUC is reported beside it and is not a performance estimate: the
    gap between the two is the leak signature, and at lag 0 it is what makes the leak visible.

    The encoder is fitted once for the whole grid. The cumulative tables do not depend on either
    swept parameter, so every point is a different read of one fit.
    """
    fitted = fit_target_encoding(train, columns, target=target)
    y_train = train[target].to_numpy(dtype="int64")
    y_val = validation[target].to_numpy(dtype="int64")
    day_train = transforms.day_index(train).to_numpy(dtype="int64")
    day_val = transforms.day_index(validation).to_numpy(dtype="int64")
    cached = {
        column: (
            level_codes(table, train[column]),
            level_codes(table, validation[column]),
            train[column].isna().to_numpy(),
            validation[column].isna().to_numpy(),
        )
        for column, table in fitted["_tables"].items()
    }

    rows: list[dict[str, Any]] = []
    for lag in lags:
        boundary_train = day_train - int(lag)
        boundary_val = day_val - int(lag)
        prior_train = prior_at(fitted, boundary_train)
        prior_val = prior_at(fitted, boundary_val)
        for smoothing in smoothings:
            for column in sorted(fitted["_tables"]):
                table = fitted["_tables"][column]
                train_codes, val_codes, train_missing, val_missing = cached[column]
                train_result = _auc(
                    y_train,
                    encode_column(
                        table, train_codes, boundary_train, prior_train, smoothing, train_missing
                    ),
                )
                val_result = _auc(
                    y_val,
                    encode_column(
                        table, val_codes, boundary_val, prior_val, smoothing, val_missing
                    ),
                )
                rows.append(
                    {
                        "column": column,
                        "lag_days": int(lag),
                        "smoothing": float(smoothing),
                        "train_auc": train_result["auc"],
                        "validation_auc": val_result["auc"],
                        "n_train_scored": train_result["n_scored"],
                        "n_train_undefined": train_result["n_undefined"],
                        "n_validation_scored": val_result["n_scored"],
                        "n_validation_undefined": val_result["n_undefined"],
                        "train_minus_validation": (
                            None
                            if train_result["auc"] is None or val_result["auc"] is None
                            else train_result["auc"] - val_result["auc"]
                        ),
                    }
                )
    return rows


def entity_target_encoding_value(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    entity_column: str,
    lags: Sequence[int],
    smoothing: float = DEFAULT_SMOOTHING,
    target: str = config.TARGET,
    decompose_at_lag: int | None = None,
) -> dict[str, Any]:
    """What a target encoding of the entity key is worth, at several lags.

    Two numbers per lag and the pair is the finding. The train AUC says how well the encoding
    separates rows the encoder was fitted on. The validation AUC says how well it separates rows
    it was not. Stage 2 measured that 0.9675 of multi-transaction entities are label-pure and
    stage 1 measured that 0.6598 of test rows are on entities the training window never saw, so
    the shape to look for is a large train number, a small validation number, and most of
    validation unscored. If that is what comes back, the encoding is a memory of the training
    labels rather than a feature, and no lag repairs it: the labels it leaks are not stale, they
    are the same entity's. ADR 0015.
    """
    # The ban in `fit_target_encoding` exists because of this measurement, so this measurement
    # cannot go through it. `_fit_unchecked` is the same fit without the exclusion list.
    fitted = _fit_unchecked(train, [entity_column], target=target)
    table = fitted["_tables"][entity_column]
    y_train = train[target].to_numpy(dtype="int64")
    y_val = validation[target].to_numpy(dtype="int64")
    day_train = transforms.day_index(train).to_numpy(dtype="int64")
    day_val = transforms.day_index(validation).to_numpy(dtype="int64")
    train_codes = level_codes(table, train[entity_column])
    val_codes = level_codes(table, validation[entity_column])
    train_missing = train[entity_column].isna().to_numpy()
    val_missing = validation[entity_column].isna().to_numpy()

    per_lag: list[dict[str, Any]] = []
    for lag in lags:
        boundary_train = day_train - int(lag)
        boundary_val = day_val - int(lag)
        train_result = _auc(
            y_train,
            encode_column(
                table,
                train_codes,
                boundary_train,
                prior_at(fitted, boundary_train),
                smoothing,
                train_missing,
            ),
        )
        val_result = _auc(
            y_val,
            encode_column(
                table,
                val_codes,
                boundary_val,
                prior_at(fitted, boundary_val),
                smoothing,
                val_missing,
            ),
        )
        per_lag.append(
            {
                "lag_days": int(lag),
                "smoothing": float(smoothing),
                "train_auc": train_result["auc"],
                "validation_auc": val_result["auc"],
                "n_train_scored": train_result["n_scored"],
                "n_train_undefined": train_result["n_undefined"],
                "n_validation_scored": val_result["n_scored"],
                "n_validation_undefined": val_result["n_undefined"],
                "validation_coverage": val_result["n_scored"] / len(y_val) if len(y_val) else None,
            }
        )
    seen = set(train[entity_column].unique())
    seen_mask = validation[entity_column].isin(seen).to_numpy()

    # The decisive decomposition. A target encoding of the entity key can look strong on
    # validation for two quite different reasons, and they call for opposite decisions. Either it
    # carries an entity's own history, which is a feature a fraud system is entitled to use, or it
    # carries nothing but the fact that the training window has seen this entity at all, which is
    # a join indicator wearing a rate's clothing. Splitting validation on that fact separates them.
    indicator_auc = _auc(y_val, seen_mask.astype("float64"))
    best_lag = int(decompose_at_lag if decompose_at_lag is not None else min(int(x) for x in lags))
    boundary_best = day_val - best_lag
    encoded_best = encode_column(
        table, val_codes, boundary_best, prior_at(fitted, boundary_best), smoothing, val_missing
    )
    on_seen = _auc(y_val[seen_mask], encoded_best[seen_mask])
    on_unseen = _auc(y_val[~seen_mask], encoded_best[~seen_mask])

    return {
        "entity_column": entity_column,
        "entity_key": list(config.ENTITY_KEY_COLUMNS),
        "n_train_entities": int(train[entity_column].nunique()),
        "n_validation_entities": int(validation[entity_column].nunique()),
        "n_validation_rows_on_a_seen_entity": int(seen_mask.sum()),
        "share_validation_rows_on_a_seen_entity": float(seen_mask.mean()),
        "decomposition": {
            "at_lag_days": best_lag,
            "note": (
                "validation split on whether the training window ever saw the entity. The "
                "indicator AUC is what the seen or unseen fact is worth on its own; the "
                "seen-rows AUC is what the encoded rate is worth where an entity history exists. "
                "If the first explains the second, the column is a join indicator."
            ),
            "seen_indicator_auc": indicator_auc["auc"],
            "auc_on_seen_rows": on_seen["auc"],
            "n_seen_rows": int(seen_mask.sum()),
            "auc_on_unseen_rows": on_unseen["auc"],
            "n_unseen_rows": int((~seen_mask).sum()),
        },
        "per_lag": per_lag,
    }
