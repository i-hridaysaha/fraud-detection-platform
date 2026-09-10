"""Loading, typing, splitting, and the train-only guard.

This is the whole data layer. It does three jobs and nothing else.

**Typing.** The joined frame is 590,540 rows by 434 columns. Read with pandas defaults it is
2,080,582,559 bytes; read through the dtype map built here it is 955,046,315. The map is passed
to read_csv rather than applied afterwards, so the wide float64 frame is never materialised.
Every dtype in it comes from a measured range, not a guess. See ADR 0005.

**Splitting.** `time_based_split` cuts on the fixed TransactionDT boundaries from ADR 0002. It
never shuffles and never samples. There is no random_state argument to pass, which is the point:
a caller cannot accidentally ask for a random split.

**The train-only guard.** `assert_train_only` raises when a frame that is about to be fitted on
contains a row at or after the training boundary. Every fitted object in every later stage goes
through it. See ADR 0006 for why this is a runtime check rather than a convention, and read the
docstring for what it cannot see.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from fraud_platform import config

F = TypeVar("F", bound=Callable[..., Any])


class TrainOnlyError(RuntimeError):
    """Raised when something would be fitted on rows outside the training window."""


def build_dtype_map(columns: Iterable[str]) -> dict[str, str]:
    """The read_csv dtype argument for a given set of column names.

    Four rules, in order: a measured null-free integer column gets the narrowest integer type its
    range fits; a string column gets `category`; TransactionAmt stays float64; everything else is
    float32. A column the spec has never seen falls through to float32 and read_csv raises if it
    turns out to hold text, which is the behaviour we want. A silent object column is how a 434
    column frame quietly triples in size.
    """
    mapping: dict[str, str] = {}
    for column in columns:
        if column in config.INTEGER_COLUMN_DTYPES:
            mapping[column] = config.INTEGER_COLUMN_DTYPES[column]
        elif column in config.CATEGORICAL_COLUMNS:
            mapping[column] = "category"
        elif column in config.EXACT_FLOAT_COLUMNS:
            mapping[column] = "float64"
        else:
            mapping[column] = config.DEFAULT_NUMERIC_DTYPE
    return mapping


def _sort_categories(frame: pd.DataFrame) -> pd.DataFrame:
    """Put every categorical column's categories in sorted order.

    read_csv assigns category codes in the order values are first seen, which depends on how the
    file happens to be ordered. Sorting makes the code for a given value a function of the
    category set alone, so a frame loaded from a subset of columns and a frame loaded whole agree
    wherever their category sets do.
    """
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            categories = frame[column].cat.categories
            if not categories.is_monotonic_increasing:
                frame[column] = frame[column].cat.reorder_categories(categories.sort_values())
    return frame


def load_raw(
    transactions_path: Path | str | None = None,
    identity_path: Path | str | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read the two training files and left join identity onto transactions on TransactionID.

    The join is left because only 0.2442 of transactions have an identity row (stage 0 audit), and
    those rows are not a random sample: coverage is 0.5477 on fraud against 0.2332 on non-fraud.
    An inner join would silently drop three quarters of the data and skew what remains.

    `columns` restricts the read on both files. TransactionID is always read so the join still
    works. Column names that belong to neither file are ignored rather than raising, so a caller
    can pass one list covering both.
    """
    transactions_path = Path(transactions_path or config.TRANSACTIONS_PATH)
    identity_path = Path(identity_path or config.IDENTITY_PATH)

    transaction_columns = _header(transactions_path)
    identity_columns = _header(identity_path)

    if columns is not None:
        wanted = set(columns) | {config.ID_COLUMN}
        transaction_columns = [c for c in transaction_columns if c in wanted]
        identity_columns = [c for c in identity_columns if c in wanted]

    transactions = pd.read_csv(
        transactions_path,
        usecols=transaction_columns,
        dtype=build_dtype_map(transaction_columns),
    )

    if identity_columns == [config.ID_COLUMN]:
        return _sort_categories(transactions)

    identity = pd.read_csv(
        identity_path,
        usecols=identity_columns,
        dtype=build_dtype_map(identity_columns),
    )
    joined = transactions.merge(identity, on=config.ID_COLUMN, how="left")
    return _sort_categories(joined)


def _header(path: Path) -> list[str]:
    """Column names without reading a row of data."""
    return list(pd.read_csv(path, nrows=0).columns)


def frame_bytes(frame: pd.DataFrame) -> int:
    """Bytes the frame occupies, counting the payload behind every column."""
    return int(frame.memory_usage(deep=True).sum())


def add_entity_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach card_start_day and entity_id, the ADR 0001 entity key.

    card_start_day is floor(TransactionDT / 86400 - D1). entity_id pastes the three key components
    into one string so the key can go into a set or a groupby without a tuple. A null component
    becomes the literal "NA" rather than dropping the row, which is what ADR 0001 decided and what
    the stage 0 audit counted, so entity counts here match reports/audit.json.

    This is the entity key, not a feature. Feature aggregation grain is a stage 4 decision.
    """
    missing = [c for c in config.ENTITY_KEY_SOURCE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"entity key needs {missing}, which the frame does not carry")

    out = frame.copy()
    out[config.CARD_START_DAY_COLUMN] = np.floor(
        out[config.TIME_COLUMN] / config.SECONDS_PER_DAY - out["D1"]
    )

    parts = [
        out[column].astype("string").fillna(config.ENTITY_NULL_TOKEN)
        for column in config.ENTITY_KEY_COLUMNS
    ]
    entity_id = parts[0]
    for part in parts[1:]:
        entity_id = entity_id + "|" + part
    out[config.ENTITY_ID_COLUMN] = entity_id
    return out


def split_masks(frame: pd.DataFrame) -> dict[str, pd.Series[bool]]:
    """Boolean mask per split, on the fixed boundaries. Exhaustive and mutually exclusive."""
    dt = frame[config.TIME_COLUMN]
    return {
        "train": dt < config.TRAIN_END_DT,
        "val": (dt >= config.TRAIN_END_DT) & (dt < config.VAL_END_DT),
        "test": dt >= config.VAL_END_DT,
    }


def time_based_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut the frame into train, val and test on TransactionDT at the ADR 0002 boundaries.

    Returns the three parts in chronological order, each sorted by TransactionDT with the original
    index preserved so a caller can trace a row back to the frame it came from.

    There is no shuffle, no sample, and no random_state. Assignment is `dt < boundary` against
    two fixed integers, so the same input always gives the same three frames and a timestamp
    shared by several transactions cannot straddle a boundary.
    """
    if config.TIME_COLUMN not in frame.columns:
        raise KeyError(f"time_based_split needs {config.TIME_COLUMN}")

    masks = split_masks(frame)
    parts = tuple(
        frame[masks[name]].sort_values(config.TIME_COLUMN, kind="stable")
        for name in config.SPLIT_NAMES
    )
    return parts[0], parts[1], parts[2]


def assert_train_only(frame: pd.DataFrame, *, what: str) -> pd.DataFrame:
    """Raise unless every row of `frame` falls strictly before the training boundary.

    Route anything that learns from data through this before it learns: encoders, imputers,
    scalers, target statistics, quantile cutpoints, vocabulary lists, a threshold read off a
    distribution. `what` names the thing being fitted and appears in the error, because the
    failure is usually a call site three layers down that nobody remembered was fitting.

    It raises rather than warns, and it raises rather than asserts, so `python -O` does not
    remove it. Two cases that a naive max() check would wave through are failures here:

    - a frame with no TransactionDT column, because the guard cannot see the timestamps and
      refuses to claim it checked them;
    - an empty frame, because "no row is out of window" is vacuously true and is far more often
      a filter that matched nothing than a deliberate fit on zero rows.

    Returns the frame so a fit can be written as one expression.

    **What it cannot catch.** This checks where rows sit in time. It says nothing about where
    their *values* came from. It will pass a frame whose columns were already computed with a
    statistic taken over the whole file, a target encoding fitted earlier on val, or an aggregate
    carried in from a previous call, because by then the leak is inside the numbers and the
    timestamps are innocent. It cannot see a fit that never calls it: an inline `df.mean()` in
    some later module is invisible here, which is why ADR 0006 pairs it with the rule that fitted
    objects are constructed through it. It cannot see label leakage inside the training window,
    where the rows are legitimate and the target is what leaked. It cannot see a mutable estimator
    that is fitted once through the guard and refitted later without it. And it trusts
    config.TRAIN_END_DT: if the boundary is wrong, this happily enforces the wrong boundary.
    """
    if config.TIME_COLUMN not in frame.columns:
        raise TrainOnlyError(
            f"cannot fit {what}: frame has no {config.TIME_COLUMN} column, so the guard cannot "
            "tell which rows it holds"
        )
    if len(frame) == 0:
        raise TrainOnlyError(
            f"cannot fit {what}: frame is empty, and an empty frame passes a window check vacuously"
        )

    dt = frame[config.TIME_COLUMN]
    if dt.isna().any():
        raise TrainOnlyError(
            f"cannot fit {what}: {int(dt.isna().sum())} rows have a null {config.TIME_COLUMN}"
        )

    outside = dt >= config.TRAIN_END_DT
    n_outside = int(outside.sum())
    if n_outside:
        raise TrainOnlyError(
            f"cannot fit {what} on {n_outside} of {len(frame)} rows at or after "
            f"{config.TIME_COLUMN} {config.TRAIN_END_DT}: max seen is {int(dt.max())}. "
            "Fit on the train split only."
        )
    return frame


def train_only(fit: F) -> F:
    """Wrap a fit function so its frame argument goes through `assert_train_only` first.

    The frame is taken from the first positional argument, or from a `frame` keyword. Use it on
    anything whose name starts with fit:

        @train_only
        def fit_amount_scaler(frame: pd.DataFrame) -> StandardScaler: ...

    The decorator is the convenient path, not the enforcing one. It removes the excuse for
    skipping the check; it cannot stop a function that was never decorated. ADR 0006.
    """

    @functools.wraps(fit)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], pd.DataFrame):
            frame = args[0]
        elif isinstance(kwargs.get("frame"), pd.DataFrame):
            frame = kwargs["frame"]
        else:
            raise TrainOnlyError(
                f"cannot guard {fit.__qualname__}: no DataFrame in the first positional argument "
                "or in a `frame` keyword"
            )
        assert_train_only(frame, what=fit.__qualname__)
        return fit(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
