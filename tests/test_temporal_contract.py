"""The temporal contract, asserted directly rather than inferred from a metric.

A model score is a terrible leak detector: a leak makes the number go up, and a number going up
is what everyone was hoping for. So these tests state the contract as three properties of the
split itself, and check them without training anything.

1. Every train timestamp is strictly before every val timestamp, which is strictly before every
   test timestamp.
2. No TransactionID appears in more than one split, and no row is lost or duplicated.
3. `assert_train_only` refuses a fit on val or test rows.

`assert_temporal_contract` below is the contract, factored out so two tests can use it: one on
the real `time_based_split`, which must pass, and one on a shuffled split, which must fail. A
test that has never been seen to fail is not a guarantee, so the failing case is a test, not a
one-off exercise. `docs/notes/stage-01.md` records the manual version of the same check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, data_loader

ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "split_summary.json"

DAY = config.SECONDS_PER_DAY


def synthetic_frame(n_rows: int = 900) -> pd.DataFrame:
    """Rows spread evenly across the whole TransactionDT range, so all three splits are populated.

    Timestamps are deterministic. The fraud labels and card values are drawn from a generator
    seeded with config.SEED so the frame is the same on every run and on every machine.
    """
    rng = np.random.default_rng(config.SEED)
    dt = np.linspace(DAY, 15_811_131, n_rows).astype(np.int64)
    return pd.DataFrame(
        {
            config.ID_COLUMN: np.arange(2_987_000, 2_987_000 + n_rows, dtype=np.int64),
            config.TARGET: rng.integers(0, 2, n_rows),
            config.TIME_COLUMN: dt,
            "card1": rng.integers(1_000, 18_396, n_rows),
            "addr1": rng.integers(100, 500, n_rows).astype("float64"),
            "D1": rng.integers(0, 640, n_rows).astype("float64"),
        }
    )


def assert_temporal_contract(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, original: pd.DataFrame
) -> None:
    """The contract. Raises AssertionError naming which property broke."""
    time_column, id_column = config.TIME_COLUMN, config.ID_COLUMN

    assert len(train) and len(val) and len(test), "a split is empty"

    assert train[time_column].max() < val[time_column].min(), (
        f"train reaches {train[time_column].max()} but val starts at {val[time_column].min()}"
    )
    assert val[time_column].max() < test[time_column].min(), (
        f"val reaches {val[time_column].max()} but test starts at {test[time_column].min()}"
    )

    ids = {
        name: set(part[id_column])
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = ids[left] & ids[right]
        assert not shared, f"{len(shared)} ids in both {left} and {right}"

    assert len(train) + len(val) + len(test) == len(original), "rows lost or duplicated"
    assert ids["train"] | ids["val"] | ids["test"] == set(original[id_column]), "ids do not cover"


def random_split(
    frame: pd.DataFrame, fractions: tuple[float, float] = (0.70, 0.85)
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The split this repo does not use, kept only so a test can watch the contract reject it."""
    shuffled = frame.sample(frac=1.0, random_state=config.SEED)
    first, second = (round(f * len(frame)) for f in fractions)
    return shuffled.iloc[:first], shuffled.iloc[first:second], shuffled.iloc[second:]


# --- the contract on the split this repo uses ------------------------------------------------


def test_time_based_split_satisfies_the_contract() -> None:
    frame = synthetic_frame()
    train, val, test = data_loader.time_based_split(frame)
    assert_temporal_contract(train, val, test, frame)


def test_a_random_split_fails_the_contract() -> None:
    """The teeth. If this ever passes, the contract above has stopped checking anything."""
    frame = synthetic_frame()
    train, val, test = random_split(frame)
    with pytest.raises(AssertionError):
        assert_temporal_contract(train, val, test, frame)


def test_split_ignores_row_order() -> None:
    """Feeding the rows in a different order gives the same three splits.

    A split that read row position instead of the timestamp would fail this, and so would one
    that sampled.
    """
    frame = synthetic_frame()
    shuffled = frame.sample(frac=1.0, random_state=config.SEED)

    ordered = data_loader.time_based_split(frame)
    reordered = data_loader.time_based_split(shuffled)
    for a, b in zip(ordered, reordered, strict=True):
        assert set(a[config.ID_COLUMN]) == set(b[config.ID_COLUMN])
        assert list(a[config.TIME_COLUMN]) == list(b[config.TIME_COLUMN])


def test_split_is_returned_in_chronological_order() -> None:
    frame = synthetic_frame()
    for part in data_loader.time_based_split(frame):
        assert part[config.TIME_COLUMN].is_monotonic_increasing


def test_split_has_no_randomness_to_ask_for() -> None:
    """No random_state, no shuffle, no sample argument. A caller cannot request one by mistake."""
    import inspect

    parameters = set(inspect.signature(data_loader.time_based_split).parameters)
    assert parameters == {"frame"}


def test_split_boundaries_are_half_open_on_the_left() -> None:
    """A row exactly on a boundary belongs to the later split, so no timestamp straddles."""
    frame = synthetic_frame(n_rows=6).copy()
    frame[config.TIME_COLUMN] = [
        config.TRAIN_END_DT - 1,
        config.TRAIN_END_DT,
        config.TRAIN_END_DT + 1,
        config.VAL_END_DT - 1,
        config.VAL_END_DT,
        config.VAL_END_DT + 1,
    ]
    train, val, test = data_loader.time_based_split(frame)
    assert list(train[config.TIME_COLUMN]) == [config.TRAIN_END_DT - 1]
    assert list(val[config.TIME_COLUMN]) == [
        config.TRAIN_END_DT,
        config.TRAIN_END_DT + 1,
        config.VAL_END_DT - 1,
    ]
    assert list(test[config.TIME_COLUMN]) == [config.VAL_END_DT, config.VAL_END_DT + 1]


def test_split_needs_the_time_column() -> None:
    frame = synthetic_frame().drop(columns=[config.TIME_COLUMN])
    with pytest.raises(KeyError):
        data_loader.time_based_split(frame)


# --- the train-only guard --------------------------------------------------------------------


def test_assert_train_only_accepts_train_rows_and_returns_the_frame() -> None:
    train, _, _ = data_loader.time_based_split(synthetic_frame())
    assert data_loader.assert_train_only(train, what="scaler") is train


@pytest.mark.parametrize("split_index", [1, 2])
def test_assert_train_only_rejects_val_and_test_rows(split_index: int) -> None:
    parts = data_loader.time_based_split(synthetic_frame())
    with pytest.raises(data_loader.TrainOnlyError, match="cannot fit scaler"):
        data_loader.assert_train_only(parts[split_index], what="scaler")


def test_assert_train_only_rejects_a_frame_holding_one_late_row() -> None:
    """One row over the line is a violation. It does not need to be a majority."""
    train, val, _ = data_loader.time_based_split(synthetic_frame())
    contaminated = pd.concat([train, val.head(1)])
    with pytest.raises(data_loader.TrainOnlyError, match="on 1 of"):
        data_loader.assert_train_only(contaminated, what="encoder")


def test_assert_train_only_rejects_a_frame_without_the_time_column() -> None:
    train, _, _ = data_loader.time_based_split(synthetic_frame())
    with pytest.raises(data_loader.TrainOnlyError, match="no TransactionDT"):
        data_loader.assert_train_only(train.drop(columns=[config.TIME_COLUMN]), what="imputer")


def test_assert_train_only_rejects_an_empty_frame() -> None:
    """An empty frame passes a max() check vacuously, which is how a bad filter goes unnoticed."""
    train, _, _ = data_loader.time_based_split(synthetic_frame())
    with pytest.raises(data_loader.TrainOnlyError, match="empty"):
        data_loader.assert_train_only(train.head(0), what="imputer")


def test_assert_train_only_rejects_null_timestamps() -> None:
    train, _, _ = data_loader.time_based_split(synthetic_frame())
    holed = train.copy()
    holed[config.TIME_COLUMN] = holed[config.TIME_COLUMN].astype("float64")
    holed.iloc[0, holed.columns.get_loc(config.TIME_COLUMN)] = np.nan
    with pytest.raises(data_loader.TrainOnlyError, match="null TransactionDT"):
        data_loader.assert_train_only(holed, what="imputer")


def test_train_only_decorator_guards_the_first_positional_frame() -> None:
    @data_loader.train_only
    def fit_mean(frame: pd.DataFrame) -> float:
        return float(frame[config.TARGET].mean())

    train, val, _ = data_loader.time_based_split(synthetic_frame())
    assert 0.0 <= fit_mean(train) <= 1.0
    with pytest.raises(data_loader.TrainOnlyError, match="fit_mean"):
        fit_mean(val)


def test_train_only_decorator_guards_a_frame_keyword() -> None:
    @data_loader.train_only
    def fit_mean(*, frame: pd.DataFrame) -> float:
        return float(frame[config.TARGET].mean())

    _, val, _ = data_loader.time_based_split(synthetic_frame())
    with pytest.raises(data_loader.TrainOnlyError, match="fit_mean"):
        fit_mean(frame=val)


def test_train_only_decorator_refuses_a_call_it_cannot_inspect() -> None:
    """No frame to check means the guard fails loudly instead of waving the call through."""

    @data_loader.train_only
    def fit_something(values: list[int]) -> int:
        return sum(values)

    with pytest.raises(data_loader.TrainOnlyError, match="no DataFrame"):
        fit_something([1, 2, 3])


# --- the committed artifact ------------------------------------------------------------------


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(SUMMARY_PATH.read_text())
    return data


def test_summary_boundaries_match_the_config(summary: dict[str, Any]) -> None:
    """The artifact and the code agree on where the split falls. If someone edits one, this fails."""
    assert summary["boundaries"]["train_end_exclusive_dt"] == config.TRAIN_END_DT
    assert summary["boundaries"]["val_end_exclusive_dt"] == config.VAL_END_DT
    assert summary["regenerate_with"] == "make split-summary"
    assert summary["seed"] == config.SEED


def test_summary_splits_are_ordered_and_exhaustive(summary: dict[str, Any]) -> None:
    splits = summary["splits"]
    assert splits["train"]["dt_max"] < splits["val"]["dt_min"]
    assert splits["val"]["dt_max"] < splits["test"]["dt_min"]
    assert sum(splits[n]["n_rows"] for n in config.SPLIT_NAMES) == summary["totals"]["n_rows"]
    assert all(
        summary["checks"][k] is True
        for k in (
            "rows_sum_to_total",
            "train_max_dt_below_val_min_dt",
            "val_max_dt_below_test_min_dt",
        )
    )


def test_summary_entity_key_matches_adr_0001(summary: dict[str, Any]) -> None:
    assert summary["entity_overlap"]["key"] == list(config.ENTITY_KEY_COLUMNS)


# --- the same contract on the real 590,540 rows ----------------------------------------------


@pytest.fixture(scope="module")
def real_frame() -> pd.DataFrame:
    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data/train_transaction.csv not present")
    columns = [
        config.ID_COLUMN,
        config.TARGET,
        config.TIME_COLUMN,
        *config.ENTITY_KEY_SOURCE_COLUMNS,
    ]
    return data_loader.load_raw(columns=columns)


@pytest.mark.needs_data
def test_real_split_satisfies_the_contract(real_frame: pd.DataFrame) -> None:
    train, val, test = data_loader.time_based_split(real_frame)
    assert_temporal_contract(train, val, test, real_frame)


@pytest.mark.needs_data
def test_real_split_reproduces_the_committed_summary(
    real_frame: pd.DataFrame, summary: dict[str, Any]
) -> None:
    """The artifact is not stale relative to the code that writes it."""
    parts = dict(zip(config.SPLIT_NAMES, data_loader.time_based_split(real_frame), strict=True))
    for name, part in parts.items():
        stored = summary["splits"][name]
        assert len(part) == stored["n_rows"]
        assert int(part[config.TIME_COLUMN].min()) == stored["dt_min"]
        assert int(part[config.TIME_COLUMN].max()) == stored["dt_max"]
        assert part[config.TARGET].mean() == pytest.approx(stored["fraud_rate"])


@pytest.mark.needs_data
def test_real_assert_train_only_rejects_the_later_splits(real_frame: pd.DataFrame) -> None:
    train, val, test = data_loader.time_based_split(real_frame)
    assert data_loader.assert_train_only(train, what="stage 1 check") is train
    for part in (val, test):
        with pytest.raises(data_loader.TrainOnlyError):
            data_loader.assert_train_only(part, what="stage 1 check")
