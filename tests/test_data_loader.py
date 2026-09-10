"""Loading, typing and the entity key.

The temporal contract lives in test_temporal_contract.py. This file covers everything else the
data layer promises: the dtype map, the left join, the derived entity key, and the rule that one
SEED constant is the only source of randomness in the repo.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import audit, config, data_loader

ROOT = Path(__file__).resolve().parents[1]
MEMORY_PROFILE_PATH = ROOT / "reports" / "memory_profile.json"

DAY = config.SECONDS_PER_DAY


@pytest.fixture
def csv_pair(tmp_path: Path) -> tuple[Path, Path]:
    """A two file stand-in with one column from each group the dtype map handles."""
    transactions = pd.DataFrame(
        {
            config.ID_COLUMN: [2_987_000, 2_987_001, 2_987_002],
            config.TARGET: [0, 1, 0],
            config.TIME_COLUMN: [DAY, 2 * DAY, 3 * DAY],
            config.AMOUNT_COLUMN: [31937.391, 12.5, 0.75],
            "ProductCD": ["W", "C", "W"],
            "card1": [1000, 18396, 1000],
            "addr1": [299.0, None, 299.0],
            "C1": [0, 4685, 1],
            "D1": [0.0, 5.0, None],
            "V1": [1.0, None, 0.0],
            "M1": ["T", None, "F"],
        }
    )
    identity = pd.DataFrame(
        {
            config.ID_COLUMN: [2_987_001],
            "id_01": [-5.0],
            "id_31": ["chrome 62.0"],
            "DeviceType": ["mobile"],
        }
    )
    transactions_path = tmp_path / "train_transaction.csv"
    identity_path = tmp_path / "train_identity.csv"
    transactions.to_csv(transactions_path, index=False)
    identity.to_csv(identity_path, index=False)
    return transactions_path, identity_path


# --- the dtype map ---------------------------------------------------------------------------


def test_dtype_map_narrows_the_measured_integer_columns() -> None:
    mapping = data_loader.build_dtype_map([config.ID_COLUMN, config.TARGET, "card1", "C14"])
    assert mapping == {
        config.ID_COLUMN: "int32",
        config.TARGET: "int8",
        "card1": "int16",
        "C14": "int16",
    }


def test_dtype_map_makes_string_columns_categorical() -> None:
    mapping = data_loader.build_dtype_map(["ProductCD", "M1", "DeviceInfo", "id_31"])
    assert set(mapping.values()) == {"category"}


def test_dtype_map_keeps_the_amount_at_float64_and_everything_else_at_float32() -> None:
    mapping = data_loader.build_dtype_map([config.AMOUNT_COLUMN, "V1", "D1", "dist1", "id_02"])
    assert mapping[config.AMOUNT_COLUMN] == "float64"
    assert {mapping[c] for c in ("V1", "D1", "dist1", "id_02")} == {"float32"}


def test_dtype_map_defaults_an_unknown_column_to_float32() -> None:
    """An unseen column is assumed numeric. If it holds text, read_csv raises, which is louder
    than a silent object column."""
    assert data_loader.build_dtype_map(["V999"]) == {"V999": "float32"}


# --- loading ---------------------------------------------------------------------------------


def test_load_raw_applies_the_dtype_map(csv_pair: tuple[Path, Path]) -> None:
    frame = data_loader.load_raw(*csv_pair)
    assert frame[config.ID_COLUMN].dtype == np.int32
    assert frame[config.TARGET].dtype == np.int8
    assert frame["card1"].dtype == np.int16
    assert frame["V1"].dtype == np.float32
    assert frame[config.AMOUNT_COLUMN].dtype == np.float64
    assert isinstance(frame["ProductCD"].dtype, pd.CategoricalDtype)


def test_load_raw_keeps_the_amount_exact(csv_pair: tuple[Path, Path]) -> None:
    """The one column held at float64. float32 would move this value by 0.000375."""
    frame = data_loader.load_raw(*csv_pair)
    assert frame[config.AMOUNT_COLUMN].iloc[0] == 31937.391


def test_load_raw_left_joins_identity(csv_pair: tuple[Path, Path]) -> None:
    """All transactions survive; identity columns are null for the rows that do not join."""
    frame = data_loader.load_raw(*csv_pair)
    assert len(frame) == 3
    assert frame["id_01"].notna().sum() == 1
    assert frame.loc[frame[config.ID_COLUMN] == 2_987_001, "id_01"].iloc[0] == -5.0
    assert isinstance(frame["DeviceType"].dtype, pd.CategoricalDtype)


def test_load_raw_sorts_categories(csv_pair: tuple[Path, Path]) -> None:
    """Category codes are a function of the category set, not of the order the file happened to
    present the values in."""
    frame = data_loader.load_raw(*csv_pair)
    assert list(frame["ProductCD"].cat.categories) == ["C", "W"]
    assert list(frame["M1"].cat.categories) == ["F", "T"]


def test_load_raw_restricts_columns_and_still_joins(csv_pair: tuple[Path, Path]) -> None:
    frame = data_loader.load_raw(*csv_pair, columns=[config.TIME_COLUMN, "id_01"])
    assert set(frame.columns) == {config.ID_COLUMN, config.TIME_COLUMN, "id_01"}


def test_load_raw_skips_the_identity_read_when_no_identity_column_is_wanted(
    csv_pair: tuple[Path, Path],
) -> None:
    frame = data_loader.load_raw(*csv_pair, columns=[config.TIME_COLUMN, config.TARGET])
    assert set(frame.columns) == {config.ID_COLUMN, config.TIME_COLUMN, config.TARGET}


def test_frame_bytes_counts_the_payload(csv_pair: tuple[Path, Path]) -> None:
    frame = data_loader.load_raw(*csv_pair)
    assert data_loader.frame_bytes(frame) > 0


# --- the entity key --------------------------------------------------------------------------


def test_add_entity_key_derives_card_start_day(csv_pair: tuple[Path, Path]) -> None:
    frame = data_loader.add_entity_key(data_loader.load_raw(*csv_pair))
    # day index 1 minus D1 0, day index 2 minus D1 5, D1 null
    assert list(frame[config.CARD_START_DAY_COLUMN][:2]) == [1.0, -3.0]
    assert pd.isna(frame[config.CARD_START_DAY_COLUMN].iloc[2])


def test_add_entity_key_matches_the_stage_0_audit_formula() -> None:
    """One formula, two modules. This is what stops them drifting apart."""
    frame = pd.DataFrame(
        {
            config.ID_COLUMN: [1, 2, 3],
            config.TARGET: [0, 0, 1],
            config.TIME_COLUMN: [10 * DAY, 12 * DAY + 500, 3 * DAY],
            "card1": [100, 100, 200],
            "addr1": [1.0, 1.0, None],
            "D1": [3.0, 5.0, None],
        }
    )
    from_loader = data_loader.add_entity_key(frame)[config.CARD_START_DAY_COLUMN]
    from_audit = audit.add_card_start_day(frame)["card_start_day"]
    pd.testing.assert_series_equal(from_loader, from_audit, check_names=False)


def test_entity_id_buckets_nulls_rather_than_dropping_rows(csv_pair: tuple[Path, Path]) -> None:
    """ADR 0001: every row belongs to exactly one entity, so the counts reconcile."""
    frame = data_loader.add_entity_key(data_loader.load_raw(*csv_pair))
    assert frame[config.ENTITY_ID_COLUMN].notna().all()
    assert frame[config.ENTITY_ID_COLUMN].str.contains(config.ENTITY_NULL_TOKEN).sum() == 2


def test_add_entity_key_names_the_columns_it_is_missing() -> None:
    frame = pd.DataFrame({config.TIME_COLUMN: [DAY], "card1": [1]})
    with pytest.raises(KeyError, match="addr1"):
        data_loader.add_entity_key(frame)


# --- one seed, read everywhere ----------------------------------------------------------------

SEED_LITERAL = re.compile(
    r"(random_state|seed|random_seed)\s*=\s*\d+|"
    r"(np\.random\.seed|random\.seed|default_rng)\s*\(\s*\d+\s*\)"
)


def test_no_module_carries_its_own_seed_literal() -> None:
    """Every stochastic component reads config.SEED.

    A literal here would be a second source of randomness that nobody remembers to change, so the
    rule is enforced by reading the source rather than by asking people to remember it. config.py
    is where the constant is defined and tests may fix a seed of their own for a fixture, so both
    are exempt.
    """
    offenders = []
    for path in sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "scripts").rglob("*.py")]):
        if path.name == "config.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if SEED_LITERAL.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not offenders, "seed literals outside config.py:\n" + "\n".join(offenders)


def test_seed_is_an_int() -> None:
    assert isinstance(config.SEED, int)


# --- config against the committed artifacts ---------------------------------------------------


@pytest.fixture(scope="module")
def memory_profile() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(MEMORY_PROFILE_PATH.read_text())
    return data


def test_categorical_columns_match_the_measured_set(memory_profile: dict[str, Any]) -> None:
    assert set(memory_profile["categorical_cardinality"]) == set(config.CATEGORICAL_COLUMNS)


def test_every_categorical_column_pays_as_a_category(memory_profile: dict[str, Any]) -> None:
    """A category costs a code array plus one copy of each distinct value. It stops paying when
    the values are nearly all distinct. The widest column here holds 1,786 of them across 590,540
    rows."""
    cardinalities = memory_profile["categorical_cardinality"]
    n_rows = memory_profile["n_rows"]
    assert max(cardinalities.values()) / n_rows < 0.01


def test_typed_load_is_the_strategy_the_profile_measured(memory_profile: dict[str, Any]) -> None:
    typed = memory_profile["strategies"]["typed"]
    assert typed["n_category_columns"] == len(config.CATEGORICAL_COLUMNS)
    assert typed["dtype_counts"]["float64"] == len(config.EXACT_FLOAT_COLUMNS)
    assert memory_profile["typed_against_naive"]["frame_bytes_ratio"] < 1.0


# --- config against the real files -------------------------------------------------------------


@pytest.mark.needs_data
def test_column_groups_partition_the_joined_frame() -> None:
    """The groups name every column exactly once, so a later stage can address the frame by
    group without wondering what it missed."""
    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data/train_transaction.csv not present")

    transaction_columns = list(pd.read_csv(config.TRANSACTIONS_PATH, nrows=0).columns)
    identity_columns = list(pd.read_csv(config.IDENTITY_PATH, nrows=0).columns)
    joined = transaction_columns + [c for c in identity_columns if c != config.ID_COLUMN]

    groups = [
        (config.ID_COLUMN,),
        (config.TARGET,),
        (config.TIME_COLUMN,),
        (config.AMOUNT_COLUMN,),
        config.PRODUCT_COLUMNS,
        config.CARD_COLUMNS,
        config.ADDRESS_COLUMNS,
        config.DISTANCE_COLUMNS,
        config.EMAIL_COLUMNS,
        config.COUNT_COLUMNS,
        config.TIMEDELTA_COLUMNS,
        config.MATCH_COLUMNS,
        config.VESTA_COLUMNS,
        config.IDENTITY_COLUMNS,
    ]
    named = [column for group in groups for column in group]

    assert len(named) == len(set(named)), "a column is named by two groups"
    assert set(named) == set(joined)
    assert len(joined) == 434


@pytest.mark.needs_data
def test_categorical_columns_are_exactly_the_string_columns() -> None:
    """Any column not in CATEGORICAL_COLUMNS is read as a number. This is the check that a new
    string column would fail before it silently became an object column."""
    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data/train_transaction.csv not present")

    frame = data_loader.load_raw()
    categorical = {c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)}
    assert categorical == set(config.CATEGORICAL_COLUMNS)


@pytest.mark.needs_data
def test_integer_columns_really_have_no_nulls() -> None:
    """The dtype map claims these 18 columns are null-free. read_csv would raise if they were
    not, so this is the check that says so in words rather than as a stack trace."""
    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data/train_transaction.csv not present")

    frame = data_loader.load_raw(columns=list(config.INTEGER_COLUMN_DTYPES))
    for column, dtype in config.INTEGER_COLUMN_DTYPES.items():
        assert frame[column].isna().sum() == 0
        assert str(frame[column].dtype) == dtype
