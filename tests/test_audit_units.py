"""Unit tests for the audit helpers, on frames small enough to check by hand.

These run in CI. Anything that reads the real CSVs is in test_audit_on_data.py and is marked
`needs_data`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_platform import audit

DAY = audit.SECONDS_PER_DAY


def frame(rows: list[tuple[int, int, int, int, float | None, float | None]]) -> pd.DataFrame:
    # read_csv gives addr1 and D1 as float64 because both carry nulls. Constructing from
    # tuples would give object dtype for an all-null column, which is not what the audit sees.
    return pd.DataFrame(
        rows, columns=["TransactionID", "isFraud", "TransactionDT", "card1", "addr1", "D1"]
    ).astype({"addr1": "float64", "D1": "float64"})


def test_add_card_start_day_subtracts_d1_from_the_day_index() -> None:
    df = frame([(1, 0, 10 * DAY, 100, 1.0, 3.0), (2, 0, 12 * DAY, 100, 1.0, 5.0)])
    out = audit.add_card_start_day(df)
    assert list(out["card_start_day"]) == [7.0, 7.0]


def test_add_card_start_day_keeps_null_d1_null() -> None:
    df = frame([(1, 0, 10 * DAY, 100, 1.0, None)])
    out = audit.add_card_start_day(df)
    assert out["card_start_day"].isna().all()


def test_basic_shape_counts_fraud() -> None:
    df = frame(
        [
            (1, 1, DAY, 100, 1.0, 0.0),
            (2, 0, 2 * DAY, 100, 1.0, 1.0),
            (3, 0, 3 * DAY, 100, 1.0, 2.0),
            (4, 0, 4 * DAY, 100, 1.0, 3.0),
        ]
    )
    out = audit.basic_shape(df, n_columns=6)
    assert out["n_rows"] == 4
    assert out["n_fraud"] == 1
    assert out["fraud_rate"] == 0.25


def test_transaction_dt_granularity_is_the_gcd_of_the_gaps() -> None:
    df = frame([(i, 0, i * 300, 100, 1.0, 0.0) for i in range(1, 6)])
    out = audit.transaction_dt_profile(df)
    assert out["granularity_units"] == 300
    assert out["is_monotonic_increasing_in_file_order"] is True


def test_label_granularity_ignores_single_transaction_entities() -> None:
    df = frame(
        [
            (1, 1, DAY, 100, 1.0, 0.0),  # card 100: all fraud
            (2, 1, 2 * DAY, 100, 1.0, 1.0),
            (3, 0, DAY, 200, 1.0, 0.0),  # card 200: all clean
            (4, 0, 2 * DAY, 200, 1.0, 1.0),
            (5, 1, DAY, 300, 1.0, 0.0),  # card 300: mixed
            (6, 0, 2 * DAY, 300, 1.0, 1.0),
            (7, 1, DAY, 400, 1.0, 0.0),  # card 400: single transaction, excluded
        ]
    )
    out = audit.label_granularity(df, ["card1"])
    assert out["n_entities_with_2_or_more_transactions"] == 3
    assert out["n_all_fraud"] == 1
    assert out["n_all_non_fraud"] == 1
    assert out["n_mixed"] == 1
    assert out["fraction_mixed"] == pytest.approx(1 / 3)


def test_entity_key_stats_purity_and_singletons() -> None:
    df = frame(
        [
            (1, 0, DAY, 100, 1.0, 0.0),
            (2, 0, 2 * DAY, 100, 1.0, 1.0),
            (3, 1, DAY, 200, 1.0, 0.0),
            (4, 0, 2 * DAY, 200, 1.0, 1.0),
            (5, 0, DAY, 300, 1.0, 0.0),
        ]
    )
    out = audit.entity_key_stats(df, "card1", ["card1"])
    assert out["n_entities"] == 3
    assert out["n_singleton_entities"] == 1
    assert out["n_entities_with_2_or_more_transactions"] == 2
    assert out["label_purity_multi_entities"] == 0.5


def test_entity_key_stats_counts_null_key_components() -> None:
    df = frame([(1, 0, DAY, 100, None, 0.0), (2, 0, 2 * DAY, 100, 1.0, 1.0)])
    out = audit.entity_key_stats(df, "card1_addr1", ["card1", "addr1"])
    assert out["n_rows_with_null_key_component"] == 1
    # a null addr1 forms its own entity rather than being dropped
    assert out["n_entities"] == 2


def test_identity_join_coverage_splits_by_label() -> None:
    df = frame(
        [
            (1, 1, DAY, 100, 1.0, 0.0),
            (2, 0, 2 * DAY, 100, 1.0, 1.0),
            (3, 0, 3 * DAY, 100, 1.0, 2.0),
        ]
    )
    out = audit.identity_join_coverage(df, pd.Series([1, 2]))
    assert out["coverage_overall"] == pytest.approx(2 / 3)
    assert out["coverage_fraud"] == 1.0
    assert out["coverage_non_fraud"] == 0.5


def test_chronological_split_keeps_the_windows_in_order_and_disjoint() -> None:
    df = frame([(i, i % 2, i * 1000, 100, 1.0, 0.0) for i in range(1, 101)])
    out = audit.chronological_split(df)
    splits = out["splits"]
    assert splits["train"]["dt_max"] < splits["val"]["dt_min"]
    assert splits["val"]["dt_max"] < splits["test"]["dt_min"]
    total = sum(splits[name]["n_rows"] for name in ("train", "val", "test"))
    assert total == len(df)


def test_chronological_split_never_straddles_a_shared_timestamp() -> None:
    # every timestamp appears twice, so a naive row-index cut would break a tie across splits
    rows = []
    for i in range(1, 51):
        rows.append((2 * i - 1, 0, i * 1000, 100, 1.0, 0.0))
        rows.append((2 * i, 1, i * 1000, 200, 1.0, 0.0))
    out = audit.chronological_split(frame(rows))
    b1 = out["boundaries"]["train_end_exclusive_dt"]
    b2 = out["boundaries"]["val_end_exclusive_dt"]
    df = frame(rows)
    for boundary in (b1, b2):
        straddling = df[df["TransactionDT"] == boundary]
        assert straddling.empty or (straddling["TransactionDT"] >= boundary).all()


def test_entity_overlap_finds_entities_absent_from_train() -> None:
    rows = [(i, 0, i * 1000, 100, 1.0, 0.0) for i in range(1, 91)]
    rows += [(i, 0, i * 1000, 999, 1.0, 0.0) for i in range(91, 101)]
    df = audit.add_card_start_day(frame(rows))
    split = audit.chronological_split(df)
    out = audit.entity_overlap(df, split, ["card1"])
    # the test window holds the tail of card 100 plus all of card 999, which train never saw
    assert out["test"]["n_entities"] == 2
    assert out["test"]["n_entities_unseen_in_train"] == 1
    assert out["test"]["share_entities_unseen_in_train"] == 0.5


def test_entity_overlap_handles_a_null_key_component() -> None:
    rows = [(i, 0, i * 1000, 100, None, 0.0) for i in range(1, 101)]
    df = audit.add_card_start_day(frame(rows))
    split = audit.chronological_split(df)
    out = audit.entity_overlap(df, split, ["card1", "addr1"])
    assert out["test"]["n_entities_unseen_in_train"] == 0


def test_d1_profile_flags_no_impossible_start_days() -> None:
    df = audit.add_card_start_day(
        frame([(1, 0, 10 * DAY, 100, 1.0, 3.0), (2, 0, 11 * DAY, 100, 1.0, 0.0)])
    )
    out = audit.d1_profile(df)
    assert out["n_negative"] == 0
    assert out["n_rows_with_card_start_day_after_transaction_day"] == 0
    assert out["share_zero"] == 0.5


def test_verify_card_start_day_counts_single_valued_groups() -> None:
    df = audit.add_card_start_day(
        frame(
            [
                (1, 0, 10 * DAY, 100, 1.0, 3.0),  # start day 7
                (2, 0, 12 * DAY, 100, 1.0, 5.0),  # start day 7, same card
                (3, 0, 10 * DAY, 200, 1.0, 3.0),  # start day 7
                (4, 0, 12 * DAY, 200, 1.0, 1.0),  # start day 11, different card
            ]
        )
    )
    out = audit.verify_card_start_day(df, ["card1"])
    assert out["n_groups_with_2_or_more_non_null"] == 2
    assert out["n_groups_single_valued"] == 1
    assert out["fraction_single_valued"] == 0.5


def test_seconds_per_day_constant() -> None:
    assert audit.SECONDS_PER_DAY == 24 * 60 * 60
    assert np.floor(audit.SECONDS_PER_DAY / 3600) == 24
