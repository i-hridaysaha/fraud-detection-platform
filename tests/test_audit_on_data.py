"""Audit checks that read the real CSVs.

Marked `needs_data` and deselected in CI: the IEEE-CIS files are not redistributable, so they
are gitignored and never reach the runner. Run locally with `make data && make test`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import audit

ROOT = Path(__file__).resolve().parents[1]
TRANSACTIONS = ROOT / "data" / "train_transaction.csv"
IDENTITY = ROOT / "data" / "train_identity.csv"
REPORT = ROOT / "reports" / "audit.json"

pytestmark = pytest.mark.needs_data


@pytest.fixture(scope="module")
def frames() -> tuple[Any, Any]:
    if not TRANSACTIONS.exists():
        pytest.skip("data/train_transaction.csv not present")
    df = audit.add_card_start_day(audit.load_transactions(str(TRANSACTIONS)))
    return df, json.loads(REPORT.read_text())


def test_row_and_column_counts_match_the_report(frames: tuple[Any, Any]) -> None:
    df, report = frames
    assert len(df) == report["shape"]["n_rows"]
    assert audit.count_columns(str(TRANSACTIONS)) == report["shape"]["n_columns"]


def test_fraud_rate_matches_the_report(frames: tuple[Any, Any]) -> None:
    df, report = frames
    assert df["isFraud"].mean() == pytest.approx(report["shape"]["fraud_rate"])


def test_transaction_dt_is_non_decreasing_in_file_order(frames: tuple[Any, Any]) -> None:
    df, _ = frames
    assert df["TransactionDT"].is_monotonic_increasing


def test_entity_key_stats_reproduce_the_report(frames: tuple[Any, Any]) -> None:
    df, report = frames
    recomputed = audit.entity_key_stats(
        df, "card1_addr1_cardday", ["card1", "addr1", "card_start_day"]
    )
    stored = report["entity_key_candidates"]["card1_addr1_cardday"]
    assert recomputed["n_entities"] == stored["n_entities"]
    assert recomputed["label_purity_multi_entities"] == pytest.approx(
        stored["label_purity_multi_entities"]
    )


def test_identity_ids_are_a_subset_of_transaction_ids(frames: tuple[Any, Any]) -> None:
    import pandas as pd

    df, _ = frames
    ids = pd.read_csv(IDENTITY, usecols=["TransactionID"])["TransactionID"]
    assert ids.isin(set(df["TransactionID"])).all()
