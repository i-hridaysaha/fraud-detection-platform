"""Checks on the committed audit artifact.

These run in CI. reports/audit.json is committed, the CSVs are not, so this is the gate that
catches an artifact edited by hand or left stale relative to the code that writes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "audit.json"


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(REPORT_PATH.read_text())
    return data


def test_report_exists_and_is_valid_json(report: dict[str, Any]) -> None:
    assert report["regenerate_with"] == "make audit"
    assert report["inputs"]["transactions"]["sha256"]


def test_every_section_carries_the_command_that_produced_it(report: dict[str, Any]) -> None:
    sections = [
        "shape",
        "transaction_dt",
        "label_granularity_card1",
        "card1_profile",
        "d1_profile",
        "identity_join",
        "chronological_split",
    ]
    for name in sections:
        assert "command" in report[name], f"{name} has no command"

    for name in ("entity_key_candidates", "label_granularity_by_candidate", "entity_overlap"):
        for candidate, block in report[name].items():
            assert "command" in block, f"{name}.{candidate} has no command"


def test_split_rows_sum_to_the_row_count(report: dict[str, Any]) -> None:
    splits = report["chronological_split"]["splits"]
    total = sum(splits[name]["n_rows"] for name in ("train", "val", "test"))
    assert total == report["shape"]["n_rows"]


def test_split_windows_are_ordered_and_disjoint(report: dict[str, Any]) -> None:
    splits = report["chronological_split"]["splits"]
    assert splits["train"]["dt_max"] < splits["val"]["dt_min"]
    assert splits["val"]["dt_max"] < splits["test"]["dt_min"]


def test_split_boundaries_are_ordered(report: dict[str, Any]) -> None:
    boundaries = report["chronological_split"]["boundaries"]
    assert boundaries["train_end_exclusive_dt"] < boundaries["val_end_exclusive_dt"]


def test_realised_split_fractions_are_close_to_the_request(report: dict[str, Any]) -> None:
    requested = report["chronological_split"]["requested_fractions"]
    splits = report["chronological_split"]["splits"]
    for want, name in zip(requested, ("train", "val", "test"), strict=True):
        assert abs(splits[name]["row_fraction"] - want) < 0.01


def test_fraud_counts_agree_with_the_rate(report: dict[str, Any]) -> None:
    shape = report["shape"]
    assert shape["n_fraud"] + shape["n_non_fraud"] == shape["n_rows"]
    assert shape["fraud_rate"] == pytest.approx(shape["n_fraud"] / shape["n_rows"])


def test_rates_are_in_the_unit_interval(report: dict[str, Any]) -> None:
    def check(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    check(value, f"{path}.{key}")
                elif (
                    isinstance(value, float)
                    and any(token in key for token in ("rate", "share", "fraction", "coverage"))
                    and "ratio" not in key
                ):
                    assert 0.0 <= value <= 1.0, f"{path}.{key} is {value}"

    check(report, "report")


def test_entity_key_purity_increases_with_key_specificity(report: dict[str, Any]) -> None:
    candidates = report["entity_key_candidates"]
    purity = [
        candidates[name]["label_purity_multi_entities"]
        for name in ("card1", "card1_addr1", "card1_addr1_cardday")
    ]
    assert purity == sorted(purity)


def test_the_chosen_key_is_the_one_the_adr_names(report: dict[str, Any]) -> None:
    # ADR 0001 picks card1 + addr1 + card_start_day. If a later stage changes the key, this
    # test fails and forces a superseding ADR rather than a quiet edit.
    assert report["entity_key_candidates"]["card1_addr1_cardday"]["key"] == [
        "card1",
        "addr1",
        "card_start_day",
    ]
