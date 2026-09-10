"""Checks on the committed stage 2 artifacts.

Most of these run in CI. The artifacts under reports/eda/ are committed, so a reviewer without
the data still gets the internal consistency of every number checked on every push: the sections
agree on how many train rows they saw, the parts of each table add up to their whole, and the
derived quantities equal the quantities they were derived from.

The handful that reopen the CSVs to recompute a number from scratch are marked `needs_data`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import config, data_loader, eda

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "reports" / "eda"
FIGURES_DIR = ROOT / "figures"
SPLIT_SUMMARY = ROOT / "reports" / "split_summary.json"

SECTIONS = (
    "target",
    "univariate",
    "missingness",
    "correlation",
    "bivariate",
    "temporal",
    "entities",
    "quality",
    "train_only",
)

FIGURES = (
    "fraud_rate_over_time.png",
    "amount_distribution.png",
    "missingness_blocks.png",
    "fraud_rate_by_decile.png",
    "psi_train_vs_test.png",
    "time_consistency.png",
)


def load(section: str) -> dict[str, Any]:
    path = EDA_DIR / f"{section}.json"
    if not path.exists():
        pytest.skip(f"{path} not present; run make eda")
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def sections() -> dict[str, dict[str, Any]]:
    return {name: load(name) for name in SECTIONS}


@pytest.fixture(scope="module")
def split_summary() -> dict[str, Any]:
    if not SPLIT_SUMMARY.exists():
        pytest.skip("reports/split_summary.json not present")
    return dict(json.loads(SPLIT_SUMMARY.read_text()))


# --- the envelope every section carries -------------------------------------------------


def test_every_section_exists_and_names_the_command_that_rebuilds_it(
    sections: dict[str, dict[str, Any]],
) -> None:
    for name, report in sections.items():
        assert report["section"] == name
        assert "make eda" in report["regenerate_with"]
        assert report["command"]


def test_every_section_saw_the_same_train_split(
    sections: dict[str, dict[str, Any]], split_summary: dict[str, Any]
) -> None:
    # If one section had quietly been run against a different frame, this is where it shows.
    counts = {report["n_train_rows"] for report in sections.values()}
    assert len(counts) == 1
    assert counts.pop() == split_summary["splits"]["train"]["n_rows"]


def test_every_section_says_it_was_computed_on_train_only(
    sections: dict[str, dict[str, Any]],
) -> None:
    for report in sections.values():
        assert "train split only" in report["computed_on"]


# --- target ------------------------------------------------------------------------------


def test_target_rate_reproduces_the_split_summary(
    sections: dict[str, dict[str, Any]], split_summary: dict[str, Any]
) -> None:
    target = sections["target"]
    for name in ("train", "val", "test"):
        assert target["per_split"][name]["fraud_rate"] == pytest.approx(
            split_summary["splits"][name]["fraud_rate"]
        )
        assert target["per_split"][name]["n_fraud"] == split_summary["splits"][name]["n_fraud"]
    assert all(target["per_split"]["reproduces_split_summary"].values())


def test_target_periods_partition_the_train_split(sections: dict[str, dict[str, Any]]) -> None:
    target = sections["target"]
    n_rows = target["n_train_rows"]
    for key in ("by_day", "by_week"):
        assert sum(row["n"] for row in target[key]) == n_rows
        assert sum(row["n_fraud"] for row in target[key]) == target["overall_train"]["n_fraud"]


def test_target_category_tables_partition_the_train_split(
    sections: dict[str, dict[str, Any]],
) -> None:
    target = sections["target"]
    for table in list(target["by_category"].values()) + list(target["by_clock"].values()):
        assert sum(row["n"] for row in table) == target["n_train_rows"]


def test_both_intervals_agree_on_the_train_fraud_rate(
    sections: dict[str, dict[str, Any]],
) -> None:
    # Wilson and the bootstrap answer the same question two ways. At this row count they should
    # land within a rounding of each other, and a gap would mean one of them is wrong.
    overall = sections["target"]["overall_train"]
    assert overall["wilson_95"]["low"] == pytest.approx(overall["bootstrap_95"]["low"], abs=2e-4)
    assert overall["wilson_95"]["high"] == pytest.approx(overall["bootstrap_95"]["high"], abs=2e-4)


def test_the_partial_week_is_excluded_from_the_second_trend_test(
    sections: dict[str, dict[str, Any]],
) -> None:
    stationarity = sections["target"]["stationarity"]
    assert (
        stationarity["weekly_complete_weeks_only"]["n_periods"]
        < stationarity["weekly"]["n_periods"]
    )
    days = sections["target"]["days_per_week"]
    for week in stationarity["weekly_complete_weeks_only"]["weeks_kept"]:
        assert days[str(week)] == 7


# --- univariate --------------------------------------------------------------------------


def test_every_column_profile_accounts_for_every_row(
    sections: dict[str, dict[str, Any]],
) -> None:
    report = sections["univariate"]
    n_rows = report["n_train_rows"]
    for profile in report["numeric"] + report["categorical"]:
        assert profile["n_total"] == n_rows
        assert profile["count"] + round(profile["missing_rate"] * n_rows) == pytest.approx(
            n_rows, abs=1
        )


def test_the_flag_lists_agree_with_the_per_column_flags(
    sections: dict[str, dict[str, Any]],
) -> None:
    report = sections["univariate"]
    profiles = {p["column"]: p for p in report["numeric"] + report["categorical"]}
    for column in report["flags"]["effectively_constant"]:
        assert profiles[column]["effectively_constant"] is True
    for column in report["flags"]["zero_inflated"]:
        assert profiles[column]["zero_share"] >= eda.ZERO_INFLATED_SHARE


def test_log1p_pulls_the_amount_skew_down_by_an_order_of_magnitude(
    sections: dict[str, dict[str, Any]],
) -> None:
    amount = sections["univariate"]["amount"]
    assert abs(amount["skew_log1p"]) < abs(amount["skew_raw"]) / 10


def test_round_and_not_round_amounts_partition_the_split(
    sections: dict[str, dict[str, Any]],
) -> None:
    split = sections["univariate"]["amount"]["round_against_not_round"]
    assert split["round"]["n"] + split["not_round"]["n"] == sections["univariate"]["n_train_rows"]


# --- missingness -------------------------------------------------------------------------


def test_missingness_blocks_partition_the_columns(sections: dict[str, dict[str, Any]]) -> None:
    report = sections["missingness"]
    seen: list[str] = []
    for block in report["blocks"]["all_blocks"]:
        seen.extend(block["columns"])
        assert len(block["columns"]) == block["n_columns"]
    assert len(seen) == len(set(seen)) == report["n_columns"]


def test_the_missing_rate_table_covers_every_column_once(
    sections: dict[str, dict[str, Any]],
) -> None:
    rows = sections["missingness"]["missing_rate_by_column"]
    assert len({row["column"] for row in rows}) == len(rows)
    assert len(rows) == sections["missingness"]["n_columns"]


def test_missingness_cells_add_up_to_the_split(sections: dict[str, dict[str, Any]]) -> None:
    report = sections["missingness"]
    n_rows = report["n_train_rows"]
    n_fraud = None
    for row in report["against_target"]["per_column"]:
        assert row["n_present"] + row["n_missing"] == n_rows
        total = row["n_fraud_present"] + row["n_fraud_missing"]
        n_fraud = total if n_fraud is None else n_fraud
        assert total == n_fraud


def test_every_label_linked_column_clears_both_thresholds(
    sections: dict[str, dict[str, Any]],
) -> None:
    report = sections["missingness"]["against_target"]
    linked = set(report["label_linked_columns"])
    for row in report["per_column"]:
        if row["column"] in linked:
            assert abs(row["gap"]) >= report["thresholds"]["abs_gap"]
            assert row["p_value"] < report["thresholds"]["p_value"]


def test_the_identity_block_is_not_one_block(sections: dict[str, dict[str, Any]]) -> None:
    # Measured, and contrary to what "the identity table joins or it does not" suggests: only
    # two of the identity columns are missing exactly when the join fails.
    join = sections["missingness"]["identity_join"]
    assert join["n_blocks_covering_identity_columns"] > 1
    assert len(join["columns_whose_mask_is_exactly_the_join"]) < join["n_identity_columns"]


# --- correlation -------------------------------------------------------------------------


def test_the_screen_threshold_leaves_room_for_the_measured_screen_error(
    sections: dict[str, dict[str, Any]],
) -> None:
    # The screen only sends pairs above SCREEN_RHO for exact computation, so a pair whose exact
    # value clears the redundancy threshold could be missed if the screen understated it by more
    # than the gap between the two thresholds. This is the check that the margin holds.
    screen = sections["correlation"]["screen"]
    margin = eda.REDUNDANT_RHO - screen["threshold"]
    assert screen["agreement_on_random_pairs"]["max_abs_difference"] < margin


def test_the_screen_is_exact_inside_a_missingness_block(
    sections: dict[str, dict[str, Any]],
) -> None:
    agreement = sections["correlation"]["screen"]["agreement_within_missingness_blocks"]
    assert agreement["n_pairs_within_tolerance"] == agreement["n_pairs_compared"]


def test_the_v_block_reduction_keeps_one_column_per_group(
    sections: dict[str, dict[str, Any]],
) -> None:
    reduction = sections["correlation"]["v_block_reduction"]
    removed = 0
    for block in reduction["blocks"]:
        assert sum(g["n_members"] for g in block["groups"]) == block["n_columns"]
        assert block["n_removed_by_reduction"] == block["n_columns"] - block["n_groups"]
        removed += block["n_removed_by_reduction"]
    assert removed == reduction["n_columns_a_reduction_would_remove"]


def test_every_reported_pair_clears_the_threshold_it_is_listed_under(
    sections: dict[str, dict[str, Any]],
) -> None:
    for key in ("near_duplicate_pairs", "redundant_pairs", "strongest_pairs_outside_v"):
        block = sections["correlation"][key]
        for pair in block["pairs"]:
            assert abs(pair["rho"]) >= block["threshold"]
    assert not any(
        pair["a"].startswith("V") or pair["b"].startswith("V")
        for pair in sections["correlation"]["strongest_pairs_outside_v"]["pairs"]
    )


# --- bivariate ---------------------------------------------------------------------------


def test_information_value_decomposition_adds_up(sections: dict[str, dict[str, Any]]) -> None:
    for row in sections["bivariate"]["per_feature"]:
        assert row["iv"] == pytest.approx(row["iv_missing_bin"] + row["iv_present_bins"])


def test_entity_identifying_columns_are_flagged_wherever_they_appear(
    sections: dict[str, dict[str, Any]],
) -> None:
    report = sections["bivariate"]
    declared = set(report["entity_identifying_columns"])
    for row in report["per_feature"]:
        assert row["entity_identifying"] is (row["column"] in declared)
    assert not any(row["entity_identifying"] for row in report["top_20_behavioural"])


def test_decile_tables_partition_the_split(sections: dict[str, dict[str, Any]]) -> None:
    report = sections["bivariate"]
    for tables in (report["deciles"], report["deciles_reference"]):
        for rows in tables.values():
            assert sum(row["n"] for row in rows) == report["n_train_rows"]


# --- temporal ----------------------------------------------------------------------------


def test_psi_and_time_consistency_cover_the_same_feature_set(
    sections: dict[str, dict[str, Any]],
) -> None:
    temporal = sections["temporal"]
    psi_columns = {row["column"] for row in temporal["psi"]["per_feature"]}
    consistency_columns = {row["column"] for row in temporal["time_consistency"]["per_feature"]}
    iv_columns = {row["column"] for row in sections["bivariate"]["per_feature"]}
    assert psi_columns == consistency_columns == iv_columns


def test_every_flagged_feature_meets_the_flag_rule(sections: dict[str, dict[str, Any]]) -> None:
    block = sections["temporal"]["time_consistency"]
    rule = block["flag_rule"]
    flagged = set(block["flagged_columns"])
    for row in block["per_feature"]:
        if row["column"] in flagged:
            assert row["early_auc"] >= rule["early_auc_at_least"]
            assert row["late_auc"] < rule["late_auc_below"]


def test_the_consistency_windows_do_not_overlap(sections: dict[str, dict[str, Any]]) -> None:
    windows = sections["temporal"]["time_consistency"]["windows"]
    assert windows["early_day_index_range"][1] < windows["late_day_index_range"][0]


def test_cold_and_warm_rows_partition_the_split(sections: dict[str, dict[str, Any]]) -> None:
    cold = sections["temporal"]["cold_entities"]
    assert cold["cold"]["n_rows"] + cold["warm"]["n_rows"] == cold["n_rows"]


def test_cold_entities_in_train_point_the_other_way_from_unseen_entities_in_test(
    sections: dict[str, dict[str, Any]], split_summary: dict[str, Any]
) -> None:
    # Two different questions that both get called cold. A first transaction inside the train
    # window carries less risk than a repeat; an entity the train window never saw carries more.
    cold = sections["temporal"]["cold_entities"]
    assert cold["cold"]["fraud_rate"] < cold["warm"]["fraud_rate"]
    overlap = split_summary["entity_overlap"]["test"]
    assert overlap["fraud_rate_on_unseen_entity_rows"] > overlap["fraud_rate_on_seen_entity_rows"]


# --- entities and quality ----------------------------------------------------------------


def test_entity_counts_match_the_split_summary(
    sections: dict[str, dict[str, Any]], split_summary: dict[str, Any]
) -> None:
    structure = sections["entities"]["structure"]
    assert structure["n_entities"] == split_summary["splits"]["train"]["n_entities"]
    assert structure["n_rows"] == split_summary["splits"]["train"]["n_rows"]


def test_the_entity_key_is_the_one_in_config(sections: dict[str, dict[str, Any]]) -> None:
    assert tuple(sections["entities"]["entity_key"]) == config.ENTITY_KEY_COLUMNS


def test_transaction_ids_are_unique_and_no_row_repeats_exactly(
    sections: dict[str, dict[str, Any]],
) -> None:
    duplicates = sections["quality"]["duplicates"]
    assert duplicates["transaction_id_is_unique"] is True
    assert duplicates["exact_rows"]["n_groups"] == 0


def test_the_wider_duplicate_definitions_find_at_least_as_much_as_the_narrower_ones(
    sections: dict[str, dict[str, Any]],
) -> None:
    duplicates = sections["quality"]["duplicates"]
    assert (
        duplicates["exact_rows"]["n_rows_in_groups"]
        <= duplicates["same_content_same_time"]["n_rows_in_groups"]
        <= duplicates["same_content"]["n_rows_in_groups"]
        <= duplicates["repeated_core_fields"]["n_rows_in_groups"]
    )


def test_no_amount_is_negative_or_zero(sections: dict[str, dict[str, Any]]) -> None:
    impossible = sections["quality"]["impossible_values"]
    assert impossible["n_negative_amounts"] == 0
    assert impossible["n_zero_amounts"] == 0
    assert impossible["n_non_finite_amounts"] == 0
    assert impossible["min_amount"] > 0


def test_email_agreement_cells_add_up_to_the_rows_with_both_present(
    sections: dict[str, dict[str, Any]],
) -> None:
    agreement = sections["quality"]["email_domains"]["agreement"]
    assert agreement["agree"]["n"] + agreement["disagree"]["n"] == agreement["n_both_present"]


# --- what an all-data EDA would have hidden ----------------------------------------------


def test_the_train_only_section_is_the_only_one_that_looks_past_the_boundary(
    sections: dict[str, dict[str, Any]], split_summary: dict[str, Any]
) -> None:
    report = sections["train_only"]
    assert report["n_rows_all"] == split_summary["totals"]["n_rows"]
    assert report["n_rows_train"] == split_summary["splits"]["train"]["n_rows"]


def test_folding_the_later_windows_in_moves_real_columns(
    sections: dict[str, dict[str, Any]],
) -> None:
    shift = sections["train_only"]["quantile_shift"]
    assert shift["n_statistics_moving_more_than_10_percent"] > 0
    worst = shift["largest_50"][0]
    assert worst["relative_shift"] > 1.0
    assert worst["on_train"] != worst["on_all_rows"]


def test_some_categorical_levels_exist_only_after_the_training_window(
    sections: dict[str, dict[str, Any]],
) -> None:
    block = sections["train_only"]["categorical_levels_only_after_train"]
    assert block["n_columns_with_unseen_levels"] > 0
    for row in block["per_column"]:
        assert (row["n_levels_only_after_train"] > 0) == (
            row["n_later_rows_on_a_level_train_never_saw"] > 0
        )


# --- figures -----------------------------------------------------------------------------


def test_every_figure_the_documents_reference_exists() -> None:
    for name in FIGURES:
        path = FIGURES_DIR / name
        assert path.exists(), f"{name} missing; run make eda-figures"
        assert path.stat().st_size > 10_000


# --- recomputation from the data ---------------------------------------------------------


@pytest.mark.needs_data
def test_the_identity_join_share_recomputes_from_the_files(
    sections: dict[str, dict[str, Any]],
) -> None:
    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data/train_transaction.csv not present")
    import pandas as pd

    frame = data_loader.load_raw(
        columns=[config.ID_COLUMN, config.TARGET, config.TIME_COLUMN, "id_01"]
    )
    train, _, _ = data_loader.time_based_split(frame)
    identity_ids = set(
        pd.read_csv(config.IDENTITY_PATH, usecols=[config.ID_COLUMN])[config.ID_COLUMN]
    )
    present = train[config.ID_COLUMN].isin(identity_ids)
    join = sections["missingness"]["identity_join"]
    assert int(present.sum()) == join["n_rows_with_identity"]
    assert float(train.loc[present, config.TARGET].mean()) == pytest.approx(
        join["fraud_rate_with_identity"]
    )


@pytest.mark.needs_data
def test_the_train_slice_the_eda_reads_passes_the_train_only_guard() -> None:
    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data/train_transaction.csv not present")
    frame = data_loader.load_raw(columns=[config.ID_COLUMN, config.TARGET, config.TIME_COLUMN])
    train, val, test = data_loader.time_based_split(frame)
    assert data_loader.assert_train_only(train, what="stage 2 test") is train
    for later in (val, test):
        with pytest.raises(data_loader.TrainOnlyError):
            data_loader.assert_train_only(later, what="stage 2 test")
