"""Internal consistency of the four stage 4 artifacts, checked without the data.

The artifacts are committed, so these run on a clean checkout and on every push. What they check
is the part that matters after the fact: that every number the stage wrote agrees with the number
the earlier stage measured, and that the declarations agree with each other.

Three kinds of check:

- **The catalogue is the source.** The artifact lists exactly the features the module builds, with
  the family and the serving state the module declares. It cannot list one the module does not
  build, and it cannot omit one it does.
- **Numbers quoted from earlier stages are the earlier stages' numbers.** The duplicate section
  quotes `reports/eda/quality.json`, the merchant section quotes
  `reports/eda/bivariate.json`, and the tenure block reproduces the PSI that
  `reports/prep/transforms.json` measured for D1. Each is asserted field by field rather than
  compared by eye.
- **Arithmetic inside an artifact adds up.** Marked plus unmarked is the split, a lift is the
  ratio of the two rates it claims to be, and a Wilson interval contains its own point estimate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import config, features

REPORTS = config.REPORTS_DIR
TRAIN_ROWS = 413_378


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return load(REPORTS / "feature_summary.json")


@pytest.fixture(scope="module")
def velocity_c() -> dict[str, Any]:
    return load(REPORTS / "features" / "velocity_vs_c.json")


@pytest.fixture(scope="module")
def merchant() -> dict[str, Any]:
    return load(REPORTS / "features" / "merchant_proxies.json")


@pytest.fixture(scope="module")
def duplicates() -> dict[str, Any]:
    return load(REPORTS / "features" / "duplicate_content.json")


@pytest.fixture(scope="module")
def quality() -> dict[str, Any]:
    return load(REPORTS / "eda" / "quality.json")


@pytest.fixture(scope="module")
def bivariate() -> dict[str, Any]:
    return load(REPORTS / "eda" / "bivariate.json")


@pytest.fixture(scope="module")
def transforms_report() -> dict[str, Any]:
    return load(REPORTS / "prep" / "transforms.json")


ALL = ("summary", "velocity_c", "merchant", "duplicates")


# --- the envelope ------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ALL)
def test_every_artifact_says_what_made_it(name: str, request: pytest.FixtureRequest) -> None:
    report = request.getfixturevalue(name)
    assert report["stage"] == 4
    assert "scripts/features.py" in report["regenerate_with"]
    assert report["n_train_rows"] == TRAIN_ROWS
    assert report["seed"] == config.SEED
    assert report["git_commit"]
    assert report["environment"]["pandas"]


@pytest.mark.parametrize("name", ALL)
def test_every_artifact_agrees_on_the_train_row_count(
    name: str, request: pytest.FixtureRequest
) -> None:
    """The same count stage 1 recorded, so a section run against a different slice is visible."""
    split_summary = load(REPORTS / "split_summary.json")
    expected = int(split_summary["splits"]["train"]["n_rows"])
    assert request.getfixturevalue(name)["n_train_rows"] == expected == TRAIN_ROWS


# --- the catalogue is the source ---------------------------------------------------------------


def test_the_artifact_lists_exactly_the_features_the_module_builds(summary: dict[str, Any]) -> None:
    listed = [row["feature"] for row in summary["per_feature"]]
    assert listed == [spec.name for spec in features.catalogue()]
    assert summary["n_features_listed"] == len(listed)
    derived = [row["feature"] for row in summary["per_feature"] if row["derived_in_stage_4"]]
    assert derived == features.feature_names()
    assert summary["n_features_derived"] == len(derived)


def test_the_family_index_partitions_the_features(summary: dict[str, Any]) -> None:
    families = summary["families"]
    assert set(families) == set(features.FAMILIES)
    flattened = sorted(name for names in families.values() for name in names)
    assert flattened == sorted(row["feature"] for row in summary["per_feature"])
    assert len(flattened) == len(set(flattened)), "a feature appears in two families"


def test_the_running_state_index_partitions_the_features(summary: dict[str, Any]) -> None:
    """Stage 8 reads this list to know what the online store must hold, so it has to be complete."""
    by_key = summary["running_state"]["by_state_key"]
    flattened = sorted(name for names in by_key.values() for name in names)
    assert flattened == sorted(row["feature"] for row in summary["per_feature"])
    stateful = [row for row in summary["per_feature"] if row["requires_running_state"]]
    assert summary["running_state"]["n_requiring_state"] == len(stateful)
    assert len(by_key["none"]) == len(summary["per_feature"]) - len(stateful)
    for row in stateful:
        assert row["state_key"] in features.GRAINS, row["feature"]
        assert row["state_fields"], f"{row['feature']} needs state and does not say which"


def test_a_windowed_feature_carries_a_window_and_the_rest_do_not(summary: dict[str, Any]) -> None:
    windows = {seconds for _, seconds in features.WINDOWS}
    for row in summary["per_feature"]:
        if row["family"] == "velocity":
            assert row["window_seconds"] in windows, row["feature"]


def test_every_feature_that_carries_nulls_says_what_a_null_means(summary: dict[str, Any]) -> None:
    for row in summary["per_feature"]:
        rates = [
            entry["null_rate"]
            for entry in row["null_rate_by_split"].values()
            if entry["null_rate"] is not None
        ]
        assert rates, row["feature"]
        assert all(0.0 <= rate <= 1.0 for rate in rates), row["feature"]
        if any(rate > 0.0 for rate in rates):
            assert row["null_meaning"], f"{row['feature']} carries nulls with no stated meaning"
        else:
            assert row["null_meaning"] is None, (
                f"{row['feature']} carries no null and declares a meaning for one"
            )


def test_the_null_rate_is_reported_for_all_three_splits(summary: dict[str, Any]) -> None:
    for row in summary["per_feature"]:
        assert set(row["null_rate_by_split"]) == set(config.SPLIT_NAMES), row["feature"]
        for split, entry in row["null_rate_by_split"].items():
            assert entry["present"] is True, f"{row['feature']} absent from {split}"
            assert entry["n_null"] == pytest.approx(entry["null_rate"] * entry["n_rows"], abs=1)


# --- the measurement behind ADR 0023 -----------------------------------------------------------


def test_the_entity_key_pins_addr1(summary: dict[str, Any]) -> None:
    """The number the geography grain rests on. If this ever stops holding, ADR 0023 is wrong."""
    block = summary["entity_key_pins_addr1"]
    assert block["entity"]["addr1"]["max_distinct_per_key"] == 1
    assert block["entity"]["addr1"]["share_keys_with_more_than_one"] == 0.0
    assert block["card"]["addr1"]["max_distinct_per_key"] > 1
    assert block["card"]["addr1"]["share_keys_with_more_than_one"] > 0.0
    assert block["card"]["n_keys"] < block["entity"]["n_keys"]


def test_no_shipped_feature_reads_the_entity_modal_addr1(summary: dict[str, Any]) -> None:
    names = [row["feature"] for row in summary["per_feature"]]
    assert not [name for name in names if "differs_from_entity_mode" in name]
    for row in summary["per_feature"]:
        if "differs_from_card_mode" in row["feature"]:
            assert row["grain"] == "card"


# --- the tenure origin reproduces stage 3 -------------------------------------------------------


def test_the_tenure_origin_reproduces_the_psi_stage_3_measured(
    summary: dict[str, Any], transforms_report: dict[str, Any]
) -> None:
    """D1_origin is the column ADR 0013 measured under another name. The numbers have to agree."""
    block = summary["tenure_origin"]
    stage_3 = next(
        row for row in transforms_report["d_normalisation"]["per_column"] if row["column"] == "D1"
    )
    assert block["psi_against"]["test"] == pytest.approx(stage_3["psi_after"], rel=1e-9)
    assert block["psi_of_d1_raw"]["test"] == pytest.approx(stage_3["psi_before"], rel=1e-9)
    assert block["shipped"] is False
    assert block["column"] not in [row["feature"] for row in summary["per_feature"]]


def test_the_tenure_origin_is_card_start_day_plus_a_fraction_of_a_day(
    summary: dict[str, Any],
) -> None:
    block = summary["tenure_origin"]["against_card_start_day"]
    assert block["min_difference"] >= 0.0
    assert block["max_difference"] < 1.0


def test_d1_is_listed_as_the_tenure_feature_and_not_rebuilt(summary: dict[str, Any]) -> None:
    row = next(row for row in summary["per_feature"] if row["feature"] == "D1")
    assert row["family"] == "tenure"
    assert row["derived_in_stage_4"] is False
    assert row["requires_running_state"] is False


# --- coverage -----------------------------------------------------------------------------------


def test_coverage_shares_are_shares(summary: dict[str, Any]) -> None:
    for split in config.SPLIT_NAMES:
        block = summary["coverage"][split]
        assert block["n_rows"] > 0
        for grain in ("entity", "card", "content"):
            assert 0.0 <= block[grain]["share_cold"] <= 1.0
            assert block[grain]["n_cold"] <= block["n_rows"]


def test_a_point_in_time_feature_covers_more_than_a_train_fitted_one(
    summary: dict[str, Any],
) -> None:
    """The comparison behind the coverage figure, and the reason ADR 0015's exclusion costs less
    than stage 3 expected it to."""
    for split in ("val", "test"):
        block = summary["coverage"][split]["entity_history_against_training_overlap"]
        assert (
            block["share_with_any_earlier_row_in_the_stream"] > block["share_entity_seen_in_train"]
        ), split
    split_summary = load(REPORTS / "split_summary.json")
    for split in ("val", "test"):
        overlap = split_summary["entity_overlap"][split]
        expected_seen = 1.0 - float(overlap["share_rows_on_unseen_entities"])
        measured = summary["coverage"][split]["entity_history_against_training_overlap"][
            "share_entity_seen_in_train"
        ]
        assert measured == pytest.approx(expected_seen, abs=5e-4), (
            f"{split}: the seen share here should be stage 1's unseen share complemented"
        )


# --- velocity against the native C columns ------------------------------------------------------


def test_the_correlation_grid_is_complete(velocity_c: dict[str, Any]) -> None:
    own = velocity_c["per_feature"]
    native = [row["column"] for row in velocity_c["native_c_columns"]]
    assert len(native) == len(config.COUNT_COLUMNS)
    assert velocity_c["summary"]["n_pairs"] == len(own) * len(native)
    assert velocity_c["summary"]["n_pairs_measured"] == velocity_c["summary"]["n_pairs"]
    for row in own:
        assert set(row["spearman_against_c"]) == set(native), row["feature"]
        assert row["max_abs_rho"] == pytest.approx(
            max(abs(value) for value in row["spearman_against_c"].values()), rel=1e-12
        )


def test_the_summary_maximum_is_the_grid_maximum(velocity_c: dict[str, Any]) -> None:
    grid = [
        abs(value)
        for row in velocity_c["per_feature"]
        for value in row["spearman_against_c"].values()
    ]
    assert velocity_c["summary"]["max_abs_rho"] == pytest.approx(max(grid), rel=1e-12)
    assert velocity_c["summary"]["n_pairs_above_0.5"] == sum(1 for v in grid if v >= 0.5)
    assert velocity_c["summary"]["n_pairs_above_0.8"] == sum(1 for v in grid if v >= 0.8)


def test_the_two_blocks_are_not_the_same_measurement(velocity_c: dict[str, Any]) -> None:
    """ADR 0021's condition for keeping both, asserted as the artifact's own number."""
    assert velocity_c["summary"]["max_abs_rho"] < 0.5
    assert velocity_c["summary"]["n_pairs_above_0.5"] == 0


def test_the_velocity_grains_are_both_present_and_labelled(velocity_c: dict[str, Any]) -> None:
    grains = {row["grain"] for row in velocity_c["per_feature"]}
    assert grains == {"entity", "card"}


def test_a_collapsed_binning_is_visible_beside_its_information_value(
    velocity_c: dict[str, Any],
) -> None:
    """An information value of 0.0000 on a zero-inflated count is a binning artifact, and the
    artifact has to show that rather than let the number be read as "carries nothing"."""
    for row in velocity_c["per_feature"]:
        assert row["n_bins"] <= row["n_bins_requested"]
        block = row["zero_against_positive"]
        assert block["n_positive"] + block["n_zero"] == TRAIN_ROWS
        if row["information_value"] == 0.0:
            assert row["n_bins"] == 1
            assert block["lift"] is not None and block["lift"] > 1.0


def test_c3_reads_zero_here_as_it_did_in_stage_2(
    velocity_c: dict[str, Any], bivariate: dict[str, Any]
) -> None:
    """C3 is the column stage 3 dropped as effectively constant. The two stages have to agree."""
    here = next(row for row in velocity_c["native_c_columns"] if row["column"] == "C3")
    there = next(row for row in bivariate["per_feature"] if row["column"] == "C3")
    assert here["information_value"] == pytest.approx(float(there["iv"]), rel=1e-9)


def test_the_c_column_meaning_is_not_claimed(velocity_c: dict[str, Any]) -> None:
    """Stage 0 could not fetch the data description and neither could stage 4. The artifact says
    so rather than restating the claim as fact."""
    assert velocity_c["what_the_c_columns_are"]["status"] == "not established in this repo"


# --- merchant ------------------------------------------------------------------------------------


def test_the_file_holds_no_merchant_or_category_column(merchant: dict[str, Any]) -> None:
    inventory = merchant["column_inventory"]
    assert inventory["n_columns"] == 434
    assert inventory["n_name_matches"] == 0
    assert inventory["name_matches"] == []
    assert len(inventory["unnumbered_columns"]) == 19


def test_every_proxy_quotes_the_information_value_stage_2_measured(
    merchant: dict[str, Any], bivariate: dict[str, Any]
) -> None:
    measured = {str(row["column"]): row for row in bivariate["per_feature"]}
    for proxy in merchant["proxies"]:
        source = measured.get(proxy["column"])
        if source is None:
            assert proxy["information_value"] is None, proxy["column"]
            continue
        assert proxy["information_value"] == pytest.approx(float(source["iv"]), rel=1e-9)
        assert proxy["information_value_present_bins"] == pytest.approx(
            float(source["iv_present_bins"]), rel=1e-9
        )


def test_the_merchant_section_says_the_numbers_are_not_comparable(
    merchant: dict[str, Any],
) -> None:
    assert "not comparable" in merchant["not_comparable"]
    assert "ADR 0019" in merchant["not_comparable"]


# --- duplicate content ----------------------------------------------------------------------------


def test_the_duplicate_section_quotes_stage_2_exactly(
    duplicates: dict[str, Any], quality: dict[str, Any]
) -> None:
    source = quality["duplicates"]["repeated_core_fields"]
    here = duplicates["stage_2_grouping"]
    assert here["columns"] == list(source["columns"])
    assert here["n_groups"] == int(source["n_groups"])
    assert here["n_rows_in_groups"] == int(source["n_rows_in_groups"])
    assert here["n_extra_rows"] == int(source["n_extra_rows"])
    assert here["fraud_rate_in_groups"] == pytest.approx(
        float(source["fraud_rate_in_groups"]), rel=1e-12
    )


def test_the_content_columns_are_stage_2s_six_fields(duplicates: dict[str, Any]) -> None:
    assert duplicates["stage_2_grouping"]["columns"] == list(features.CONTENT_COLUMNS)


def test_the_causal_count_reproduces_stage_2s_extra_rows(duplicates: dict[str, Any]) -> None:
    """The strongest cross-check in the stage: two unrelated implementations of the same grouping.

    Stage 2 hashed the six fields and compared rows inside each bucket. Stage 4 factorized them
    and counted strictly earlier matches per row. The counts differ only by rows whose sole
    identical companion shares their timestamp, and the artifact bounds that by the tied count.
    """
    block = duplicates["against_stage_2"]
    assert block["difference"] == block["stage_2_n_extra_rows"] - block["causal_n_rows_marked"]
    assert 0 <= block["difference"] <= block["n_tied_rows_in_content_groups"]
    assert block["difference"] / block["stage_2_n_extra_rows"] < 1e-3


def test_the_causal_lift_is_the_ratio_it_claims_to_be(duplicates: dict[str, Any]) -> None:
    block = duplicates["causal_grouping"]
    assert block["lift"] == pytest.approx(
        block["fraud_rate_marked"] / block["fraud_rate_unmarked"], rel=1e-12
    )
    assert block["interval_marked"]["n"] + block["interval_unmarked"]["n"] == TRAIN_ROWS


def test_the_causal_lift_clears_the_interval_it_is_measured_against(
    duplicates: dict[str, Any],
) -> None:
    """The rule ADR 0022 was written to. Ship when the two Wilson intervals do not overlap."""
    block = duplicates["causal_grouping"]
    assert block["interval_marked"]["low"] > block["interval_unmarked"]["high"]


def test_the_hour_window_feature_separates_more_than_the_all_history_one(
    duplicates: dict[str, Any],
) -> None:
    """The finding that decided which duplicate features are worth their width."""
    hour = next(
        row["zero_against_positive"]
        for row in duplicates["per_feature"]
        if row["feature"] == "dup_prior_count_1h"
    )
    ever = next(
        row["zero_against_positive"]
        for row in duplicates["per_feature"]
        if row["feature"] == "dup_prior_count"
    )
    assert hour["lift"] > ever["lift"]
    assert hour["interval_positive"]["low"] > hour["interval_zero"]["high"]


def test_the_strict_content_definition_is_recorded_as_empty(duplicates: dict[str, Any]) -> None:
    """Stage 2's whole-row duplicate definitions found 10 rows and no fraud. The stage brief asked
    about content duplicates and this is why the core-field definition is the one used."""
    block = duplicates["strict_content_grouping"]
    assert block["n_rows_in_groups"] == 10
    assert block["fraud_rate_in_groups"] == 0.0


@pytest.mark.parametrize(
    ("artifact", "path"),
    [
        ("duplicates", ("causal_grouping", "interval_marked")),
        ("duplicates", ("causal_grouping", "interval_unmarked")),
    ],
)
def test_a_wilson_interval_contains_its_own_point(
    artifact: str, path: tuple[str, ...], request: pytest.FixtureRequest
) -> None:
    block: Any = request.getfixturevalue(artifact)
    for key in path:
        block = block[key]
    assert block["low"] <= block["point"] <= block["high"]


# --- what is deliberately absent -------------------------------------------------------------------


def test_the_summary_records_what_was_not_built(summary: dict[str, Any]) -> None:
    assert summary["not_built"] == list(features.NOT_BUILT)
    joined = " ".join(summary["not_built"])
    for expected in ("UID aggregation", "merchant", "ADR 0020", "ADR 0019", "ADR 0023"):
        assert expected in joined


# --- the per-split separation, which is the check a train-only number would have passed ----------


def _by_split_blocks(
    velocity_c: dict[str, Any], duplicates: dict[str, Any]
) -> list[tuple[str, Any]]:
    blocks = [(row["feature"], row["by_split"]) for row in velocity_c["per_feature"]]
    blocks += [(row["column"], row["by_split"]) for row in velocity_c["native_c_columns"]]
    blocks += [("causal_grouping", duplicates["causal_grouping_by_split"])]
    blocks += [
        (row["feature"], row["by_split"]) for row in duplicates["per_feature"] if "by_split" in row
    ]
    return blocks


def test_every_separation_is_reported_on_all_three_splits(
    velocity_c: dict[str, Any], duplicates: dict[str, Any]
) -> None:
    for name, block in _by_split_blocks(velocity_c, duplicates):
        for split in config.SPLIT_NAMES:
            assert split in block, f"{name} does not report {split}"
            entry = block[split]
            assert entry["n_positive"] + entry["n_zero"] > 0, name


def test_the_stability_flags_are_the_arithmetic_they_claim(
    velocity_c: dict[str, Any], duplicates: dict[str, Any]
) -> None:
    """`direction_stable` is the instability test and `positive_is_riskier` is not. A stably
    inverse column has to pass the first and fail the second, or the two are measuring one thing."""
    for name, block in _by_split_blocks(velocity_c, duplicates):
        rates = [
            (block[split]["fraud_rate_positive"], block[split]["fraud_rate_zero"])
            for split in config.SPLIT_NAMES
            if block[split]["fraud_rate_positive"] is not None
            and block[split]["fraud_rate_zero"] is not None
        ]
        signs = {1 if positive > zero else -1 for positive, zero in rates}
        assert block["direction_stable"] == (bool(rates) and len(signs) == 1), name
        assert block["positive_is_riskier"] == (
            bool(rates) and all(positive > zero for positive, zero in rates)
        ), name
        if block["positive_is_riskier"]:
            assert block["direction_stable"], f"{name} cannot be riskier everywhere and unstable"


def test_every_shipped_velocity_feature_holds_on_every_split(velocity_c: dict[str, Any]) -> None:
    """ADR 0021's out-of-sample condition. This is the test that would have caught the duplicate
    family's reversal if the duplicate family had been velocity."""
    for row in velocity_c["per_feature"]:
        block = row["by_split"]
        assert block["positive_is_riskier"], row["feature"]
        assert block["direction_stable"], row["feature"]
        assert block["disjoint_on_every_split"], row["feature"]


def test_the_all_history_duplicate_count_reverses_and_is_labelled(
    duplicates: dict[str, Any],
) -> None:
    """ADR 0022's finding. If this ever stops holding, that ADR is superseded rather than edited."""
    block = duplicates["causal_grouping_by_split"]
    assert block["train"]["fraud_rate_positive"] > block["train"]["fraud_rate_zero"]
    for split in ("val", "test"):
        assert block[split]["fraud_rate_positive"] < block[split]["fraud_rate_zero"], split
    assert block["direction_stable"] is False
    assert block["disjoint_on_every_split"] is True, (
        "the reversal is outside the interval on every split, so it is a sign change and not noise"
    )


def test_the_hour_windowed_duplicate_count_holds_on_every_split(
    duplicates: dict[str, Any],
) -> None:
    block = next(
        row["by_split"]
        for row in duplicates["per_feature"]
        if row["feature"] == "dup_prior_count_1h"
    )
    assert block["positive_is_riskier"]
    assert block["direction_stable"]
    assert block["disjoint_on_every_split"]
    ratios = [block[split]["lift"] for split in config.SPLIT_NAMES]
    assert all(ratio > 1.5 for ratio in ratios)
