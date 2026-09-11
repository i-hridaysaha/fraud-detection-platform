"""Unit tests for the stage 2 analysis primitives, on frames small enough to check by hand.

These run in CI. Anything that reads the real CSVs or the committed artifacts is in
test_eda_artifacts.py and is marked `needs_data`.

Several of these assert against a value worked out by hand rather than against whatever the
function currently returns, which is the only version of the test worth having: a test that
records today's output cannot fail when the meaning changes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, eda

# --- univariate ------------------------------------------------------------------------


def test_numeric_profile_counts_nulls_separately_from_zeros() -> None:
    series = pd.Series([0.0, 0.0, 1.0, None, 3.0])
    profile = eda.numeric_profile(series, "x")
    assert profile["count"] == 4
    assert profile["missing_rate"] == pytest.approx(0.2)
    assert profile["zero_share"] == pytest.approx(0.5)
    assert profile["median"] == pytest.approx(0.5)


def test_numeric_profile_flags_a_column_one_value_dominates() -> None:
    series = pd.Series([7.0] * 995 + [1.0] * 5)
    profile = eda.numeric_profile(series, "x")
    assert profile["top_value_share"] == pytest.approx(0.995)
    assert profile["effectively_constant"] is True
    assert profile["single_value_dominated"] is True


def test_numeric_profile_of_an_all_null_column_says_so_and_stops() -> None:
    profile = eda.numeric_profile(pd.Series([None, None], dtype="float64"), "x")
    assert profile["all_null"] is True
    assert "percentiles" not in profile


def test_numeric_profile_leaves_undefined_moments_as_none() -> None:
    profile = eda.numeric_profile(pd.Series([1.0, 2.0]), "x")
    assert profile["std"] is not None
    assert profile["kurtosis"] is None


def test_categorical_profile_ignores_categories_no_row_uses() -> None:
    # A category set fixed over the whole file leaves empty levels in any slice of it. Counting
    # them would overstate cardinality on every split.
    dtype = pd.CategoricalDtype(categories=["a", "b", "c", "d"])
    series = pd.Series(["a", "a", "b"], dtype=dtype)
    profile = eda.categorical_profile(series, "c")
    assert profile["cardinality"] == 2
    assert profile["top_value"] == "a"


def test_categorical_rare_tail_counts_from_the_least_frequent_upwards() -> None:
    series = pd.Series(["a"] * 980 + ["b"] * 15 + ["c"] * 3 + ["d"] * 2)
    profile = eda.categorical_profile(series, "c")
    # d covers 0.002 of the rows and c another 0.003. Adding b would take the tail to 0.020,
    # past the 0.01 target, so the tail is two categories wide.
    assert profile["n_categories_in_rare_tail"] == 2
    assert profile["rare_tail_row_share"] == pytest.approx(0.005)
    assert profile["n_singleton_categories"] == 0


# --- rates and intervals ---------------------------------------------------------------


def test_wilson_interval_brackets_the_point_and_stays_inside_zero_and_one() -> None:
    interval = eda.wilson_interval(1, 20)
    assert interval["low"] > 0.0
    assert interval["low"] < interval["point"] < interval["high"]
    assert interval["high"] < 1.0


def test_wilson_interval_of_an_empty_cell_is_null_rather_than_a_number() -> None:
    assert eda.wilson_interval(0, 0)["point"] is None


def test_bootstrap_interval_is_reproducible_and_contains_the_point() -> None:
    labels = np.array([0] * 950 + [1] * 50)
    first = eda.bootstrap_rate_interval(labels, 2000, seed=42)
    second = eda.bootstrap_rate_interval(labels, 2000, seed=42)
    assert first == second
    assert first["low"] < first["point"] < first["high"]


def test_two_proportion_test_finds_no_difference_between_equal_rates() -> None:
    assert eda.two_proportion_test(50, 1000, 50, 1000)["p_value"] == pytest.approx(1.0)


def test_trend_tests_see_a_monotone_series_as_monotone() -> None:
    rising = eda.trend_tests([0, 1, 2, 3, 4], [0.01, 0.02, 0.03, 0.04, 0.05], [1000] * 5)
    assert rising["kendall_tau"] == pytest.approx(1.0)
    assert rising["ols_slope_per_period"] == pytest.approx(0.01)
    assert rising["rate_range"] == pytest.approx(0.04)


def test_trend_tests_separate_no_trend_from_no_variation() -> None:
    # A series that bounces has no monotone trend and is still not one rate. Kendall says the
    # first, chi-square says the second, and reporting only one of them would hide the other.
    bouncing = eda.trend_tests([0, 1, 2, 3], [0.01, 0.09, 0.01, 0.09], [5000] * 4)
    assert abs(float(bouncing["kendall_tau"])) < 0.5
    assert float(bouncing["chi2_p_value"]) < 1e-6


def test_rate_by_group_keeps_missing_as_its_own_level() -> None:
    frame = pd.DataFrame({"g": ["a", "a", None, None], "isFraud": [1, 0, 1, 1]})
    rows = {row["value"]: row for row in eda.rate_by_group(frame, "g", "isFraud")}
    assert rows["(missing)"]["n"] == 2
    assert rows["(missing)"]["fraud_rate"] == pytest.approx(1.0)
    assert sum(row["n"] for row in rows.values()) == len(frame)


# --- missingness -----------------------------------------------------------------------


def test_missingness_blocks_group_columns_missing_on_the_same_rows() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [4.0, None, 6.0],
            "c": [None, 8.0, 9.0],
            "d": [1.0, 2.0, 3.0],
        }
    )
    blocks = {tuple(b["columns"]): b for b in eda.missingness_blocks(frame.isna())}
    assert ("a", "b") in blocks
    assert blocks[("a", "b")]["n_rows_missing"] == 1
    assert ("c",) in blocks
    assert ("d",) in blocks


def test_missingness_against_target_skips_columns_with_nothing_to_compare() -> None:
    frame = pd.DataFrame({"a": [1.0, None, 3.0, 4.0], "full": [1.0, 2.0, 3.0, 4.0]})
    target = pd.Series([0, 1, 0, 0])
    rows = eda.missingness_against_target(frame.isna(), target)
    assert [row["column"] for row in rows] == ["a"]
    assert rows[0]["fraud_rate_missing"] == pytest.approx(1.0)
    assert rows[0]["fraud_rate_present"] == pytest.approx(0.0)


def test_missingness_flag_needs_both_a_gap_and_a_p_value() -> None:
    # 2,000 rows, a tiny gap, a large sample. The test alone would call it a difference; the
    # absolute gate is what stops the flag.
    n = 2000
    missing = pd.Series([True] * n + [False] * n)
    labels = np.array([1] * 70 + [0] * (n - 70) + [1] * 68 + [0] * (n - 68))
    rows = eda.missingness_against_target(pd.DataFrame({"a": missing}), pd.Series(labels))
    assert abs(rows[0]["gap"]) < eda.MNAR_ABS_GAP
    assert rows[0]["label_linked"] is False


# --- correlation -----------------------------------------------------------------------


def test_spearman_matrix_matches_pandas_when_no_value_is_missing() -> None:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    frame["e"] = frame["a"] * 3.0 + 1.0
    rho, counts = eda.spearman_matrix(eda.rank_matrix(frame), chunk_rows=100)
    reference = frame.corr(method="spearman").to_numpy()
    assert np.abs(rho - reference).max() < 1e-6
    assert counts[0, 1] == 500


def test_spearman_matrix_is_exact_when_two_columns_share_a_missingness_mask() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    b = rng.normal(size=400)
    hidden = rng.random(400) < 0.3
    frame = pd.DataFrame({"a": np.where(hidden, np.nan, a), "b": np.where(hidden, np.nan, b)})
    rho, _ = eda.spearman_matrix(eda.rank_matrix(frame), chunk_rows=97)
    reference = frame.corr(method="spearman").to_numpy()[0, 1]
    assert abs(rho[0, 1] - reference) < 1e-6


def test_correlated_groups_puts_a_scaled_copy_with_its_original() -> None:
    names = ["a", "b", "c"]
    rho = np.array([[1.0, 0.999, 0.1], [0.999, 1.0, 0.1], [0.1, 0.1, 1.0]])
    groups = eda.correlated_groups(names, rho, 0.95, [3.0, 2.0, 1.0])
    assert [g["representative"] for g in groups] == ["a", "c"]
    assert groups[0]["members"] == ["a", "b"]


def test_correlated_groups_follows_the_priority_order() -> None:
    names = ["a", "b"]
    rho = np.array([[1.0, 0.99], [0.99, 1.0]])
    groups = eda.correlated_groups(names, rho, 0.95, [1.0, 5.0])
    assert groups[0]["representative"] == "b"


def test_near_duplicate_pairs_reads_the_exact_values_not_the_screen() -> None:
    exact = {"a|b": 0.995, "a|c": 0.5, "b|V9": 0.999}
    counts = np.full((4, 4), 100, dtype="int64")
    index_of = {"a": 0, "b": 1, "c": 2, "V9": 3}
    everything = eda.near_duplicate_pairs(exact, counts, index_of, 0.99, 10)
    assert {(p["a"], p["b"]) for p in everything["pairs"]} == {("a", "b"), ("b", "V9")}
    without_v = eda.near_duplicate_pairs(exact, counts, index_of, 0.99, 10, exclude_prefix="V")
    assert {(p["a"], p["b"]) for p in without_v["pairs"]} == {("a", "b")}


# --- bivariate and drift ---------------------------------------------------------------


def test_information_value_is_zero_when_the_column_says_nothing() -> None:
    values = pd.Series(list(range(1000)))
    target = pd.Series([0, 1] * 500)
    assert eda.information_value(values, target)["iv"] < 0.01


def test_information_value_rises_when_the_column_separates_the_classes() -> None:
    values = pd.Series(list(range(1000)))
    target = pd.Series([0] * 500 + [1] * 500)
    assert eda.information_value(values, target)["iv"] > 1.0


def test_information_value_splits_out_the_missing_bin_exactly() -> None:
    values = pd.Series([None] * 200 + list(range(200)))
    target = pd.Series([1] * 200 + [0] * 200)
    result = eda.information_value(values, target)
    assert result["iv"] == pytest.approx(result["iv_missing_bin"] + result["iv_present_bins"])
    assert result["iv_missing_bin"] > 0.0


def test_decile_table_bins_add_up_to_the_rows_it_was_given() -> None:
    values = pd.Series([None] * 10 + list(range(100)))
    target = pd.Series([0] * 110)
    rows = eda.decile_table(values, target)
    assert sum(row["n"] for row in rows) == 110
    assert rows[0]["bin"] == "(missing)"


def test_psi_of_a_distribution_against_itself_is_zero() -> None:
    values = pd.Series(np.random.default_rng(2).normal(size=5000))
    assert eda.population_stability_index(values, values)["psi"] == pytest.approx(0.0, abs=1e-9)


def test_psi_rises_when_the_later_distribution_moves() -> None:
    rng = np.random.default_rng(3)
    train = pd.Series(rng.normal(size=5000))
    shifted = pd.Series(rng.normal(loc=2.0, size=5000))
    assert eda.population_stability_index(train, shifted)["psi"] > 0.5


def test_psi_bins_on_train_and_does_not_rebin_on_the_later_split() -> None:
    # If the later split were rebinned, a pure scale change would look like no drift at all.
    train = pd.Series(np.linspace(0.0, 1.0, 2000))
    scaled = pd.Series(np.linspace(10.0, 11.0, 2000))
    assert eda.population_stability_index(train, scaled)["psi"] > 1.0


def test_psi_handles_a_categorical_column_without_trying_to_bin_it() -> None:
    train = pd.Series(pd.Categorical(["a"] * 80 + ["b"] * 20))
    later = pd.Series(pd.Categorical(["a"] * 20 + ["b"] * 80))
    assert eda.population_stability_index(train, later)["psi"] > 0.5


def test_time_consistency_reports_a_status_instead_of_raising_on_one_class() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    result = eda.time_consistency(
        values, pd.Series([0, 0, 0, 0]), values, pd.Series([0, 1, 0, 1]), seed=42
    )
    assert result["status"] == "one class in one window"
    assert result["early_auc"] is None


def test_time_consistency_catches_a_relationship_that_inverts() -> None:
    rng = np.random.default_rng(4)
    n = 3000
    early_y = pd.Series((rng.random(n) < 0.3).astype(int))
    late_y = pd.Series((rng.random(n) < 0.3).astype(int))
    # The same column predicts one way early and the opposite way late.
    early_x = pd.Series(early_y.to_numpy() + rng.normal(scale=0.4, size=n))
    late_x = pd.Series(-late_y.to_numpy() + rng.normal(scale=0.4, size=n))
    result = eda.time_consistency(early_x, early_y, late_x, late_y, seed=42)
    assert result["early_auc"] > 0.7
    assert result["late_auc"] < 0.3


# --- class balance ---------------------------------------------------------------------


def test_hanley_mcneil_standard_error_falls_as_positives_are_added() -> None:
    small = eda.hanley_mcneil_se(0.9, 1000, 30000)
    large = eda.hanley_mcneil_se(0.9, 4000, 120000)
    assert large < small
    assert math.isfinite(small)


def test_detectable_difference_shrinks_as_the_sample_grows() -> None:
    small = eda.detectable_auc_difference(0.9, 1000, 30000)
    large = eda.detectable_auc_difference(0.9, 4000, 120000)
    assert large["min_detectable_difference_unpaired"] < small["min_detectable_difference_unpaired"]


def test_recall_half_width_is_the_binomial_one() -> None:
    result = eda.recall_precision_of_positives(2500, 0.6)
    assert result["se"] == pytest.approx(math.sqrt(0.6 * 0.4 / 2500))


# --- entities and duplicates -----------------------------------------------------------


def test_entity_structure_counts_entities_and_their_gaps() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["a", "a", "a", "b"],
            "isFraud": [0, 1, 0, 0],
            "TransactionDT": [100, 200, 400, 999],
        }
    )
    out = eda.entity_structure(frame, "entity_id", "isFraud", "TransactionDT")
    assert out["n_entities"] == 2
    assert out["singleton_entity_share"] == pytest.approx(0.5)
    assert out["label_purity"]["share_mixed"] == pytest.approx(1.0)
    assert out["seconds_between_transactions"]["all"]["n"] == 2


def test_cold_warm_within_calls_the_first_row_of_each_entity_cold() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["a", "a", "b"],
            "isFraud": [1, 0, 0],
            "TransactionDT": [100, 200, 150],
        }
    )
    out = eda.cold_warm_within(frame, "entity_id", "isFraud", "TransactionDT")
    assert out["cold"]["n_rows"] == 2
    assert out["warm"]["n_rows"] == 1
    assert out["cold"]["fraud_rate"] == pytest.approx(0.5)


def test_duplicate_groups_treats_two_nulls_in_the_same_column_as_equal() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0, 2.0],
            "b": [None, None, 5.0],
            "isFraud": [1, 1, 0],
        }
    )
    out = eda.duplicate_groups(frame, ["a", "b"], "isFraud")
    assert out["n_groups"] == 1
    assert out["n_rows_in_groups"] == 2
    assert out["fraud_rate_in_groups"] == pytest.approx(1.0)
    assert out["fraud_rate_outside_groups"] == pytest.approx(0.0)


def test_duplicate_groups_finds_nothing_when_every_row_differs() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "isFraud": [0, 0, 0]})
    out = eda.duplicate_groups(frame, ["a"], "isFraud")
    assert out["n_groups"] == 0
    assert out["n_rows_in_groups"] == 0
    assert out["fraud_rate_in_groups"] is None


# --- the pieces the driver script leans on ----------------------------------------------


def test_histogram_edges_and_counts_describe_the_column_it_was_given() -> None:
    hist = eda.histogram(pd.Series([0.0, 1.0, 2.0, 3.0, None]), bins=4)
    assert hist["n"] == 4
    assert sum(hist["counts"]) == 4
    assert len(hist["edges"]) == 5
    assert hist["edges"][0] == pytest.approx(0.0)
    assert hist["edges"][-1] == pytest.approx(3.0)


def test_block_overlap_matrix_puts_each_blocks_own_rate_on_the_diagonal() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, None, None, 4.0],
            "b": [1.0, None, None, 4.0],
            "c": [None, None, 3.0, 4.0],
            "d": [None, None, 3.0, 4.0],
        }
    )
    mask = frame.isna()
    blocks = eda.missingness_blocks(mask)
    overlap = eda.block_overlap_matrix(mask, blocks)
    matrix = overlap["share_missing_in_both"]
    for index, block in enumerate(overlap["blocks"]):
        assert matrix[index][index] == pytest.approx(blocks[index]["missing_rate"])
        assert block["n_columns"] == blocks[index]["n_columns"]
    # a and c are both missing on row 1 only, which is a quarter of the frame.
    assert matrix[0][1] == pytest.approx(0.25)


def test_screened_pairs_returns_the_strong_pairs_strongest_first() -> None:
    names = ["a", "b", "c"]
    rho = np.array([[1.0, 0.9, -0.99], [0.9, 1.0, 0.2], [-0.99, 0.2, 1.0]])
    assert eda.screened_pairs(names, rho, 0.85) == [("a", "c"), ("a", "b")]
    assert eda.screened_pairs(names, rho, 0.995) == []


def test_exact_spearman_agrees_with_pandas_on_the_pairs_it_is_handed() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    frame["c"] = frame["a"] ** 3
    exact = eda.exact_spearman(frame, [("a", "c"), ("a", "b")])
    assert exact["a|c"] == pytest.approx(1.0)
    assert exact["a|b"] == pytest.approx(
        float(frame[["a", "b"]].corr(method="spearman").to_numpy()[0, 1])
    )


def test_compare_against_pandas_reports_the_worst_pair_it_saw() -> None:
    rng = np.random.default_rng(8)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    rho, _ = eda.spearman_matrix(eda.rank_matrix(frame), chunk_rows=64)
    result = eda.compare_against_pandas(frame, [("a", "b")], rho, {"a": 0, "b": 1})
    assert result["n_pairs_compared"] == 1
    assert result["n_pairs_within_tolerance"] == 1
    assert result["worst_pair"]["a"] == "a"


def test_binning_a_constant_column_gives_one_bin_and_keeps_missing_apart() -> None:
    values = pd.Series([5.0, 5.0, 5.0, None])
    result = eda.information_value(values, pd.Series([0, 1, 0, 1]))
    assert result["n_bins"] == 2
    table = eda.decile_table(values, pd.Series([0, 1, 0, 1]))
    assert {row["bin"] for row in table} == {"(missing)", "(single bin)"}


def test_information_value_has_no_missing_term_when_nothing_is_missing() -> None:
    result = eda.information_value(pd.Series(range(100)), pd.Series([0, 1] * 50))
    assert result["iv_missing_bin"] == 0.0
    assert result["iv_present_bins"] == pytest.approx(result["iv"])


def test_entity_structure_reports_no_gaps_when_every_entity_appears_once() -> None:
    frame = pd.DataFrame({"entity_id": ["a", "b"], "isFraud": [0, 1], "TransactionDT": [1, 2]})
    out = eda.entity_structure(frame, "entity_id", "isFraud", "TransactionDT")
    assert out["seconds_between_transactions"]["all"] == {"n": 0}
    assert out["label_purity"]["n_entities_multi"] == 0
    assert out["label_purity"]["share_mixed"] is None


# --- the edges a 434 column frame actually contains ---------------------------------------


def test_categorical_profile_of_an_all_null_column_says_so_and_stops() -> None:
    profile = eda.categorical_profile(pd.Series([None, None], dtype="object"), "c")
    assert profile["all_null"] is True
    assert profile["cardinality"] == 0


def test_bootstrap_interval_of_an_empty_selection_is_null_rather_than_a_number() -> None:
    assert eda.bootstrap_rate_interval(np.array([], dtype="int64"), 100, seed=1)["point"] is None


def test_two_proportion_test_declines_to_answer_on_an_empty_cell() -> None:
    assert eda.two_proportion_test(0, 0, 5, 100)["p_value"] is None
    # Both cells at rate zero leaves the pooled standard error at zero, and no z to report.
    assert eda.two_proportion_test(0, 50, 0, 50)["p_value"] is None


def test_rate_by_group_handles_a_categorical_key() -> None:
    frame = pd.DataFrame(
        {
            "g": pd.Series(["a", "b", "a"], dtype=pd.CategoricalDtype(["a", "b", "unused"])),
            "isFraud": [1, 0, 0],
        }
    )
    rows = {row["value"]: row for row in eda.rate_by_group(frame, "g", "isFraud")}
    assert set(rows) == {"a", "b"}
    assert rows["a"]["n"] == 2


def test_compare_against_pandas_on_no_usable_pairs_says_it_compared_nothing() -> None:
    frame = pd.DataFrame({"a": [1.0, 1.0, 1.0], "b": [1.0, 2.0, 3.0]})
    rho, _ = eda.spearman_matrix(eda.rank_matrix(frame), chunk_rows=2)
    # A constant column has no rank variation, so pandas returns NaN and the pair is dropped.
    assert eda.compare_against_pandas(frame, [("a", "b")], rho, {"a": 0, "b": 1}) == {
        "n_pairs_compared": 0
    }


def test_information_value_of_a_column_with_no_fraud_at_all_still_returns() -> None:
    result = eda.information_value(pd.Series(range(100)), pd.Series([0] * 100))
    assert math.isfinite(result["iv"])


def test_time_consistency_says_so_when_a_window_holds_no_values() -> None:
    empty = pd.Series([None] * 10, dtype="float64")
    result = eda.time_consistency(
        empty, pd.Series([0, 1] * 5), pd.Series(range(10)), pd.Series([0, 1] * 5), seed=42
    )
    assert result["status"] == "all null in one window"


def test_time_consistency_maps_a_late_category_onto_the_early_category_set() -> None:
    # A level the early window never saw must not become a different code in the late window,
    # or the model would score it as whatever level happens to share its position.
    rng = np.random.default_rng(9)
    n = 800
    early_y = pd.Series((rng.random(n) < 0.3).astype(int))
    late_y = pd.Series((rng.random(n) < 0.3).astype(int))
    early_x = pd.Series(
        pd.Categorical(
            np.where(early_y.to_numpy() == 1, "risky", "safe"), categories=["risky", "safe"]
        )
    )
    late_x = pd.Series(
        pd.Categorical(
            np.where(late_y.to_numpy() == 1, "risky", "new"), categories=["new", "risky", "safe"]
        )
    )
    result = eda.time_consistency(early_x, early_y, late_x, late_y, seed=42)
    assert result["status"] == "ok"
    assert result["late_auc"] is not None


# --- a count feature against the label, per split --------------------------------------------------
# Written in stage 4's driver and moved into the package in stage 5, because the graph features are
# counts too and are judged by the same comparison. Tested here for the first time.


def _count_frame() -> tuple[pd.Series, np.ndarray, dict[str, pd.Series]]:
    """A count that is riskier when positive on train, and a label that follows it on train only."""
    rng = np.random.default_rng(config.SEED)
    n = 3_000
    values = pd.Series(rng.integers(0, 3, n).astype("float64"))
    split = np.repeat(["train", "val", "test"], n // 3)
    labels = np.where(
        (split == "train") & (values.to_numpy() > 0),
        rng.random(n) < 0.20,
        rng.random(n) < 0.05,
    ).astype("int64")
    masks = {name: pd.Series(split == name) for name in ("train", "val", "test")}
    return values, labels, masks


def test_zero_against_positive_reports_both_sides_and_their_intervals() -> None:
    values, labels, masks = _count_frame()
    block = eda.zero_against_positive(values, labels, masks["train"].to_numpy(dtype=bool))
    train = masks["train"].to_numpy(dtype=bool)
    assert block["n_positive"] == int((train & (values.to_numpy() > 0)).sum())
    assert block["n_zero"] == int((train & (values.to_numpy() == 0)).sum())
    assert block["n_positive"] + block["n_zero"] == int(train.sum())
    assert block["share_positive"] == pytest.approx(block["n_positive"] / train.sum())
    assert block["n_distinct_train"] == 3
    assert block["fraud_rate_positive"] > block["fraud_rate_zero"]
    assert block["lift"] == pytest.approx(block["fraud_rate_positive"] / block["fraud_rate_zero"])
    for side in ("positive", "zero"):
        interval = block[f"interval_{side}"]
        assert interval["low"] <= block[f"fraud_rate_{side}"] <= interval["high"]
        assert interval["n"] == block[f"n_{side}"]


def test_zero_against_positive_on_an_empty_side_returns_no_rate_and_no_lift() -> None:
    values = pd.Series([1.0, 2.0, 3.0])
    labels = np.array([1, 0, 0], dtype="int64")
    block = eda.zero_against_positive(values, labels, np.ones(3, dtype=bool))
    assert block["n_zero"] == 0
    assert block["fraud_rate_zero"] is None
    assert block["lift"] is None
    assert block["interval_zero"]["n"] == 0
    empty = eda.zero_against_positive(values, labels, np.zeros(3, dtype=bool))
    assert empty["share_positive"] is None


def test_zero_against_positive_by_split_flags_a_train_only_separation_as_unstable() -> None:
    """The case the block exists for: a lift on train that the later windows do not carry."""
    values, labels, masks = _count_frame()
    out = eda.zero_against_positive_by_split(values, labels, masks)
    assert set(out) >= {"train", "val", "test", "definitions"}
    assert out["train"]["fraud_rate_positive"] > out["train"]["fraud_rate_zero"]
    assert out["positive_is_riskier"] is False
    assert out["disjoint_on_every_split"] is False
    assert isinstance(out["direction_stable"], bool)


def test_zero_against_positive_by_split_passes_a_separation_that_holds_everywhere() -> None:
    rng = np.random.default_rng(config.SEED)
    n = 6_000
    values = pd.Series(rng.integers(0, 2, n).astype("float64"))
    labels = np.where(values.to_numpy() > 0, rng.random(n) < 0.30, rng.random(n) < 0.02).astype(
        "int64"
    )
    split = np.repeat(["train", "val", "test"], n // 3)
    masks = {name: pd.Series(split == name) for name in ("train", "val", "test")}
    out = eda.zero_against_positive_by_split(values, labels, masks)
    assert out["positive_is_riskier"] is True
    assert out["direction_stable"] is True
    assert out["disjoint_on_every_split"] is True


def test_zero_against_positive_by_split_with_no_defined_rate_passes_nothing() -> None:
    values = pd.Series([np.nan, np.nan])
    labels = np.array([0, 1], dtype="int64")
    masks = {"train": pd.Series([True, False]), "val": pd.Series([False, True])}
    out = eda.zero_against_positive_by_split(values, labels, masks)
    assert out["direction_stable"] is False
    assert out["disjoint_on_every_split"] is False
    assert out["positive_is_riskier"] is False
