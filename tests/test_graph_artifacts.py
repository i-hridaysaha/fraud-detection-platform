"""Internal consistency of reports/graph_summary.json, checked without the data.

The artifact is committed, so these run on a clean checkout and on every push. Three kinds of
check, in the stage 2 to stage 4 tradition:

- **The module is the source.** The artifact lists exactly the features the module computes, the
  candidate set it declares, the switch it defaults to, and the link columns it builds on.
- **Numbers quoted from earlier stages are the earlier stages' numbers.** The lag is stage 3's
  from `reports/encoding_spec.json`; the label count is stage 1's train fraud count from
  `reports/split_summary.json`; the link column cardinalities and null rates are stage 2's from
  `reports/eda/`.
- **Arithmetic inside the artifact adds up.** Components partition the train split, the giant
  component is the largest, a decile table partitions its split, a Wilson interval contains its
  point, and the decision block is the rule applied to the per-feature flags.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import config, encoders, graph_features

REPORTS = config.REPORTS_DIR
TRAIN_ROWS = 413_378


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def summary() -> dict[str, Any]:
    return load(REPORTS / "graph_summary.json")


@pytest.fixture(scope="module")
def encoding_spec() -> dict[str, Any]:
    return load(REPORTS / "encoding_spec.json")


@pytest.fixture(scope="module")
def split_summary() -> dict[str, Any]:
    return load(REPORTS / "split_summary.json")


@pytest.fixture(scope="module")
def univariate() -> dict[str, Any]:
    return load(REPORTS / "eda" / "univariate.json")


@pytest.fixture(scope="module")
def missingness() -> dict[str, Any]:
    return load(REPORTS / "eda" / "missingness.json")


# --- the envelope -------------------------------------------------------------------------------


def test_the_artifact_says_what_made_it(summary: dict[str, Any]) -> None:
    assert summary["stage"] == 5
    assert "scripts/graph.py" in summary["regenerate_with"]
    assert summary["n_train_rows"] == TRAIN_ROWS
    assert summary["seed"] == config.SEED
    assert summary["git_commit"]
    assert summary["environment"]["pandas"]


def test_the_train_row_count_is_stage_1s(
    summary: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    assert summary["n_train_rows"] == int(split_summary["splits"]["train"]["n_rows"])


# --- the module is the source -------------------------------------------------------------------


def test_the_config_block_is_the_module_default(summary: dict[str, Any]) -> None:
    recorded = summary["config"]
    assert recorded["enabled"] is graph_features.DEFAULT_ENABLED
    assert recorded["link_columns"] == list(graph_features.LINK_COLUMNS)
    assert recorded["features"] == list(graph_features.CANDIDATE_FEATURES)
    assert recorded["all_features"] == list(graph_features.ALL_FEATURES)
    assert recorded["built_here_with"]["features"] == list(graph_features.ALL_FEATURES)
    assert recorded["built_here_with"]["enabled"] is True


def test_the_artifact_lists_exactly_the_features_the_module_computes(
    summary: dict[str, Any],
) -> None:
    listed = [row["feature"] for row in summary["per_feature"]]
    assert listed == [spec.name for spec in graph_features.catalogue()]
    for row, spec in zip(summary["per_feature"], graph_features.catalogue(), strict=True):
        assert row["label_derived"] == spec.label_derived
        assert row["in_candidate_set"] == (spec.name in graph_features.CANDIDATE_FEATURES)
        assert row["definition"] == spec.definition


def test_the_decision_block_is_the_rule_applied_to_the_flags(summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    assert decision["default_enabled"] is graph_features.DEFAULT_ENABLED
    assert decision["candidate_features"] == list(graph_features.CANDIDATE_FEATURES)
    passing = [
        row["feature"] for row in summary["per_feature"] if row["stage_4_ship_rule"]["passes"]
    ]
    assert decision["features_passing_the_stage_4_rule"] == passing
    # The switch is off exactly because nothing passes. If something passes one day, this fails
    # and the default is the thing to revisit.
    assert bool(passing) == decision["default_enabled"]
    assert decision["provided_by_stage_4"] == graph_features.PROVIDED_BY_STAGE_4
    assert set(decision["not_reachable"]) == set(graph_features.not_reachable())


def test_the_ship_rule_flags_are_the_comparison_they_claim(summary: dict[str, Any]) -> None:
    for row in summary["per_feature"]:
        comparison = row["zero_against_positive_by_split"]
        rule = row["stage_4_ship_rule"]
        assert rule["direction_stable"] == comparison["direction_stable"]
        assert rule["disjoint_on_every_split"] == comparison["disjoint_on_every_split"]
        assert rule["passes"] == (
            comparison["direction_stable"] and comparison["disjoint_on_every_split"]
        )


# --- numbers quoted from earlier stages ---------------------------------------------------------


def test_the_lag_is_stage_3s(summary: dict[str, Any], encoding_spec: dict[str, Any]) -> None:
    chosen = int(encoding_spec["encoders"]["target"]["chosen_lag_days"])
    assert summary["lag"]["lag_days"] == chosen
    assert summary["component_fraud_rate"]["labels"]["lag_days"] == chosen
    assert "encoding_spec.json" in summary["lag"]["source"]


def test_the_label_source_is_the_train_split(
    summary: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    labels = summary["component_fraud_rate"]["labels"]
    train = split_summary["splits"]["train"]
    assert labels["n_rows"] == int(train["n_rows"])
    assert labels["n_fraud"] == int(train["n_fraud"])
    assert labels["max_ts"] < config.TRAIN_END_DT
    assert summary["component_fraud_rate"]["n_labels_attached_by_end_of_stream"] == labels["n_rows"]


def test_the_link_column_profile_is_stage_2s(
    summary: dict[str, Any],
    univariate: dict[str, Any],
    missingness: dict[str, Any],
    encoding_spec: dict[str, Any],
) -> None:
    """Cardinality on train and null rate on train were measured in stage 2 and stage 3."""
    profile = summary["link_columns"]
    categorical = {row["column"]: row for row in univariate["categorical"]}
    for column in ("P_emaildomain", "R_emaildomain"):
        assert profile[column]["n_values"] == int(categorical[column]["cardinality"]), column
        # Stage 2 reports the top value's share of present rows; the graph reports its share of
        # all rows, because a null row is a row the value does not link.
        present = 1.0 - float(categorical[column]["missing_rate"])
        assert profile[column]["top_value_share_of_rows"] == pytest.approx(
            float(categorical[column]["top_value_share"]) * present, abs=1e-9
        ), column
        assert profile[column]["top_value"] == categorical[column]["top_value"]
    missing = {row["column"]: row for row in missingness["missing_rate_by_column"]}
    for column in ("addr1", "P_emaildomain", "R_emaildomain", "dist1", "DeviceInfo"):
        recorded = column if column != "DeviceInfo" else graph_features.DEVICE_COLUMN
        assert profile[recorded]["null_rate"] == pytest.approx(
            float(missing[column]["missing_rate"]), abs=1e-9
        ), column
    assert profile["card1"]["null_rate"] == 0.0
    free_text = next(
        row
        for row in encoding_spec["encoders"]["free_text"]["coverage"]
        if row["column"] == "DeviceInfo"
    )
    assert profile[graph_features.DEVICE_COLUMN]["n_values"] == int(
        free_text["train"]["n_levels_normalised"]
    )


def test_the_hub_sweep_starts_at_stage_3s_rare_tail_line(summary: dict[str, Any]) -> None:
    points = summary["hub_sweep"]["points"]
    assert points[0]["hub_share"] == encoders.RARE_TAIL_SHARE
    assert points[0]["is_stage_3_rare_tail_line"] is True
    assert all(point["is_stage_3_rare_tail_line"] is False for point in points[1:])


# --- arithmetic inside the artifact -------------------------------------------------------------


def test_the_end_of_train_components_partition_the_split(summary: dict[str, Any]) -> None:
    graph = summary["end_of_train_graph"]
    assert graph["n_transactions"] == TRAIN_ROWS
    giant = graph["giant_component"]
    assert giant["n_transactions"] == graph["largest_ten_in_transactions"][0]
    assert giant["share_of_transactions"] == pytest.approx(giant["n_transactions"] / TRAIN_ROWS)
    assert giant["n_nodes"] >= giant["n_transactions"]
    assert graph["size_in_transactions"]["n"] == graph["n_components"]
    assert graph["size_in_transactions"]["max"] == giant["n_transactions"]
    assert graph["share_transactions_in_a_component_of_size_1"] == pytest.approx(
        graph["n_transactions_in_a_component_of_size_1"] / TRAIN_ROWS
    )
    assert sum(graph["n_value_nodes_by_column"].values()) == graph["n_value_nodes_present"]
    for column, block in graph["degree_by_column"].items():
        assert block["n"] == graph["n_value_nodes_by_column"][column]


def test_the_giant_component_is_the_finding(summary: dict[str, Any]) -> None:
    """The number the stage's decision rests on. If it stops holding, the default is wrong."""
    graph = summary["end_of_train_graph"]
    assert graph["giant_component"]["share_of_transactions"] > 0.99
    assert graph["n_components"] < 100


def test_the_hub_sweep_fragments_monotonically(summary: dict[str, Any]) -> None:
    points = summary["hub_sweep"]["points"]
    shares = [point["hub_share"] for point in points]
    assert shares == sorted(shares, reverse=True)
    giant = [point["giant_component_share_of_transactions"] for point in points]
    assert giant == sorted(giant, reverse=True)
    lone = [point["share_transactions_in_a_component_of_size_1"] for point in points]
    assert lone == sorted(lone)
    for point in points:
        for column, block in point["hubs_by_column"].items():
            assert 0 <= block["n_hubs"] <= block["n_values"], column
            assert 0.0 <= block["share_rows_through_hubs"] <= 1.0, column


def test_the_link_variants_cover_every_column(summary: dict[str, Any]) -> None:
    variants = summary["link_variants"]
    assert set(variants["leave_one_out"]) == set(graph_features.LINK_COLUMNS)
    assert set(variants["single_column"]) == set(graph_features.LINK_COLUMNS)
    assert set(variants["pairs_with_the_anchor"]) == set(graph_features.LINK_COLUMNS) - {
        graph_features.ANCHOR_COLUMN
    }
    for column, block in variants["leave_one_out"].items():
        assert column not in block["link_columns"]
        assert len(block["link_columns"]) == len(graph_features.LINK_COLUMNS) - 1
    for column, block in variants["single_column"].items():
        assert block["link_columns"] == [column]
        assert block["giant_component_share_of_transactions"] == pytest.approx(
            summary["link_columns"][column]["top_value_share_of_rows"], abs=1e-9
        ), "a single-column graph's giant component is that column's heaviest value"


def test_every_decile_table_partitions_its_split(
    summary: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    rows_by_split = {
        split: int(split_summary["splits"][split]["n_rows"]) for split in config.SPLIT_NAMES
    }
    for row in summary["per_feature"]:
        for split, table in row["fraud_rate_by_decile"].items():
            assert sum(bin_["n"] for bin_ in table) == rows_by_split[split], (
                row["feature"],
                split,
            )
            for bin_ in table:
                interval = bin_["interval"]
                assert interval["low"] - 1e-12 <= bin_["fraud_rate"] <= interval["high"] + 1e-12, (
                    row["feature"]
                )


def test_every_zero_against_positive_block_adds_up(summary: dict[str, Any]) -> None:
    for row in summary["per_feature"]:
        comparison = row["zero_against_positive_by_split"]
        for split in config.SPLIT_NAMES:
            block = comparison[split]
            n_defined = block["n_positive"] + block["n_zero"]
            null_rate = row["null_rate_by_split"][split]
            assert n_defined <= summary["n_train_rows"] if split == "train" else True
            if null_rate == 0.0 and split == "train":
                assert n_defined == TRAIN_ROWS, row["feature"]
            for side in ("positive", "zero"):
                interval = block[f"interval_{side}"]
                rate = block[f"fraud_rate_{side}"]
                if rate is not None:
                    # A Wilson bound on zero successes lands a rounding error above zero.
                    assert interval["low"] - 1e-12 <= rate <= interval["high"] + 1e-12, (
                        row["feature"],
                        split,
                    )


def test_the_clock_test_names_the_clocks(summary: dict[str, Any]) -> None:
    """The component features are counts of the stream before the row and the artifact says so."""
    by_name = {row["feature"]: row for row in summary["per_feature"]}
    for name in (graph_features.COMPONENT_SIZE, graph_features.COMPONENT_N_LABELLED):
        assert by_name[name]["clock_test"]["spearman_with_transaction_dt_train"] > 0.9, name
        assert by_name[name]["drift"]["test"] > 1.0, name
    for name in graph_features.CANDIDATE_FEATURES:
        assert by_name[name]["clock_test"]["spearman_with_transaction_dt_train"] < 0.9, name


def test_the_component_rate_never_exceeds_the_train_rate(
    summary: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    """One component, so the rate is the lagged training rate and cannot be larger than it."""
    train_rate = float(split_summary["splits"]["train"]["fraud_rate"])
    block = summary["component_fraud_rate"]
    assert block[graph_features.COMPONENT_FRAUD_RATE]["max"] <= train_rate + 1e-3
    assert block["rate_against_others_rate"]["max_abs_difference"] < 0.01


def test_the_decomposition_covers_its_split(summary: dict[str, Any]) -> None:
    block = summary["component_fraud_rate"]
    for name in graph_features.LABEL_FEATURES:
        for split in ("val", "test"):
            decomposition = block[name][f"{split}_decomposition"]
            total = block[name]["by_split"][split]["n_scored"]
            assert (
                decomposition["auc_on_seen_rows"]["n_scored"]
                + decomposition["auc_on_unseen_rows"]["n_scored"]
                == total
            ), (name, split)
            assert 0.0 < decomposition["share_rows_on_a_seen_entity"] < 1.0


def test_the_others_rate_is_the_entitys_own_labels_reversed(summary: dict[str, Any]) -> None:
    """The ADR 0025 finding: on seen entities the subtraction reads the entity's labels, sign
    flipped; on unseen entities there is nothing to read."""
    block = summary["component_fraud_rate"][graph_features.COMPONENT_FRAUD_RATE_OTHERS]
    seen = block["test_decomposition"]["auc_on_seen_rows"]["auc"]
    unseen = block["test_decomposition"]["auc_on_unseen_rows"]["auc"]
    assert seen < 0.1
    assert 0.4 < unseen < 0.6


def test_the_transductive_block_quantifies_the_leak(summary: dict[str, Any]) -> None:
    graphs = {entry["graph"]: entry for entry in summary["transductive_leak"]["graphs"]}
    six = graphs["six_column_graph"]
    assert six["hub_share"] is None
    assert six["component_size"]["share_rows_where_transductive_exceeds_causal"] == 1.0
    for split in config.SPLIT_NAMES:
        auc = six["component_fraud_rate"]["auc_by_split"][split]
        # One component: the leave-one-out rate is the row's own label, reversed. Train is not
        # exactly one component (23, of which 17 are lone transactions), so its AUC is a
        # rounding error above zero rather than zero.
        assert auc["transductive_leave_one_out"]["auc"] < 1e-4, split
        assert auc["transductive_leave_one_out"]["n_scored"] > 0
        # And the final component size is one number for every row bar the 23 stragglers.
        assert six["component_size"]["auc_by_split"][split]["transductive"]["auc"] == pytest.approx(
            0.5, abs=1e-3
        )
    final = six["final_graph_over_every_row"]
    assert final["n_transactions"] == 590_540
    assert final["giant_component"]["share_of_transactions"] == pytest.approx(
        final["giant_component"]["n_transactions"] / final["n_transactions"]
    )
    assert final["giant_component"]["share_of_transactions"] > 0.99
    fragmented = graphs["hub_excluded_graph"]
    assert fragmented["hub_share"] == summary["hub_sweep"]["points"][-1]["hub_share"]
    assert fragmented["final_graph_over_every_row"]["n_transactions"] == 590_540
    assert fragmented["final_graph_over_every_row"]["n_components"] > final["n_components"]
    for split in ("val", "test"):
        auc = fragmented["component_fraud_rate"]["auc_by_split"][split]
        assert auc["transductive_leave_one_out"]["auc"] > auc["causal"]["auc"], split


def test_the_hub_excluded_rates_cover_the_sweep(summary: dict[str, Any]) -> None:
    sweep = [point["hub_share"] for point in summary["hub_sweep"]["points"]]
    points = summary["hub_excluded_rates"]["points"]
    assert [point["hub_share"] for point in points] == sweep
    for point in points:
        for name in (graph_features.COMPONENT_SIZE, *graph_features.LABEL_FEATURES):
            block = point[name]
            assert set(block["by_split"]) == set(config.SPLIT_NAMES), name
            assert set(block["drift"]) == {"val", "test"}, name
            for split in ("val", "test"):
                decomposition = block[f"{split}_decomposition"]
                assert (
                    decomposition["auc_on_seen_rows"]["n_scored"]
                    + decomposition["auc_on_unseen_rows"]["n_scored"]
                    == block["by_split"][split]["n_scored"]
                ), (name, split)
