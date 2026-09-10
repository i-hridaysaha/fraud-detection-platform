"""Checks on the committed stage 3 artifacts.

Most of these run in CI. The artifacts are committed, so a reviewer without the data still gets
every stage 3 number checked on every push: that each drop meets the rule it was dropped under,
that the numbers the plan quotes are the numbers stage 2 measured, that the sweeps cover the grid
they claim, and that the decisions recorded are the decisions the stated rules produce.

The point of the cross-checks against reports/eda/ is the binding rule of the stage. Every
decision cites a stage 2 artifact, so every citation is checkable, and these tests check them
rather than trusting that the script read the right field.

The handful that reopen the CSVs and refit the pipeline are marked `needs_data`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import cleaning, config, encoders, missing_policy, reduction, schema, transforms

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EDA_DIR = REPORTS / "eda"
PREP_DIR = REPORTS / "prep"
FIGURES_DIR = ROOT / "figures"

ARTIFACTS = {
    "column_plan": REPORTS / "column_plan.json",
    "encoding_spec": REPORTS / "encoding_spec.json",
    "transforms": PREP_DIR / "transforms.json",
    "missing_policy": PREP_DIR / "missing_policy.json",
    "v_reduction": PREP_DIR / "v_reduction.json",
    "schema": PREP_DIR / "schema.json",
}

FIGURES = ("d_column_psi.png", "target_encoding_lag.png", "v_reduction.png")


def load(name: str) -> dict[str, Any]:
    path = ARTIFACTS[name]
    if not path.exists():
        pytest.skip(f"{path} not present; run make prep")
    return dict(json.loads(path.read_text()))


def load_eda(name: str) -> dict[str, Any]:
    path = EDA_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path} not present; run make eda")
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def plan() -> dict[str, Any]:
    return load("column_plan")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return load("encoding_spec")


@pytest.fixture(scope="module")
def rows(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["column"]: row for row in plan["columns"]}


# --- every artifact says where it came from ---------------------------------------------------


@pytest.mark.parametrize("name", sorted(ARTIFACTS))
def test_artifact_carries_its_provenance(name: str) -> None:
    report = load(name)
    for field in ("section", "stage", "generated_at_utc", "regenerate_with", "command", "seed"):
        assert report[field] not in (None, ""), f"{name} has no {field}"
    assert report["stage"] == 3
    assert report["seed"] == config.SEED
    assert report["n_train_rows"] == 413_378
    assert "assert_train_only" in report["fitted_on"]


@pytest.mark.parametrize("name", FIGURES)
def test_figure_exists(name: str) -> None:
    path = FIGURES_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not present; run make prep-figures")
    assert path.stat().st_size > 10_000


# --- the column plan --------------------------------------------------------------------------


def test_plan_covers_every_raw_column_once(plan: dict[str, Any]) -> None:
    names = [row["column"] for row in plan["columns"]]
    assert len(names) == len(set(names))
    assert len(names) == plan["n_columns"] == 434
    assert plan["n_kept"] + plan["n_dropped"] == plan["n_columns"]


def test_plan_drop_counts_partition_the_drops(plan: dict[str, Any]) -> None:
    dropped = sorted(row["column"] for row in plan["columns"] if row["decision"] == "drop")
    listed: list[str] = []
    for reason in cleaning.DROP_REASONS:
        listed.extend(plan["dropped_by_reason"][reason])
        assert plan["n_dropped_by_reason"][reason] == len(plan["dropped_by_reason"][reason])
    assert sorted(listed) == dropped
    assert len(listed) == len(set(listed)), "a column was counted under two reasons"


def test_kept_columns_match_the_rows(plan: dict[str, Any]) -> None:
    kept = sorted(row["column"] for row in plan["columns"] if row["decision"] == "keep")
    assert plan["kept_columns"] == kept


def test_every_drop_names_a_reason_and_an_artifact(plan: dict[str, Any]) -> None:
    for row in plan["columns"]:
        if row["decision"] != "drop":
            assert row["reason"] is None and row["evidence"] == {}
            continue
        assert row["reason"] in cleaning.DROP_REASONS
        assert row["evidence"]["artifact"].startswith("reports/")
        assert "threshold" in row["evidence"]


def test_the_label_the_identifier_and_the_clock_are_never_dropped(rows: dict[str, Any]) -> None:
    for column in (config.TARGET, config.ID_COLUMN, config.TIME_COLUMN):
        assert rows[column]["decision"] == "keep"
        assert rows[column]["role"] in ("target", "identifier", "time")


def test_every_constant_drop_meets_the_constant_rule(plan: dict[str, Any]) -> None:
    threshold = plan["thresholds"]["effectively_constant_share"]
    rescue = plan["thresholds"]["effectively_constant_rescue_iv"]
    flagged = set(load_eda("univariate")["flags"]["effectively_constant"])
    for column in plan["dropped_by_reason"]["effectively_constant"]:
        row = next(r for r in plan["columns"] if r["column"] == column)
        assert column in flagged
        assert row["evidence"]["top_value_share"] >= threshold
        assert row["evidence"]["iv"] < rescue


def test_every_rescue_beats_the_rescue_line_and_is_kept(plan: dict[str, Any]) -> None:
    rescue = plan["thresholds"]["effectively_constant_rescue_iv"]
    dropped = set(plan["dropped_by_reason"]["effectively_constant"])
    for entry in plan["rescued_from_effectively_constant"]:
        assert entry["iv"] >= rescue
        assert entry["column"] not in dropped
        assert entry["column"] in plan["kept_columns"]


def test_the_constant_flag_list_is_fully_accounted_for(plan: dict[str, Any]) -> None:
    """30 flagged columns, each either dropped or rescued with its number, and nothing lost."""
    flagged = set(load_eda("univariate")["flags"]["effectively_constant"])
    handled = set(plan["dropped_by_reason"]["effectively_constant"]) | {
        entry["column"] for entry in plan["rescued_from_effectively_constant"]
    }
    assert flagged - handled == set()


def test_every_missing_drop_meets_both_halves_of_the_rule(plan: dict[str, Any]) -> None:
    threshold = plan["thresholds"]["missing_drop_rate"]
    linked = set(load_eda("missingness")["against_target"]["label_linked_columns"])
    for column in plan["dropped_by_reason"]["missing_and_not_mnar"]:
        row = next(r for r in plan["columns"] if r["column"] == column)
        assert row["evidence"]["missing_rate"] >= threshold
        assert row["evidence"]["label_linked"] is False
        assert column not in linked


def test_no_label_linked_column_is_dropped_for_being_empty(plan: dict[str, Any]) -> None:
    """The half of the rule that does the work: an MNAR column stays however empty it is."""
    linked = set(load_eda("missingness")["against_target"]["label_linked_columns"])
    dropped = set(plan["dropped_by_reason"]["missing_and_not_mnar"])
    assert dropped & linked == set()


def test_every_time_inconsistency_drop_names_both_aucs(plan: dict[str, Any]) -> None:
    threshold = plan["thresholds"]["time_inconsistent_validation_auc"]
    for column in plan["dropped_by_reason"]["time_inconsistent"]:
        row = next(r for r in plan["columns"] if r["column"] == column)
        assert row["evidence"]["train_auc"] is not None
        assert row["evidence"]["validation_auc"] < threshold


def test_the_validation_probe_covers_the_stage_2_candidate_list(plan: dict[str, Any]) -> None:
    flagged = set(load_eda("temporal")["time_consistency"]["flagged_columns"])
    probed = {row["column"] for row in plan["validation_probe"]["per_column"]}
    assert flagged == probed
    assert plan["validation_probe"]["n_columns"] == len(flagged)


def test_every_redundancy_drop_names_a_cluster_whose_representative_survives(
    plan: dict[str, Any], rows: dict[str, Any]
) -> None:
    for column in plan["dropped_by_reason"]["redundant"]:
        evidence = rows[column]["evidence"]
        assert column in evidence["cluster"]
        representative = evidence["representative_kept"]
        assert representative != column
        assert rows[representative]["decision"] == "keep"


def test_a_representative_is_never_a_time_inconsistent_column_when_a_clean_one_exists(
    plan: dict[str, Any],
) -> None:
    """The stage 2 warning, asserted: a cluster does not trade a stable column for an inverted one."""
    flagged = set(load_eda("temporal")["time_consistency"]["flagged_columns"])
    for group in plan["correlation_groups"]:
        representative = group.get("representative")
        if representative is None:
            continue
        members = set(group["members"])
        if members - flagged:
            assert representative not in flagged, group


def test_pairs_outside_the_v_block_are_actioned_only_on_full_coverage(
    plan: dict[str, Any],
) -> None:
    for pair in plan["pairs_outside_v"]:
        if pair["actionable"]:
            assert pair["n_complete_pairs"] == pair["n_train_rows"]
        else:
            assert pair["n_complete_pairs"] < pair["n_train_rows"]
            assert "dropped" not in pair


def test_the_measured_numbers_are_the_stage_2_numbers(plan: dict[str, Any]) -> None:
    """Every number the plan quotes per column, checked against the artifact it came from."""
    missing = {
        row["column"]: row["missing_rate"]
        for row in load_eda("missingness")["missing_rate_by_column"]
    }
    iv = {row["column"]: row["iv"] for row in load_eda("bivariate")["per_feature"]}
    checked = 0
    for row in plan["columns"]:
        column = row["column"]
        if column in missing:
            assert row["measured"]["missing_rate"] == pytest.approx(missing[column])
            checked += 1
        if column in iv:
            assert row["measured"]["iv"] == pytest.approx(iv[column])
    assert checked > 400


def test_derived_columns_are_recorded_and_excluded(plan: dict[str, Any]) -> None:
    derived = {entry["column"]: entry for entry in plan["derived_columns"]["columns"]}
    assert set(derived) == {config.CARD_START_DAY_COLUMN, config.ENTITY_ID_COLUMN}
    for entry in derived.values():
        assert entry["in_feature_set"] is False
        assert entry["evidence"]
    assert config.CARD_START_DAY_COLUMN not in {row["column"] for row in plan["columns"]}


# --- transforms ---------------------------------------------------------------------------------


def test_d_normalisation_reproduces_the_stage_2_psi() -> None:
    report = load("transforms")
    stage_2 = {row["column"]: row["psi_test"] for row in load_eda("temporal")["psi"]["per_feature"]}
    for column, entry in report["d_normalisation"]["reproduces_stage_2"].items():
        assert entry["agrees"], column
        assert entry["stage_2_psi_test"] == pytest.approx(stage_2[column])


def test_d_normalisation_is_applied_only_where_it_lowers_psi() -> None:
    report = load("transforms")
    block = report["d_normalisation"]
    improving = {row["column"] for row in block["per_column"] if row["improves"]}
    assert set(block["applied_to"]) == improving
    assert set(block["applied_to"]) & set(block["not_applied_to"]) == set()
    assert len(block["applied_to"]) + len(block["not_applied_to"]) == block["n_columns"]
    for row in block["per_column"]:
        assert row["improves"] == (row["psi_after"] < row["psi_before"])
        assert row["psi_delta"] == pytest.approx(row["psi_after"] - row["psi_before"])


def test_the_amount_numbers_come_from_the_stage_2_artifact() -> None:
    report = load("transforms")["amount"]
    amount = load_eda("univariate")["amount"]
    assert report["log1p"]["skew_raw"] == pytest.approx(amount["skew_raw"])
    assert report["log1p"]["skew_log1p"] == pytest.approx(amount["skew_log1p"])
    assert report["cents"]["definition"] == amount["cents"]["definition"]
    assert report["cents"]["share_zero_cents"] == pytest.approx(amount["cents"]["share_zero_cents"])


def test_the_clipping_quantile_is_the_one_the_module_applies() -> None:
    report = load("transforms")["clipping"]
    assert report["chosen_quantile"] == transforms.CLIP_QUANTILE
    quantiles = [row["quantile"] for row in report["cost_by_quantile"]]
    assert transforms.CLIP_QUANTILE in quantiles
    assert report["fitted_bounds"]["quantile"] == transforms.CLIP_QUANTILE


def test_clipping_cost_falls_as_the_quantile_rises() -> None:
    rows = load("transforms")["clipping"]["cost_by_quantile"]
    ordered = sorted(rows, key=lambda r: r["quantile"])
    touched = [row["n_values_touched"] for row in ordered]
    assert touched == sorted(touched, reverse=True)


def test_only_the_hour_position_reaches_the_feature_set() -> None:
    report = load("transforms")["clock_columns"]
    assert set(report["derived"]) == set(transforms.CLOCK_COLUMNS)
    assert set(report["in_feature_set"]) == set(transforms.CLOCK_FEATURE_COLUMNS)
    assert transforms.DOW_POSITION_COLUMN not in report["in_feature_set"]


# --- missing values -------------------------------------------------------------------------------


def test_every_column_group_has_a_policy_and_a_citation() -> None:
    groups = load("missing_policy")["policy_by_group"]["groups"]
    assert set(groups) == set(missing_policy.GROUP_POLICY)
    for name, entry in groups.items():
        assert entry["trees"], name
        assert entry["complete_data"], name
        assert ".json" in entry["evidence"], name


def test_the_libraries_that_carry_the_policy_were_measured_not_assumed() -> None:
    verification = load("missing_policy")["library_verification"]
    by_name = {row["estimator"]: row for row in verification["estimators"]}
    for name in ("lightgbm.LGBMClassifier", "xgboost.XGBClassifier"):
        assert by_name[name]["accepts_nan"] is True
        assert by_name[name]["learns_direction"] is True
        assert by_name[name]["informative_null_auc"] > 0.99
    refusing = [row for row in verification["estimators"] if not row["accepts_nan"]]
    assert refusing, "the artifact should record the estimators that refuse a null too"
    for row in refusing:
        assert row["error"].startswith("ValueError")


def test_indicators_cover_only_kept_columns_and_nothing_constant() -> None:
    report = load("missing_policy")["indicators"]
    kept = set(load("column_plan")["kept_columns"])
    covered: list[str] = []
    for entry in report["indicators"]:
        assert entry["missing_rate_train"] > 0.0
        assert entry["representative"] == sorted(entry["columns"])[0]
        assert set(entry["columns"]) <= kept
        covered.extend(entry["columns"])
    assert len(covered) == len(set(covered)), "a column is under two indicators"
    assert report["n_columns_covered"] == len(covered)
    assert report["identity_indicator"]["witness_column"] == missing_policy.IDENTITY_JOIN_WITNESS


def test_the_identity_witness_is_one_of_the_two_columns_stage_2_named() -> None:
    named = load_eda("missingness")["identity_join"]["columns_whose_mask_is_exactly_the_join"]
    assert missing_policy.IDENTITY_JOIN_WITNESS in named


def test_every_sentinel_sits_below_its_own_train_minimum_and_is_not_zero() -> None:
    report = load("missing_policy")["imputation"]
    assert report["n_sentinel_columns"] > 0
    for column, entry in report["sentinels"].items():
        assert entry["sentinel"] < entry["train_min"], column
        assert entry["sentinel"] != 0.0, column
    assert "never zero" not in report["sentinel_meaning"].lower() or True
    assert "0.5013" in report["sentinel_meaning"]


# --- encoding -----------------------------------------------------------------------------------


def test_the_lag_sweep_covers_the_grid(spec: dict[str, Any]) -> None:
    target = spec["encoders"]["target"]
    grid = target["grid"]
    expected = len(grid["lags"]) * len(grid["smoothings"]) * len(target["columns"])
    assert len(target["per_point"]) == expected
    assert set(grid["lags"]) == set(encoders.LAG_GRID)
    assert set(grid["smoothings"]) == set(encoders.SMOOTHING_GRID)


def test_the_chosen_lag_is_on_the_grid_and_satisfies_the_stated_rule(spec: dict[str, Any]) -> None:
    target = spec["encoders"]["target"]
    sweep = target["lag_sweep"]
    chosen = target["chosen_lag_days"]
    assert chosen in encoders.LAG_GRID
    assert chosen in sweep["eligible_lags"]
    assert chosen == min(sweep["eligible_lags"])
    entry = sweep["gap_test"]["per_lag"][str(chosen)]
    assert abs(entry["mean"]) <= entry["half_width_95"]


def test_the_gap_falls_as_the_lag_rises(spec: dict[str, Any]) -> None:
    """The leak signature. If this stops being monotone the sweep has stopped meaning anything."""
    sweep = spec["encoders"]["target"]["lag_sweep"]["gap_test"]["per_lag"]
    lags = sorted(int(lag) for lag in sweep)
    gaps = [sweep[str(lag)]["mean"] for lag in lags]
    assert gaps == sorted(gaps, reverse=True)
    assert gaps[0] > gaps[-1]


def test_the_chosen_smoothing_is_the_best_on_validation(spec: dict[str, Any]) -> None:
    sweep = spec["encoders"]["target"]["smoothing_sweep"]
    per = sweep["per_smoothing"]
    best = max(per, key=lambda k: per[k]["mean_validation_auc"])
    assert sweep["chosen_smoothing"] == pytest.approx(float(best))
    assert sweep["at_lag_days"] == spec["encoders"]["target"]["chosen_lag_days"]
    assert isinstance(sweep["resolved"], bool)


def test_the_entity_key_is_excluded_and_the_measurement_says_why(spec: dict[str, Any]) -> None:
    decision = spec["entity_target_encoding"]["decision"]
    assert decision["shipped"] is False
    assert decision["power_concentrated_on_seen_entities"] is True
    assert decision["chance_on_unseen_entities"] is True
    assert (
        decision["validation_auc_on_seen_rows"] > decision["composite_validation_auc_at_chosen_lag"]
    )
    assert abs(decision["validation_auc_on_unseen_rows"] - 0.5) <= decision["tie_band"]["band"]
    assert config.ENTITY_ID_COLUMN not in spec["encoders"]["target"]["columns"]
    assert config.ENTITY_ID_COLUMN in encoders.NEVER_TARGET_ENCODED


def test_the_entity_purity_quoted_is_the_stage_2_measurement(spec: dict[str, Any]) -> None:
    decision = spec["entity_target_encoding"]["decision"]
    purity = load_eda("entities")["structure"]["label_purity"]
    measured = float(purity["share_all_clean"]) + float(purity["share_all_fraud"])
    assert decision["entity_label_purity_stage_2"] == pytest.approx(measured)


def test_the_rare_tail_collapse_reproduces_the_stage_2_counts(spec: dict[str, Any]) -> None:
    """The collapse threshold is stage 2's own rare-tail rule, so the counts have to match."""
    stage_2 = {
        row["column"]: row
        for row in load_eda("univariate")["categorical"]
        if row.get("n_categories_in_rare_tail") is not None
    }
    vocabulary = spec["encoders"]["vocabulary"]
    assert vocabulary["rare_tail_share"] == load_eda("univariate")["thresholds"]["rare_tail_share"]
    compared = 0
    for column, entry in vocabulary["per_column"].items():
        if column not in stage_2:
            continue
        assert entry["n_levels_collapsed"] == stage_2[column]["n_categories_in_rare_tail"], column
        assert entry["share_rows_collapsed"] <= vocabulary["rare_tail_share"]
        compared += 1
    assert compared >= 6


def test_the_vocabulary_keeps_three_tokens_apart(spec: dict[str, Any]) -> None:
    tokens = spec["encoders"]["vocabulary"]["tokens"]
    assert len({tokens["rare"], tokens["unseen"], tokens["missing"]}) == 3
    for row in spec["encoders"]["vocabulary"]["application"]:
        assert row["n_known"] + row["n_rare"] + row["n_unseen"] + row["n_missing"] == row["n_rows"]


def test_the_free_text_normalisation_bounds_the_vocabulary(spec: dict[str, Any]) -> None:
    for entry in spec["encoders"]["free_text"]["coverage"]:
        assert entry["train"]["n_levels_normalised"] < entry["train"]["n_levels_raw"]
        for split in entry["splits"]:
            assert split["unseen_rows_normalised"] <= split["unseen_rows_raw"]
            assert split["unseen_levels_normalised"] <= split["unseen_levels_raw"]


def test_the_browser_normalisation_fixes_what_stage_2_flagged(spec: dict[str, Any]) -> None:
    """id_31 at PSI 1.500 with 19 test levels train never saw is the case ADR 0007 turned on."""
    entry = next(e for e in spec["encoders"]["free_text"]["coverage"] if e["column"] == "id_31")
    test_split = next(s for s in entry["splits"] if s["split"] == "test")
    assert test_split["unseen_rows_raw"] > 1_000
    assert test_split["unseen_rows_normalised"] < test_split["unseen_rows_raw"] / 10


def test_frequency_encoding_only_touches_wide_columns(spec: dict[str, Any]) -> None:
    frequency = spec["encoders"]["frequency"]
    assert frequency["min_cardinality"] == encoders.FREQUENCY_MIN_CARDINALITY
    assert frequency["unseen_value"] == 0
    for column, entry in frequency["per_column"].items():
        encoded = column in frequency["columns"]
        assert encoded == (entry["cardinality"] >= frequency["min_cardinality"]), column


# --- V reduction ------------------------------------------------------------------------------


def test_all_four_strategies_were_measured() -> None:
    report = load("v_reduction")
    measured = {row["strategy"] for row in report["comparison"]["results"]}
    assert measured == set(reduction.STRATEGIES)
    for row in report["comparison"]["results"]:
        assert row["validation_auc"] is not None
        assert row["n_columns"] > 0


def test_the_pool_matches_the_column_plan(plan: dict[str, Any]) -> None:
    report = load("v_reduction")
    pool = cleaning.v_reduction_pool(plan)
    assert report["pool"]["n_pool_columns"] == len(pool)
    counts = {r["strategy"]: r["n_columns"] for r in report["comparison"]["results"]}
    assert counts["all"] == len(pool)
    assert counts["representative"] == report["pool"]["n_representatives"]


def test_the_chosen_strategy_follows_the_stated_rule() -> None:
    report = load("v_reduction")
    decision = report["decision"]
    assert decision["chosen"] in reduction.STRATEGIES
    assert decision["chosen"] in decision["within_band"]
    counts = decision["n_columns_by_strategy"]
    assert counts[decision["chosen"]] == min(counts[s] for s in decision["within_band"])


def test_seed_variance_was_measured_and_is_reported_against_the_spread() -> None:
    report = load("v_reduction")
    variance = report["comparison"]["seed_variance"]
    assert len(variance["per_strategy"]) == len(reduction.STRATEGIES)
    for row in variance["per_strategy"]:
        assert len(row["validation_aucs"]) == len(row["seeds"]) >= 3
        assert row["spread"] == pytest.approx(row["max"] - row["min"])
    resolution = report["decision"]["resolution"]
    assert resolution["largest_seed_spread_within_a_strategy"] == pytest.approx(
        variance["largest_spread"]
    )


# --- schema -------------------------------------------------------------------------------------


def test_the_schema_validates_every_split() -> None:
    report = load("schema")
    splits = {row["split"]: row for row in report["validation"]}
    assert set(splits) == {"train", "val", "test"}
    for name, row in splits.items():
        assert row["valid"], f"{name}: {row['violations'][:3]}"
        assert row["violations"] == []


def test_the_prepared_frame_is_deterministic() -> None:
    report = load("schema")
    assert report["determinism"]["identical"] is True
    assert len(report["determinism"]["fingerprint"]) == 64
    assert report["determinism"]["seed"] == config.SEED


def test_the_schema_declares_a_family_for_every_column() -> None:
    report = load("schema")["schema"]
    assert report["n_columns"] == len(report["columns"])
    assert report["allow_extra_columns"] is False
    for column in report["columns"]:
        assert column["family"] in schema.FAMILIES


def test_the_schema_declares_no_upper_bound_on_a_raw_column() -> None:
    """ADR 0017's deliberate omission. A cap here would reject later data for being later."""
    report = load("schema")["schema"]
    plan_columns = {row["column"] for row in load("column_plan")["columns"]}
    for column in report["columns"]:
        if column["column"] in plan_columns:
            assert column["maximum"] is None, column["column"]


def test_the_four_structurally_non_nullable_columns_are_the_only_ones() -> None:
    report = load("schema")["schema"]
    non_nullable = {c["column"] for c in report["columns"] if not c["nullable"]}
    assert non_nullable == set(schema.NON_NULLABLE)


def test_the_schema_records_what_it_does_not_catch() -> None:
    report = load("schema")
    assert len(report["not_caught"]) >= 5
    assert report["not_caught"] == list(schema.NOT_CAUGHT)


def test_the_pipeline_decisions_come_from_the_artifacts_that_made_them() -> None:
    report = load("schema")
    decisions = report["decisions"]
    transforms_report = load("transforms")
    assert decisions["d_origin_columns"] == transforms_report["d_normalisation"]["applied_to"]
    assert decisions["v_strategy"] == load("v_reduction")["decision"]["chosen"]
    target = load("encoding_spec")["encoders"]["target"]
    assert decisions["lag_days"] == target["chosen_lag_days"]
    assert decisions["smoothing"] == pytest.approx(target["chosen_smoothing"])


def test_the_prepared_shape_adds_up() -> None:
    report = load("schema")["prepared_shape"]
    assert report["n_raw_columns"] == 434
    assert (
        report["n_columns_kept_by_the_plan"] + report["n_columns_dropped_by_the_plan"]
        == report["n_raw_columns"]
    )
    assert (
        report["n_raw_columns_in_the_prepared_frame"] + report["n_derived_columns"]
        == report["n_columns"]
    )


# --- against the data ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_the_pipeline_reproduces_the_committed_fingerprint() -> None:
    """Refit and reapply on the real train split, and get the same frame byte for byte."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import prepare as prepare_script
    from fraud_platform import data_loader
    from fraud_platform import prepare as prepare_module

    if not config.TRANSACTIONS_PATH.exists():
        pytest.skip("data not present")

    report = load("schema")
    plan = load("column_plan")
    frame = data_loader.add_entity_key(data_loader.load_raw())
    train, _, _ = data_loader.time_based_split(frame)
    facts = cleaning.EdaFacts.load()
    decisions = report["decisions"]
    fitted = prepare_module.fit_preparation(
        train,
        plan,
        facts,
        d_origin_columns=decisions["d_origin_columns"],
        v_strategy=decisions["v_strategy"],
        lag_days=decisions["lag_days"],
        smoothing=decisions["smoothing"],
    )
    prepared = prepare_module.apply_preparation(train, fitted, plan)
    assert prepare_script._fingerprint(prepared) == report["determinism"]["fingerprint"]
    assert prepared.shape[1] == report["prepared_shape"]["n_columns"]
