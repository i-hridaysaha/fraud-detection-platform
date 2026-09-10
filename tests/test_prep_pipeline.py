"""The stage 3 code paths that only run against a whole frame, exercised on a small one.

`tests/test_prep_units.py` checks the pieces. This checks the assemblies: the column plan builder,
the sweeps, the three V-reduction strategies and the pipeline end to end. All of it runs on a
generated frame and a generated set of stage 2 facts, so it needs no CSV and it runs in CI.

The stage 2 facts are built here rather than read from `reports/eda/`, which is deliberate. A test
that reads the committed artifacts checks that this frame produced this plan. A test that builds
its own facts checks that the rules do what they say, including on inputs the real data does not
happen to contain: a column that is constant and worthless, one that is constant and useful, one
that is empty and label-linked, one that is empty and is not, and one that inverts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import cleaning, config, encoders, prepare, reduction, transforms

DAY = config.SECONDS_PER_DAY
FEATURES = ("C1", "C2", "V1", "V2", "V3", "D1", "card1", "card4", "DeviceInfo")


def frame(n_days: int = 60, per_day: int = 30) -> pd.DataFrame:
    """A frame with one column of each shape the plan's four rules are meant to catch."""
    rng = np.random.default_rng(config.SEED)
    n = n_days * per_day
    days = np.repeat(np.arange(1, n_days + 1), per_day)
    y = (rng.random(n) < 0.06).astype("int64")

    v1 = rng.normal(size=n)
    data: dict[str, Any] = {
        config.ID_COLUMN: np.arange(2_987_000, 2_987_000 + n, dtype="int64"),
        config.TARGET: y,
        config.TIME_COLUMN: (days * DAY + rng.integers(0, DAY, n)).astype("int64"),
        config.AMOUNT_COLUMN: np.round(rng.lognormal(4.0, 1.0, n), 2),
        "C1": rng.integers(0, 40, n).astype("float64"),
        "C2": rng.integers(0, 40, n).astype("float64"),
        "V1": v1,
        # V2 is a near copy of V1, so a correlation group has something to reduce.
        "V2": v1 + rng.normal(scale=0.01, size=n),
        "V3": rng.normal(size=n),
        "D1": rng.integers(0, 200, n).astype("float64"),
        "card1": rng.integers(1_000, 1_060, n).astype("float64"),
        "card4": pd.Series(rng.choice(["visa", "mastercard", "amex"], size=n), dtype="object"),
        "DeviceInfo": pd.Series(
            rng.choice(["Windows", "iOS Device", "SM-J700M Build/MMB29K", None], size=n),
            dtype="object",
        ),
    }
    out = pd.DataFrame(data)
    out.loc[out.index[rng.random(n) < 0.25], "V3"] = np.nan
    out.loc[out.index[rng.random(n) < 0.30], "D1"] = np.nan
    return out.sort_values(config.TIME_COLUMN, kind="stable").reset_index(drop=True)


def split(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two halves in time order, standing in for train and validation."""
    boundary = int(base[config.TIME_COLUMN].quantile(0.7))
    return (
        base[base[config.TIME_COLUMN] < boundary].copy(),
        base[base[config.TIME_COLUMN] >= boundary].copy(),
    )


def facts(train: pd.DataFrame) -> cleaning.EdaFacts:
    """A stage 2 fact set built to exercise each rule, in the shape EdaFacts reads."""
    columns = list(train.columns)
    numeric = [
        {
            "column": c,
            "top_value_share": 0.999 if c in ("C2", "V1") else 0.4,
            "top_value": 0.0,
        }
        for c in columns
        if c not in ("card4", "DeviceInfo")
    ]
    categorical = [
        {"column": c, "top_value_share": 0.4, "top_value": "x"} for c in ("card4", "DeviceInfo")
    ]
    univariate = {
        "n_train_rows": len(train),
        "numeric": numeric,
        "categorical": categorical,
        # C2 is constant and worthless, V1 is constant and useful. The rescue has to tell them
        # apart, and no other rule may reach either of them first.
        "flags": {"effectively_constant": ["C2", "V1"]},
    }
    missingness = {
        # V3 is nearly empty and label-linked, D1 is nearly empty and is not. The rule keeps the
        # first and drops the second, which is the whole of its content.
        "missing_rate_by_column": [
            {"column": c, "missing_rate": 0.97 if c in ("V3", "D1") else 0.0} for c in columns
        ],
        "against_target": {
            "label_linked_columns": ["V3"],
            "per_column": [
                {"column": "V3", "gap": 0.12, "p_value": 1e-9},
                {"column": "D1", "gap": 0.001, "p_value": 0.4},
            ],
        },
        "identity_join": {"n_rows_with_identity": 0},
        "blocks": {
            "all_blocks": [
                {"columns": ["V3"], "missing_rate": 0.97, "n_columns": 1},
                {"columns": ["V1", "V2"], "missing_rate": 0.0, "n_columns": 2},
            ]
        },
    }
    bivariate = {
        "per_feature": [
            {
                "column": c,
                "iv": {"C2": 0.001, "V1": 0.9, "V2": 0.5, "V3": 0.4}.get(c, 0.3),
                "iv_present_bins": {"C2": 0.001, "V1": 0.9, "V2": 0.5, "V3": 0.4}.get(c, 0.3),
            }
            for c in columns
        ]
    }
    temporal = {
        "time_consistency": {
            "per_feature": [{"column": "C1", "early_auc": 0.6, "late_auc": 0.45}],
            "flagged_columns": ["C1"],
        },
        "psi": {"per_feature": [{"column": c, "psi_val": 0.01, "psi_test": 0.01} for c in columns]},
    }
    correlation = {
        "v_block_reduction": {
            "blocks": [
                {
                    "block_missing_rate": 0.0,
                    "groups": [
                        {
                            "representative": "V1",
                            "n_members": 2,
                            "members": ["V1", "V2"],
                            "min_abs_rho_to_representative": 0.99,
                        }
                    ],
                }
            ]
        },
        "strongest_pairs_outside_v": {
            "pairs": [
                {"a": "C1", "b": "C2", "rho": 0.98, "n_complete_pairs": len(train)},
                {"a": "D1", "b": "card1", "rho": 0.99, "n_complete_pairs": 10},
            ]
        },
    }
    quality = {
        "impossible_values": {"columns_with_negative_values": ["V1", "V2", "V3"]},
        "device_info": {},
        "email_domains": {},
    }
    return cleaning.EdaFacts(
        {
            "univariate": univariate,
            "missingness": missingness,
            "bivariate": bivariate,
            "temporal": temporal,
            "correlation": correlation,
            "quality": quality,
        }
    )


@pytest.fixture(scope="module")
def built() -> tuple[pd.DataFrame, pd.DataFrame, cleaning.EdaFacts, dict[str, Any]]:
    base = frame()
    train, validation = split(base)
    known = facts(train)
    probe = {"C1": {"train_auc": 0.60, "validation_auc": 0.44}}
    plan = cleaning.build_column_plan(known, probe, list(train.columns))
    return train, validation, known, plan


# --- the plan builder ---------------------------------------------------------------------------


def test_each_rule_catches_the_column_it_was_written_for(built: Any) -> None:
    _, _, _, plan = built
    rows = {row["column"]: row for row in plan["columns"]}
    assert rows["C2"]["reason"] == "effectively_constant"
    assert rows["D1"]["reason"] == "missing_and_not_mnar"
    assert rows["C1"]["reason"] == "time_inconsistent"
    assert rows["V2"]["reason"] == "redundant"


def test_a_constant_column_with_signal_is_rescued(built: Any) -> None:
    _, _, _, plan = built
    rows = {row["column"]: row for row in plan["columns"]}
    assert rows["V1"]["decision"] == "keep"
    assert rows["V1"]["measured"]["rescued_from_effectively_constant"] is True
    rescued = {entry["column"] for entry in plan["rescued_from_effectively_constant"]}
    assert rescued == {"V1"}


def test_a_label_linked_column_survives_a_missing_rate_of_0_97(built: Any) -> None:
    _, _, _, plan = built
    rows = {row["column"]: row for row in plan["columns"]}
    assert rows["V3"]["decision"] == "keep"
    assert rows["V3"]["measured"]["kept_despite_missing_rate"] is True
    assert rows["V3"]["measured"]["mnar_evidence"]["missing_rate"] == 0.97


def test_a_column_matching_two_rules_is_counted_once(built: Any) -> None:
    """C2 is both constant and correlated with C1. It is dropped as constant and only as that.

    The redundancy rule then declines to act on the pair rather than recording a second match:
    one of the two is already gone, so there is no cluster left to choose a representative from.
    """
    _, _, _, plan = built
    rows = {row["column"]: row for row in plan["columns"]}
    assert rows["C2"]["reason"] == "effectively_constant"
    pair = next(e for e in plan["pairs_outside_v"] if e["pair"] == ["C1", "C2"])
    assert "already dropped" in pair["note"]
    assert plan["n_dropped_by_reason"]["effectively_constant"] == 1
    listed = [c for reason in cleaning.DROP_REASONS for c in plan["dropped_by_reason"][reason]]
    assert len(listed) == len(set(listed)) == plan["n_dropped"]


def test_a_pair_measured_on_a_fraction_of_rows_is_not_actioned(built: Any) -> None:
    _, _, _, plan = built
    by_pair = {tuple(entry["pair"]): entry for entry in plan["pairs_outside_v"]}
    assert by_pair[("D1", "card1")]["actionable"] is False
    assert "dropped" not in by_pair[("D1", "card1")]
    assert by_pair[("C1", "C2")]["actionable"] is True


def test_a_group_whose_members_were_all_dropped_records_that(built: Any) -> None:
    _, _, known, _ = built
    probe: dict[str, dict[str, Any]] = {
        c: {"train_auc": 0.6, "validation_auc": 0.4} for c in ("V1", "V2")
    }
    known.time_consistency_flagged = ["V1", "V2"]
    plan = cleaning.build_column_plan(known, probe, ["V1", "V2", config.TARGET])
    group = plan["correlation_groups"][0]
    assert group["representative"] is None
    assert "every member was dropped" in group["note"]
    known.time_consistency_flagged = ["C1"]


def test_feature_columns_excludes_the_label_the_id_and_the_clock(built: Any) -> None:
    _, _, _, plan = built
    features = cleaning.feature_columns(plan)
    for column in (config.TARGET, config.ID_COLUMN, config.TIME_COLUMN):
        assert column not in features
    assert config.AMOUNT_COLUMN in features


def test_the_v_reduction_pool_holds_redundancy_drops_as_well_as_survivors(built: Any) -> None:
    _, _, _, plan = built
    assert cleaning.v_reduction_pool(plan) == ["V1", "V2", "V3"]


def test_a_column_stage_2_never_profiled_gets_null_measurements(built: Any) -> None:
    _, _, known, _ = built
    plan = cleaning.build_column_plan(known, {}, ["C1", "brand_new_column"])
    row = next(r for r in plan["columns"] if r["column"] == "brand_new_column")
    assert row["decision"] == "keep"
    assert row["measured"]["missing_rate"] is None
    assert row["measured"]["iv"] is None


# --- transforms over a whole frame --------------------------------------------------------------


def test_d_origin_psi_reports_both_sides_and_the_delta(built: Any) -> None:
    train, validation, _, _ = built
    rows = transforms.d_origin_psi(train, validation, ["D1"])
    assert len(rows) == 1
    row = rows[0]
    assert row["origin_column"] == "D1_origin"
    assert row["psi_delta"] == pytest.approx(row["psi_after"] - row["psi_before"])
    assert row["improves"] == (row["psi_after"] < row["psi_before"])
    assert row["missing_rate_train"] == pytest.approx(float(train["D1"].isna().mean()))


def test_d_origin_psi_skips_a_column_the_frame_does_not_carry(built: Any) -> None:
    train, validation, _, _ = built
    assert transforms.d_origin_psi(train, validation, ["D99"]) == []


def test_clip_cost_touches_fewer_rows_as_the_quantile_rises(built: Any) -> None:
    train, _, _, _ = built
    rows = transforms.clip_cost(
        train, [config.AMOUNT_COLUMN, "C1"], (0.95, 0.99, 0.999), config.TARGET
    )
    touched = [row["n_values_touched"] for row in rows]
    assert touched == sorted(touched, reverse=True)
    for row in rows:
        assert row["share_of_fraud_rows_touched"] is not None
        assert {entry["column"] for entry in row["per_column"]} == {config.AMOUNT_COLUMN, "C1"}


# --- encoder sweeps -------------------------------------------------------------------------------


def test_the_lag_sweep_covers_the_grid_and_reports_both_aucs(built: Any) -> None:
    train, validation, _, _ = built
    rows = encoders.target_encoding_sweep(train, validation, ["card4"], (0, 7), (1.0, 20.0))
    assert len(rows) == 4
    for row in rows:
        assert row["lag_days"] in (0, 7)
        assert row["smoothing"] in (1.0, 20.0)
        assert row["train_auc"] is not None
        assert row["validation_auc"] is not None
        assert row["train_minus_validation"] == pytest.approx(
            row["train_auc"] - row["validation_auc"]
        )
    undefined = {row["lag_days"]: row["n_train_undefined"] for row in rows}
    assert undefined[7] > undefined[0]


def test_the_entity_probe_decomposes_validation_on_entity_overlap(built: Any) -> None:
    train, validation, _, _ = built
    train = train.assign(**{config.ENTITY_ID_COLUMN: train["card1"].astype("string")})
    validation = validation.assign(
        **{config.ENTITY_ID_COLUMN: validation["card1"].astype("string")}
    )
    result = encoders.entity_target_encoding_value(
        train, validation, config.ENTITY_ID_COLUMN, (0, 7), decompose_at_lag=7
    )
    assert result["decomposition"]["at_lag_days"] == 7
    assert result["decomposition"]["n_seen_rows"] + result["decomposition"]["n_unseen_rows"] == len(
        validation
    )
    assert result["n_train_entities"] == train[config.ENTITY_ID_COLUMN].nunique()
    assert [row["lag_days"] for row in result["per_lag"]] == [0, 7]


def test_normalisation_coverage_reports_every_split(built: Any) -> None:
    train, validation, _, _ = built
    rows = encoders.normalisation_coverage(train, {"val": validation})
    by_column = {row["column"]: row for row in rows}
    assert set(by_column) == {"DeviceInfo"}
    entry = by_column["DeviceInfo"]
    assert entry["train"]["n_levels_normalised"] <= entry["train"]["n_levels_raw"]
    split_row = entry["splits"][0]
    assert split_row["split"] == "val"
    assert split_row["unseen_rows_normalised"] <= split_row["unseen_rows_raw"]


def test_the_vocabulary_application_report_partitions_the_rows(built: Any) -> None:
    train, validation, _, _ = built
    fitted = encoders.fit_category_vocabulary(train, ["card4", "DeviceInfo"])
    rows = encoders.vocabulary_application_report(validation, fitted, "val")
    assert {row["column"] for row in rows} == {"card4", "DeviceInfo"}
    for row in rows:
        assert row["split"] == "val"
        assert row["n_known"] + row["n_rare"] + row["n_unseen"] + row["n_missing"] == row["n_rows"]


def test_is_categorical_like_catches_a_string_column_not_only_an_object_one() -> None:
    """The bug this predicate exists for: the normalisers return string columns, not object ones."""
    assert encoders.is_categorical_like(pd.Series(["a", "b"], dtype="string"))
    assert encoders.is_categorical_like(pd.Series(["a", "b"], dtype="object"))
    assert encoders.is_categorical_like(pd.Series(pd.Categorical(["a", "b"])))
    assert not encoders.is_categorical_like(pd.Series([1.0, 2.0]))


# --- the V reduction -------------------------------------------------------------------------------


def blocks(train: pd.DataFrame) -> list[dict[str, Any]]:
    return reduction.v_blocks(
        [
            {"columns": ["V1", "V2", "V3"], "missing_rate": 0.1, "n_columns": 3},
            {"columns": ["C1"], "missing_rate": 0.0, "n_columns": 1},
        ],
        ["V1", "V2", "V3"],
    )


def test_v_blocks_drops_a_block_with_nothing_to_reduce(built: Any) -> None:
    train, _, _, _ = built
    found = blocks(train)
    assert len(found) == 1
    assert found[0]["columns"] == ["V1", "V2", "V3"]
    assert found[0]["block"] == "V1"


def test_block_pca_keeps_fewer_components_than_columns_and_respects_missingness(
    built: Any,
) -> None:
    train, validation, _, _ = built
    fitted = reduction.fit_block_pca(train, blocks(train))
    block = fitted["blocks"]["V1"]
    assert 1 <= block["n_components"] <= 3
    assert block["cumulative_explained_variance"] >= 0.0
    projected = reduction.apply_block_pca(validation, fitted)
    assert projected.shape[0] == len(validation)
    # A row where any block column is null has no complete vector to project.
    incomplete = validation[["V1", "V2", "V3"]].isna().any(axis=1)
    assert projected.loc[incomplete.to_numpy()].isna().all().all()


def test_block_mean_is_one_column_per_block_and_null_where_the_block_is_null(built: Any) -> None:
    train, validation, _, _ = built
    fitted = reduction.fit_block_mean(train, blocks(train))
    out = reduction.apply_block_mean(validation, fitted)
    assert list(out.columns) == ["vblockmean_V1"]
    all_null = validation[["V1", "V2", "V3"]].isna().all(axis=1)
    if all_null.any():
        assert out.loc[all_null.to_numpy(), "vblockmean_V1"].isna().all()


def test_a_reduction_applied_to_a_frame_missing_a_block_column_skips_it(built: Any) -> None:
    train, validation, _, _ = built
    fitted = reduction.fit_block_mean(train, blocks(train))
    assert reduction.apply_block_mean(validation.drop(columns=["V3"]), fitted).empty
    pca = reduction.fit_block_pca(train, blocks(train))
    assert reduction.apply_block_pca(validation.drop(columns=["V3"]), pca).empty


def test_compare_strategies_measures_all_four_and_the_seed_spread(built: Any) -> None:
    train, validation, _, _ = built
    result = reduction.compare_strategies(
        train,
        validation,
        ["V1", "V2", "V3"],
        ["V1", "V3"],
        blocks(train),
        seeds=(config.SEED, 7),
    )
    measured = {row["strategy"]: row for row in result["results"]}
    assert set(measured) == set(reduction.STRATEGIES)
    assert measured["all"]["n_columns"] == 3
    assert measured["representative"]["n_columns"] == 2
    assert measured["block_mean"]["n_columns"] == 1
    for row in result["seed_variance"]["per_strategy"]:
        assert len(row["validation_aucs"]) == 2
        assert row["spread"] == pytest.approx(row["max"] - row["min"])
    assert result["seed_variance"]["largest_spread"] is not None


def test_a_probe_on_no_columns_returns_nothing_rather_than_raising(built: Any) -> None:
    train, validation, _, _ = built
    y = train[config.TARGET].to_numpy()
    result = reduction.probe(train[[]], y, validation[[]], validation[config.TARGET].to_numpy())
    assert result == {"n_columns": 0, "train_auc": None, "validation_auc": None}


# --- the pipeline end to end -----------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["representative", "pca", "block_mean"])
def test_the_pipeline_runs_under_every_v_strategy_and_is_deterministic(
    built: Any, strategy: str
) -> None:
    train, validation, known, plan = built
    fitted = prepare.fit_preparation(
        train,
        plan,
        known,
        d_origin_columns=["D1"],
        v_strategy=strategy,
        lag_days=3,
        smoothing=10.0,
        clip_columns=[config.AMOUNT_COLUMN],
    )
    once = prepare.apply_preparation(validation, fitted, plan)
    twice = prepare.apply_preparation(validation, fitted, plan)
    pd.testing.assert_frame_equal(once, twice)
    assert list(once.columns) == sorted(once.columns)
    assert len(once) == len(validation)
    assert fitted["v_reduction"]["strategy"] == strategy

    pool = cleaning.v_reduction_pool(plan)
    if strategy == "representative":
        assert "V2" not in once.columns
    else:
        assert not set(pool) & set(once.columns)
        prefix = reduction.PCA_PREFIX if strategy == "pca" else reduction.BLOCK_MEAN_PREFIX
        assert any(str(c).startswith(prefix) for c in once.columns)


def test_the_pipeline_derives_what_the_stage_said_it_would(built: Any) -> None:
    train, _, known, plan = built
    fitted = prepare.fit_preparation(
        train,
        plan,
        known,
        d_origin_columns=[],
        v_strategy="representative",
        lag_days=3,
        smoothing=10.0,
    )
    out = prepare.apply_preparation(train, fitted, plan)
    for column in (
        transforms.AMOUNT_LOG_COLUMN,
        transforms.AMOUNT_CENTS_COLUMN,
        transforms.HOUR_POSITION_COLUMN,
        encoders.normalised_name("DeviceInfo"),
        encoders.vocabulary_name("card4"),
    ):
        assert column in out.columns, column
    assert "D1_origin" not in out.columns


def test_the_pipeline_refuses_to_fit_past_the_training_boundary(built: Any) -> None:
    from fraud_platform.data_loader import TrainOnlyError

    train, _, known, plan = built
    beyond = train.copy()
    beyond[config.TIME_COLUMN] = config.TRAIN_END_DT + np.arange(len(beyond), dtype="int64")
    with pytest.raises(TrainOnlyError):
        prepare.fit_preparation(
            beyond,
            plan,
            known,
            d_origin_columns=[],
            v_strategy="representative",
            lag_days=3,
            smoothing=10.0,
        )


def test_a_column_matching_two_rules_records_the_second_under_also_matched(built: Any) -> None:
    """D1 made constant as well as empty: dropped once, with the second match recorded."""
    _, _, known, _ = built
    known.constant_flags = {"D1"}
    known.information_value["D1"]["iv"] = 0.001
    try:
        plan = cleaning.build_column_plan(known, {}, ["D1", config.TARGET])
    finally:
        known.constant_flags = {"C2", "V1"}
        known.information_value["D1"]["iv"] = 0.3
    row = next(r for r in plan["columns"] if r["column"] == "D1")
    assert row["reason"] == "effectively_constant"
    assert row["also_matched"] == ["missing_and_not_mnar"]
    assert plan["n_dropped"] == 1
    assert plan["n_dropped_by_reason"]["missing_and_not_mnar"] == 0


def test_a_rule_naming_a_column_the_frame_does_not_carry_is_ignored(built: Any) -> None:
    """A stage 2 flag list can name a column the plan was not asked about. That is not an error."""
    _, _, known, _ = built
    known.constant_flags = {"C2", "V1", "not_in_this_frame"}
    try:
        plan = cleaning.build_column_plan(
            known, {"also_not_here": {"validation_auc": 0.1}}, ["C1", config.TARGET]
        )
    finally:
        known.constant_flags = {"C2", "V1"}
    assert {row["column"] for row in plan["columns"]} == {"C1", config.TARGET}
    assert plan["n_dropped"] == 0


def test_an_outside_v_pair_with_full_coverage_drops_one_and_keeps_the_other(built: Any) -> None:
    _, _, known, _ = built
    known.constant_flags = set()
    plan = cleaning.build_column_plan(known, {}, ["C1", "C2", config.TARGET])
    known.constant_flags = {"C2", "V1"}
    entry = next(e for e in plan["pairs_outside_v"] if e["pair"] == ["C1", "C2"])
    assert entry["actionable"] is True
    assert entry["representative_kept"] != entry["dropped"]
    assert "note" not in entry
    rows = {row["column"]: row for row in plan["columns"]}
    assert rows[entry["dropped"]]["reason"] == "redundant"
    assert rows[entry["representative_kept"]]["decision"] == "keep"


def test_a_probe_entry_for_a_column_that_survives_is_recorded(built: Any) -> None:
    _, _, known, _ = built
    probe = {"C1": {"train_auc": 0.6, "validation_auc": 0.58}}
    plan = cleaning.build_column_plan(known, probe, ["C1", config.TARGET])
    row = next(r for r in plan["columns"] if r["column"] == "C1")
    assert row["decision"] == "keep"
    assert row["measured"]["survived_time_consistency_on_validation"] is True
    assert row["measured"]["validation_probe"]["validation_auc"] == 0.58


def test_all_columns_lists_everything_stage_2_profiled(built: Any) -> None:
    _, _, known, _ = built
    listed = known.all_columns()
    assert listed == sorted(listed)
    assert "C1" in listed and "DeviceInfo" in listed
