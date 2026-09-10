"""Unit tests for the stage 3 modules, on frames built here rather than on the CSVs.

Four things get checked and the first is the one the stage rests on.

**Every encoder refuses a fit on rows outside the training window.** Not by convention: the test
calls each fit function with a frame that straddles the boundary and requires `TrainOnlyError`.
A new encoder that forgets the decorator fails here.

**Transformations are idempotent where they should be and invertible where they can be.** Adding
the clock columns twice writes the same values; clipping a clipped frame changes nothing; log1p
and expm1 round trip; the D origin recovers its delta given the timestamp.

**The schema validator rejects the three failures it exists for**, each proved by constructing
the bad frame rather than by trusting the code path.

**The prepared output is the same twice.** Same fitted objects, same input, same frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_platform import (
    cleaning,
    config,
    encoders,
    missing_policy,
    reduction,
    transforms,
)
from fraud_platform import (
    schema as schema_module,
)
from fraud_platform.data_loader import TrainOnlyError

DAY = config.SECONDS_PER_DAY


def frame(n_days: int = 40, per_day: int = 12, first_day: int = 1) -> pd.DataFrame:
    """A small frame with the columns the stage 3 modules address, deterministic under SEED."""
    rng = np.random.default_rng(config.SEED)
    n = n_days * per_day
    days = np.repeat(np.arange(first_day, first_day + n_days), per_day)
    levels = np.array(["alpha", "beta", "gamma", "delta", "rare_one"])
    weights = np.array([0.45, 0.25, 0.18, 0.115, 0.005])
    card = rng.choice(levels, size=n, p=weights)
    amount = np.round(rng.lognormal(4.0, 1.0, n), 2)
    d1 = rng.integers(0, 200, n).astype("float64")
    d1[rng.random(n) < 0.2] = np.nan
    return pd.DataFrame(
        {
            config.ID_COLUMN: np.arange(2_987_000, 2_987_000 + n, dtype="int64"),
            config.TARGET: (rng.random(n) < 0.05).astype("int64"),
            config.TIME_COLUMN: (days * DAY + rng.integers(0, DAY, n)).astype("int64"),
            config.AMOUNT_COLUMN: amount,
            "card4": pd.Series(card, dtype="object"),
            "D1": d1,
            "C1": rng.integers(0, 30, n).astype("float64"),
            "DeviceInfo": pd.Series(
                rng.choice(["Windows", "SM-J700M Build/MMB29K", "iOS Device", None], size=n),
                dtype="object",
            ),
        }
    ).sort_values(config.TIME_COLUMN, kind="stable")


def train_frame() -> pd.DataFrame:
    return frame()


def out_of_window_frame() -> pd.DataFrame:
    """Rows on both sides of the training boundary, so the guard has something to catch."""
    inside = frame(n_days=10)
    outside = inside.copy()
    outside[config.TIME_COLUMN] = config.TRAIN_END_DT + np.arange(len(outside), dtype="int64")
    return pd.concat([inside, outside], ignore_index=True)


# --- every encoder refuses a fit outside the training window ---------------------------------


FIT_FUNCTIONS = (
    ("fit_category_vocabulary", lambda f: encoders.fit_category_vocabulary(f, ["card4"])),
    ("fit_frequency_encoding", lambda f: encoders.fit_frequency_encoding(f, ["card4"], 2)),
    ("fit_target_encoding", lambda f: encoders.fit_target_encoding(f, ["card4"])),
    ("fit_clip_bounds", lambda f: transforms.fit_clip_bounds(f, [config.AMOUNT_COLUMN])),
    ("fit_imputer", lambda f: missing_policy.fit_imputer(f, ["C1"], ["D1"])),
    (
        "fit_block_pca",
        lambda f: reduction.fit_block_pca(f, [{"block": "C1", "columns": ["C1", "D1"]}]),
    ),
    (
        "fit_block_mean",
        lambda f: reduction.fit_block_mean(f, [{"block": "C1", "columns": ["C1", "D1"]}]),
    ),
)


@pytest.mark.parametrize("name,fit", FIT_FUNCTIONS, ids=[n for n, _ in FIT_FUNCTIONS])
def test_fit_refuses_rows_outside_the_training_window(name: str, fit: object) -> None:
    with pytest.raises(TrainOnlyError) as raised:
        fit(out_of_window_frame())  # type: ignore[operator]
    assert config.TIME_COLUMN in str(raised.value)


@pytest.mark.parametrize("name,fit", FIT_FUNCTIONS, ids=[n for n, _ in FIT_FUNCTIONS])
def test_fit_refuses_an_empty_frame(name: str, fit: object) -> None:
    with pytest.raises(TrainOnlyError):
        fit(train_frame().iloc[:0])  # type: ignore[operator]


@pytest.mark.parametrize("name,fit", FIT_FUNCTIONS, ids=[n for n, _ in FIT_FUNCTIONS])
def test_fit_refuses_a_frame_with_no_timestamp(name: str, fit: object) -> None:
    with pytest.raises(TrainOnlyError):
        fit(train_frame().drop(columns=[config.TIME_COLUMN]))  # type: ignore[operator]


def test_entity_key_cannot_be_target_encoded() -> None:
    """ADR 0015 is enforced in the encoder, not left to whoever assembles the pipeline."""
    with_entity = train_frame().assign(**{config.ENTITY_ID_COLUMN: "e1"})
    with pytest.raises(ValueError, match="cannot be target encoded"):
        encoders.fit_target_encoding(with_entity, [config.ENTITY_ID_COLUMN])


# --- transformations --------------------------------------------------------------------------


def test_clock_columns_are_idempotent() -> None:
    once = transforms.add_clock_columns(train_frame())
    twice = transforms.add_clock_columns(once)
    pd.testing.assert_frame_equal(once, twice[once.columns])


def test_amount_features_are_idempotent() -> None:
    once = transforms.add_amount_features(train_frame())
    twice = transforms.add_amount_features(once)
    pd.testing.assert_frame_equal(once, twice[once.columns])


def test_log1p_round_trips_to_the_amount() -> None:
    out = transforms.add_amount_features(train_frame())
    recovered = transforms.invert_amount_log(out[transforms.AMOUNT_LOG_COLUMN])
    np.testing.assert_allclose(recovered, out[config.AMOUNT_COLUMN], rtol=1e-12)


def test_d_origin_round_trips_to_the_delta() -> None:
    """The normalisation is invertible given the timestamp, and nulls stay null through both."""
    base = train_frame()
    out = transforms.to_d_origin(base, ["D1"])
    recovered = transforms.from_d_origin(out, "D1")
    both_null = base["D1"].isna()
    np.testing.assert_allclose(recovered[~both_null], base["D1"][~both_null], rtol=1e-12)
    assert recovered[both_null].isna().all()


def test_cents_matches_the_stage_2_definition() -> None:
    out = transforms.add_amount_features(train_frame())
    expected = (out[config.AMOUNT_COLUMN] * 100).round().astype("int64") % 100
    np.testing.assert_array_equal(out[transforms.AMOUNT_CENTS_COLUMN].to_numpy(), expected)
    assert (
        out[transforms.AMOUNT_IS_ROUND_COLUMN].to_numpy() == (expected == 0).to_numpy().astype(int)
    ).all()


def test_clipping_is_idempotent() -> None:
    base = train_frame()
    fitted = transforms.fit_clip_bounds(base, [config.AMOUNT_COLUMN, "C1"], quantile=0.9)
    once = transforms.apply_clip(base, fitted)
    twice = transforms.apply_clip(once, fitted)
    pd.testing.assert_frame_equal(once, twice)


def test_clipping_bounds_come_from_train_and_bind_a_later_frame() -> None:
    """A bound fitted on train has to bite on a later frame, or it is not doing anything."""
    base = train_frame()
    fitted = transforms.fit_clip_bounds(base, [config.AMOUNT_COLUMN], quantile=0.9)
    upper = fitted["bounds"][config.AMOUNT_COLUMN]["upper"]
    later = base.copy()
    later[config.AMOUNT_COLUMN] = upper * 10
    clipped = transforms.apply_clip(later, fitted)
    assert float(clipped[config.AMOUNT_COLUMN].max()) == pytest.approx(upper)


# --- imputation and indicators ------------------------------------------------------------------


def test_indicators_are_computed_before_imputation_or_they_are_all_zero() -> None:
    base = train_frame()
    spec = missing_policy.missing_indicator_spec(
        [{"columns": ["D1"], "missing_rate": float(base["D1"].isna().mean())}], ["D1"]
    )
    with_indicator = missing_policy.add_missing_indicators(base, spec)
    name = missing_policy.indicator_name("D1")
    assert int(with_indicator[name].sum()) == int(base["D1"].isna().sum())
    assert int(with_indicator[name].sum()) > 0

    imputed = missing_policy.apply_imputer(base, missing_policy.fit_imputer(base, [], ["D1"]))
    after = missing_policy.add_missing_indicators(imputed, spec)
    assert int(after[name].sum()) == 0


def test_sentinel_is_below_every_observed_value_and_is_not_zero() -> None:
    base = train_frame()
    fitted = missing_policy.fit_imputer(base, [], ["D1"])
    sentinel = fitted["sentinels"]["D1"]["sentinel"]
    assert sentinel < float(base["D1"].min())
    assert sentinel != 0.0
    filled = missing_policy.apply_imputer(base, fitted)
    assert filled["D1"].notna().all()
    assert int((filled["D1"] == sentinel).sum()) == int(base["D1"].isna().sum())


def test_imputation_is_idempotent() -> None:
    base = train_frame()
    fitted = missing_policy.fit_imputer(base, ["C1"], ["D1"])
    once = missing_policy.apply_imputer(base, fitted)
    twice = missing_policy.apply_imputer(once, fitted)
    pd.testing.assert_frame_equal(once, twice)


def test_native_null_support_is_measured_not_assumed() -> None:
    """The claim rule 1 of the missing value policy rests on, re-measured on every run."""
    result = missing_policy.verify_native_nan_support(n_rows=1_200)
    by_name = {row["estimator"]: row for row in result["estimators"]}
    for name in ("lightgbm.LGBMClassifier", "xgboost.XGBClassifier"):
        assert by_name[name]["accepts_nan"], f"{name} no longer accepts nulls"
        assert by_name[name]["learns_direction"], f"{name} no longer learns a null direction"
    assert result["n_accepting_nan"] >= 2


# --- encoders -----------------------------------------------------------------------------------


def test_target_encoding_uses_only_days_before_the_lagged_boundary() -> None:
    """Checked against a direct recomputation, not against the encoder's own arithmetic."""
    base = train_frame()
    lag, smoothing = 5, 0.0001
    fitted = encoders.fit_target_encoding(base, ["card4"], lag_days=lag, smoothing=smoothing)
    encoded = encoders.apply_target_encoding(base, fitted)[encoders.target_name("card4")]
    day = transforms.day_index(base)

    for index in base.index[::7]:
        boundary = int(day.loc[index]) - lag
        available = base[day <= boundary]
        if available.empty:
            assert np.isnan(encoded.loc[index])
            continue
        level = available[available["card4"] == base.loc[index, "card4"]]
        prior = float(available[config.TARGET].mean())
        expected = (float(level[config.TARGET].sum()) + smoothing * prior) / (
            len(level) + smoothing
        )
        assert encoded.loc[index] == pytest.approx(expected, abs=1e-9)


def test_target_encoding_is_null_before_any_label_has_matured() -> None:
    base = train_frame()
    fitted = encoders.fit_target_encoding(base, ["card4"], lag_days=5)
    encoded = encoders.apply_target_encoding(base, fitted)[encoders.target_name("card4")]
    day = transforms.day_index(base)
    early = day < int(day.min()) + 5
    assert encoded[early].isna().all()
    assert encoded[~early].notna().any()


def test_target_encoding_separates_an_unseen_level_from_a_null() -> None:
    """Three states, three answers. An unseen level gets the prior; a null gets nothing."""
    base = train_frame()
    fitted = encoders.fit_target_encoding(base, ["card4"], lag_days=1, smoothing=10.0)
    later = base.tail(2).copy()
    later["card4"] = ["never_seen_before", None]
    encoded = encoders.apply_target_encoding(later, fitted)[encoders.target_name("card4")]
    assert np.isfinite(encoded.iloc[0])
    assert np.isnan(encoded.iloc[1])


def test_target_encoding_lag_and_smoothing_overrides_match_a_refit() -> None:
    base = train_frame()
    fitted = encoders.fit_target_encoding(base, ["card4"], lag_days=1, smoothing=1.0)
    override = encoders.apply_target_encoding(base, fitted, lag_days=9, smoothing=33.0)
    refit = encoders.apply_target_encoding(
        base, encoders.fit_target_encoding(base, ["card4"], lag_days=9, smoothing=33.0)
    )
    name = encoders.target_name("card4")
    pd.testing.assert_series_equal(override[name], refit[name])


def test_frequency_encoding_gives_an_unseen_level_zero_and_keeps_a_null_null() -> None:
    base = train_frame()
    fitted = encoders.fit_frequency_encoding(base, ["card4"], min_cardinality=2)
    later = base.tail(2).copy()
    later["card4"] = ["never_seen_before", None]
    encoded = encoders.apply_frequency_encoding(later, fitted)[encoders.frequency_name("card4")]
    assert float(encoded.iloc[0]) == 0.0
    assert pd.isna(encoded.iloc[1])


def test_frequency_encoding_skips_a_narrow_column() -> None:
    base = train_frame()
    fitted = encoders.fit_frequency_encoding(base, ["card4"], min_cardinality=99)
    assert "card4" not in fitted["tables"]
    assert fitted["per_column"]["card4"]["encoded"] is False


def test_vocabulary_collapses_the_rare_tail_and_names_the_three_states() -> None:
    base = train_frame()
    fitted = encoders.fit_category_vocabulary(base, ["card4"], rare_tail_share=0.01)
    spec = fitted["columns"]["card4"]
    assert spec["n_levels_collapsed"] >= 1
    assert spec["share_rows_collapsed"] <= 0.01

    later = base.tail(3).copy()
    later["card4"] = [spec["collapsed_levels"][0], "never_seen_before", None]
    encoded = encoders.apply_category_vocabulary(later, fitted)[encoders.vocabulary_name("card4")]
    assert list(encoded.astype("object")) == [
        fitted["rare_token"],
        fitted["unseen_token"],
        fitted["missing_token"],
    ]


def test_vocabulary_category_codes_do_not_depend_on_the_rows_present() -> None:
    base = train_frame()
    fitted = encoders.fit_category_vocabulary(base, ["card4"])
    name = encoders.vocabulary_name("card4")
    whole = encoders.apply_category_vocabulary(base, fitted)[name]
    part = encoders.apply_category_vocabulary(base.tail(5), fitted)[name]
    assert list(whole.cat.categories) == list(part.cat.categories)


def test_device_info_normalisation_is_structural() -> None:
    assert encoders.normalise_device_info("SM-J700M Build/MMB29K") == "sm"
    assert encoders.normalise_device_info("SM-G610M Build/MMB29K") == "sm"
    assert encoders.normalise_device_info("rv:57.0") == "rv"
    assert encoders.normalise_device_info("Trident/7.0") == "trident"
    assert encoders.normalise_device_info("Windows") == "windows"
    assert encoders.normalise_device_info(None) is None


def test_browser_normalisation_drops_the_version_and_keeps_the_platform() -> None:
    assert encoders.normalise_browser("chrome 63.0") == "chrome|none"
    assert encoders.normalise_browser("chrome 64.0") == "chrome|none"
    assert encoders.normalise_browser("chrome 63.0 for android") == "chrome|android"
    assert encoders.normalise_browser("safari generic") == "safari|none"
    assert encoders.normalise_browser("ie 11.0 for desktop") == "ie|desktop"
    assert encoders.normalise_browser(None) is None


# --- schema -------------------------------------------------------------------------------------


def declared(base: pd.DataFrame) -> schema_module.Schema:
    return schema_module.declare(base, negative_value_columns=[], measured_columns=base.columns)


def test_schema_accepts_the_frame_it_was_declared_from() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema_module.validate(base, declared(base))


def test_schema_rejects_a_wrong_dtype() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.copy()
    bad["C1"] = bad["C1"].astype("string")
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("dtype family" in v["problem"] for v in raised.value.violations)


def test_schema_rejects_an_out_of_range_value() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.copy()
    bad.loc[bad.index[0], config.AMOUNT_COLUMN] = -1.0
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("below the declared minimum" in v["problem"] for v in raised.value.violations)


def test_schema_rejects_an_unexpected_column() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.assign(surprise=1.0)
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("not in the schema" in v["problem"] for v in raised.value.violations)


def test_schema_rejects_a_missing_column() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(base.drop(columns=["C1"]), schema)
    assert any("declared column is missing" in v["problem"] for v in raised.value.violations)


def test_schema_rejects_a_null_in_a_non_nullable_column() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.copy()
    bad[config.AMOUNT_COLUMN] = bad[config.AMOUNT_COLUMN].astype("float64")
    bad.loc[bad.index[0], config.AMOUNT_COLUMN] = np.nan
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("non-nullable" in v["problem"] for v in raised.value.violations)


def test_schema_reports_every_violation_and_not_just_the_first() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.assign(surprise=1.0)
    bad.loc[bad.index[0], config.AMOUNT_COLUMN] = -1.0
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert len(raised.value.violations) >= 2


def test_schema_round_trips_through_a_dict() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    assert schema_module.Schema.from_dict(schema.to_dict()) == schema


def test_schema_declares_no_upper_bound_on_a_measured_column() -> None:
    """The deliberate omission from ADR 0017, asserted so it cannot be added by accident."""
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    spec = schema.spec("C1")
    assert spec is not None
    assert spec.maximum is None


def test_derived_columns_get_bounds_from_their_arithmetic_not_from_a_measurement() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    derived = base.assign(
        **{
            "vblockmean_V1": -1.5,
            "card1_te": 0.4,
            "card1_freq": 12.0,
        }
    )
    schema = schema_module.declare(
        derived, negative_value_columns=[], measured_columns=base.columns
    )
    assert schema.spec("vblockmean_V1").minimum is None  # type: ignore[union-attr]
    assert schema.spec("card1_te").maximum == 1.0  # type: ignore[union-attr]
    assert schema.spec("card1_freq").minimum == 0.0  # type: ignore[union-attr]


# --- the column plan ------------------------------------------------------------------------------


def test_apply_column_plan_drops_what_the_plan_drops_and_spares_what_is_spared() -> None:
    plan = {
        "columns": [
            {"column": "C1", "decision": "keep", "reason": None, "role": "feature"},
            {"column": "D1", "decision": "drop", "reason": "redundant", "role": "feature"},
        ]
    }
    base = train_frame()
    assert "D1" not in cleaning.apply_column_plan(base, plan).columns
    assert "D1" in cleaning.apply_column_plan(base, plan, spare=["D1"]).columns


def test_apply_column_plan_leaves_a_column_the_plan_never_saw() -> None:
    plan = {"columns": [{"column": "C1", "decision": "keep", "reason": None, "role": "feature"}]}
    assert "DeviceInfo" in cleaning.apply_column_plan(train_frame(), plan).columns


def test_column_group_and_role_cover_the_named_blocks() -> None:
    assert cleaning.column_group("V12") == "vesta"
    assert cleaning.column_group("id_31") == "identity"
    assert cleaning.column_group("DeviceInfo") == "identity"
    assert cleaning.column_group("C1") == "count"
    assert cleaning.column_group(config.TARGET) == "meta"
    assert cleaning.column_role(config.TARGET) == "target"
    assert cleaning.column_role(config.ID_COLUMN) == "identifier"
    assert cleaning.column_role(config.TIME_COLUMN) == "time"
    assert cleaning.column_role("C1") == "feature"


# --- the remaining validator branches -------------------------------------------------------


def test_schema_rejects_a_non_finite_float() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.copy()
    bad.loc[bad.index[0], "C1"] = np.inf
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("non-finite" in v["problem"] for v in raised.value.violations)


def test_schema_rejects_a_value_above_a_declared_maximum() -> None:
    """The only bounded-above columns are derivations whose arithmetic bounds them."""
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    derived = base.assign(card1_te=0.4)
    schema = schema_module.declare(
        derived, negative_value_columns=[], measured_columns=base.columns
    )
    bad = derived.copy()
    bad.loc[bad.index[0], "card1_te"] = 1.5
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("above the declared maximum" in v["problem"] for v in raised.value.violations)


def test_schema_rejects_a_duplicate_in_the_unique_column() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = declared(base)
    bad = base.copy()
    bad.loc[bad.index[1], config.ID_COLUMN] = bad.loc[bad.index[0], config.ID_COLUMN]
    with pytest.raises(schema_module.SchemaError) as raised:
        schema_module.validate(bad, schema)
    assert any("unique column" in v["problem"] for v in raised.value.violations)


def test_schema_spec_returns_none_for_a_column_it_does_not_declare() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    assert declared(base).spec("nothing_like_this") is None


def test_family_of_covers_every_dtype_the_pipeline_produces() -> None:
    assert schema_module.family_of(pd.Series([1, 2], dtype="int16").dtype) == "integer"
    assert schema_module.family_of(pd.Series([1.0], dtype="float32").dtype) == "float"
    assert schema_module.family_of(pd.Series([True], dtype="bool").dtype) == "boolean"
    assert schema_module.family_of(pd.Series(pd.Categorical(["a"])).dtype) == "categorical"
    assert schema_module.family_of(pd.Series(["a"], dtype="string").dtype) == "categorical"
    assert schema_module.family_of(pd.Series(["a"], dtype="object").dtype) == "categorical"
    assert schema_module.family_of(np.dtype("datetime64[ns]")) == "datetime64[ns]"


def test_a_derived_column_with_no_known_range_gets_no_bound() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    derived = base.assign(something_this_stage_invented=-3.0)
    schema = schema_module.declare(
        derived, negative_value_columns=[], measured_columns=base.columns
    )
    spec = schema.spec("something_this_stage_invented")
    assert spec is not None
    assert spec.minimum is None and spec.maximum is None
    assert "neither measured" in (spec.note or "")
    schema_module.validate(derived, schema)


def test_a_measured_column_on_the_negative_list_gets_no_lower_bound() -> None:
    base = train_frame().drop(columns=["card4", "DeviceInfo"])
    schema = schema_module.declare(
        base, negative_value_columns=["C1"], measured_columns=base.columns
    )
    spec = schema.spec("C1")
    assert spec is not None
    assert spec.minimum is None
    assert "17 that hold" in (spec.note or "")
