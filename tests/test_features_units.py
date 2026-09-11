"""The pieces of `fraud_platform.features` that are not the causal rule.

`tests/test_causality.py` checks the rule. This checks everything around it: the config switch
the native column groups sit behind, the catalogue that the artifact is generated from, the one
fitted object and its guard, the null and zero convention, and the dtypes.

Everything runs on a generated frame, so it runs in CI with no data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, features
from fraud_platform.data_loader import TrainOnlyError

DAY = config.SECONDS_PER_DAY


# --- the catalogue is the source, not a second list ------------------------------------------


def test_the_catalogue_and_the_built_frame_hold_the_same_features(
    built_features: pd.DataFrame,
) -> None:
    assert list(built_features.columns) == features.feature_names()


def test_every_spec_declares_a_known_family_and_grain() -> None:
    for spec in features.catalogue():
        assert spec.family in features.FAMILIES, spec.name
        assert spec.grain in (*features.GRAINS, "row"), spec.name
        assert spec.definition.strip(), spec.name


def test_a_feature_that_needs_state_says_what_state() -> None:
    for spec in features.catalogue():
        if spec.requires_running_state:
            assert spec.state_key in features.GRAINS, spec.name
            assert spec.state_fields, f"{spec.name} needs state and does not say which"
        else:
            assert spec.state_key == "none", spec.name
            assert not spec.state_fields, spec.name


def test_a_windowed_feature_carries_its_window() -> None:
    windowed = [spec for spec in features.catalogue() if spec.family == "velocity"]
    assert windowed
    declared = {spec.window_seconds for spec in windowed}
    assert declared == {seconds for _, seconds in features.WINDOWS}


def test_every_nullable_feature_says_what_a_null_means(built_features: pd.DataFrame) -> None:
    """A null with no stated meaning is the failure this asserts against, not a missing value."""
    for spec in features.catalogue():
        if not spec.derived:
            continue
        has_nulls = bool(built_features[spec.name].isna().any())
        if has_nulls:
            assert spec.null_meaning, f"{spec.name} carries nulls and declares no meaning for them"


def test_a_count_feature_declares_no_null_meaning_and_carries_none(
    built_features: pd.DataFrame,
) -> None:
    """The convention in one test: a count of history is zero, never null."""
    counts = [
        "rec_prior_count",
        "dev_n_distinct_so_far",
        "geo_n_distinct_addr1_on_card",
        "dup_prior_count",
        "dup_prior_count_1h",
        *[
            features.velocity_name("count", s, g)
            for s, _ in features.WINDOWS
            for g in ("entity", "card")
        ],
    ]
    for name in counts:
        assert not built_features[name].isna().any(), f"{name} is a count and carries a null"
        assert (built_features[name] >= 0).all(), name
        spec = next(spec for spec in features.catalogue() if spec.name == name)
        assert spec.null_meaning is None, name


def test_the_catalogue_follows_the_config() -> None:
    cfg = features.DEFAULT_CONFIG.with_families("velocity")
    names = features.feature_names(cfg)
    assert names
    assert all(name.startswith("vel_") for name in names)

    one_grain = features.FeatureConfig(velocity_grains=("entity",))
    velocity = [spec.name for spec in features.catalogue(one_grain) if spec.family == "velocity"]
    assert velocity
    assert all(name.endswith("_entity") for name in velocity)
    assert len(velocity) == 3 * len(features.WINDOWS)

    with_origin = features.FeatureConfig(include_d_origin_tenure=True)
    assert "D1_origin" in features.feature_names(with_origin)
    assert "D1_origin" not in features.feature_names()


def test_the_tenure_native_column_is_listed_and_not_recomputed(
    built_features: pd.DataFrame,
) -> None:
    """D1 is the tenure feature. Listing it is the contribution; copying it would be a second
    source for one number."""
    spec = next(spec for spec in features.catalogue() if spec.name == "D1")
    assert spec.family == "tenure"
    assert not spec.derived
    assert "D1" not in built_features.columns


# --- the native group switch ------------------------------------------------------------------


def native_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.TIME_COLUMN: [1, 2],
            "C1": [1.0, 2.0],
            "C14": [1.0, 2.0],
            "D1": [1.0, 2.0],
            "D15": [1.0, 2.0],
            "M1": ["T", "F"],
            "V1": [1.0, 2.0],
            "vblockmean_V12": [0.1, 0.2],
            "vpca_V12_0": [0.1, 0.2],
            "TransactionAmt": [10.0, 20.0],
        }
    )


def test_switching_a_group_off_drops_exactly_that_group() -> None:
    frame = native_frame()
    kept = features.select_native_groups(frame, features.FeatureConfig(native_groups=("count",)))
    assert set(kept.columns) == {config.TIME_COLUMN, "C1", "C14", "TransactionAmt"}


def test_the_vesta_switch_covers_the_columns_stage_3_left_behind() -> None:
    """ADR 0016 replaced the V columns with block means, so a switch on the raw names alone would
    switch nothing off."""
    frame = native_frame()
    without = features.select_native_groups(
        frame, features.FeatureConfig(native_groups=("count", "timedelta", "match"))
    )
    assert "V1" not in without.columns
    assert "vblockmean_V12" not in without.columns
    assert "vpca_V12_0" not in without.columns
    assert "C1" in without.columns


def test_all_groups_on_changes_nothing() -> None:
    frame = native_frame()
    assert list(features.select_native_groups(frame, features.DEFAULT_CONFIG).columns) == list(
        frame.columns
    )


def test_an_unknown_group_raises() -> None:
    with pytest.raises(KeyError, match="unknown native group"):
        features.native_group_columns(native_frame(), "merchant")


# --- the one fitted object -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("what", "make"),
    [
        (
            "straddles the boundary",
            lambda: pd.DataFrame(
                {
                    config.TIME_COLUMN: [config.TRAIN_END_DT - 1, config.TRAIN_END_DT + 1],
                    "dist1": [1.0, 2.0],
                }
            ),
        ),
        ("is empty", lambda: pd.DataFrame({config.TIME_COLUMN: [], "dist1": []})),
        ("has no timestamp", lambda: pd.DataFrame({"dist1": [1.0, 2.0]})),
    ],
)
def test_fitting_the_buckets_on_a_frame_that_(what: str, make: object) -> None:
    with pytest.raises(TrainOnlyError):
        features.fit_dist_buckets(make())  # type: ignore[operator]


def test_the_bucket_count_reported_is_the_count_the_edges_produce() -> None:
    frame = pd.DataFrame({config.TIME_COLUMN: [1_000] * 6, "dist1": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]})
    fitted = features.fit_dist_buckets(frame)
    assert fitted["n_buckets_requested"] == features.N_DIST_BUCKETS
    assert fitted["n_buckets"] < fitted["n_buckets_requested"], (
        "a column concentrated on two values cannot be cut into ten equal parts"
    )
    assert fitted["n_buckets"] == len(fitted["edges"]) + 1


def test_a_bucket_index_is_monotone_in_the_value_and_null_stays_null() -> None:
    frame = pd.DataFrame(
        {config.TIME_COLUMN: [1_000] * 10, "dist1": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    )
    fitted = features.fit_dist_buckets(frame)
    scored = pd.DataFrame({config.TIME_COLUMN: [1] * 4, "dist1": [0.5, 5.0, 100.0, np.nan]})
    buckets = features.apply_dist_buckets(scored, fitted)
    assert buckets[0] <= buckets[1] <= buckets[2]
    assert buckets[0] < buckets[2]
    assert np.isnan(buckets[3])


# --- what build_features refuses ---------------------------------------------------------------


def test_a_frame_without_the_entity_key_raises(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=config.ENTITY_ID_COLUMN):
        features.build_features(feature_frame.drop(columns=[config.ENTITY_ID_COLUMN]))


def test_an_unknown_family_raises(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="unknown feature families"):
        features.build_features(
            feature_frame, dist_buckets, features.DEFAULT_CONFIG.with_families("merchant")
        )


def test_an_unknown_grain_raises(feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]) -> None:
    cfg = features.FeatureConfig(velocity_grains=("merchant",))
    with pytest.raises(ValueError, match="unknown velocity grains"):
        features.build_features(feature_frame, dist_buckets, cfg)


def test_geography_without_the_fitted_edges_raises(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="dist1 bucket edges"):
        features.build_features(
            feature_frame, None, features.DEFAULT_CONFIG.with_families("geography")
        )


def test_a_family_that_needs_no_fit_runs_without_one(feature_frame: pd.DataFrame) -> None:
    only_velocity = features.build_features(
        feature_frame, None, features.DEFAULT_CONFIG.with_families("velocity")
    )
    assert len(only_velocity.columns) == 3 * len(features.WINDOWS) * 2


# --- shapes, dtypes and determinism ------------------------------------------------------------


def test_the_built_frame_lines_up_with_the_input(
    feature_frame: pd.DataFrame, built_features: pd.DataFrame
) -> None:
    assert len(built_features) == len(feature_frame)
    assert built_features.index.equals(feature_frame.index)


def test_counts_are_integers_and_the_rest_are_floats(built_features: pd.DataFrame) -> None:
    for name in built_features.columns:
        dtype = built_features[name].dtype
        if built_features[name].isna().any():
            assert pd.api.types.is_float_dtype(dtype), name
        else:
            assert pd.api.types.is_numeric_dtype(dtype), name
    assert built_features["rec_prior_count"].dtype == np.dtype("int32")


def test_building_twice_gives_the_same_frame(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]
) -> None:
    first = features.build_features(feature_frame, dist_buckets)
    second = features.build_features(feature_frame, dist_buckets)
    pd.testing.assert_frame_equal(first, second)


def test_row_order_does_not_change_the_answer(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any], built_features: pd.DataFrame
) -> None:
    """The stream is sorted internally, so a caller handing over a shuffled frame gets the same
    values back against the same rows."""
    shuffled = feature_frame.sample(frac=1.0, random_state=config.SEED)
    again = features.build_features(shuffled, dist_buckets)
    pd.testing.assert_frame_equal(again.loc[feature_frame.index], built_features)


def test_add_features_keeps_the_frame_and_the_switch(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]
) -> None:
    cfg = features.FeatureConfig(native_groups=())
    joined = features.add_features(feature_frame, dist_buckets, cfg)
    assert "D1" not in joined.columns
    assert config.AMOUNT_COLUMN in joined.columns
    assert set(features.feature_names(cfg)) <= set(joined.columns)


# --- the keys -----------------------------------------------------------------------------------


def test_two_rows_null_in_the_same_field_share_a_content_key() -> None:
    """Stage 2's rule, from eda.duplicate_groups. Reproducing it is what makes the counts here
    comparable with quality.json."""
    frame = pd.DataFrame(
        {
            "card1": [1.0, 1.0, 1.0],
            "card2": [np.nan, np.nan, 5.0],
            "addr1": [1.0, 1.0, 1.0],
            config.AMOUNT_COLUMN: [10.0, 10.0, 10.0],
            "ProductCD": ["W", "W", "W"],
            "P_emaildomain": [None, None, None],
        }
    )
    codes = features.content_codes(frame)
    assert codes[0] == codes[1]
    assert codes[2] != codes[0]


def test_a_row_with_no_identity_record_has_no_device_combination() -> None:
    frame = pd.DataFrame(
        {
            "DeviceInfo_norm": ["windows", None, None],
            "DeviceType": ["desktop", None, "mobile"],
            "id_30": ["windows 10", None, None],
            "id_31_norm": ["chrome|none", None, None],
        }
    )
    codes = features.device_codes(frame)
    assert codes[1] == -1
    assert codes[0] != -1
    assert codes[2] != -1, "a partly present combination is a combination"


# --- what the artifact reads --------------------------------------------------------------------


def test_null_rates_report_an_absent_column_as_absent(built_features: pd.DataFrame) -> None:
    rates = features.null_rates(built_features, ["rec_prior_count", "not_a_column"])
    assert rates["rec_prior_count"]["present"] is True
    assert rates["rec_prior_count"]["n_null"] == 0
    assert rates["not_a_column"]["present"] is False
    assert rates["not_a_column"]["null_rate"] is None


def test_the_state_footprint_covers_all_three_keyspaces(feature_frame: pd.DataFrame) -> None:
    footprint = features.state_footprint(feature_frame)
    for grain in ("entity", "card", "content"):
        assert footprint[grain]["n_keys"] > 0
    assert footprint["card"]["n_keys"] < footprint["entity"]["n_keys"], (
        "the card grain is coarser than the entity grain by construction"
    )
    assert footprint["card"]["distinct_addr1_per_key"]["max"] >= 1
    assert footprint["entity"]["distinct_devices_per_key"]["max"] >= 1


# --- what is deliberately absent ----------------------------------------------------------------


def test_the_module_ships_no_uid_aggregation() -> None:
    """ADR 0020. Not behind a switch, so there is no switch to find and no name to import."""
    public = [name for name in dir(features) if not name.startswith("_")]
    assert not [name for name in public if "uid" in name.lower()]
    assert not [name for name in public if "agg" in name.lower()]
    assert any("UID aggregation" in entry for entry in features.NOT_BUILT)


def test_the_entity_grain_addr1_deviation_is_not_shipped() -> None:
    """ADR 0023. addr1 is a component of the entity key, so the feature is identically zero."""
    assert "geo_addr1_differs_from_entity_mode" not in features.feature_names()
    assert any("addr1 is a component of the entity key" in entry for entry in features.NOT_BUILT)


def test_addr1_cannot_vary_inside_an_entity(feature_frame: pd.DataFrame) -> None:
    """The measurement behind ADR 0023, on the generated frame: the key pins the column."""
    distinct = feature_frame.groupby(config.ENTITY_ID_COLUMN, observed=True)["addr1"].nunique(
        dropna=True
    )
    assert int(distinct.max()) <= 1
    by_card = feature_frame.groupby("card1", observed=True)["addr1"].nunique(dropna=True)
    assert int(by_card.max()) > 1, "the card grain is where the deviation can be non-zero"


def test_the_tenure_origin_switch_builds_the_column_it_declares(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]
) -> None:
    """Off by default on ADR 0013's measurement, and still reachable, because stage 4 measured
    its PSI and a switch nobody can turn on is not a switch."""
    cfg = features.FeatureConfig(
        families=("tenure",), include_d_origin_tenure=True, native_groups=()
    )
    built = features.build_features(feature_frame, dist_buckets, cfg)
    assert list(built.columns) == ["D1_origin"]
    expected = feature_frame[config.TIME_COLUMN] / DAY - feature_frame["D1"]
    pd.testing.assert_series_equal(built["D1_origin"], expected, check_names=False)

    off = features.build_features(
        feature_frame, dist_buckets, features.FeatureConfig(families=("tenure",))
    )
    assert list(off.columns) == []


def test_the_card_grain_needs_card1(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="card grain needs card1"):
        features.card_codes(feature_frame.drop(columns=["card1"]))


def test_the_content_grain_needs_one_of_its_columns(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="content grain needs"):
        features.content_codes(feature_frame.drop(columns=list(features.CONTENT_COLUMNS)))


def test_a_frame_with_no_identity_columns_has_no_device_combinations(
    feature_frame: pd.DataFrame,
) -> None:
    stripped = feature_frame.drop(
        columns=[c for c in features.DEVICE_COMBINATION_COLUMNS if c in feature_frame.columns]
    )
    assert (features.device_codes(stripped) == -1).all()


def test_an_empty_frame_gives_empty_features(dist_buckets: dict[str, Any]) -> None:
    """Not a real case, and the guards are there so it is not a crash either."""
    columns = [
        config.TIME_COLUMN,
        config.AMOUNT_COLUMN,
        config.ENTITY_ID_COLUMN,
        "card1",
        "addr1",
        "addr2",
        "dist1",
        *features.CONTENT_COLUMNS,
    ]
    empty = pd.DataFrame({column: pd.Series(dtype="float64") for column in dict.fromkeys(columns)})
    empty[config.ENTITY_ID_COLUMN] = pd.Series(dtype="string")
    empty["ProductCD"] = pd.Series(dtype="string")
    empty["P_emaildomain"] = pd.Series(dtype="string")
    built = features.build_features(empty, dist_buckets)
    assert len(built) == 0
    assert list(built.columns) == features.feature_names()
    footprint = features.state_footprint(empty)
    assert footprint["entity"]["n_keys"] == 0


def test_entity_codes_refuse_a_frame_without_the_key(feature_frame: pd.DataFrame) -> None:
    """`build_features` checks the required columns first, so this guard needs its own call."""
    with pytest.raises(KeyError, match="add_entity_key"):
        features.entity_codes(feature_frame.drop(columns=[config.ENTITY_ID_COLUMN]))


def test_the_config_serialises_to_what_the_artifact_records() -> None:
    payload = features.DEFAULT_CONFIG.to_dict()
    assert payload["families"] == list(features.FAMILIES)
    assert payload["velocity_grains"] == ["entity", "card"]
    assert payload["windows"] == [
        {"suffix": suffix, "seconds": seconds} for suffix, seconds in features.WINDOWS
    ]
    assert payload["include_d_origin_tenure"] is False


def test_every_spec_serialises_with_the_fields_the_artifact_reads() -> None:
    """`reports/feature_summary.json` is generated from these dicts, and stage 8 reads
    `state_key` and `state_fields` out of it. A renamed field here is a broken artifact."""
    expected = {
        "feature",
        "family",
        "grain",
        "derived_in_stage_4",
        "window_seconds",
        "requires_running_state",
        "state_key",
        "state_fields",
        "definition",
        "null_meaning",
    }
    for spec in features.catalogue():
        payload = spec.to_dict()
        assert set(payload) == expected, spec.name
        assert payload["feature"] == spec.name
        assert payload["requires_running_state"] == (payload["state_key"] != "none")
