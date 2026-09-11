"""The pieces of `fraud_platform.graph_features` that are not the causal rule.

`tests/test_causality.py` checks the rule by brute force. This checks everything around it: the
switch, the candidate set and the catalogue, the label source and its guard, the null and zero
convention, the union-find's counters, the static graph the artifact is summarised from, and the
hub masking the sweep uses. Everything runs on the generated frame, so it runs in CI with no data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, features, graph_features
from fraud_platform.data_loader import TrainOnlyError

DAY = config.SECONDS_PER_DAY


@pytest.fixture(scope="module")
def all_on() -> graph_features.GraphConfig:
    return graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)


@pytest.fixture(scope="module")
def labels(feature_frame: pd.DataFrame) -> graph_features.LabelSource:
    boundary = int(feature_frame[config.TIME_COLUMN].quantile(0.7))
    return graph_features.label_source(
        feature_frame[feature_frame[config.TIME_COLUMN] < boundary], lag_days=3
    )


@pytest.fixture(scope="module")
def built(
    feature_frame: pd.DataFrame,
    labels: graph_features.LabelSource,
    all_on: graph_features.GraphConfig,
) -> pd.DataFrame:
    return graph_features.build_graph_features(feature_frame, labels, all_on)


# --- the switch and the candidate set -----------------------------------------------------------


def test_the_switch_is_off_by_default_and_off_builds_nothing(feature_frame: pd.DataFrame) -> None:
    assert graph_features.DEFAULT_CONFIG.enabled is False
    out = graph_features.build_graph_features(feature_frame)
    assert out.shape == (len(feature_frame), 0)
    assert out.index.equals(feature_frame.index)


def test_switching_on_builds_the_candidate_set_and_nothing_label_derived(
    feature_frame: pd.DataFrame,
) -> None:
    cfg = graph_features.DEFAULT_CONFIG.switched(True)
    out = graph_features.build_graph_features(feature_frame)
    assert out.shape[1] == 0
    out = graph_features.build_graph_features(feature_frame, cfg=cfg)
    assert list(out.columns) == list(graph_features.CANDIDATE_FEATURES)
    assert not set(out.columns) & set(graph_features.LABEL_FEATURES)
    assert graph_features.COMPONENT_SIZE not in out.columns


def test_the_candidate_set_is_a_subset_of_everything_and_excludes_the_clock() -> None:
    assert set(graph_features.CANDIDATE_FEATURES) < set(graph_features.ALL_FEATURES)
    assert set(graph_features.LABEL_FEATURES) < set(graph_features.ALL_FEATURES)
    assert not set(graph_features.CANDIDATE_FEATURES) & set(graph_features.LABEL_FEATURES)
    assert graph_features.COMPONENT_SIZE not in graph_features.CANDIDATE_FEATURES


def test_a_label_feature_cannot_be_built_without_a_label_source(
    feature_frame: pd.DataFrame,
) -> None:
    cfg = graph_features.GraphConfig(enabled=True).with_features(
        graph_features.COMPONENT_FRAUD_RATE
    )
    with pytest.raises(ValueError, match="LabelSource"):
        graph_features.build_graph_features(feature_frame, cfg=cfg)


def test_an_unknown_feature_raises(feature_frame: pd.DataFrame) -> None:
    cfg = graph_features.GraphConfig(enabled=True).with_features("graph_pagerank")
    with pytest.raises(ValueError, match="unknown graph features"):
        graph_features.build_graph_features(feature_frame, cfg=cfg)


def test_the_config_serialises_to_what_the_artifact_records() -> None:
    recorded = graph_features.DEFAULT_CONFIG.to_dict()
    assert recorded["enabled"] is graph_features.DEFAULT_ENABLED
    assert recorded["link_columns"] == list(graph_features.LINK_COLUMNS)
    assert recorded["anchor_column"] == graph_features.ANCHOR_COLUMN
    assert recorded["features"] == list(graph_features.CANDIDATE_FEATURES)
    assert recorded["all_features"] == list(graph_features.ALL_FEATURES)
    assert recorded["label_features"] == list(graph_features.LABEL_FEATURES)


# --- the catalogue is the source ---------------------------------------------------------------


def test_the_catalogue_lists_every_feature_in_the_order_the_frame_carries_them(
    built: pd.DataFrame, all_on: graph_features.GraphConfig
) -> None:
    assert [spec.name for spec in graph_features.catalogue(all_on)] == list(
        graph_features.ALL_FEATURES
    )
    assert list(built.columns) == graph_features.graph_feature_names(all_on)


def test_every_spec_says_whether_it_is_label_derived_and_what_state_it_needs() -> None:
    for spec in graph_features.catalogue():
        assert spec.definition.strip(), spec.name
        assert spec.state.strip(), spec.name
        assert spec.label_derived == (spec.name in graph_features.LABEL_FEATURES), spec.name
        assert spec.to_dict()["requires_running_state"] is True


def test_every_nullable_feature_says_what_a_null_means(built: pd.DataFrame) -> None:
    for spec in graph_features.catalogue():
        carries_null = bool(built[spec.name].isna().any())
        if carries_null:
            assert spec.null_meaning, f"{spec.name} carries nulls with no stated meaning"
        else:
            assert spec.null_meaning is None, spec.name


def test_a_count_is_an_integer_and_the_rest_are_floats(built: pd.DataFrame) -> None:
    counts = (
        graph_features.COMPONENT_SIZE,
        graph_features.COMPONENT_N_LABELLED,
        graph_features.degree_name(graph_features.ANCHOR_COLUMN),
    )
    for name in counts:
        assert built[name].dtype == np.int32, name
        assert (built[name] >= 0).all(), name
    for name in built.columns:
        if name not in counts:
            assert built[name].dtype == np.float64, name


def test_the_degree_is_null_exactly_where_the_value_is_null(
    feature_frame: pd.DataFrame, built: pd.DataFrame
) -> None:
    for column in graph_features.DEGREE_COLUMNS:
        name = graph_features.degree_name(column)
        pd.testing.assert_series_equal(
            built[name].isna(), feature_frame[column].isna(), check_names=False
        )
    assert not built[graph_features.degree_name(graph_features.ANCHOR_COLUMN)].isna().any()
    pd.testing.assert_series_equal(
        built[graph_features.ENTITIES_ON_DEVICE].isna(),
        feature_frame[graph_features.DEVICE_COLUMN].isna(),
        check_names=False,
    )


def test_a_rate_is_null_exactly_where_its_count_is_zero(built: pd.DataFrame) -> None:
    rate = built[graph_features.COMPONENT_FRAUD_RATE]
    count = built[graph_features.COMPONENT_N_LABELLED]
    pd.testing.assert_series_equal(rate.isna(), count == 0, check_names=False)
    assert ((rate.dropna() >= 0.0) & (rate.dropna() <= 1.0)).all()


def test_the_first_row_of_the_stream_sees_an_empty_graph(
    feature_frame: pd.DataFrame, built: pd.DataFrame
) -> None:
    first = int(np.argmin(feature_frame[config.TIME_COLUMN].to_numpy()))
    row = built.iloc[first]
    assert row[graph_features.COMPONENT_SIZE] == 0
    assert row[graph_features.degree_name("card1")] == 0.0
    assert row[graph_features.COMPONENT_N_LABELLED] == 0
    assert np.isnan(row[graph_features.COMPONENT_FRAUD_RATE])


def test_the_component_of_the_card_holds_at_least_the_cards_own_history(
    feature_frame: pd.DataFrame, built: pd.DataFrame
) -> None:
    """The component is anchored on card1, so it is never smaller than the card's degree."""
    assert (
        built[graph_features.COMPONENT_SIZE] >= built[graph_features.degree_name("card1")]
    ).all()


def test_the_others_rate_reads_fewer_labels_than_the_full_rate(built: pd.DataFrame) -> None:
    rate = built[graph_features.COMPONENT_FRAUD_RATE]
    others = built[graph_features.COMPONENT_FRAUD_RATE_OTHERS]
    assert others.isna().sum() >= rate.isna().sum()
    assert (others.notna() <= rate.notna()).all()
    assert (rate != others).any(), "the generated frame is meant to make the two differ"


def test_the_build_records_how_many_labels_it_attached(
    built: pd.DataFrame, labels: graph_features.LabelSource
) -> None:
    attached = built.attrs["n_labels_attached"]
    assert 0 < attached <= labels.n_rows


def test_building_twice_gives_the_same_frame(
    feature_frame: pd.DataFrame,
    labels: graph_features.LabelSource,
    all_on: graph_features.GraphConfig,
    built: pd.DataFrame,
) -> None:
    again = graph_features.build_graph_features(feature_frame, labels, all_on)
    pd.testing.assert_frame_equal(again, built)


def test_row_order_does_not_change_the_answer(
    feature_frame: pd.DataFrame,
    labels: graph_features.LabelSource,
    all_on: graph_features.GraphConfig,
    built: pd.DataFrame,
) -> None:
    shuffled = feature_frame.sample(frac=1.0, random_state=config.SEED)
    out = graph_features.build_graph_features(shuffled, labels, all_on)
    pd.testing.assert_frame_equal(out.sort_index(), built)


def test_add_graph_features_keeps_the_frame(
    feature_frame: pd.DataFrame,
    labels: graph_features.LabelSource,
    all_on: graph_features.GraphConfig,
) -> None:
    out = graph_features.add_graph_features(feature_frame, labels, all_on)
    assert list(out.columns) == [*feature_frame.columns, *graph_features.ALL_FEATURES]
    pd.testing.assert_frame_equal(out[feature_frame.columns], feature_frame)


# --- the inputs the graph refuses ---------------------------------------------------------------


def test_a_missing_link_column_raises(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match="link columns"):
        graph_features.build_graph_features(
            feature_frame.drop(columns=["R_emaildomain"]),
            cfg=graph_features.GraphConfig(enabled=True),
        )


def test_the_feature_stream_needs_the_anchor_among_the_links(feature_frame: pd.DataFrame) -> None:
    cfg = graph_features.GraphConfig(enabled=True, link_columns=("addr1", "dist1"))
    with pytest.raises(ValueError, match="anchor"):
        graph_features.make_graph_stream(feature_frame, cfg)
    assert graph_features.make_graph_stream(feature_frame, cfg, require_anchor=False).n_rows == len(
        feature_frame
    )


def test_a_degree_column_must_be_a_link_column(feature_frame: pd.DataFrame) -> None:
    cfg = graph_features.GraphConfig(enabled=True, link_columns=("card1", "addr1"))
    with pytest.raises(ValueError, match="degree column"):
        graph_features.build_graph_features(feature_frame, cfg=cfg)


def test_a_null_anchor_raises(feature_frame: pd.DataFrame) -> None:
    holed = feature_frame.copy()
    holed.loc[holed.index[3], "card1"] = np.nan
    with pytest.raises(ValueError, match="null-free"):
        graph_features.build_graph_features(holed, cfg=graph_features.GraphConfig(enabled=True))


def test_a_frame_without_the_entity_key_raises(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError, match=config.ENTITY_ID_COLUMN):
        graph_features.build_graph_features(
            feature_frame.drop(columns=[config.ENTITY_ID_COLUMN]),
            cfg=graph_features.GraphConfig(enabled=True),
        )


def test_an_empty_frame_gives_empty_features(
    feature_frame: pd.DataFrame, all_on: graph_features.GraphConfig
) -> None:
    out = graph_features.build_graph_features(feature_frame.iloc[0:0], cfg=all_on.with_features())
    assert out.shape == (0, 0)


# --- the label source ---------------------------------------------------------------------------


def test_the_label_source_is_train_only_and_sorted(feature_frame: pd.DataFrame) -> None:
    source = graph_features.label_source(feature_frame, lag_days=2)
    assert source.lag_days == 2
    assert source.n_rows == len(feature_frame)
    assert (np.diff(source.ts) >= 0).all()
    assert source.to_dict()["n_fraud"] == int(feature_frame[config.TARGET].sum())


def test_the_label_source_refuses_a_negative_lag(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        graph_features.label_source(feature_frame, lag_days=-1)


def test_the_label_source_needs_the_label_and_the_id(feature_frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        graph_features.label_source(feature_frame.drop(columns=[config.TARGET]), lag_days=1)


@pytest.mark.parametrize(
    ("what", "shift"), [("a frame straddling the boundary", 5 * DAY), ("a later frame", 0)]
)
def test_the_label_source_goes_through_the_guard(
    feature_frame: pd.DataFrame, what: str, shift: int
) -> None:
    late = feature_frame.copy()
    late[config.TIME_COLUMN] = late[config.TIME_COLUMN] + config.TRAIN_END_DT - shift
    assert (late[config.TIME_COLUMN] >= config.TRAIN_END_DT).any(), what
    if shift:
        assert (late[config.TIME_COLUMN] < config.TRAIN_END_DT).any(), what
    with pytest.raises(TrainOnlyError):
        graph_features.label_source(late, lag_days=1)


def test_a_label_for_a_transaction_the_frame_does_not_carry_attaches_to_nothing(
    feature_frame: pd.DataFrame, all_on: graph_features.GraphConfig
) -> None:
    """A truncated frame with the full label source: the labels of absent rows are ignored."""
    source = graph_features.label_source(feature_frame, lag_days=1)
    early = feature_frame[feature_frame[config.TIME_COLUMN] < 10 * DAY]
    out = graph_features.build_graph_features(early, source, all_on)
    assert out.attrs["n_labels_attached"] < source.n_rows
    assert out[graph_features.COMPONENT_N_LABELLED].max() <= len(early)


def test_lag_zero_still_excludes_the_row_itself_and_its_ties() -> None:
    """With no lag, a label matures the moment it is strictly earlier: same instant never counts."""
    frame = pd.DataFrame(
        {
            config.ID_COLUMN: [1, 2, 3],
            config.TARGET: [1, 1, 0],
            config.TIME_COLUMN: [DAY, DAY, DAY + 1],
            "card1": [1.0, 1.0, 1.0],
            "addr1": [np.nan] * 3,
            "P_emaildomain": [None] * 3,
            "R_emaildomain": [None] * 3,
            graph_features.DEVICE_COLUMN: [None] * 3,
            "dist1": [np.nan] * 3,
            config.ENTITY_ID_COLUMN: ["a", "b", "c"],
        }
    )
    source = graph_features.label_source(frame, lag_days=0)
    cfg = graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)
    out = graph_features.build_graph_features(frame, source, cfg)
    assert out[graph_features.COMPONENT_N_LABELLED].tolist() == [0, 0, 2]
    assert out[graph_features.COMPONENT_FRAUD_RATE].tolist()[2] == 1.0


# --- the union-find ------------------------------------------------------------------------------


def test_the_union_find_carries_its_counters_through_a_merge() -> None:
    uf = graph_features.UnionFind(4)
    for node in range(4):
        uf.touch(node, is_transaction=node % 2 == 0)
    uf.label(0, 1)
    uf.union(0, 1)
    uf.union(2, 3)
    assert uf.find(0) == uf.find(1) != uf.find(2) == uf.find(3)
    root = uf.find(0)
    assert (uf.n_nodes[root], uf.n_tx[root], uf.n_labelled[root], uf.y_labelled[root]) == (
        2,
        1,
        1,
        1,
    )
    uf.union(1, 3)
    root = uf.find(2)
    assert (uf.n_nodes[root], uf.n_tx[root], uf.n_labelled[root], uf.y_labelled[root]) == (
        4,
        2,
        1,
        1,
    )
    uf.union(0, 3)
    assert uf.n_nodes[uf.find(0)] == 4, "a repeated union must not double count"


def test_an_untouched_node_has_no_size() -> None:
    uf = graph_features.UnionFind(2)
    assert uf.n_nodes == [0, 0]
    uf.touch(0, is_transaction=True)
    uf.touch(0, is_transaction=True)
    assert uf.n_nodes[0] == 1
    assert uf.n_tx[0] == 1


# --- the static graph the artifact summarises ---------------------------------------------------


def test_the_static_summary_partitions_the_rows(feature_frame: pd.DataFrame) -> None:
    graph = graph_features.static_graph(feature_frame, graph_features.GraphConfig(enabled=True))
    summary = graph_features.component_summary(graph)
    assert summary["n_transactions"] == len(feature_frame)
    assert sum(graph.component_sizes().tolist()) == len(feature_frame)
    assert summary["giant_component"]["n_transactions"] == summary["largest_ten_in_transactions"][0]
    assert summary["size_in_nodes"]["max"] >= summary["size_in_transactions"]["max"]
    assert summary["n_value_nodes_present"] == sum(
        graph.value_degrees(column).size for column in graph.cfg.link_columns
    )


def test_the_static_graph_can_leave_the_anchor_out(feature_frame: pd.DataFrame) -> None:
    cfg = graph_features.GraphConfig(enabled=True, link_columns=("addr1",))
    graph = graph_features.static_graph(feature_frame, cfg)
    summary = graph_features.component_summary(graph)
    assert summary["n_components"] >= feature_frame["addr1"].nunique()


def test_the_static_graph_over_a_single_column_is_that_columns_groups(
    feature_frame: pd.DataFrame,
) -> None:
    cfg = graph_features.GraphConfig(enabled=True, link_columns=("card1",))
    graph = graph_features.static_graph(feature_frame, cfg)
    expected = sorted(feature_frame.groupby("card1").size().tolist(), reverse=True)
    assert graph.component_sizes().tolist() == expected


def test_the_hub_set_is_fitted_on_train_and_names_what_it_removes(
    feature_frame: pd.DataFrame,
) -> None:
    hubs = graph_features.fit_hub_values(feature_frame, ("P_emaildomain", "card1"), share=0.3)
    counts = feature_frame["P_emaildomain"].value_counts()
    expected = [value for value, count in counts.items() if count / len(feature_frame) >= 0.3]
    assert hubs["values"]["P_emaildomain"] == expected
    assert hubs["report"]["P_emaildomain"]["n_hubs"] == len(expected)
    assert hubs["report"]["P_emaildomain"]["n_values"] == counts.size
    assert hubs["values"]["card1"] == []
    assert hubs["share"] == 0.3
    late = feature_frame.copy()
    late[config.TIME_COLUMN] = late[config.TIME_COLUMN] + config.TRAIN_END_DT
    with pytest.raises(TrainOnlyError):
        graph_features.fit_hub_values(late, ("card1",), share=0.3)
    with pytest.raises(ValueError, match="share"):
        graph_features.fit_hub_values(feature_frame, ("card1",), share=0.0)


def test_an_excluded_value_links_nothing_and_reads_as_absent(
    feature_frame: pd.DataFrame, labels: graph_features.LabelSource
) -> None:
    """A hub card has no node: its degree is zero and its component is empty, on every row."""
    cfg = graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)
    heaviest = feature_frame["card1"].value_counts().index[0]
    hubs = {
        "share": 0.0,
        "n_train_rows": len(feature_frame),
        "values": {"card1": [float(heaviest)]},
        "report": {"card1": {"n_hubs": 1, "n_values": 1, "share_rows_through_hubs": 0.0}},
    }
    out = graph_features.build_graph_features(feature_frame, labels, cfg, hubs)
    on_hub = (feature_frame["card1"] == heaviest).to_numpy()
    assert on_hub.sum() > 1
    assert (out.loc[on_hub, graph_features.degree_name("card1")] == 0).all()
    assert (out.loc[on_hub, graph_features.COMPONENT_SIZE] == 0).all()
    assert out.loc[on_hub, graph_features.COMPONENT_FRAUD_RATE].isna().all()
    plain = graph_features.build_graph_features(feature_frame, labels, cfg)
    assert (plain.loc[on_hub, graph_features.degree_name("card1")] > 0).any()


def test_a_hub_masked_static_graph_has_more_components(feature_frame: pd.DataFrame) -> None:
    cfg = graph_features.GraphConfig(enabled=True)
    plain = graph_features.component_summary(graph_features.static_graph(feature_frame, cfg))
    hubs = graph_features.fit_hub_values(feature_frame, cfg.link_columns, share=0.05)
    masked_graph = graph_features.static_graph(feature_frame, cfg, hubs)
    masked = graph_features.component_summary(masked_graph)
    assert masked["n_components"] > plain["n_components"]
    assert masked_graph.hub_report == hubs["report"]
    assert all(column in masked_graph.hub_report for column in cfg.link_columns)


def test_the_link_column_profile_reports_the_heaviest_value(feature_frame: pd.DataFrame) -> None:
    profile = graph_features.link_column_profile(feature_frame, ("P_emaildomain", "card1"))
    counts = feature_frame["P_emaildomain"].value_counts()
    assert profile["P_emaildomain"]["top_value"] == str(counts.index[0])
    assert profile["P_emaildomain"]["top_value_share_of_rows"] == pytest.approx(
        counts.iloc[0] / len(feature_frame)
    )
    assert profile["card1"]["null_rate"] == 0.0


def test_spread_of_nothing_is_nothing() -> None:
    assert graph_features.spread(np.array([], dtype="int64"))["n"] == 0
    assert graph_features.spread(np.array([3, 1, 2], dtype="int64"))["max"] == 3


# --- what the module refuses to offer ----------------------------------------------------------


def test_no_full_dataset_component_function_is_offered() -> None:
    assert "full-dataset components" in graph_features.not_reachable()
    assert not hasattr(graph_features, "connected_components")


def test_the_addr1_per_card_count_is_stage_4s_and_not_rebuilt(built: pd.DataFrame) -> None:
    for description, name in graph_features.PROVIDED_BY_STAGE_4.items():
        assert name in features.feature_names(), description
        assert name not in built.columns


def test_the_day_index_is_stage_3s(feature_frame: pd.DataFrame) -> None:
    expected = np.floor(feature_frame[config.TIME_COLUMN].to_numpy() / DAY).astype("int64")
    assert graph_features.day_of(feature_frame).tolist() == expected.tolist()
