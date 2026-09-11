"""The test the feature stage exists to pass: every feature reads only rows strictly before t.

`fraud_platform.features` computes its columns with prefix sums and searchsorted over a stream
sorted by key and time, which is fast and is not obviously correct. This module recomputes each
feature the slow, obvious way, by filtering the whole frame down to the rows with
`TransactionDT < t` and reducing them in Python, and asserts the two agree.

The reference implementation below shares nothing with the module. It does not use searchsorted,
it does not sort, it does not factorize, and it builds its own notion of a device combination and
a content key out of the raw columns. Two implementations of the same definition agreeing is
worth more than one implementation agreeing with itself.

Three properties are checked and they are not the same property:

1. **Brute force agreement.** For a sample of rows, every feature equals the reference value
   computed over `TransactionDT < t`.
2. **Truncation invariance.** Features computed on the whole stream equal features computed on
   the stream truncated at a time T, for every row before T. A feature that read anything from
   the future, a global mean, a whole-group aggregate, a fold-wide statistic, would break this
   and would not necessarily break the first check on a frame where the future happens to look
   like the past. This is the check that catches the transductive shape ADR 0020 refuses.
3. **Tie exclusion.** A transaction sharing its timestamp with another on the same key does not
   see it. The test also asserts that `side="right"` would see it, so the choice in ADR 0018 is
   visible as a difference rather than asserted as a convention.

Floating point: counts, flags, timestamps and second differences are compared exactly. The
columns that add or average amounts are compared on a tolerance, and the reason is worth being
precise about, because it is not the reason it would usually be.

The module accumulates every running quantity inside its own key: a restarted cumulative sum for
the window totals, Welford for the running mean and spread. That is what makes truncation
invariance exact rather than approximate, and the truncation tests below compare bit for bit with
no tolerance at all. The tolerance here is against the **reference**, which adds a Python list
with `sum()` and takes a two-pass `statistics.stdev`. Adding the same numbers in a different
order gives a different last bit: 662.8500000000001 against 662.85 is a real observed
disagreement between the two, and it is a fact about float64 addition rather than about which
rows were read.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, encoders, features, graph_features

DAY = config.SECONDS_PER_DAY

# Absolute and relative tolerance for the columns that add or average amounts. Both sides compute
# the same quantity by a different route: the module accumulates per key, the reference calls
# `sum()` and `statistics.stdev` over a list. On amounts up to 31,937.391 the two differ in the
# last few places of float64 and nowhere else, and 1e-6 is far below any difference a model could
# read. The truncation tests use no tolerance, which is where the causal claim actually lives.
AMOUNT_TOLERANCE = 1e-6
TOLERANT_COLUMNS: frozenset[str] = frozenset(
    {
        "amt_entity_mean",
        "amt_entity_std",
        "amt_deviation_from_entity_mean",
        "amt_zscore_within_entity",
    }
)

# Velocity sums are prefix differences too, so they take the same tolerance for the same reason.
TOLERANT_PREFIXES: tuple[str, ...] = ("vel_amount_sum_", "vel_amount_mean_")

_NULL = "\x00null"


# --- the reference implementation -----------------------------------------------------------


def _key(row: pd.Series, columns: Sequence[str]) -> tuple[Any, ...]:
    """A tuple key with nulls folded to one token, which is stage 2's rule for identical content."""
    out: list[Any] = []
    for column in columns:
        value = row[column]
        out.append(_NULL if pd.isna(value) else value)
    return tuple(out)


def _device_key(row: pd.Series) -> tuple[Any, ...] | None:
    """The device combination, or None when the row carries no identity record at all."""
    present = [c for c in features.DEVICE_COMBINATION_COLUMNS if c in row.index]
    if not any(not pd.isna(row[c]) for c in present):
        return None
    return _key(row, present)


def content_keys(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    """One tuple key per row, built with a plain loop over the columns.

    The keys are built once per frame rather than once per sampled row. That is a change of
    running time and not of definition: the tuple for a row is the same tuple `_key` would build
    for it, and nothing here consults another row.
    """
    columns = [c for c in features.CONTENT_COLUMNS if c in frame.columns]
    blocks = [
        [_NULL if pd.isna(value) else value for value in frame[column].tolist()]
        for column in columns
    ]
    return list(zip(*blocks, strict=True))


def device_keys(frame: pd.DataFrame) -> list[tuple[Any, ...] | None]:
    """One device combination per row, or None where the row carries no identity record."""
    columns = [c for c in features.DEVICE_COMBINATION_COLUMNS if c in frame.columns]
    raw = [frame[column].tolist() for column in columns]
    out: list[tuple[Any, ...] | None] = []
    for values in zip(*raw, strict=True):
        if all(pd.isna(value) for value in values):
            out.append(None)
        else:
            out.append(tuple(_NULL if pd.isna(value) else value for value in values))
    return out


def _mode(values: list[float]) -> float | None:
    """Most frequent value, ties broken on the smaller value. None on an empty list."""
    if not values:
        return None
    counts: dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def reference_row(
    frame: pd.DataFrame,
    position: int,
    dist_buckets: dict[str, Any],
    cfg: features.FeatureConfig,
    keys: tuple[list[tuple[Any, ...]], list[tuple[Any, ...] | None]] | None = None,
) -> dict[str, Any]:
    """Every feature for one row, computed by filtering the whole frame to `TransactionDT < t`.

    Slow on purpose. Nothing here is incremental, nothing is sorted, and the filter is written
    out as a boolean mask over the whole frame so that "strictly before t" is the only thing the
    reader has to check.
    """
    row = frame.iloc[position]
    moment = int(row[config.TIME_COLUMN])
    before_mask = (frame[config.TIME_COLUMN] < moment).to_numpy(dtype=bool)
    before = frame[before_mask]
    content, devices = keys if keys is not None else (content_keys(frame), device_keys(frame))
    positions_before = np.flatnonzero(before_mask)
    out: dict[str, Any] = {}

    grain_column = {"entity": config.ENTITY_ID_COLUMN, "card": "card1"}

    for grain in cfg.velocity_grains:
        column = grain_column[grain]
        same_key = before[before[column] == row[column]]
        for suffix, window in cfg.windows:
            inside = same_key[same_key[config.TIME_COLUMN] >= moment - window]
            amounts = inside[config.AMOUNT_COLUMN].tolist()
            out[features.velocity_name("count", suffix, grain)] = len(amounts)
            out[features.velocity_name("amount_sum", suffix, grain)] = float(sum(amounts))
            out[features.velocity_name("amount_mean", suffix, grain)] = (
                float(sum(amounts) / len(amounts)) if amounts else np.nan
            )

    prior = before[before[config.ENTITY_ID_COLUMN] == row[config.ENTITY_ID_COLUMN]]
    prior_ts = sorted(int(value) for value in prior[config.TIME_COLUMN])
    since = float(moment - prior_ts[-1]) if prior_ts else np.nan
    mean_gap = (
        float((prior_ts[-1] - prior_ts[0]) / (len(prior_ts) - 1)) if len(prior_ts) >= 2 else np.nan
    )
    out["rec_seconds_since_prev"] = since
    out["rec_prior_count"] = len(prior_ts)
    out["rec_mean_gap"] = mean_gap
    out["rec_gap_deviation"] = since - mean_gap

    prior_amounts = [float(value) for value in prior[config.AMOUNT_COLUMN]]
    amount = float(row[config.AMOUNT_COLUMN])
    mean = float(sum(prior_amounts) / len(prior_amounts)) if prior_amounts else np.nan
    std = float(statistics.stdev(prior_amounts)) if len(prior_amounts) >= 2 else np.nan
    out["amt_entity_mean"] = mean
    out["amt_entity_std"] = std
    out["amt_deviation_from_entity_mean"] = amount - mean
    out["amt_zscore_within_entity"] = (
        (amount - mean) / std if len(prior_amounts) >= 2 and std > 0 else np.nan
    )

    repeat_window = prior[prior[config.TIME_COLUMN] >= moment - cfg.amount_repeat_window_seconds]
    repeat_amounts = [float(value) for value in repeat_window[config.AMOUNT_COLUMN]]
    out["amt_repeat_within_24h"] = (
        (1.0 if amount in repeat_amounts else 0.0) if repeat_amounts else np.nan
    )

    prior_entity = frame[config.ENTITY_ID_COLUMN].to_numpy()
    own_entity = prior_entity[position]
    prior_positions = [i for i in positions_before if prior_entity[i] == own_entity]
    prior_devices = [devices[i] for i in prior_positions if devices[i] is not None]
    own_device = devices[position]
    out["dev_n_distinct_so_far"] = len(set(prior_devices))
    out["dev_combination_is_new"] = (
        (0.0 if own_device in set(prior_devices) else 1.0)
        if own_device is not None and prior_devices
        else np.nan
    )

    card_prior = before[before["card1"] == row["card1"]]
    for column, name in (
        ("addr1", "geo_addr1_differs_from_card_mode"),
        ("addr2", "geo_addr2_differs_from_card_mode"),
    ):
        seen = [float(value) for value in card_prior[column] if not pd.isna(value)]
        modal = _mode(seen)
        own = row[column]
        out[name] = (
            (0.0 if float(own) == modal else 1.0)
            if modal is not None and not pd.isna(own)
            else np.nan
        )
    out["geo_n_distinct_addr1_on_card"] = len(
        {float(value) for value in card_prior["addr1"] if not pd.isna(value)}
    )
    edges = list(dist_buckets["edges"])
    own_dist = row[dist_buckets["column"]]
    out["geo_dist1_bucket"] = (
        np.nan if pd.isna(own_dist) else float(sum(1 for edge in edges if edge <= float(own_dist)))
    )

    own_content = content[position]
    all_ts = frame[config.TIME_COLUMN].to_numpy(dtype="int64")
    identical = [int(all_ts[i]) for i in positions_before if content[i] == own_content]
    out["dup_prior_count"] = len(identical)
    out["dup_seconds_since_prev"] = float(moment - max(identical)) if identical else np.nan
    out["dup_prior_count_1h"] = sum(
        1 for value in identical if value >= moment - cfg.duplicate_window_seconds
    )
    return out


# --- helpers ---------------------------------------------------------------------------------


def _tolerant(name: str) -> bool:
    return name in TOLERANT_COLUMNS or name.startswith(TOLERANT_PREFIXES)


def _assert_equal(name: str, built: Any, expected: Any, position: int) -> None:
    built_is_null = built is None or (isinstance(built, float) and np.isnan(built))
    expected_is_null = expected is None or (isinstance(expected, float) and np.isnan(expected))
    assert built_is_null == expected_is_null, (
        f"{name} at row {position}: null disagreement, built {built!r}, reference {expected!r}"
    )
    if built_is_null:
        return
    if _tolerant(name):
        assert float(built) == pytest.approx(
            float(expected), rel=AMOUNT_TOLERANCE, abs=AMOUNT_TOLERANCE
        ), f"{name} at row {position}: {built} against {expected}"
    else:
        assert float(built) == float(expected), (
            f"{name} at row {position}: {built} against {expected}"
        )


def _sample_positions(frame: pd.DataFrame, built: pd.DataFrame, n_random: int) -> list[int]:
    """A sample that is not purely random: every branch the features have gets a row.

    Brute force is slow, so the sample is small. Choosing it at random would leave the interesting
    rows out most of the time, so the cold rows, the tie rows and the rows with a repeat are
    included by construction and the rest is drawn with a fixed seed.
    """
    positions: set[int] = set()
    cold = np.flatnonzero(built["rec_prior_count"].to_numpy() == 0)
    positions.update(int(p) for p in cold[:10])
    ts = frame[config.TIME_COLUMN].to_numpy()
    tied = np.flatnonzero(pd.Series(ts).duplicated(keep=False).to_numpy())
    positions.update(int(p) for p in tied[:20])
    for column in ("dup_prior_count", "vel_count_24h_entity", "dev_n_distinct_so_far"):
        if column in built.columns:
            hot = np.flatnonzero(built[column].to_numpy() > 0)
            positions.update(int(p) for p in hot[:10])
    rng = np.random.default_rng(config.SEED)
    positions.update(
        int(p) for p in rng.choice(len(frame), size=min(n_random, len(frame)), replace=False)
    )
    return sorted(positions)


def _build(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], features.FeatureConfig]:
    """Build the features for a frame, fitting the one fitted object on its own early rows."""
    cfg = features.DEFAULT_CONFIG
    train = frame[frame[config.TIME_COLUMN] < int(frame[config.TIME_COLUMN].quantile(0.7))]
    buckets = features.fit_dist_buckets(train)
    return features.build_features(frame, buckets, cfg), buckets, cfg


# --- 1. brute force agreement ---------------------------------------------------------------


def test_every_feature_matches_a_brute_force_recomputation(
    feature_frame: pd.DataFrame,
    built_features: pd.DataFrame,
    dist_buckets: dict[str, Any],
) -> None:
    frame, built, buckets, cfg = (
        feature_frame,
        built_features,
        dist_buckets,
        features.DEFAULT_CONFIG,
    )
    positions = _sample_positions(frame, built, n_random=60)
    assert len(positions) >= 60

    keys = (content_keys(frame), device_keys(frame))
    for position in positions:
        expected = reference_row(frame, position, buckets, cfg, keys)
        assert set(expected) == set(built.columns), (
            "the reference implementation and the module disagree on which features exist: "
            f"{sorted(set(expected) ^ set(built.columns))}"
        )
        for name, value in expected.items():
            _assert_equal(name, built.iloc[position][name], value, position)


def test_the_sample_covers_the_branches_it_claims_to(
    feature_frame: pd.DataFrame, built_features: pd.DataFrame
) -> None:
    """A sample that happened to hold no tie and no cold row would pass the test above for free."""
    frame, built = feature_frame, built_features
    positions = _sample_positions(frame, built, n_random=60)
    sampled = built.iloc[positions]
    assert (sampled["rec_prior_count"] == 0).any(), "no cold row in the sample"
    assert (sampled["rec_prior_count"] > 0).any(), "no warm row in the sample"
    assert (sampled["dev_n_distinct_so_far"] > 0).any(), "no row with device history"
    assert sampled["amt_repeat_within_24h"].notna().any(), "no row with a repeat window"
    assert sampled["geo_addr1_differs_from_card_mode"].notna().any(), "no row with an addr mode"
    ts = frame[config.TIME_COLUMN].to_numpy()
    assert pd.Series(ts[positions]).isin(ts[pd.Series(ts).duplicated(keep=False)]).any(), (
        "no tied timestamp in the sample"
    )


# --- 2. truncation invariance ----------------------------------------------------------------


def test_truncating_the_stream_does_not_change_earlier_rows(
    feature_frame: pd.DataFrame, built_features: pd.DataFrame, dist_buckets: dict[str, Any]
) -> None:
    """The property a global statistic breaks and a brute-force check can miss.

    A feature that read a whole-group aggregate would still agree with a reference that read the
    same whole-group aggregate. It would not survive having the future removed.

    The comparison is exact, with no tolerance, and that is the point of accumulating inside each
    key. An earlier version of this module read its window sums off a global `np.cumsum`, and
    this test failed on the real frame by 2e-4 on the running spread of entities that repeat one
    amount. The values were immaterial and the dependency was real.
    """
    frame, full, buckets, cfg = (
        feature_frame,
        built_features,
        dist_buckets,
        features.DEFAULT_CONFIG,
    )
    boundary = int(frame[config.TIME_COLUMN].quantile(0.6))
    early = frame[frame[config.TIME_COLUMN] < boundary]
    truncated = features.build_features(early, buckets, cfg)

    assert 0 < len(early) < len(frame)
    for column in full.columns:
        pd.testing.assert_series_equal(
            truncated[column],
            full.loc[early.index, column],
            check_names=False,
            check_dtype=True,
        )


def test_appending_a_later_row_does_not_change_the_rows_before_it(
    feature_frame: pd.DataFrame, built_features: pd.DataFrame, dist_buckets: dict[str, Any]
) -> None:
    """The same property from the other side: adding the future must not rewrite the past."""
    frame, full, buckets, cfg = (
        feature_frame,
        built_features,
        dist_buckets,
        features.DEFAULT_CONFIG,
    )
    extra = frame.iloc[[0]].copy()
    extra[config.TIME_COLUMN] = int(frame[config.TIME_COLUMN].max()) + 10 * DAY
    extra.index = [len(frame)]
    extended = features.build_features(
        pd.concat([frame, extra]).sort_values(config.TIME_COLUMN, kind="stable"), buckets, cfg
    )
    for column in full.columns:
        pd.testing.assert_series_equal(
            extended.loc[frame.index, column], full[column], check_names=False
        )


# --- 3. ties -----------------------------------------------------------------------------------


def test_a_transaction_does_not_see_a_simultaneous_one_on_the_same_key(
    feature_frame: pd.DataFrame, built_features: pd.DataFrame
) -> None:
    frame, built = feature_frame, built_features
    ts = frame[config.TIME_COLUMN].to_numpy()
    entity = frame[config.ENTITY_ID_COLUMN].to_numpy()
    tied = pd.DataFrame({"ts": ts, "entity": entity}).duplicated(keep=False).to_numpy()
    assert tied.sum() >= 2, "the frame is meant to contain ties on one entity"

    for position in np.flatnonzero(tied):
        moment = int(ts[position])
        same = (entity == entity[position]) & (ts == moment)
        others = int(same.sum()) - 1
        assert others >= 1
        strictly_before = int(((entity == entity[position]) & (ts < moment)).sum())
        assert int(built.iloc[position]["rec_prior_count"]) == strictly_before, (
            "a tied transaction was counted as prior history"
        )


def test_side_right_would_have_counted_the_tie() -> None:
    """The choice in ADR 0018 is a difference, not a convention. This is the difference."""
    ts = np.array([100, 200, 200, 200, 500], dtype="int64")
    lo, hi = features.trailing_bounds(ts, window=1_000)
    left_counts = (hi - lo).tolist()
    right_hi = np.searchsorted(ts, ts, side="right")
    right_counts = (right_hi - lo).tolist()
    assert left_counts == [0, 1, 1, 1, 4]
    assert right_counts == [1, 4, 4, 4, 5]
    assert left_counts != right_counts


# --- the grouped primitive against the single-key one ----------------------------------------


def test_the_grouped_bounds_equal_a_loop_over_the_single_key_primitive() -> None:
    rng = np.random.default_rng(config.SEED)
    codes = rng.integers(0, 7, 400).astype("int64")
    ts = rng.integers(0, 5 * DAY, 400).astype("int64")
    order = np.lexsort((ts, codes))
    codes, ts = codes[order], ts[order]

    for window in (600, 3_600, 86_400, 10 * DAY):
        lo, hi = features.grouped_trailing_bounds(codes, ts, window)
        counts = hi - lo
        expected = np.empty_like(counts)
        for code in np.unique(codes):
            mask = codes == code
            own_lo, own_hi = features.trailing_bounds(ts[mask], window)
            expected[mask] = own_hi - own_lo
        assert (counts == expected).all(), f"window {window} disagrees with the primitive"


def test_a_window_wider_than_the_stream_cannot_reach_into_the_previous_key() -> None:
    codes = np.array([0, 0, 1, 1], dtype="int64")
    ts = np.array([10, 20, 30, 40], dtype="int64")
    lo, hi = features.grouped_trailing_bounds(codes, ts, window=1_000_000)
    assert (hi - lo).tolist() == [0, 1, 0, 1]


def test_prior_bounds_are_the_expanding_case_of_the_trailing_ones() -> None:
    codes = np.array([0, 0, 0, 1], dtype="int64")
    ts = np.array([10, 10, 40, 5], dtype="int64")
    lo, hi = features.grouped_prior_bounds(codes, ts)
    assert (hi - lo).tolist() == [0, 0, 2, 0]


def test_packing_refuses_a_timestamp_it_cannot_pack() -> None:
    codes = np.array([0, 1], dtype="int64")
    with pytest.raises(ValueError, match="preserve order"):
        features.grouped_prior_bounds(codes, np.array([0, 1 << 33], dtype="int64"))
    with pytest.raises(ValueError, match="non-negative"):
        features.grouped_prior_bounds(
            np.array([-1, 0], dtype="int64"), np.array([1, 2], dtype="int64")
        )


# --- on the real data ------------------------------------------------------------------------


@pytest.mark.needs_data
def test_brute_force_agreement_on_the_real_frame() -> None:
    """The same check on the file, on a sample, because the frame is 590,540 rows.

    Every row costs a full scan of the frame in the reference implementation, so this samples
    rather than sweeping. The generated frame above is where coverage of the branches comes from;
    this is where coverage of the actual value distributions comes from.
    """
    from fraud_platform import data_loader

    columns = [
        config.ID_COLUMN,
        config.TARGET,
        config.TIME_COLUMN,
        config.AMOUNT_COLUMN,
        "card1",
        "card2",
        "addr1",
        "addr2",
        "dist1",
        "D1",
        "ProductCD",
        "P_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
    ]
    frame = data_loader.add_entity_key(data_loader.load_raw(columns=columns))
    frame = encoders.add_normalised_free_text(frame)
    # The first 20 days of the training window. The reference is O(rows) per sampled row, so the
    # frame it scans has to be small enough that the check finishes, and taking a prefix of the
    # stream rather than a slice out of the middle keeps every entity's history inside it intact
    # from its own first row.
    frame = frame[frame[config.TIME_COLUMN] < 20 * DAY].reset_index(drop=True)
    assert len(frame) > 50_000

    built, buckets, cfg = _build(frame)
    rng = np.random.default_rng(config.SEED)
    warm = np.flatnonzero(built["rec_prior_count"].to_numpy() > 0)
    cold = np.flatnonzero(built["rec_prior_count"].to_numpy() == 0)
    positions = sorted(
        {int(p) for p in rng.choice(warm, size=30, replace=False)}
        | {int(p) for p in rng.choice(cold, size=10, replace=False)}
    )

    keys = (content_keys(frame), device_keys(frame))
    for position in positions:
        expected = reference_row(frame, position, buckets, cfg, keys)
        for name, value in expected.items():
            _assert_equal(name, built.iloc[position][name], value, position)


@pytest.mark.needs_data
def test_truncation_invariance_on_the_real_frame() -> None:
    from fraud_platform import data_loader

    columns = [
        config.ID_COLUMN,
        config.TARGET,
        config.TIME_COLUMN,
        config.AMOUNT_COLUMN,
        "card1",
        "card2",
        "addr1",
        "addr2",
        "dist1",
        "D1",
        "ProductCD",
        "P_emaildomain",
        "DeviceType",
        "DeviceInfo",
        "id_30",
        "id_31",
    ]
    frame = data_loader.add_entity_key(data_loader.load_raw(columns=columns))
    frame = encoders.add_normalised_free_text(frame)
    train, val, test = data_loader.time_based_split(frame)
    del test

    full, buckets, cfg = _build(frame)
    train_only_features = features.build_features(train, buckets, cfg)
    for column in full.columns:
        pd.testing.assert_series_equal(
            train_only_features[column],
            full.loc[train.index, column],
            check_names=False,
        )
    assert len(val) > 0


# --- 4. the graph ------------------------------------------------------------------------------
#
# Stage 5 adds structural features over the graph that shared values draw between transactions,
# and the same three properties are checked for them. The reference below rebuilds the graph from
# scratch with networkx over the rows with `TransactionDT < t`, which shares nothing with the
# union-find in `fraud_platform.graph_features`: no incremental state, no node id arithmetic, no
# maturation pointer. The lag is applied by filtering on the day index in the plainest way.


def _graph_before(
    before: pd.DataFrame, link_columns: Sequence[str]
) -> tuple[nx.Graph, dict[Any, Any]]:
    """The transaction-to-value graph over `before`, plus the entity of every transaction node."""
    graph = nx.Graph()
    entity_of: dict[Any, Any] = {}
    ids = before[config.ID_COLUMN].tolist()
    entities = before[config.ENTITY_ID_COLUMN].tolist()
    values = {column: before[column].tolist() for column in link_columns}
    for i, transaction_id in enumerate(ids):
        node = ("tx", transaction_id)
        graph.add_node(node)
        entity_of[node] = entities[i]
        for column in link_columns:
            value = values[column][i]
            if not pd.isna(value):
                graph.add_edge(node, (column, value))
    return graph, entity_of


def graph_reference_row(
    frame: pd.DataFrame,
    position: int,
    labels: graph_features.LabelSource,
    cfg: graph_features.GraphConfig,
) -> dict[str, Any]:
    """Every graph feature for one row, from a graph rebuilt over `TransactionDT < t`."""
    row = frame.iloc[position]
    moment = int(row[config.TIME_COLUMN])
    before = frame[frame[config.TIME_COLUMN] < moment]
    graph, entity_of = _graph_before(before, cfg.link_columns)
    out: dict[str, Any] = {}

    anchor = (graph_features.ANCHOR_COLUMN, row[graph_features.ANCHOR_COLUMN])
    component: set[Any] = nx.node_connected_component(graph, anchor) if anchor in graph else set()
    transactions = [node for node in component if node[0] == "tx"]
    out[graph_features.COMPONENT_SIZE] = len(transactions)

    for column in graph_features.DEGREE_COLUMNS:
        value = row[column]
        node = (column, value)
        out[graph_features.degree_name(column)] = (
            np.nan if pd.isna(value) else float(graph.degree(node)) if node in graph else 0.0
        )

    device = (graph_features.DEVICE_COLUMN, row[graph_features.DEVICE_COLUMN])
    if pd.isna(device[1]):
        out[graph_features.ENTITIES_ON_DEVICE] = np.nan
    elif device in graph:
        out[graph_features.ENTITIES_ON_DEVICE] = float(
            len({entity_of[neighbour] for neighbour in graph.neighbors(device)})
        )
    else:
        out[graph_features.ENTITIES_ON_DEVICE] = 0.0

    # The lag, applied as a filter: a label counts when its transaction's day is at least
    # lag_days before this row's day. Labels come from the LabelSource alone.
    label_of = {
        int(i): (int(t), int(y)) for i, t, y in zip(labels.ids, labels.ts, labels.y, strict=True)
    }
    day = moment // DAY
    own_entity = row[config.ENTITY_ID_COLUMN]
    matured: list[tuple[int, Any]] = []
    for node in transactions:
        labelled = label_of.get(int(node[1]))
        if labelled is None:
            continue
        label_ts, label_y = labelled
        if label_ts // DAY <= day - labels.lag_days:
            matured.append((label_y, entity_of[node]))
    others = [y for y, entity in matured if entity != own_entity]
    out[graph_features.COMPONENT_N_LABELLED] = len(matured)
    out[graph_features.COMPONENT_FRAUD_RATE] = (
        sum(y for y, _ in matured) / len(matured) if matured else np.nan
    )
    out[graph_features.COMPONENT_FRAUD_RATE_OTHERS] = (
        sum(others) / len(others) if others else np.nan
    )
    return out


def distinct_addr1_per_card_from_the_graph(frame: pd.DataFrame, position: int) -> int:
    """The typed degree of the card1 node towards addr1 nodes, over `TransactionDT < t`.

    This is the graph reading of "distinct addr1 values seen for this card1 so far". Stage 4
    ships that number as `geo_n_distinct_addr1_on_card`, so the module does not emit it twice; the
    test below asserts the two readings agree.
    """
    row = frame.iloc[position]
    before = frame[frame[config.TIME_COLUMN] < int(row[config.TIME_COLUMN])]
    graph, _ = _graph_before(before, ("card1", "addr1"))
    card = ("card1", row["card1"])
    if card not in graph:
        return 0
    addr_nodes = {
        neighbour
        for transaction in graph.neighbors(card)
        for neighbour in graph.neighbors(transaction)
        if neighbour[0] == "addr1"
    }
    return len(addr_nodes)


@pytest.fixture(scope="module")
def graph_cfg() -> graph_features.GraphConfig:
    return graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)


@pytest.fixture(scope="module")
def graph_labels(feature_frame: pd.DataFrame) -> graph_features.LabelSource:
    """Labels from the first 70 percent of the generated stream, maturing under a 3 day lag.

    A prefix rather than every row, so the graph holds transactions whose label the feature may
    not read, which is the state a validation row is in. The lag is a test parameter; the stage's
    own lag is read from reports/encoding_spec.json by the driver.
    """
    boundary = int(feature_frame[config.TIME_COLUMN].quantile(0.7))
    train = feature_frame[feature_frame[config.TIME_COLUMN] < boundary]
    return graph_features.label_source(train, lag_days=3)


@pytest.fixture(scope="module")
def built_graph(
    feature_frame: pd.DataFrame,
    graph_labels: graph_features.LabelSource,
    graph_cfg: graph_features.GraphConfig,
) -> pd.DataFrame:
    return graph_features.build_graph_features(feature_frame, graph_labels, graph_cfg)


def _graph_sample_positions(frame: pd.DataFrame, built: pd.DataFrame, n_random: int) -> list[int]:
    """Cold cards, tied timestamps, rows with a device, rows with a matured label, plus a draw."""
    positions: set[int] = set()
    cold = np.flatnonzero(built[graph_features.COMPONENT_SIZE].to_numpy() == 0)
    positions.update(int(p) for p in cold[:10])
    ts = frame[config.TIME_COLUMN].to_numpy()
    tied = np.flatnonzero(pd.Series(ts).duplicated(keep=False).to_numpy())
    positions.update(int(p) for p in tied[:20])
    with_device = np.flatnonzero(built[graph_features.ENTITIES_ON_DEVICE].notna().to_numpy())
    positions.update(int(p) for p in with_device[:10])
    labelled = np.flatnonzero(built[graph_features.COMPONENT_N_LABELLED].to_numpy() > 0)
    positions.update(int(p) for p in labelled[:10])
    positions.update(int(p) for p in labelled[-10:])
    rng = np.random.default_rng(config.SEED)
    positions.update(
        int(p) for p in rng.choice(len(frame), size=min(n_random, len(frame)), replace=False)
    )
    return sorted(positions)


def test_every_graph_feature_matches_a_brute_force_rebuild(
    feature_frame: pd.DataFrame,
    built_graph: pd.DataFrame,
    graph_labels: graph_features.LabelSource,
    graph_cfg: graph_features.GraphConfig,
) -> None:
    positions = _graph_sample_positions(feature_frame, built_graph, n_random=60)
    assert len(positions) >= 60
    for position in positions:
        expected = graph_reference_row(feature_frame, position, graph_labels, graph_cfg)
        assert set(expected) == set(built_graph.columns), (
            f"the reference and the module disagree on which graph features exist: "
            f"{sorted(set(expected) ^ set(built_graph.columns))}"
        )
        for name, value in expected.items():
            _assert_equal(name, built_graph.iloc[position][name], value, position)


def test_the_graph_sample_covers_the_branches_it_claims_to(
    feature_frame: pd.DataFrame, built_graph: pd.DataFrame
) -> None:
    positions = _graph_sample_positions(feature_frame, built_graph, n_random=60)
    sampled = built_graph.iloc[positions]
    assert (sampled[graph_features.COMPONENT_SIZE] == 0).any(), "no cold card in the sample"
    assert (sampled[graph_features.COMPONENT_SIZE] > 0).any(), "no warm card in the sample"
    assert sampled[graph_features.ENTITIES_ON_DEVICE].notna().any(), "no row with a device"
    assert sampled[graph_features.ENTITIES_ON_DEVICE].isna().any(), "no row without a device"
    assert (sampled[graph_features.COMPONENT_N_LABELLED] > 0).any(), "no matured label"
    assert (sampled[graph_features.COMPONENT_N_LABELLED] == 0).any(), "no row before any label"
    assert sampled[graph_features.COMPONENT_FRAUD_RATE_OTHERS].notna().any()
    ts = feature_frame[config.TIME_COLUMN].to_numpy()
    assert pd.Series(ts[positions]).isin(ts[pd.Series(ts).duplicated(keep=False)]).any(), (
        "no tied timestamp in the sample"
    )


def test_truncating_the_stream_does_not_change_earlier_graph_rows(
    feature_frame: pd.DataFrame,
    built_graph: pd.DataFrame,
    graph_labels: graph_features.LabelSource,
    graph_cfg: graph_features.GraphConfig,
) -> None:
    """The property a full-dataset component computation breaks first.

    A component computed over the whole stream joins a row to every later row it will ever share
    a value with, so truncating the stream would shrink it. The comparison is exact: every count
    is an integer and every rate is a ratio of two integers computed the same way on both sides.
    """
    boundary = int(feature_frame[config.TIME_COLUMN].quantile(0.6))
    early = feature_frame[feature_frame[config.TIME_COLUMN] < boundary]
    truncated = graph_features.build_graph_features(early, graph_labels, graph_cfg)
    assert 0 < len(early) < len(feature_frame)
    for column in built_graph.columns:
        pd.testing.assert_series_equal(
            truncated[column],
            built_graph.loc[early.index, column],
            check_names=False,
            check_dtype=True,
        )


def test_appending_a_later_row_does_not_change_earlier_graph_rows(
    feature_frame: pd.DataFrame,
    built_graph: pd.DataFrame,
    graph_labels: graph_features.LabelSource,
    graph_cfg: graph_features.GraphConfig,
) -> None:
    extra = feature_frame.iloc[[0]].copy()
    extra[config.TIME_COLUMN] = int(feature_frame[config.TIME_COLUMN].max()) + 10 * DAY
    extra[config.ID_COLUMN] = int(feature_frame[config.ID_COLUMN].max()) + 1
    extra.index = [len(feature_frame)]
    extended = graph_features.build_graph_features(
        pd.concat([feature_frame, extra]).sort_values(config.TIME_COLUMN, kind="stable"),
        graph_labels,
        graph_cfg,
    )
    for column in built_graph.columns:
        pd.testing.assert_series_equal(
            extended.loc[feature_frame.index, column], built_graph[column], check_names=False
        )


def test_a_transaction_does_not_see_a_simultaneous_one_sharing_a_node(
    feature_frame: pd.DataFrame, built_graph: pd.DataFrame
) -> None:
    """The tie rule on the graph: rows at one instant on one card do not count each other."""
    ts = feature_frame[config.TIME_COLUMN].to_numpy()
    card = feature_frame["card1"].to_numpy()
    tied = pd.DataFrame({"ts": ts, "card": card}).duplicated(keep=False).to_numpy()
    assert tied.sum() >= 2, "the frame is meant to contain ties on one card"
    degree = built_graph[graph_features.degree_name("card1")].to_numpy()
    for position in np.flatnonzero(tied):
        strictly_before = int(((card == card[position]) & (ts < ts[position])).sum())
        assert int(degree[position]) == strictly_before, (
            "a tied transaction was counted into the card's degree"
        )


def test_a_label_matures_only_after_the_lag() -> None:
    """Two rows on one card. The first is labelled fraud; the second reads it only once the lag
    has passed, whatever the graph looks like in between."""
    lag = 5
    first_day = 10
    base = {
        config.ID_COLUMN: [1, 2, 3],
        config.TARGET: [1, 0, 0],
        "card1": [7.0, 7.0, 7.0],
        "addr1": [1.0, 1.0, 1.0],
        "P_emaildomain": [None, None, None],
        "R_emaildomain": [None, None, None],
        graph_features.DEVICE_COLUMN: [None, None, None],
        "dist1": [np.nan, np.nan, np.nan],
        config.ENTITY_ID_COLUMN: ["a", "b", "b"],
    }
    frame = pd.DataFrame(base)
    frame[config.TIME_COLUMN] = np.array(
        [first_day * DAY, (first_day + lag - 1) * DAY + 100, (first_day + lag) * DAY + 100],
        dtype="int64",
    )
    labels = graph_features.label_source(frame.iloc[[0]], lag_days=lag)
    cfg = graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)
    built = graph_features.build_graph_features(frame, labels, cfg)
    rate = built[graph_features.COMPONENT_FRAUD_RATE].tolist()
    assert np.isnan(rate[0]), "a row cannot read a label from its own edges"
    assert np.isnan(rate[1]), "the label is one day short of maturing"
    assert rate[2] == 1.0
    assert built[graph_features.COMPONENT_N_LABELLED].tolist() == [0, 0, 1]
    # Entity b is not entity a, so the others rate reads the same label; a row of entity a would not.
    assert built[graph_features.COMPONENT_FRAUD_RATE_OTHERS].tolist()[2] == 1.0


def test_the_label_source_refuses_rows_outside_the_training_window(
    feature_frame: pd.DataFrame,
) -> None:
    from fraud_platform.data_loader import TrainOnlyError

    late = feature_frame.copy()
    late[config.TIME_COLUMN] = late[config.TIME_COLUMN] + config.TRAIN_END_DT
    with pytest.raises(TrainOnlyError):
        graph_features.label_source(late, lag_days=3)
    with pytest.raises(TrainOnlyError):
        graph_features.label_source(feature_frame.iloc[0:0], lag_days=3)


def test_the_graph_reading_of_distinct_addr1_per_card_is_stage_4s_column(
    feature_frame: pd.DataFrame, built_features: pd.DataFrame
) -> None:
    """One number, one source. The brief asks the graph for the distinct addr1 values a card has
    used; stage 4 already emits it, and this asserts the graph's reading is the same number."""
    stage_4 = built_features["geo_n_distinct_addr1_on_card"].to_numpy()
    rng = np.random.default_rng(config.SEED)
    positions = sorted(
        {int(p) for p in rng.choice(len(feature_frame), size=60, replace=False)}
        | {int(p) for p in np.flatnonzero(stage_4 > 1)[:20]}
    )
    assert any(stage_4[p] > 1 for p in positions)
    for position in positions:
        assert distinct_addr1_per_card_from_the_graph(feature_frame, position) == int(
            stage_4[position]
        ), f"row {position}"


@pytest.mark.needs_data
def test_graph_brute_force_agreement_on_the_real_frame() -> None:
    """The same rebuild on a prefix of the file, on a sample, because the reference is O(rows)."""
    from fraud_platform import data_loader

    columns = [
        config.ID_COLUMN,
        config.TARGET,
        config.TIME_COLUMN,
        "card1",
        "addr1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "dist1",
        "D1",
    ]
    frame = data_loader.add_entity_key(data_loader.load_raw(columns=columns))
    frame = encoders.add_normalised_free_text(frame)
    frame = frame[frame[config.TIME_COLUMN] < 12 * DAY].reset_index(drop=True)
    assert len(frame) > 30_000
    labels = graph_features.label_source(frame[frame[config.TIME_COLUMN] < 9 * DAY], lag_days=3)
    cfg = graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)
    built = graph_features.build_graph_features(frame, labels, cfg)

    rng = np.random.default_rng(config.SEED)
    warm = np.flatnonzero(built[graph_features.COMPONENT_N_LABELLED].to_numpy() > 0)
    cold = np.flatnonzero(built[graph_features.COMPONENT_SIZE].to_numpy() == 0)
    with_device = np.flatnonzero(built[graph_features.ENTITIES_ON_DEVICE].notna().to_numpy())
    positions = sorted(
        {int(p) for p in rng.choice(warm, size=12, replace=False)}
        | {int(p) for p in rng.choice(cold, size=6, replace=False)}
        | {int(p) for p in rng.choice(with_device, size=6, replace=False)}
    )
    for position in positions:
        expected = graph_reference_row(frame, position, labels, cfg)
        for name, value in expected.items():
            _assert_equal(name, built.iloc[position][name], value, position)


@pytest.mark.needs_data
def test_graph_truncation_invariance_on_the_real_frame() -> None:
    from fraud_platform import data_loader

    columns = [
        config.ID_COLUMN,
        config.TARGET,
        config.TIME_COLUMN,
        "card1",
        "addr1",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceInfo",
        "dist1",
        "D1",
    ]
    frame = data_loader.add_entity_key(data_loader.load_raw(columns=columns))
    frame = encoders.add_normalised_free_text(frame)
    train, val, _ = data_loader.time_based_split(frame)
    labels = graph_features.label_source(train, lag_days=3)
    cfg = graph_features.GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)
    full = graph_features.build_graph_features(frame, labels, cfg)
    train_only_features = graph_features.build_graph_features(train, labels, cfg)
    for column in full.columns:
        pd.testing.assert_series_equal(
            train_only_features[column], full.loc[train.index, column], check_names=False
        )
    assert len(val) > 0
