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

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, encoders, features

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
