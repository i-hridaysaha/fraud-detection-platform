"""The parity test: the batch pipeline and the online store agree on every feature of every row.

`fraud_platform.features` and `fraud_platform.graph_features` compute the features over a whole
stream with prefix sums, searchsorted and a forward scan in timestamp blocks.
`fraud_platform.online_store` computes the same definitions one transaction at a time from
per-key state, on an in-process backend and on Redis. This module runs the same chronological
stream through both and asserts the feature rows are equal, which is the check the design
document promised in stage 0: "the same transaction scored offline and online must produce the
same feature vector".

**The tolerance, stated.** Counts, flags and second differences are compared exactly, null mask
included. The columns that add or average amounts are compared at a relative tolerance of
`FLOAT_TOLERANCE`, and the reason it exists is not the usual one: the store carries the batch
path's running prefix on every ring entry and folds the amount moments in the batch path's
order, so the two are expected to agree bit for bit, and `scripts/parity.py` records the largest
difference actually observed. The tolerance is here so the assertion states what it accepts
rather than relying on a property it has not measured.

**The stream has the cases in it.** The generated frame from `conftest.py` carries tied
timestamps on one entity and on one card, cold keys, repeated amounts inside a day, rows with no
identity record and rows with part of one, and repeated content. A test below asserts each of
those branches was reached, so an agreement is not an agreement on the easy rows only.

**The test can fail.** Stage 8 broke the store's upper window bound from `< t` to `<= t` and the
batch path's tie exclusion the other way round, ran this module, and read the failures. The
notes for the stage carry both outputs. A parity test that has never failed is decoration.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, encoders, features, graph_features, online_store

# Relative tolerance on the columns that add or average amounts. Every other column is exact.
FLOAT_TOLERANCE = 1e-9

TOLERANT_COLUMNS: frozenset[str] = frozenset(
    {
        "amt_entity_mean",
        "amt_entity_std",
        "amt_deviation_from_entity_mean",
        "amt_zscore_within_entity",
    }
)
TOLERANT_PREFIXES: tuple[str, ...] = ("vel_amount_sum_", "vel_amount_mean_")

REDIS_URL = os.environ.get("FRAUD_REDIS_URL", "redis://localhost:6379/0")
BACKENDS: tuple[str, ...] = ("memory", "fakeredis", "redis")


# --- backends -----------------------------------------------------------------------------------


def _redis_client() -> Any | None:
    """A client on a reachable server, or None. A server nobody started is a skip, not a fail."""
    import redis

    client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.2, socket_timeout=1.0)
    try:
        client.ping()
    except redis.exceptions.RedisError:
        return None
    return client


def make_backend(name: str) -> online_store.StateBackend:
    if name == "memory":
        return online_store.MemoryBackend()
    if name == "fakeredis":
        import fakeredis

        return online_store.RedisBackend(fakeredis.FakeRedis(), prefix="parity-test")
    client = _redis_client()
    if client is None:
        pytest.skip(f"no Redis server at {REDIS_URL}")
    return online_store.RedisBackend(client, prefix="parity-test")


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Iterator[online_store.StateBackend]:
    built = make_backend(str(request.param))
    built.flush()
    yield built
    built.flush()


# --- the comparison ----------------------------------------------------------------------------


def _tolerant(name: str) -> bool:
    return name in TOLERANT_COLUMNS or name.startswith(TOLERANT_PREFIXES)


def compare(batch: pd.DataFrame, online: pd.DataFrame) -> dict[str, float]:
    """Assert row-for-row agreement and return the largest absolute difference per column."""
    assert list(online.columns) == list(batch.columns)
    assert online.index.equals(batch.index)
    largest: dict[str, float] = {}
    for name in batch.columns:
        expected = batch[name].to_numpy(dtype="float64")
        got = online[name].to_numpy(dtype="float64")
        null_expected = np.isnan(expected)
        null_got = np.isnan(got)
        disagree = np.flatnonzero(null_expected != null_got)
        assert disagree.size == 0, (
            f"{name}: null mask differs on {disagree.size} rows, first at position "
            f"{disagree[0]} (batch {expected[disagree[0]]}, online {got[disagree[0]]})"
        )
        present = ~null_expected
        if not present.any():
            largest[name] = 0.0
            continue
        gap = np.abs(expected[present] - got[present])
        largest[name] = float(gap.max())
        if _tolerant(name):
            allowed = FLOAT_TOLERANCE * np.maximum(np.abs(expected[present]), 1.0)
            off = np.flatnonzero(gap > allowed)
        else:
            off = np.flatnonzero(gap != 0.0)
        assert off.size == 0, (
            f"{name}: {off.size} rows differ, first at present-row {off[0]} "
            f"(batch {expected[present][off[0]]!r}, online {got[present][off[0]]!r})"
        )
    return largest


def batch_features(frame: pd.DataFrame, dist_buckets: dict[str, Any]) -> pd.DataFrame:
    """The batch path: stage 4's features and stage 5's candidate counters, side by side."""
    causal = features.build_features(frame, dist_buckets)
    graph = graph_features.build_graph_features(frame, None, online_store.GRAPH_ON)
    return pd.concat([causal, graph], axis=1)


@pytest.fixture(scope="module")
def batch(feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]) -> pd.DataFrame:
    return batch_features(feature_frame, dist_buckets)


# --- parity ---------------------------------------------------------------------------------------


def test_the_online_store_reproduces_the_batch_pipeline(
    feature_frame: pd.DataFrame,
    dist_buckets: dict[str, Any],
    batch: pd.DataFrame,
    backend: online_store.StateBackend,
) -> None:
    store = online_store.OnlineFeatureStore(backend, dist_buckets)
    online = online_store.replay(store, feature_frame)
    assert online.shape == batch.shape
    compare(batch, online)


def test_the_stream_reaches_every_branch(feature_frame: pd.DataFrame, batch: pd.DataFrame) -> None:
    """An agreement means something only if the hard cases are in the stream."""
    tie_key = [config.ENTITY_ID_COLUMN, config.TIME_COLUMN]
    assert feature_frame.duplicated(subset=tie_key).any()
    assert feature_frame.duplicated(subset=["card1", config.TIME_COLUMN]).any()
    # Tied rows on one entity do not see each other: the prior count is one number for the block.
    tied = feature_frame[feature_frame.duplicated(subset=tie_key, keep=False)]
    for _, block in batch.loc[tied.index].groupby(tied[config.ENTITY_ID_COLUMN]):
        assert block["rec_prior_count"].nunique() == 1 and len(block) >= 2
    assert (batch["rec_prior_count"] == 0).any() and (batch["rec_prior_count"] >= 2).any()
    for name in (
        "amt_repeat_within_24h",
        "dev_combination_is_new",
        "geo_addr1_differs_from_card_mode",
        "geo_addr2_differs_from_card_mode",
    ):
        values = batch[name].dropna()
        assert set(values.unique()) == {0.0, 1.0}, name
        assert batch[name].isna().any(), name
    assert (batch["dup_prior_count"] > 0).any()
    assert (batch["vel_count_24h_entity"] > 0).any() and (batch["vel_count_1h_card"] > 0).any()
    assert (batch["geo_dist1_bucket"].isna()).any() and batch["geo_dist1_bucket"].notna().any()
    assert (batch["graph_n_entities_on_device"] > 1).any()
    assert batch["graph_degree_addr1"].isna().any()


def test_get_features_writes_nothing(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any], backend: online_store.StateBackend
) -> None:
    """Phase one is a read. Two reads of the same transaction see the same state."""
    store = online_store.OnlineFeatureStore(backend, dist_buckets)
    rows = list(online_store.transactions(feature_frame))
    first = store.get_features(rows[0])
    second = store.get_features(rows[0])
    assert pd.Series(first).equals(pd.Series(second))
    assert first["rec_prior_count"] == 0 and first["vel_count_24h_entity"] == 0
    store.commit(rows[0])
    later = dict(rows[0])
    later[config.TIME_COLUMN] = int(rows[0][config.TIME_COLUMN]) + 1
    assert store.get_features(later)["rec_prior_count"] == 1
    assert store.get_features(later)["rec_prior_count"] == 1


def test_a_read_at_the_committed_timestamp_excludes_the_tie(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any], backend: online_store.StateBackend
) -> None:
    """The batch rule on ties, one call at a time: a simultaneous row is not knowable."""
    store = online_store.OnlineFeatureStore(backend, dist_buckets)
    row = next(iter(online_store.transactions(feature_frame)))
    store.commit(row)
    tied = dict(row)
    assert store.get_features(tied)["rec_prior_count"] == 0
    assert store.get_features(tied)["vel_count_24h_card"] == 0
    assert store.get_features(tied)["dup_prior_count"] == 0
    store.commit(tied)
    later = dict(row)
    later[config.TIME_COLUMN] = int(row[config.TIME_COLUMN]) + 60
    got = store.get_features(later)
    assert got["rec_prior_count"] == 2
    assert got["vel_count_10m_entity"] == 2
    assert got["dup_prior_count"] == 2
    assert got["rec_mean_gap"] == 0.0


def test_an_older_transaction_is_refused(
    feature_frame: pd.DataFrame, dist_buckets: dict[str, Any], backend: online_store.StateBackend
) -> None:
    store = online_store.OnlineFeatureStore(backend, dist_buckets)
    row = next(iter(online_store.transactions(feature_frame)))
    store.commit(row)
    earlier = dict(row)
    earlier[config.TIME_COLUMN] = int(row[config.TIME_COLUMN]) - 1
    with pytest.raises(online_store.OutOfOrderError):
        store.get_features(earlier)
    with pytest.raises(online_store.OutOfOrderError):
        store.commit(earlier)


def test_the_component_features_are_not_held(dist_buckets: dict[str, Any]) -> None:
    cfg = online_store.GRAPH_ON.with_features(graph_features.COMPONENT_SIZE)
    with pytest.raises(NotImplementedError, match="ADR 0024"):
        online_store.OnlineFeatureStore(online_store.MemoryBackend(), dist_buckets, graph_cfg=cfg)


def test_the_store_lists_the_catalogue_in_the_catalogue_order(
    dist_buckets: dict[str, Any],
) -> None:
    store = online_store.OnlineFeatureStore(online_store.MemoryBackend(), dist_buckets)
    assert store.feature_names == features.feature_names() + list(
        graph_features.graph_feature_names(online_store.GRAPH_ON)
    )
    running = [spec.name for spec in features.catalogue() if spec.requires_running_state]
    assert set(running) <= set(store.feature_names)


# --- the real stream ----------------------------------------------------------------------------

DAY = config.SECONDS_PER_DAY
STREAM_COLUMNS: tuple[str, ...] = (
    config.ID_COLUMN,
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
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
)


def real_stream(n_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The first `n_days` of the file with the entity key and the normalised columns attached.

    A prefix of the stream rather than a slice out of the middle, so every key's history inside
    it is complete from its own first row, and the dist1 edges fitted on the train split.
    """
    from fraud_platform import data_loader

    frame = data_loader.add_entity_key(data_loader.load_raw(columns=list(STREAM_COLUMNS)))
    frame = encoders.add_normalised_free_text(frame)
    buckets = features.fit_dist_buckets(frame[data_loader.split_masks(frame)["train"]])
    frame = frame[frame[config.TIME_COLUMN] < (1 + n_days) * DAY].reset_index(drop=True)
    return frame, buckets


@pytest.mark.needs_data
@pytest.mark.parametrize("name", ["memory", "fakeredis"])
def test_parity_on_the_real_stream(name: str) -> None:
    """The same check on the file, on a prefix long enough to hold every window several times."""
    frame, buckets = real_stream(n_days=4)
    assert len(frame) > 5_000
    # The file's ties inside this prefix sit on the card and on the device node; the entity
    # and content grains have none in the first four days, so the generated frame covers those.
    assert frame.duplicated(subset=["card1", config.TIME_COLUMN]).any()
    assert frame.duplicated(subset=[graph_features.DEVICE_COLUMN, config.TIME_COLUMN]).any()
    batch = batch_features(frame, buckets)
    # Two branches the generated frame is too sparse to reach: the hour-window duplicate and
    # the ten-minute entity window.
    assert (batch["dup_prior_count_1h"] > 0).any()
    assert (batch["vel_count_10m_entity"] > 0).any()
    backend = make_backend(name)
    backend.flush()
    store = online_store.OnlineFeatureStore(backend, buckets)
    compare(batch, online_store.replay(store, frame))
    backend.flush()
