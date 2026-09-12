"""Unit tests for the online store: the keys, the state, the fold, the ring and the backends.

The parity test is where the definitions are checked against the batch path. This module checks
the machinery underneath: that a key built one transaction at a time partitions the rows the way
the batch coders do, that a state survives the trip through Redis fields, that the fold is
Welford, that pruning leaves the prefix base behind, and that a commit which loses the race on
Redis is retried rather than lost.
"""

from __future__ import annotations

import math
from typing import Any

import fakeredis
import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, features, online_store
from fraud_platform.online_store import (
    KeyState,
    MemoryBackend,
    OnlineFeatureStore,
    RedisBackend,
    RingEntry,
    canonical,
    content_key,
    device_combination,
    fold,
    fresh_state,
    window,
)

DAY = config.SECONDS_PER_DAY


def _txn(t: int, amount: float, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        config.ENTITY_ID_COLUMN: "1000|100.0|5.0",
        config.TIME_COLUMN: t,
        config.AMOUNT_COLUMN: amount,
        "card1": 1000,
        "addr1": 100.0,
    }
    row.update(extra)
    return row


# --- keys -----------------------------------------------------------------------------------------


def test_canonical_merges_integer_and_float_spellings_and_keeps_nulls_apart() -> None:
    assert canonical(np.int16(10023)) == canonical(10023.0) == canonical(np.float32(10023))
    assert canonical("W") == "W"
    assert canonical(None) is None and canonical(float("nan")) is None
    assert canonical(pd.NA) is None
    assert canonical(0.1) != canonical(0.2)


def test_the_content_key_partitions_rows_like_the_batch_coder(feature_frame: pd.DataFrame) -> None:
    codes = features.content_codes(feature_frame)
    keys = [content_key(txn) for txn in _rows(feature_frame)]
    _assert_same_partition(codes, keys)


def test_the_device_combination_partitions_rows_like_the_batch_coder(
    feature_frame: pd.DataFrame,
) -> None:
    codes = features.device_codes(feature_frame)
    keys = [device_combination(txn) for txn in _rows(feature_frame)]
    undefined = codes == -1
    assert all(key is None for key, off in zip(keys, undefined, strict=True) if off)
    assert all(key is not None for key, off in zip(keys, undefined, strict=True) if not off)
    _assert_same_partition(codes[~undefined], [k for k in keys if k is not None])


def test_the_device_combination_is_computed_from_the_raw_columns_when_the_row_lacks_the_norm(
    feature_frame: pd.DataFrame,
) -> None:
    rows = _rows(feature_frame)
    stripped = [{k: v for k, v in row.items() if not k.endswith("_norm")} for row in rows]
    assert [device_combination(a) for a in rows] == [device_combination(b) for b in stripped]


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [dict(zip(frame.columns, values, strict=True)) for values in frame.to_numpy(object)]


def _assert_same_partition(codes: np.ndarray, keys: list[Any]) -> None:
    by_code: dict[int, Any] = {}
    by_key: dict[Any, int] = {}
    for code, key in zip(codes.tolist(), keys, strict=True):
        assert by_code.setdefault(code, key) == key
        assert by_key.setdefault(key, code) == code


# --- state ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("grain", online_store.GRAINS)
def test_a_state_survives_the_trip_through_redis_fields(grain: str) -> None:
    state = fresh_state(grain)
    state.settled["n"] = 3
    state.pending_ts = 90_000
    state.pending = [{"t": 90_000, "a": 12.5, "d": None}]
    state.ring = [RingEntry(86_400, 0, 10.0, 10.0), RingEntry(90_000, 1, None, None)]
    state.pruned_prefix = 662.8500000000001
    state.seq = 2
    ring = [RingEntry.from_member(e.member(), float(e.ts)) for e in state.ring]
    back = KeyState.from_fields(state.to_fields(), ring, grain)
    assert back == state
    assert KeyState.from_fields({}, [], grain) == fresh_state(grain)


def test_ring_members_sort_by_arrival_inside_a_tie() -> None:
    first = RingEntry(100, 9, 1.0, 1.0).member()
    second = RingEntry(100, 10, 1.0, 2.0).member()
    assert first < second


def test_the_entity_fold_is_welford() -> None:
    rng = np.random.default_rng(config.SEED)
    amounts = rng.uniform(0.25, 3000.0, 50)
    entries = [{"t": 1000 + i, "a": float(a), "d": None} for i, a in enumerate(amounts)]
    out = fold("entity", fresh_state("entity").settled, entries)
    assert out["n"] == 50 and out["first"] == 1000 and out["last"] == 1049
    assert math.isclose(out["mean"], float(np.mean(amounts)), rel_tol=1e-12)
    assert math.isclose(math.sqrt(out["m2"] / 49), float(np.std(amounts, ddof=1)), rel_tol=1e-12)


def test_the_fold_does_not_touch_its_input() -> None:
    settled = fresh_state("card").settled
    fold("card", settled, [{"t": 1, "addr1": "100.0", "addr2": None}])
    assert settled == fresh_state("card").settled


# --- the ring ---------------------------------------------------------------------------------------


def test_pruning_leaves_the_prefix_base_behind() -> None:
    """A window whose lower bound is the ring's first entry subtracts what the batch subtracts."""
    store = OnlineFeatureStore(MemoryBackend(), {"column": "dist1", "edges": []})
    amounts = [10.0, 20.0, 30.0, 40.0, 50.0]
    moments = [0, 1_000, DAY + 500, DAY + 1_500, 2 * DAY + 100]
    for t, a in zip(moments, amounts, strict=True):
        store.commit(_txn(t, a))
    state = store.backend.load([("entity", "1000|100.0|5.0")])[("entity", "1000|100.0|5.0")]
    # The last commit, at 2 days + 100 s, dropped everything older than a day before it.
    assert [e.ts for e in state.ring] == [DAY + 500, DAY + 1_500, 2 * DAY + 100]
    assert state.pruned_prefix == 10.0 + 20.0
    count, total, _ = window(state, 2 * DAY + 200, DAY)
    assert count == 3 and total == (10.0 + 20.0 + 30.0 + 40.0 + 50.0) - (10.0 + 20.0)
    count, total, _ = window(state, 2 * DAY + 1_200, DAY)
    assert count == 2 and total == (10.0 + 20.0 + 30.0 + 40.0 + 50.0) - (10.0 + 20.0 + 30.0)
    count, total, _ = window(state, 3 * DAY + 2_000, DAY)
    assert count == 0 and total == 0.0


def test_the_read_at_a_later_time_folds_pending_without_writing() -> None:
    store = OnlineFeatureStore(MemoryBackend(), {"column": "dist1", "edges": []})
    store.commit(_txn(1_000, 10.0))
    store.commit(_txn(1_000, 30.0))
    got = store.get_features(_txn(2_000, 5.0))
    assert got["rec_prior_count"] == 2 and got["amt_entity_mean"] == 20.0
    state = store.backend.load([("entity", "1000|100.0|5.0")])[("entity", "1000|100.0|5.0")]
    assert state.settled["n"] == 0 and len(state.pending) == 2


# --- backends ---------------------------------------------------------------------------------------


def test_the_memory_backend_hands_out_copies() -> None:
    backend = MemoryBackend()
    key = ("entity", "k")
    backend.transaction([key], lambda states: states)
    loaded = backend.load([key])[key]
    loaded.settled["n"] = 99
    assert backend.load([key])[key].settled["n"] == 0


def test_a_redis_commit_that_loses_the_race_is_retried() -> None:
    client = fakeredis.FakeRedis()
    backend = RedisBackend(client, prefix="units")
    key = ("addr1_node", "100.0")
    calls = {"n": 0}

    def update(states: dict[tuple[str, str], KeyState]) -> dict[tuple[str, str], KeyState]:
        calls["n"] += 1
        if calls["n"] == 1:
            # Another writer lands a state on the watched key between the read and the EXEC.
            other = fresh_state(key[0])
            other.seq = 7
            client.hset(backend.hash_key(key), mapping=other.to_fields())
        out = online_store.copy_state(states[key])
        out.seq += 1
        return {key: out}

    backend.transaction([key], update)
    assert calls["n"] == 2
    assert backend.load([key])[key].seq == 8


def test_the_redis_backend_prunes_the_sorted_set_and_flushes_its_prefix_only() -> None:
    client = fakeredis.FakeRedis()
    client.set("other:key", "1")
    backend = RedisBackend(client, prefix="units")
    store = OnlineFeatureStore(backend, {"column": "dist1", "edges": []})
    store.commit(_txn(0, 1.0))
    store.commit(_txn(2 * DAY, 2.0))
    assert client.zcard(backend.ring_key(("entity", "1000|100.0|5.0"))) == 1
    assert client.zcard(backend.ring_key(("card", "1000.0"))) == 1
    backend.flush()
    assert client.keys("units:*") == [] and client.get("other:key") == b"1"


def test_transactions_keep_frame_order_inside_a_tie(feature_frame: pd.DataFrame) -> None:
    ids = [txn[config.ID_COLUMN] for txn in online_store.transactions(feature_frame)]
    ts = feature_frame.set_index(config.ID_COLUMN)[config.TIME_COLUMN].loc[ids]
    assert ts.is_monotonic_increasing
    tied = feature_frame.loc[10:12, config.ID_COLUMN].tolist()
    assert [i for i in ids if i in tied] == tied
