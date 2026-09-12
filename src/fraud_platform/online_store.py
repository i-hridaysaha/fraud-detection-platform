"""The online feature store: stage 4's definitions and stage 5's counters, one transaction at a time.

`fraud_platform.features` computes every behavioural feature over a whole stream with prefix
sums and searchsorted. An authorisation endpoint does not have a stream; it has one transaction
and whatever it has remembered about the keys that transaction carries. This module is the second
implementation of the same definitions, written for that shape, and `tests/test_parity.py` is
the test that the two agree row for row. Nothing here is imported by the batch path and nothing
in the batch path is imported into the arithmetic here beyond the constants that name the
features and the windows, so an agreement between the two is evidence rather than tautology.

**Two calls, in this order, never one.**

    features = store.get_features(txn)   # state as of now, this transaction excluded
    ...score, decide, respond...
    store.commit(txn)                     # fold this transaction in, for the next one

`get_features` writes nothing. `commit` returns nothing. The batch definition of every feature is
"rows strictly before t", and the split is what makes that definition auditable at the call
site: a feature vector produced by `get_features` cannot contain the transaction it describes,
because the transaction has not been written yet, and a reviewer can see that from the two lines
rather than from a flag inside one. ADR 0033.

**Same-timestamp ties are excluded here too.** The batch path reads a block of rows sharing a key
and a timestamp against the state before the block, then adds the whole block. The store keeps
the same shape as two levels of state per key: `settled`, everything strictly before the latest
committed timestamp, and `pending`, the rows at that timestamp. A read at the pending timestamp
sees `settled` only; a read at a later timestamp sees both. Nothing else reproduces the batch
number on a tied row, and the parity test's stream has ties in it.

**What is windowed and what is not.** The velocity, repeat-amount and duplicate features read a
trailing window, so each key holds a ring of `(timestamp, payload)` entries bounded by the widest
window rather than by the key's lifetime. Everything else the catalogue lists as running state is
a handful of scalars per key: three for recency, three for the amount moments, one counter per
address value, one set of device combinations. Both are laid out the way
`reports/feature_summary.json` says they have to be, and `state_footprint` there is the measured
bound on the two structures that look unbounded.

**Exactly, not approximately.** The window sums are prefix differences in the batch path, so each
ring entry carries the key's running prefix and the store subtracts the same two numbers. The
amount moments are Welford in the same order. The parity test's tolerance on the float columns
is therefore a statement about the test, not a concession to the implementation, and the
artifact records the largest difference actually observed.

**Two backends, one set of definitions.** `MemoryBackend` holds `KeyState` objects in a dict.
`RedisBackend` holds the scalars in a hash and the ring in a sorted set scored by timestamp, and
commits under `WATCH` so two writers on one key cannot lose an update. Neither backend computes a
feature: `read_features` and `folded` are pure functions of the loaded states, and a backend
only loads and stores. That is how the semantics are kept identical. ADR 0034.

**What the store does not hold.** The stage 5 component features. A connected component is a
fact about every edge ever added and cannot be windowed or held per key (ADR 0024), and the
fraud rate over it is label-derived and was dropped on a measurement (ADR 0025). Asking for one
raises. The four candidate graph features are counters and are here.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, encoders, features, graph_features, transforms

# --- choices --------------------------------------------------------------------------------

GRAINS: tuple[str, ...] = ("entity", "card", "content", "addr1_node", "device_node")

# The grains that hold a ring. The two amount rings serve velocity and the repeat flag; the
# content ring holds timestamps only.
WINDOWED_GRAINS: tuple[str, ...] = ("entity", "card", "content")

# The key namespace in Redis. One prefix so a flush of the store is one SCAN.
REDIS_PREFIX = "fp"

# A null component of a key is a level of its own, as in `features.content_codes`.
_NULL = "\x00null"

_NOT_HELD_WHY = (
    "a component is a fact about every edge ever added and cannot be held per key (ADR 0024); "
    "the rate over it is label-derived and was dropped on a measurement (ADR 0025)"
)
NOT_HELD: dict[str, str] = dict.fromkeys(
    (graph_features.COMPONENT_SIZE, *graph_features.LABEL_FEATURES), _NOT_HELD_WHY
)

# The graph block, switched on: the store holds the candidate counters whatever stage 6 shipped,
# because the parity test covers the definitions and not the stack.
GRAPH_ON = graph_features.GraphConfig(enabled=True)


class OutOfOrderError(ValueError):
    """A transaction older than the key's latest committed one. The stream is chronological."""


# --- values ---------------------------------------------------------------------------------


def is_null(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and bool(np.isnan(value))


def canonical(value: Any) -> str | None:
    """One string per distinct value, so a key built from it equals another iff the values do.

    Numbers go through float so an int16 card1 and a float64 card1 land on the same key; the
    batch path factorizes a whole column at once and never mixes the two, so the merge is
    harmless there and necessary here.
    """
    if is_null(value):
        return None
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | float | np.integer | np.floating):
        return repr(float(value))
    return str(value)


def _get(txn: Mapping[str, Any], column: str) -> Any:
    value = txn.get(column)
    return None if is_null(value) else value


def _float(txn: Mapping[str, Any], column: str) -> float | None:
    value = _get(txn, column)
    return None if value is None else float(value)


def normalised_value(txn: Mapping[str, Any], column: str) -> str | None:
    """The stage 3 normalised value of a free-text column, computed here if the row lacks it."""
    name = encoders.normalised_name(column)
    if name in txn:
        value = _get(txn, name)
        return None if value is None else str(value)
    return encoders.FREE_TEXT_NORMALISERS[column](_get(txn, column))


def device_combination(txn: Mapping[str, Any]) -> str | None:
    """The `features.device_codes` tuple as one string, None when every component is null."""
    parts: list[str] = []
    known = False
    for column in features.DEVICE_COMBINATION_COLUMNS:
        if column.endswith(encoders.NORMALISED_SUFFIX):
            value = normalised_value(txn, column[: -len(encoders.NORMALISED_SUFFIX)])
        else:
            value = canonical(_get(txn, column))
        known |= value is not None
        parts.append(_NULL if value is None else value)
    return "|".join(parts) if known else None


def content_key(txn: Mapping[str, Any]) -> str:
    """The `features.CONTENT_COLUMNS` tuple as one string, nulls as a level."""
    return "|".join(canonical(_get(txn, column)) or _NULL for column in features.CONTENT_COLUMNS)


def keys_of(txn: Mapping[str, Any]) -> dict[str, str | None]:
    """Every key the transaction touches, by grain. None where the row carries no value."""
    if config.ENTITY_ID_COLUMN not in txn or is_null(txn[config.ENTITY_ID_COLUMN]):
        raise KeyError(
            f"a transaction needs {config.ENTITY_ID_COLUMN}; build it with "
            "data_loader.add_entity_key so the key is the one the batch path uses"
        )
    return {
        "entity": str(txn[config.ENTITY_ID_COLUMN]),
        # factorize with use_na_sentinel=False makes a null card1 a level of its own.
        "card": canonical(_get(txn, "card1")) or _NULL,
        "content": content_key(txn),
        "addr1_node": canonical(_get(txn, "addr1")),
        "device_node": normalised_value(txn, "DeviceInfo"),
    }


# --- the state --------------------------------------------------------------------------------


@dataclass
class RingEntry:
    ts: int
    seq: int
    amount: float | None
    prefix: float | None

    def member(self) -> str:
        """The sorted-set member: the sequence number first so ties keep arrival order."""
        return f"{self.seq:012d}|{json.dumps({'a': self.amount, 'p': self.prefix})}"

    @classmethod
    def from_member(cls, member: str, score: float) -> RingEntry:
        seq, payload = member.split("|", 1)
        data = json.loads(payload)
        return cls(ts=int(score), seq=int(seq), amount=data["a"], prefix=data["p"])


@dataclass
class KeyState:
    """One key's memory: scalars settled strictly before `pending_ts`, the rows at it, the ring.

    `pruned_prefix` is the running prefix of the newest ring entry that was dropped as older than
    the widest window, so a window whose lower bound is the first ring entry still subtracts the
    same number the batch path does. `seq` numbers every committed row on the key.
    """

    settled: dict[str, Any]
    pending_ts: int | None = None
    pending: list[dict[str, Any]] = field(default_factory=list)
    ring: list[RingEntry] = field(default_factory=list)
    pruned_prefix: float = 0.0
    seq: int = 0

    def to_fields(self) -> dict[str, str]:
        return {
            "settled": json.dumps(self.settled),
            "pending_ts": "" if self.pending_ts is None else str(self.pending_ts),
            "pending": json.dumps(self.pending),
            "pruned_prefix": repr(self.pruned_prefix),
            "seq": str(self.seq),
        }

    @classmethod
    def from_fields(
        cls, fields: Mapping[str, str], ring: Sequence[RingEntry], grain: str
    ) -> KeyState:
        if not fields:
            return fresh_state(grain)
        return cls(
            settled=json.loads(fields["settled"]),
            pending_ts=int(fields["pending_ts"]) if fields["pending_ts"] else None,
            pending=json.loads(fields["pending"]),
            ring=list(ring),
            pruned_prefix=float(fields["pruned_prefix"]),
            seq=int(fields["seq"]),
        )


def fresh_state(grain: str) -> KeyState:
    settled: dict[str, Any]
    if grain == "entity":
        settled = {"n": 0, "first": None, "last": None, "mean": 0.0, "m2": 0.0, "devices": []}
    elif grain == "card":
        settled = {"n": 0, "addr1": {}, "addr2": {}}
    elif grain == "content":
        settled = {"n": 0, "last": None}
    elif grain == "addr1_node":
        settled = {"n": 0}
    elif grain == "device_node":
        settled = {"n": 0, "entities": []}
    else:
        raise KeyError(f"unknown grain {grain!r}, expected one of {list(GRAINS)}")
    return KeyState(settled=settled)


def copy_state(state: KeyState) -> KeyState:
    return KeyState(
        settled=json.loads(json.dumps(state.settled)),
        pending_ts=state.pending_ts,
        pending=json.loads(json.dumps(state.pending)),
        ring=[RingEntry(e.ts, e.seq, e.amount, e.prefix) for e in state.ring],
        pruned_prefix=state.pruned_prefix,
        seq=state.seq,
    )


# --- the fold: what commit does to the scalars ---------------------------------------------------


def fold(
    grain: str, settled: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The scalars after `entries` (one timestamp block, arrival order) are folded in.

    Pure: returns a new dictionary. This is the serving-side arithmetic of `features._recency`,
    `features._entity_scan`, `features._geography`, `features._duplicate` and the stage 5
    degree counters, one row at a time instead of one block at a time.
    """
    out: dict[str, Any] = json.loads(json.dumps(settled))
    for entry in entries:
        ts = int(entry["t"])
        out["n"] = int(out["n"]) + 1
        if grain == "entity":
            if out["first"] is None:
                out["first"] = ts
            out["last"] = ts
            amount = float(entry["a"])
            # Welford, in the batch path's order: the delta against the old mean, the mean, then
            # the second moment against the new mean.
            delta = amount - float(out["mean"])
            out["mean"] = float(out["mean"]) + delta / int(out["n"])
            out["m2"] = float(out["m2"]) + delta * (amount - float(out["mean"]))
            device = entry.get("d")
            if device is not None and device not in out["devices"]:
                out["devices"].append(device)
        elif grain == "card":
            for label in ("addr1", "addr2"):
                value = entry.get(label)
                if value is not None:
                    out[label][value] = int(out[label].get(value, 0)) + 1
        elif grain == "content":
            out["last"] = ts
        elif grain == "device_node":
            entity = entry["e"]
            if entity not in out["entities"]:
                out["entities"].append(entity)
    return out


def effective(grain: str, state: KeyState, t: int) -> dict[str, Any]:
    """The scalars a read at `t` may see: settled only at the pending timestamp, both after it."""
    if state.pending_ts is None or t == state.pending_ts:
        return state.settled
    if t < state.pending_ts:
        raise OutOfOrderError(
            f"a read at {t} on a key whose latest committed timestamp is {state.pending_ts}"
        )
    return fold(grain, state.settled, state.pending)


# --- the ring: what the windows read ---------------------------------------------------------


def _first_at_or_after(ring: Sequence[RingEntry], moment: int) -> int:
    """Index of the first entry with `ts >= moment`: searchsorted with side="left"."""
    lo, hi = 0, len(ring)
    while lo < hi:
        mid = (lo + hi) // 2
        if ring[mid].ts < moment:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _prefix_before(state: KeyState, index: int) -> float:
    if index == 0:
        return state.pruned_prefix
    prefix = state.ring[index - 1].prefix
    return 0.0 if prefix is None else float(prefix)


def window(state: KeyState, t: int, seconds: int) -> tuple[int, float, list[RingEntry]]:
    """Count, prefix-difference total and entries of `[t - seconds, t)`. Ties at `t` excluded.

    `_first_at_or_after(ring, t)` is `searchsorted(ts, t, side="left")`: the first entry at `t`
    is outside the window, and so is everything at `t` behind it.
    """
    if state.ring and state.ring[-1].ts > t:
        raise OutOfOrderError(
            f"a read at {t} on a key whose ring holds a later timestamp {state.ring[-1].ts}"
        )
    lo = _first_at_or_after(state.ring, t - seconds)
    hi = _first_at_or_after(state.ring, t)
    total = _prefix_before(state, hi) - _prefix_before(state, lo)
    return hi - lo, total, list(state.ring[lo:hi])


# --- the backends -------------------------------------------------------------------------------

StateKey = tuple[str, str]
Update = Callable[[dict[StateKey, KeyState]], dict[StateKey, KeyState]]


class StateBackend:
    """What a backend has to do: load some keys' states, and apply one update to them atomically."""

    def load(self, keys: Sequence[StateKey]) -> dict[StateKey, KeyState]:
        raise NotImplementedError

    def transaction(self, keys: Sequence[StateKey], update: Update) -> None:
        raise NotImplementedError

    def flush(self) -> None:
        raise NotImplementedError


class MemoryBackend(StateBackend):
    """In-process states in a dict. The fallback, and what the Redis backend is tested against."""

    def __init__(self) -> None:
        self._states: dict[StateKey, KeyState] = {}
        self._lock = threading.Lock()

    def _snapshot(self, keys: Sequence[StateKey]) -> dict[StateKey, KeyState]:
        return {
            key: copy_state(self._states[key]) if key in self._states else fresh_state(key[0])
            for key in keys
        }

    def load(self, keys: Sequence[StateKey]) -> dict[StateKey, KeyState]:
        with self._lock:
            return self._snapshot(keys)

    def transaction(self, keys: Sequence[StateKey], update: Update) -> None:
        with self._lock:
            self._states.update(update(self._snapshot(keys)))

    def flush(self) -> None:
        with self._lock:
            self._states.clear()

    @property
    def n_keys(self) -> int:
        return len(self._states)


class RedisBackend(StateBackend):
    """Scalars in a hash, the ring in a sorted set scored by timestamp, commits under WATCH.

    The sorted set is the ring's native shape: an append is one ZADD, pruning is one ZREM of
    the entries that fell out of the widest window, and both happen without the ring being
    rewritten. A read fetches the ring whole, which is bounded by the key's activity inside the
    widest window plus whatever the last commit had not pruned yet.

    Optimistic concurrency: `transaction` watches every key it will write, reads, computes the
    update in Python, then writes inside MULTI/EXEC. If another writer touched a key in between,
    EXEC fails and the update is recomputed from the new state. Same state in, same state out as
    the memory backend, because the update function is the same function.
    """

    def __init__(self, client: Any, prefix: str = REDIS_PREFIX, max_retries: int = 20) -> None:
        self.client = client
        self.prefix = prefix
        self.max_retries = max_retries

    def hash_key(self, key: StateKey) -> str:
        return f"{self.prefix}:h:{key[0]}:{key[1]}"

    def ring_key(self, key: StateKey) -> str:
        return f"{self.prefix}:z:{key[0]}:{key[1]}"

    def _read(self, client: Any, keys: Sequence[StateKey]) -> dict[StateKey, KeyState]:
        """One hash and one sorted set per key, through whatever client is handed in.

        A pipeline built with `transaction=False` collects the replies for one round trip; a
        pipeline in its WATCH state answers each command as it is issued. Either works here.
        """
        replies: list[Any] = []
        for key in keys:
            replies.append(client.hgetall(self.hash_key(key)))
            replies.append(client.zrange(self.ring_key(key), 0, -1, withscores=True))
        if hasattr(client, "execute") and not getattr(client, "watching", False):
            replies = client.execute()
        out: dict[StateKey, KeyState] = {}
        for i, key in enumerate(keys):
            fields = {_text(k): _text(v) for k, v in replies[2 * i].items()}
            ring = [RingEntry.from_member(_text(m), s) for m, s in replies[2 * i + 1]]
            out[key] = KeyState.from_fields(fields, ring, key[0])
        return out

    def load(self, keys: Sequence[StateKey]) -> dict[StateKey, KeyState]:
        return self._read(self.client.pipeline(transaction=False), keys)

    def transaction(self, keys: Sequence[StateKey], update: Update) -> None:
        from redis.exceptions import WatchError

        watched = [self.hash_key(key) for key in keys] + [self.ring_key(key) for key in keys]
        for _attempt in range(self.max_retries):
            with self.client.pipeline(transaction=True) as pipe:
                pipe.watch(*watched)
                before = self._read(pipe, keys)
                after = update(before)
                pipe.multi()
                for key, state in after.items():
                    pipe.hset(self.hash_key(key), mapping=state.to_fields())
                    old = before[key]
                    kept = {entry.seq for entry in state.ring}
                    dropped = [entry.member() for entry in old.ring if entry.seq not in kept]
                    if dropped:
                        pipe.zrem(self.ring_key(key), *dropped)
                    known = {entry.seq for entry in old.ring}
                    added = {
                        entry.member(): entry.ts for entry in state.ring if entry.seq not in known
                    }
                    if added:
                        pipe.zadd(self.ring_key(key), added)
                try:
                    pipe.execute()
                    return
                except WatchError:
                    continue
        raise RuntimeError(f"commit lost the race {self.max_retries} times on {list(keys)}")

    def flush(self) -> None:
        cursor = 0
        while True:
            cursor, found = self.client.scan(cursor=cursor, match=f"{self.prefix}:*", count=1000)
            if found:
                self.client.delete(*found)
            if cursor == 0:
                break


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


# --- the store ----------------------------------------------------------------------------------


class OnlineFeatureStore:
    """`get_features` then `commit`, on any backend, with the batch path's definitions."""

    def __init__(
        self,
        backend: StateBackend,
        dist_buckets: Mapping[str, Any] | None = None,
        cfg: features.FeatureConfig = features.DEFAULT_CONFIG,
        graph_cfg: graph_features.GraphConfig = GRAPH_ON,
    ) -> None:
        unknown = [name for name in cfg.families if name not in features.FAMILIES]
        if unknown:
            raise ValueError(f"unknown feature families {unknown}")
        held = [name for name in graph_cfg.features if name in NOT_HELD]
        if graph_cfg.enabled and held:
            raise NotImplementedError(
                "the store does not hold " + "; ".join(f"{n}: {NOT_HELD[n]}" for n in held)
            )
        if "geography" in cfg.families and dist_buckets is None:
            raise ValueError(
                "the geography family needs the dist1 bucket edges from features.fit_dist_buckets"
            )
        self.backend = backend
        self.cfg = cfg
        self.graph_cfg = graph_cfg
        self.dist_buckets = dist_buckets
        self.widest = max(
            [seconds for _, seconds in cfg.windows] + [cfg.amount_repeat_window_seconds]
        )
        self.feature_names: list[str] = features.feature_names(cfg) + (
            graph_features.graph_feature_names(graph_cfg) if graph_cfg.enabled else []
        )

    # -- keys ------------------------------------------------------------------------------------

    def keys(self, txn: Mapping[str, Any]) -> list[StateKey]:
        named = keys_of(txn)
        return [(grain, str(named[grain])) for grain in GRAINS if named[grain] is not None]

    def _window_seconds(self, grain: str) -> int:
        return self.cfg.duplicate_window_seconds if grain == "content" else self.widest

    # -- phase one ------------------------------------------------------------------------------

    def get_features(self, txn: Mapping[str, Any]) -> dict[str, float | int]:
        """Every feature for `txn`, from the state strictly before it. Writes nothing."""
        return self.read_features(txn, self.backend.load(self.keys(txn)))

    def read_features(
        self, txn: Mapping[str, Any], states: Mapping[StateKey, KeyState]
    ) -> dict[str, float | int]:
        """The pure part of `get_features`: features from already-loaded states."""
        t = int(txn[config.TIME_COLUMN])
        amount = float(txn[config.AMOUNT_COLUMN])
        named = keys_of(txn)
        by_grain = {grain: states.get((grain, str(key))) for grain, key in named.items()}
        out: dict[str, float | int] = {}
        nan = math.nan
        cfg = self.cfg

        if "velocity" in cfg.families:
            for grain in cfg.velocity_grains:
                state = by_grain[grain]
                assert state is not None
                for suffix, seconds in cfg.windows:
                    count, total, _ = window(state, t, seconds)
                    out[features.velocity_name("count", suffix, grain)] = count
                    out[features.velocity_name("amount_sum", suffix, grain)] = total
                    out[features.velocity_name("amount_mean", suffix, grain)] = (
                        total / count if count > 0 else nan
                    )

        entity = by_grain["entity"]
        assert entity is not None
        scalars = effective("entity", entity, t)
        n_prior = int(scalars["n"])

        if "recency" in cfg.families:
            since = float(t - int(scalars["last"])) if n_prior > 0 else nan
            mean_gap = (
                float(int(scalars["last"]) - int(scalars["first"])) / float(n_prior - 1)
                if n_prior >= 2
                else nan
            )
            out["rec_seconds_since_prev"] = since
            out["rec_prior_count"] = n_prior
            out["rec_mean_gap"] = mean_gap
            out["rec_gap_deviation"] = since - mean_gap

        if "tenure" in cfg.families and cfg.include_d_origin_tenure:
            d1 = _float(txn, "D1")
            out[transforms.d_origin_name("D1")] = (
                nan if d1 is None else t / config.SECONDS_PER_DAY - d1
            )

        if "geography" in cfg.families:
            card = by_grain["card"]
            assert card is not None
            card_scalars = effective("card", card, t)
            for label, name in (
                ("addr1", "geo_addr1_differs_from_card_mode"),
                ("addr2", "geo_addr2_differs_from_card_mode"),
            ):
                mode = _mode(card_scalars[label])
                level = canonical(_get(txn, label))
                out[name] = (
                    nan if mode is None or level is None else (0.0 if level == mode else 1.0)
                )
            out["geo_n_distinct_addr1_on_card"] = len(card_scalars["addr1"])
            assert self.dist_buckets is not None
            dist = _float(txn, str(self.dist_buckets["column"]))
            edges = np.asarray(self.dist_buckets["edges"], dtype="float64")
            out["geo_dist1_bucket"] = (
                nan if dist is None else float(np.searchsorted(edges, dist, side="right"))
            )

        if "device" in cfg.families:
            device = device_combination(txn)
            seen = scalars["devices"]
            out["dev_combination_is_new"] = (
                nan if device is None or not seen else (0.0 if device in seen else 1.0)
            )
            out["dev_n_distinct_so_far"] = len(seen)

        if "amount" in cfg.families:
            seen_mean = float(scalars["mean"]) if n_prior > 0 else nan
            seen_std = (
                math.sqrt(max(float(scalars["m2"]), 0.0) / (n_prior - 1)) if n_prior >= 2 else nan
            )
            out["amt_entity_mean"] = seen_mean
            out["amt_entity_std"] = seen_std
            out["amt_deviation_from_entity_mean"] = amount - seen_mean
            out["amt_zscore_within_entity"] = (
                (amount - seen_mean) / seen_std if seen_std > 0.0 else nan
            )
            _, _, recent = window(entity, t, cfg.amount_repeat_window_seconds)
            recent_amounts = {entry.amount for entry in recent}
            out["amt_repeat_within_24h"] = (
                nan if not recent_amounts else (1.0 if amount in recent_amounts else 0.0)
            )

        if "duplicate" in cfg.families:
            content = by_grain["content"]
            assert content is not None
            content_scalars = effective("content", content, t)
            n_same = int(content_scalars["n"])
            out["dup_prior_count"] = n_same
            out["dup_seconds_since_prev"] = (
                float(t - int(content_scalars["last"])) if n_same > 0 else nan
            )
            count_1h, _, _ = window(content, t, cfg.duplicate_window_seconds)
            out["dup_prior_count_1h"] = count_1h

        if self.graph_cfg.enabled:
            card = by_grain["card"]
            assert card is not None
            wanted = set(self.graph_cfg.features)
            degrees = {
                graph_features.degree_name("card1"): int(effective("card", card, t)["n"]),
                graph_features.degree_name("addr1"): _node_count(
                    "addr1_node", by_grain["addr1_node"], t
                ),
                graph_features.degree_name(graph_features.DEVICE_COLUMN): _node_count(
                    "device_node", by_grain["device_node"], t
                ),
            }
            for name, value in degrees.items():
                if name in wanted:
                    out[name] = value
            if graph_features.ENTITIES_ON_DEVICE in wanted:
                node = by_grain["device_node"]
                out[graph_features.ENTITIES_ON_DEVICE] = (
                    nan
                    if node is None
                    else float(len(effective("device_node", node, t)["entities"]))
                )

        return {name: out[name] for name in self.feature_names if name in out}

    # -- phase two ------------------------------------------------------------------------------

    def commit(self, txn: Mapping[str, Any]) -> None:
        """Fold `txn` into every key it touches. Returns nothing: there is nothing to read here."""
        self.backend.transaction(self.keys(txn), lambda states: self.folded(txn, states))

    def folded(
        self, txn: Mapping[str, Any], states: Mapping[StateKey, KeyState]
    ) -> dict[StateKey, KeyState]:
        """The pure part of `commit`: every touched key's next state."""
        t = int(txn[config.TIME_COLUMN])
        amount = float(txn[config.AMOUNT_COLUMN])
        entries = self._entries(txn, t, amount)
        out: dict[StateKey, KeyState] = {}
        for key, state in states.items():
            grain = key[0]
            nxt = copy_state(state)
            if nxt.pending_ts is not None and t < nxt.pending_ts:
                raise OutOfOrderError(
                    f"a commit at {t} on a key whose latest committed timestamp is {nxt.pending_ts}"
                )
            if nxt.pending_ts is None or t > nxt.pending_ts:
                nxt.settled = fold(grain, nxt.settled, nxt.pending)
                nxt.pending = [entries[grain]]
                nxt.pending_ts = t
            else:
                nxt.pending.append(entries[grain])
            if grain in WINDOWED_GRAINS:
                carries_amount = grain != "content"
                prefix = _prefix_before(nxt, len(nxt.ring)) + amount if carries_amount else None
                nxt.ring.append(
                    RingEntry(
                        ts=t, seq=nxt.seq, amount=amount if carries_amount else None, prefix=prefix
                    )
                )
                # Everything older than the widest window falls out. The newest entry to fall
                # out leaves its prefix behind, as the base the next window subtracts.
                keep = _first_at_or_after(nxt.ring, t - self._window_seconds(grain))
                if keep > 0:
                    dropped = nxt.ring[keep - 1]
                    if dropped.prefix is not None:
                        nxt.pruned_prefix = float(dropped.prefix)
                    nxt.ring = nxt.ring[keep:]
            nxt.seq += 1
            out[key] = nxt
        return out

    def _entries(self, txn: Mapping[str, Any], t: int, amount: float) -> dict[str, dict[str, Any]]:
        return {
            "entity": {"t": t, "a": amount, "d": device_combination(txn)},
            "card": {
                "t": t,
                "addr1": canonical(_get(txn, "addr1")),
                "addr2": canonical(_get(txn, "addr2")),
            },
            "content": {"t": t},
            "addr1_node": {"t": t},
            "device_node": {"t": t, "e": str(txn[config.ENTITY_ID_COLUMN])},
        }


def _mode(counts: Mapping[str, int]) -> str | None:
    """Most frequent value so far, ties broken on the smaller value. `features._mode_of`."""
    best: str | None = None
    best_count = -1
    for value, count in counts.items():
        if count > best_count or (count == best_count and best is not None and _less(value, best)):
            best = value
            best_count = int(count)
    return best


def _less(a: str, b: str) -> bool:
    try:
        return float(a) < float(b)
    except ValueError:
        return a < b


def _node_count(grain: str, state: KeyState | None, t: int) -> float:
    return math.nan if state is None else float(effective(grain, state, t)["n"])


# --- replay --------------------------------------------------------------------------------------


def stream_order(frame: pd.DataFrame) -> np.ndarray:
    """Positions in stream order: by timestamp, ties in frame order.

    The batch path sorts with a stable lexsort, so two rows sharing a key and a timestamp keep
    the order the frame gave them; the same stable sort here is what makes the two Welford
    folds add the same numbers in the same order.
    """
    return np.argsort(frame[config.TIME_COLUMN].to_numpy(dtype="int64"), kind="stable")


def transactions(frame: pd.DataFrame) -> Iterable[dict[str, Any]]:
    """The frame's rows as transactions, in stream order."""
    columns = list(frame.columns)
    values = frame.to_numpy(dtype=object)
    for position in stream_order(frame):
        yield dict(zip(columns, values[position], strict=True))


def replay(store: OnlineFeatureStore, frame: pd.DataFrame) -> pd.DataFrame:
    """Run the frame through `get_features` then `commit`, row by row, and collect the rows.

    Returns a frame indexed like `frame` so it lines up with `features.build_features`.
    """
    rows: dict[Any, dict[str, float | int]] = {}
    for position, txn in zip(stream_order(frame), transactions(frame), strict=True):
        rows[frame.index[position]] = store.get_features(txn)
        store.commit(txn)
    out = pd.DataFrame.from_dict(rows, orient="index")
    return out.reindex(frame.index)[store.feature_names]
