# 0034: Redis sorted sets for the shared store, an in-process dict as the fallback, one set of definitions

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 8

## Context

The online store holds, per key, a handful of scalars (a count, two timestamps, a running mean
and second moment, a set of device combinations, a counter per address value) and a ring of
timestamped entries for the trailing windows. The service needs that state to survive a process
restart and to be shared between worker processes, which means a store outside the process; the
test suite, the parity script and a developer without a server need the same store without one.
The brief asks for a Redis sorted-set implementation and an in-process fallback with identical
semantics, and the question this ADR answers is where the semantics live so that "identical"
is a fact about the code rather than a hope about two implementations.

## Options considered

**Two stores, each computing the features from its own state.** A Redis class that folds and
reads with Redis commands (INCRBYFLOAT for the moments, ZCOUNT for the windows) and a Python
class that does the same in a dict. Rejected: it is two implementations of the definitions,
and the parity test would then have to cover both against the batch path, which is three
implementations of one thing. It also loses exactness: INCRBYFLOAT on a running mean is not
Welford, and a window sum from ZRANGEBYSCORE and a client-side add is not the batch path's prefix
difference. Both would agree with the batch path to rounding, not to the bit.

**Server-side Lua for the fold, Python for the fallback.** Atomic and one round trip per
commit, and still two implementations of the arithmetic. Rejected for the same reason, with a
practical one added: the in-memory Redis the tests use has no Lua without an optional
dependency, so the path CI exercises would not be the path the service runs.

**One set of pure functions over a `KeyState`, and backends that only load and store.** Chosen.
`online_store.read_features` and `online_store.folded` take loaded states and return features
and next states; they contain every definition and touch no storage. `MemoryBackend` keeps
`KeyState` objects in a dict. `RedisBackend` keeps each key's scalars in a hash and its ring in a
sorted set scored by timestamp, and runs a commit as WATCH, read, compute, MULTI, write, EXEC,
retrying when another writer got there first. The two backends cannot disagree on a feature
because neither computes one.

## Decision

The sorted set is the ring and nothing else. Its score is the timestamp, its member carries the
sequence number first (so ties keep arrival order under Redis's lexicographic tie-break) and
then the amount and the key's running prefix at that entry. An append is one ZADD; pruning
below the widest window is one ZREM of the entries that fell out; the ring never travels back
to be rewritten. A read fetches the ring whole, which is bounded by the key's activity inside
the widest window plus whatever the last commit had not pruned yet, and the window arithmetic
(`searchsorted` bounds, prefix difference, tie exclusion) runs in Python on it, the same code
the fallback runs on its list.

The scalars are one hash per key with the settled block, the pending block and its timestamp,
the pruned prefix and the sequence counter, serialised as JSON strings. The fold that turns
pending rows into settled scalars is the same `fold` function on both backends.

Commits are optimistic. `RedisBackend.transaction` watches every hash and ring it will write,
reads them, applies the update, and writes inside MULTI/EXEC; a WatchError means another writer
touched a key in between, and the update is recomputed from the new state. Twenty retries
before giving up. `MemoryBackend.transaction` takes a lock. Both hand back the same next state
for the same current state because the update is the same function.

The store is Redis when the service sees `FRAUD_REDIS_URL` and in-process otherwise. The test
suite runs the Redis backend against `fakeredis` in CI and against a real server when one
answers at `FRAUD_REDIS_URL`, skipping visibly when none does.

## Evidence

`reports/parity.json`, regenerated with `make parity`: 134,339 rows, 40 features, largest
absolute difference 0.0 on the in-process backend and 0.0 on Redis, against the batch path.
`tests/test_parity.py` runs the same comparison on three backends (memory, fakeredis, redis)
and passes on all three; `tests/test_online_store_units.py` round-trips a state through the
Redis field encoding for every grain, checks that pruning leaves the prefix base behind, and
makes a commit lose the WATCH race once and land on the second attempt.

`reports/latency.json`, block `summary.store_p50_ms`, regenerated with `make serving-latency`:
per transaction, `get_features` at 0.041 ms in-process and 0.183 ms on Redis, `commit` at 0.039
ms in-process and 0.90 ms on Redis, medians of the five-process p50s. The Redis commit is the
WATCH round trip, one HGETALL and one ZRANGE per key answered one at a time inside it, and the
MULTI/EXEC; three keys per transaction on the served stack.

`reports/parity.json`, `backends`: the full replay costs 0.99 ms per row in-process and 2.30 ms
on Redis, which includes building each transaction from the frame.

## Consequences

- One place to read the definitions and one place to change them. A change to a feature is a
  change to `read_features` or `fold`, and both backends carry it at once.
- The Redis commit is about a millisecond of round trips on a local server. A single Lua
  script per commit would make it one round trip and would be the first thing to add if the
  store's share of the request mattered; on the served path it is about a fortieth of the
  request (`reports/latency.json`).
- Reading the ring whole is the price of exactness. A key with thousands of transactions in a
  day would return thousands of entries per read; `reports/feature_summary.json`'s
  `state_footprint` is the measured bound on this data, and a key past it would need the window
  arithmetic moved server-side.
- The state under a key is JSON in a hash, so a schema change to `KeyState` is a migration of
  every key. The sequence counter and the pending block are what a migration has to preserve to
  keep the tie semantics.
- `fakeredis` is a dev dependency and the CI path. It reproduces the commands this backend uses
  (hash, sorted set, WATCH, MULTI, SCAN) and the real-server test is the check that it does.
