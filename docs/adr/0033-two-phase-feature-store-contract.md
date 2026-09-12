# 0033: The online store reads and writes in two separate calls

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 8

## Context

Stage 4 defined every behavioural feature as a function of the rows strictly before the row's
own timestamp, and stage 5 kept the rule for the graph counters. The batch path enforces it in
the arithmetic: `searchsorted(ts, ts, side="left")` stops before the row itself, and a block of
rows sharing a key and a timestamp is read against the state before the block and then added
whole. `tests/test_causality.py` recomputes every feature by brute force and asserts truncation
invariance, so on the batch side the rule is a tested property rather than a convention.

Stage 8 moves the same definitions into an online store that holds state per key and answers
one transaction at a time. The rule has to survive the move, and it has to be visible where the
store is called, because the failure this stage exists to prevent, the online path drifting from
the batch path, is silent: a store that folds the transaction in before reading would return
numbers that look right, load-test fine, and disagree with every training row by exactly one
transaction.

## Options considered

**One call, `features_for(txn)`, that reads the state and updates it.** The shape most feature
store clients have, and the shape a first draft reaches for. Rejected because the order of the
two operations inside it is invisible from outside: a reviewer reading the endpoint cannot tell
whether the returned vector excludes the transaction, and neither can a test without reaching
into the store's internals. The exclusion becomes a property of an implementation detail rather
than of the call sequence, and a refactor can flip it without any call site changing.

**One call with a flag, `features_for(txn, commit=True)`.** Same call, the order made
configurable. Rejected for the same reason with one more: the flag's default becomes the
contract, and a caller who omits it gets whichever behaviour the default encodes.

**Two calls, `get_features(txn)` then `commit(txn)`, never one that does both.** Chosen.
`get_features` writes nothing and `commit` returns nothing, so the only way a feature vector can
describe the state after the transaction is for the caller to have committed first, which is
visible as two lines in the wrong order. The store can be tested for the property directly:
two reads of the same transaction return the same vector, and a read at the timestamp of the
latest committed row sees the state before that row.

## Decision

`fraud_platform.online_store.OnlineFeatureStore` exposes `get_features` and `commit` and nothing
that combines them. Inside, every key holds two levels of state: `settled`, the scalars over
rows strictly before the latest committed timestamp, and `pending`, the rows at that timestamp.
A read at the pending timestamp sees `settled` only; a read at a later timestamp sees both. That
is the batch path's block semantics one row at a time, and it is what makes a tied row agree
with the batch number.

The service calls `get_features` before the row is prepared and `commit` after the score is out
(`serving.score_transaction`). A transaction older than a key's latest committed one is refused
(`OutOfOrderError`, HTTP 409) rather than folded in: the definitions are "strictly before t" and
an older row cannot be given the state it would have seen without rewriting the key's history.
The store therefore guarantees parity for a per-key ordered stream and says so loudly when it is
handed anything else.

## Evidence

`reports/parity.json`, regenerated with `make parity`; `tests/test_parity.py`, run on every
push.

The first 30 days of the file, 134,339 rows, replayed through `get_features` then `commit` on
both backends, against `features.build_features` and `graph_features.build_graph_features` over
the same rows: 40 features, largest absolute difference 0.0 on every feature, null masks
identical, on the in-process backend and on Redis. The stream holds 32 rows tied with an earlier
row on the card grain, 3 on the entity grain, 278 on the addr1 node and 2,382 on the device
node, which is where the two-level state is exercised.

The test can fail, and was made to. With the store's upper window bound changed from `< t` to
`<= t`, `tests/test_parity.py` failed on the generated stream at the tied rows, positions 11 and
12, `vel_count_10m_entity` batch 0 against online 1, and on the real four-day prefix on 9 rows of
`vel_count_10m_card`. With the batch path's upper bound changed to `side="right"`, every one of
the 900 generated rows failed on the same column, batch 1 against online 0. Both breaks were
reverted; the outputs are in `docs/notes/stage-08.md`.

`tests/test_parity.py::test_get_features_writes_nothing` and
`test_a_read_at_the_committed_timestamp_excludes_the_tie` are the direct assertions of the
contract, on all three backends.

## Consequences

- Every caller carries two lines, and a reviewer can see the order. That is the point.
- The guarantee is conditional on per-key ordered arrival. Concurrent clients replaying one
  stream break it whenever two in-flight rows share a key; the load test supplies the ordering
  from the client side and counts what it cost (`loadtest/results.json`, `dispatch_stalls`). A
  deployment would get it from a partitioned queue or would have to accept a documented
  deviation, and that deviation is not implemented here.
- A read followed by no commit is a legitimate use (a shadow score, a what-if) and costs
  nothing; a commit followed by no read is a backfill. Both are the same two calls.
- The two-level state makes a commit at a new timestamp fold the pending block first, so the
  fold order inside a tie is arrival order, the same order the batch path's stable sort gives.
  A store that received tied rows in a different order from the file would fold the Welford
  moments in a different order and agree with the batch path to rounding rather than to the bit.
