# Serving: the online store, the parity test, and the scoring service

Stage 6 shipped a model and stage 7 priced the shortcuts it refused. This stage puts the model
behind an endpoint, and the claim it makes is not about the endpoint. It is that the feature a
transaction gets at authorisation time is the feature the training row got, to the bit, and
that a test would say so if it stopped being true. Train/serve skew is the way a system that
validated well fails without anyone noticing, because nothing in the ordinary path compares the
two. The parity test is that comparison, and this document is arranged around it.

Every number here is read from a committed artifact: `reports/serving_bundle.json` (`make
register`), `reports/parity.json` (`make parity`), `reports/latency.json` (`make
serving-latency`) and `loadtest/results.json` (`make loadtest`). `tests/test_serving_artifacts.py`
holds each artifact to the modules it describes.

## 0. What was held fixed

The model is the stage 6 booster, unchanged: xgboost, `prepared|native=count,timedelta,vesta`,
186 columns, module-default hyperparameters, seed 42. The thresholds are stage 6's: the band
edges 0.05 and 0.5 on the calibrated probability, derived from the cost matrix of ADR 0029, and
the tuned raw-score threshold 0.2519 from the largest validation F1. The stage 3 pipeline is
refitted on the training split at registration and checked against the stage 3 artifact's
fingerprint by the existing test; the schema is the stage 3 schema with the label removed.
Nothing here learns from data.

## 1. The online store

`fraud_platform.online_store` is a second implementation of the stage 4 definitions and the
stage 5 counters, written for one transaction at a time instead of a stream. It holds state per
key on five grains: the ADR 0001 entity, the card, the six-field content key of ADR 0022, and
the addr1 and device value nodes of stage 5. The 35 features `reports/feature_summary.json`
lists as needing running state are all here, plus `geo_dist1_bucket` from the fitted edges and
the four stage 5 candidate counters; the component features are refused, for the reason ADR
0024 gave, and the store says so when asked for one.

Two calls, in this order, never one that does both:

    features = store.get_features(txn)   # state strictly before this transaction
    store.commit(txn)                    # fold it in, for the next one

`get_features` writes nothing and `commit` returns nothing. The split is what makes the "strictly
before t" rule visible at the call site rather than buried in an implementation (ADR 0033).
Inside, each key keeps two levels of state, the scalars settled strictly before the latest
committed timestamp and the rows at that timestamp, so a read at a tied timestamp sees what
the batch path's block scan sees. The window features read a ring per key, bounded by the
widest window, where every entry carries the key's running prefix so the window sum is the
same prefix difference the batch path takes; the amount moments are Welford in the same order.
That is what makes the agreement below exact rather than approximate.

Two backends, one set of definitions: an in-process dict, and Redis with the scalars in a hash
and the ring in a sorted set scored by timestamp, commits under WATCH. Neither backend computes
a feature; both load and store a `KeyState` that two pure functions read and fold (ADR 0034).

A transaction older than a key's latest committed one is refused. The definitions cannot give
it the state it would have seen, and the store would rather say so (HTTP 409) than fold it in
and drift. The guarantee is therefore for a per-key ordered stream, which is what the file is
and what a partitioned queue would supply.

## 2. The parity test

`tests/test_parity.py` runs the same chronological stream through `features.build_features`
plus `graph_features.build_graph_features` and through `get_features` then `commit`, row by
row, and asserts the feature rows are equal. Counts, flags, timestamps and buckets are compared
exactly, null masks included; the columns that add or average amounts at a relative tolerance
of 1e-9, stated in the test. It runs on the generated stream from `tests/conftest.py`, which was
built to hold tied timestamps, cold keys, repeated amounts and repeated content, and a test
asserts each of those branches was reached; on all three backends; and under `needs_data` on
the first four days of the file.

`scripts/parity.py` is the same comparison as an artifact, on the first 30 days of the file:
134,339 rows, 70,221 entities, 8,641 cards, 40 features. Largest absolute difference over every
feature and every row, 0.0, on the in-process backend and on Redis. Rows outside tolerance, 0.
The stream holds 32 rows tied with an earlier row on the card, 3 on the entity, 278 on the
addr1 node and 2,382 on the device node, so the tie rule was exercised on the file and not only
on the generated frame. The batch path built the 40 columns in 1.1 seconds; the replay took
132 seconds in-process (0.99 ms per row) and 309 seconds on Redis (2.30 ms per row), which is
the honest cost of computing one row at a time from per-key state through a client library.

**The test fails when it should.** Two deliberate breaks, one on each side, both reverted:

- Online path: the ring's upper bound changed from `< t` to `<= t`, so a read includes rows at
  its own timestamp. The generated stream failed at positions 11 and 12, the forced tie block,
  `vel_count_10m_entity` batch 0 against online 1; the real four-day prefix failed on 9 rows of
  `vel_count_10m_card`, batch 14 against online 15 on the first. Six tests failed, the three
  parity runs and the three tie tests, one per backend.
- Batch path: `grouped_trailing_bounds` changed to `side="right"` on its upper bound, the change
  ADR 0018 ruled out. All 900 generated rows failed on the same column, batch 1 against online 0:
  the batch path was counting each row in its own window and the store was not.

The outputs are in `docs/notes/stage-08.md`. Both breaks are one token each; the test caught
both on the first run.

## 3. The service

`fraud_platform.service` is FastAPI with one scoring endpoint. `POST /score` takes the raw
columns of one transaction and returns the raw score, the calibrated probability, the band
(`approve`, `review`, `block`), whether the score clears the tuned threshold, and on request the
top SHAP factors and the store's feature vector. `GET /health` and `GET /model` say which
version is being served.

The model comes from the MLflow registry by name and alias, `models:/fraud-detector@production`,
never from a path. MLflow's registry stages are deprecated in the installed 3.16
(`transition_model_version_stage` carries `deprecated(since="2.9.0")`, checked in stage 8), so
the alias is the stage. `scripts/register_model.py` refits the stage 3 pipeline on train (10.0
seconds), packs it with the booster and a JSON of the column list, schema, calibrator
breakpoints, band edges, tuned threshold and dist1 edges, logs the three files as one model
version under `src/fraud_platform/serving_model.py` as the model's code, and points the alias at
it. The bundle is 3.7 MB of booster, 85.6 MB of fitted pipeline (the target-encoding tables
hold the training split's cumulative labels per level per day) and 78 KB of JSON. The
calibrator is stored as its 138 isotonic breakpoints and applied with `np.interp`; on the
validation scores that reproduces sklearn's `predict` with a largest difference of 0.0.

The registration checks itself against stage 6 and records the result
(`reports/serving_bundle.json`):

| Check | Result |
| --- | --- |
| band edges as raw scores, this fit against `reports/operating_points.json` | 0.039616 and 0.477274 both times, difference 0.0 |
| 500 test rows scored one at a time through the serving path against this script's batch scoring | largest difference 0.0 |
| this script's batch scoring against the stage 6 cache `scores_shipped.parquet` | largest difference 0.0 |
| the SHAP example, `TransactionID` 3556851, 20 factors against `reports/shap/shap_example.json` | score 0.9966 both, base value -3.3691 both, largest contribution difference 0.0, top factor C14 both |

The factors come from the booster's own `pred_contribs`, TreeSHAP with the tree-path-dependent
weighting, which is what `shap.TreeExplainer` computed for stage 6; the check above is the
evidence that the two are the same numbers.

The payload is validated three ways, in order. A column the file does not have is refused
before anything is built, and so is a label. The row goes through the stage 1 dtype map, so a
count arriving as text is refused where it is cast. Then the stage 3 pipeline runs on the
one-row frame and `schema.validate` holds the result to the stage 3 contract with the label
removed: 246 of the 247 declared columns, every family, every null rule, every bound, with
every violation listed in the 422 rather than the first. A measured null-free integer column
that arrives null reaches the schema as a float and is refused as the family mismatch it is.

The store is read before the row is prepared and written after the score is out. The shipped
stack does not read the store's features, and that is stated plainly rather than papered over:
stage 6's ablation found the causal block did not ship (ADR 0028), so the served model reads the
prepared columns and the store's vector travels beside it, returned on request. The path is
wired so that a bundle whose column list names a store feature reads it from the vector
(`serving.matrix`), which is what a stack with the causal block on would do, and a test covers
that join.

## 4. Latency

`reports/latency.json`. Five fresh interpreter processes, each timing every model on the same
300 test rows, single-row `predict_proba`, milliseconds; ranges are across the five processes.
Every model pinned to one thread, the way the served booster is pinned (ADR 0035); the two
families stage 6 did not ship refitted from the stage 6 cache at the shipped configuration.

| Model, one thread | p50 range | p99 range | ratio to xgboost, p50 |
| --- | --- | --- | --- |
| xgboost, shipped | 0.077 to 0.081 | 0.146 to 0.158 | 1 |
| random forest | 3.55 to 3.59 | 3.76 to 4.41 | 43.9 to 46.2 |
| logistic regression | 0.033 to 0.034 | 0.041 to 0.043 | 0.410 to 0.430 |

The between-process spread was under 5 percent on the medians here, not the factor of two the
stage brief warned of; the ratios moved less than the points, as it said they would.

The pin: the same booster unpinned (every core) on the same single rows is slower by 1.49 to
1.59 in every process; on a batch of 1,000 rows it is faster by 4.00 to 4.75. Single-row
serving pins; a batch scorer must not.

The request, step by step, on the first 100 of those rows as raw payloads, medians of the five
process p50s, without the store:

| Step | p50, ms |
| --- | --- |
| payload to a typed one-row frame with the entity key | 4.60 |
| the stage 3 pipeline on that frame | 23.8 |
| the stage 3 schema | 3.41 |
| the model's matrix | 2.08 |
| the booster | 0.081 |
| calibrate and band | 0.007 |
| the five factors | 1.81 |
| whole path, five factors, no store | 37.6 |

The store on top: in-process, 0.041 ms to read and 0.039 ms to commit; Redis on localhost,
0.183 ms to read and 0.90 ms to commit. The request is the pandas pipeline on a one-row frame,
not the model, and the stage halved it from where it started: the stage 3 apply functions
inserted derived columns one at a time and the validator went through pandas per column, and
on 413,378 rows that cost nothing worth noticing. The first measurement of the serving path was
65 ms; the same pipeline attaching its columns in one insertion is 37.6 ms with identical
values and the identical stage 3 fingerprint. What remains is per-column pandas overhead in
the encoders, and a row-native preparation would be a second implementation of stage 3 with
its own parity obligation; that is left as the next step, not taken.

## 5. Load test

`loadtest/results.json`, rendered as `loadtest/results.md`. One uvicorn worker on the Redis
store, the first 2,000 rows of the test split replayed in chronological order over HTTP, one
connection per client, concurrency 1 to 32. The dispatcher hands rows out in stream order and
holds the next one back while an in-flight row shares its entity, card or content key, which
is the per-key ordering the store needs, done by the client and counted.

| Clients | In flight | Throughput, req/s | Client p50, ms | Client p99, ms | Errors |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 25.2 | 38.1 | 84.2 | 0 |
| 2 | 2.0 | 22.6 | 83.9 | 133.5 | 0 |
| 4 | 3.8 | 21.6 | 174.7 | 261.7 | 0 |
| 8 | 6.9 | 20.3 | 365.4 | 526.4 | 0 |
| 16 | 10.1 | 20.2 | 495.2 | 924.4 | 0 |
| 32 | 11.1 | 20.1 | 508.5 | 1322.9 | 0 |

One worker saturates at its first client. Peak throughput 25.2 req/s at one client; every
level after it is lower, and the client p50 grows with the number of clients, 38 ms to 84 to
175 to 365. The path is Python and pandas under one interpreter lock, so a second concurrent
request adds contention and no parallelism; the pinned booster is 0.08 ms of it and is not
where the ceiling comes from. Four workers, same rows, same sweep: 24.8 req/s at one client,
46.6 at two, 59.7 at four, peak 62.1 at eight, then 52.1 and 51.8. Processes scale it, threads
do not, and the four-worker curve flattens where the dispatcher's ordering holds effective
concurrency to 6.0 of 8 and 8.6 of 16 in flight.

Zero errors at every level of both sweeps. The first run of the sweep had 41 errors at 32
clients, all client-side: a client held back by the dispatcher for longer than uvicorn's
five-second keep-alive found its connection closed. The server's keep-alive timeout is set to
300 seconds for the load test and the artifact is the rerun; the first run is in the notes.

## 6. What this does and does not demonstrate

- The parity guarantee is real and measured to the bit, on 134,339 rows and 40 features, on both
  backends, with ties in the stream; and the test that guards it has been seen to fail.
- The latency profile is real: 37.6 ms per request without the store, 0.08 ms of which is the
  booster, on this laptop.
- The throughput number is a laptop number. One machine, the client on the same ten cores as
  the server, a historical file replayed as fast as the server takes it. 25 req/s per worker
  says what this path costs per request on this hardware and nothing about a machine it was
  not measured on.
- The store's guarantee holds for a per-key ordered stream. A deployment supplies the ordering
  or accepts a documented deviation; neither is built here.
- The served model does not read the store. The store exists so that the model which would
  read it, or the stage 9 candidate that might, is served by the same path with the same
  guarantee; the guarantee is on the feature vector and is not a claim about the shipped
  score.
- This is a single-machine replay of a file, not production traffic. Nothing here measures a
  real request mix, a real network, or a store under a real key distribution over months.

## What stage 9 inherits

- A registered bundle under an alias, and a service that loads by alias. A promotion is a
  registration plus one alias move, and the smoke checks in `scripts/register_model.py` are the
  first gate a candidate passes.
- The parity test as the second gate: a candidate whose stack reads the store is served by the
  same path, and the path's feature vector is the batch path's.
- The store's feature vector on every scored request, which is the input a drift monitor on the
  features would read.

## What is not established

- The serving latency of a row-native preparation. The pandas one-row path is 37.6 ms and the
  model 0.08 ms; the gap is measured, the fix is not built.
- Whether pinning helps or hurts under concurrent load. The benchmark isolates the single-row
  call; the load test runs pinned only.
- What the store does with an out-of-order arrival other than refuse it. A late-arrival policy
  with a bounded tolerance is a design not written.
- The store's memory over months. `state_footprint` bounds the structures on this file; nothing
  here ran long enough to measure growth.
- Throughput on any machine but this one.
