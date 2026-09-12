# Stage 8: serving

Stages 4 and 5 built the features over a stream with prefix sums and searchsorted, and stage 6
shipped a model on the prepared columns. This stage builds the path an authorisation request
takes: a store that holds per-key state and answers one transaction at a time, a service in
front of the model, and the test that says the two paths agree. The interesting parts were
where they nearly did not, and what the request turned out to be made of.

## Breaking the parity test on purpose

A parity test that has never failed is decoration, so before trusting it I broke each side once
and read what it said. Both breaks are one token. Both were reverted the same minute, and the
suite was green again before anything else was touched.

**The online side.** In `online_store.window`, the ring's upper bound is
`_first_at_or_after(state.ring, t)`, the store's `searchsorted(side="left")`: the first entry at
`t` is outside the window. I changed it to `t + 1`, which pulls the row's own timestamp in.
`pytest tests/test_parity.py -m "not needs_data"`:

    E   AssertionError: vel_count_10m_entity: 2 rows differ, first at present-row 11 (batch np.float64(0.0), online np.float64(1.0))
    E   assert 2 == 0
    FAILED tests/test_parity.py::test_the_online_store_reproduces_the_batch_pipeline[memory]
    FAILED tests/test_parity.py::test_the_online_store_reproduces_the_batch_pipeline[fakeredis]
    FAILED tests/test_parity.py::test_the_online_store_reproduces_the_batch_pipeline[redis]
    FAILED tests/test_parity.py::test_a_read_at_the_committed_timestamp_excludes_the_tie[memory]
    FAILED tests/test_parity.py::test_a_read_at_the_committed_timestamp_excludes_the_tie[fakeredis]
    FAILED tests/test_parity.py::test_a_read_at_the_committed_timestamp_excludes_the_tie[redis]
    6 failed, 9 passed, 2 deselected in 2.40s

Rows 11 and 12 are the forced tie block in the generated frame: three rows on one entity at one
timestamp. The batch path gives them all a ten-minute count of 0, because none of them can see
the others; the broken store gave the second and third a count of 1, because the first had been
committed at the same timestamp and the bound now let it through. The tie test failed on its
first velocity assertion for the same reason:

    >       assert store.get_features(tied)["vel_count_24h_card"] == 0
    E       assert 1 == 0

The same break on the real four-day prefix:

    E   AssertionError: vel_count_10m_card: 9 rows differ, first at present-row 942 (batch np.float64(14.0), online np.float64(15.0))
    E    +  where 9 = array([  942,   945,  1055,  3187,  4672,  7085,  8182, 11802, 15647]).size
    FAILED tests/test_parity.py::test_parity_on_the_real_stream[memory] - Asserti...

Nine rows in 16,129, every one a card with another transaction at the same second. That is the
whole population the rule moves on the file's first four days, and the test found all nine.

**The batch side.** In `features.grouped_trailing_bounds`, the upper bound is
`np.searchsorted(keys, keys, side="left")`. I changed it to `side="right"`, the choice ADR 0018
rejected, which counts every row in its own window:

    E   AssertionError: vel_count_10m_entity: 900 rows differ, first at present-row 0 (batch np.float64(1.0), online np.float64(0.0))
    E   assert 900 == 0
    FAILED tests/test_parity.py::test_the_online_store_reproduces_the_batch_pipeline[memory]
    FAILED tests/test_parity.py::test_the_online_store_reproduces_the_batch_pipeline[fakeredis]
    FAILED tests/test_parity.py::test_the_online_store_reproduces_the_batch_pipeline[redis]
    3 failed, 12 passed, 2 deselected in 2.36s

All 900 rows, off by exactly one, on the first column checked. The store was still right; the
test does not know which side is right, only that they differ, and that is enough.

What I take from the two: the test's failure message names the column, the row and both values,
which is what a person fixing a real skew would need first. And a break that moves nine rows in
sixteen thousand is caught as surely as one that moves all of them, because the comparison is
exact and the stream has the ties in it.

## Bit for bit, and why

I expected to write a tolerance and explain it. The store's window sums are the batch path's
prefix differences (every ring entry carries the key's running prefix, so the store subtracts
the same two floats), the amount moments are Welford in the same order, and the stable sort on
ties gives both sides the same fold order. On 134,339 rows and 40 features the largest
difference on either backend is 0.0. The tolerance is still in the test, at 1e-9 relative on
the amount columns, because an assertion should say what it accepts; the artifact says what was
observed.

The one place this could have gone wrong is the ring's prefix base. When entries fall out of
the widest window, the prefix of the newest one to fall out has to stay behind as what the next
window subtracts, or a window whose lower bound is the ring's first entry would subtract zero
where the batch subtracts the group's history to that point. A unit test holds that case; the
first draft got it right by construction and I only noticed how easy it would have been to get
wrong when writing the test.

## The number that changed my mind: 65 milliseconds

The first timed request through the serving path took 65 ms, and the booster was 0.3 ms of it.
Everything else was pandas on a one-row frame: 434 columns through the stage 3 pipeline, 155
derived columns each inserted with `out[name] = values`, then a validator that touched 247
columns through Series objects. On 413,378 rows none of that shows; on one row the insertion is
the work.

The fix that was mine to make was to attach derived columns in one insertion per step and to
run the validator's per-column checks in numpy, with identical values and dtypes and the same
stage 3 fingerprint (`test_prep_artifacts` asserts the committed one, and it passed). That took
the path to 37.6 ms with five factors. The remaining time is per-column overhead inside the
encoders, and the fix for that is a row-native preparation, which is a second implementation
of stage 3 and would need its own parity check. I left it unbuilt and wrote down where the
milliseconds are (`reports/latency.json`, `summary.serving_path_p50_ms`).

The lesson is not about pandas. It is that a batch pipeline reused unchanged for serving is
correct by construction and slow by construction, and both are worth knowing before choosing.

## One worker saturates at one client

I expected the one-worker sweep to climb for a few levels before flattening. It peaked at the
first client, 25.2 req/s, and every level after was lower, with the p50 growing 38, 84, 175,
365 ms as the clients doubled. The request is Python under the interpreter lock; a second
concurrent request buys contention and nothing else, and the pinned booster is 0.08 ms of the
37 and not where the ceiling is. Four workers took the peak to 62.1 req/s at eight clients,
which is the scaling one gets from processes and the flattening one gets from the dispatcher
holding rows back to keep keys ordered (6.0 of 8 requests actually in flight at that level).

The first run of the sweep had 41 errors at 32 clients. All were client-side: uvicorn's keep-alive
closes an idle connection after five seconds, and a client the dispatcher held back for longer
than that came back to a closed socket. The load test now sets the server's keep-alive to 300
seconds and the artifact is the rerun, with zero errors at every level. I kept the first run's
counts here because "zero errors" is only believable next to the run that had some.

## The spread that was not there

The brief warned that single-row latency moves by a factor of two between invocations. Across
five fresh processes on this machine the xgboost medians ran 0.077 to 0.081 ms, a spread of 5
percent, and the ratios between models moved less than the points. I ran the benchmark as
repeats anyway and report the spread, because the claim is about the machine and not the
model and the next machine may show it. What did hold was the direction of the pin: the
unpinned single-row call was slower in every process, by 1.49 to 1.59, and on a batch of a
thousand rows faster by 4.0 to 4.75. The comment in `serving.py` says that and points at the
artifact; it does not say anything about concurrent requests, because nothing here measured
that.

## Smaller things

- MLflow 3.16 refuses its filesystem backend unless opted into, and marks registry stages
  deprecated since 2.9. The registry is a SQLite file under `mlruns/` and the alias
  `production` is the stage. Both are recorded in `serving.tracking_uri` and
  `serving.registry_uri`.
- The model-from-code path needs the scorer's methods to win the method resolution order over
  `PythonModel`'s no-op `load_context`. The first registration loaded a bundle that was `None`.
- The store holds the stage 5 counters only when the served stack reads them. Their keys are
  hub values (one device string is shared by a large share of the stream), and holding them
  for a stack that ignores them would have made every request touch two keys nobody reads, and
  every concurrent replay collide on them.
- Concurrent clients replaying one stream produce out-of-order arrivals on shared keys, which
  the store refuses. The load generator supplies per-key ordering from the client side and
  counts the stalls. A deployment gets it from a partitioned queue; the note is here so the
  next person does not learn it from a wall of 409s.
- The random forest's p50 on one thread is 3.57 ms, forty-five times the booster's, and 15.8
  ms unpinned. The family stage 6 shipped is also the one that serves.
