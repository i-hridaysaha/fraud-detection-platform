# 0018: Causal windows are searchsorted over the sorted stream, and ties are excluded

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 4

## Context

Every behavioural feature in stage 4 is an aggregate over a trailing window of one key's
transactions. The binding rule of the whole project is that the aggregate for a transaction at
time t reads only rows strictly before t. Not before t within the fold, and not before t plus
whatever the window function happens to include: strictly before t, on the whole stream, because
that is the information an authorisation endpoint has when it is asked for a score.

The rule is easy to state and easy to break by accident, and the ways it breaks are not exotic.
A window that includes its right edge counts the transaction being scored. A window centred
rather than trailing reads the next hour. A same-timestamp companion is included by a `>=`
where a `>` was meant. None of these raises, none of these looks wrong in a diff, and every one
of them produces a better offline number than the correct version does.

So the question this ADR settles is not whether to respect the rule. It is where in the code the
rule lives: in a parameter that a later edit can change, or in the arithmetic.

## Options considered

**A rolling join, or `pandas.DataFrame.rolling` with a time offset and `closed=`.** The obvious
choice, and it computes the right numbers when it is configured correctly. Rejected on where the
correctness sits. `closed="left"` is right and `closed="both"` is wrong, they differ by one word,
the wrong one is the pandas default for an offset window, and nothing in the type system or the
output distinguishes them. The exclusion of the present row would be a keyword argument, and a
keyword argument is exactly the kind of thing that gets changed by someone tidying a call site.
There is also a performance argument, since a per-entity rolling apply over the 217,850 entities stage 0 counted is
slow, but the performance argument is not why this was rejected.

**A self-join on the entity with a `TransactionDT <` predicate, aggregated by group.** Correct by
construction, because the inequality is written out. Rejected on cost: the join is quadratic in
the size of the largest entity and this dataset has entities with 319 transactions and cards with
14,932, so the intermediate frame is large enough to matter on 590,540 rows. It is a reasonable
choice at a smaller scale and it is what the brute-force reference in
`tests/test_causality.py` does, on a sample, precisely because it is obvious.

**`np.searchsorted` twice over the time-sorted stream.** Chosen. The window becomes two indices
and the exclusion becomes the choice of `side`:

```python
lo = np.searchsorted(ts, ts - window, side="left")   # first row at or after t - window
hi = np.searchsorted(ts, ts, side="left")            # first row at t
count = hi - lo
total = prefix[hi] - prefix[lo]
```

There is no window argument to misconfigure and no `closed=` to set. The slice `[lo, hi)` cannot
contain the current row, because `hi` is the index of the first row whose timestamp equals `t`
and the current row is at or after that index. Including the present row would require changing
`side="left"` to `side="right"` on the upper bound, which is a deliberate edit to a line whose
only purpose is that choice, and `tests/test_causality.py` asserts the difference the two produce
rather than asserting the convention.

`fraud_platform.features.trailing_bounds` is those four lines for a single key's stream.
`grouped_trailing_bounds` is the same thing for every key at once, by packing the group code and
the timestamp into one ascending int64 key with a stride of 2**32, which exceeds the maximum
TransactionDT stage 0 measured at 15,811,131. The lower search target is clamped to the group's
own start so a window wider than a key's first timestamp cannot reach into the previous key's
rows. A test loops the single-key primitive per group and asserts the two agree.

## Decision

Trailing windows are computed with `np.searchsorted` over a stream sorted by key then time, with
`side="left"` on both bounds. Expanding windows use the same upper bound and the key's first row
as the lower bound, so the tie exclusion is the same code in both cases.

**Same-timestamp ties are excluded.** A transaction sharing its timestamp with another on the same
key does not see it, in either direction. The reason is the serving path and not the arithmetic:
at authorisation you do not know about a transaction being authorised at the same instant
somewhere else, and a feature that assumes you do is a feature that will read differently online
than it did offline. Stage 0 measured 573,349 distinct TransactionDT values across 590,540 rows,
so ties are common enough that the choice moves real numbers rather than being a formality.

The same rule governs the sequential features that need a set or a counter rather than a sum.
`fraud_platform.features._blocks` walks the stream in blocks of equal `(key, timestamp)`, every
row of a block reads the state as it stood before the block, and the whole block updates the state
afterwards. Row by row, the first of two simultaneous transactions would inform the second.

**Everything accumulates inside its own key.** A restarted cumulative sum for the window totals
and Welford for the running mean and spread, rather than one global `np.cumsum` and a difference
of sums of squares. This is part of the same decision and it was not the first implementation. See
Evidence.

## Evidence

`tests/test_causality.py`, which is the artifact for this ADR. It runs in CI on a generated frame
built to contain every branch, and it runs on the real file under `needs_data`. Regenerate with
`.venv/bin/pytest tests/test_causality.py`.

Three properties, and they are not the same property:

| Property | What it catches |
| --- | --- |
| Brute-force agreement | every feature equals a reference recomputed over `TransactionDT < t` by filtering the whole frame, on a sample chosen to include cold rows, tied rows and rows with history |
| Truncation invariance | features on the whole stream equal features on the stream cut at time T, for every row before T, compared bit for bit |
| Tie exclusion | a tied transaction is not in its companion's history, and `side="right"` is shown to produce a different answer |

The reference implementation shares nothing with the module: it does not sort, does not use
searchsorted, does not factorize, and builds its own notion of a device combination and a content
key out of the raw columns.

**Four deliberate breaks, applied, run, and reverted.**

| Break | Change | Tests failed | First failure |
| --- | --- | --- | --- |
| Trailing upper bound | `side="left"` to `side="right"` in `grouped_trailing_bounds` | 3 | brute-force agreement |
| Expanding upper bound | the same in `grouped_prior_bounds` | 5 | brute-force agreement, and the tie test |
| Transductive mean | the amount mean taken over the whole entity rather than the prefix | 2 | brute-force agreement, and truncation invariance |
| Mode over the block | the addr counter updated before the block is read instead of after | 1 | brute-force agreement |

The third is the one worth reading. It is the shape ADR 0020 refuses, and truncation invariance is
the test that catches it independently of whether the future happens to resemble the past.

**Why the accumulation is per key.** Truncation invariance failed on the real frame with a global
`np.cumsum`, by 2e-4 on `amt_zscore_within_entity` for entities that repeat one amount. A global
cumulative sum carries a rounding error proportional to the total it has accumulated, so the last
bits of a ten-minute window sum depended on every amount earlier in the file, and after the
difference was taken, on amounts from rows later than the one being computed. The values were
immaterial to any model and the dependency was real, so the arithmetic changed rather than the
test's tolerance: `grouped_prefix_sum` restarts at every key, and the running mean and spread come
from Welford, which is also the three scalars a serving path would hold. The truncation tests now
compare with no tolerance.

The remaining tolerance in the suite, 1e-6, is against the reference and not against the past: the
reference adds a Python list with `sum()` and the module accumulates per key, and adding the same
numbers in a different order gives a different last bit. The observed disagreement is
662.8500000000001 against 662.85.

## Consequences

- The exclusion of the present row and of its ties is a property of two `side="left"` arguments
  whose only purpose is that exclusion, checked by a test that shows what the other choice does.
  It is not a convention anyone has to remember.
- Features are computed over the whole stream, train and later windows together, and that is
  correct rather than tolerated. A validation row reads earlier training rows, as it would in
  production. A training row cannot read a validation row, because validation rows are later in
  time. There is therefore no `assert_train_only` on the feature build, and the docstring says why
  rather than leaving its absence to be noticed.
- Stage 8 inherits an implementation whose state is small and explicit: two indices into a ring
  for a window, three scalars for a running moment, a set for novelty. The catalogue in
  `reports/feature_summary.json` names the state per feature.
- A window wider than 2**32 seconds, or a timestamp beyond it, raises rather than silently
  wrapping the packed key. That is a guard against a change in the file rather than a live risk:
  the measured maximum is 15,811,131.
- The tie exclusion costs measurable signal and that is accepted, not hidden. A pair of
  simultaneous transactions on one card is a plausible card-testing shape and this implementation
  refuses to see it at the moment it happens. It sees it one transaction later.
- This does not say a rolling join is wrong. It says that on this codebase the exclusion should
  not be a keyword argument, and that a slower obvious implementation is worth keeping as the
  test's reference rather than as the pipeline.
