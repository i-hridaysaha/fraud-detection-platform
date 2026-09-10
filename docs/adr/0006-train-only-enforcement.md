# 0006: Train-only enforcement as a runtime guard

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 1

## Context

ADR 0002 fixed a chronological split. That decides which rows a model may be trained on. It does
not decide anything about the objects fitted alongside the model, and those are where leakage
usually gets in: a scaler fitted on the whole frame, an imputer whose fill value is the global
median, a target encoding computed before the split, a rare-category threshold read off every row
in the file, a quantile cut for binning. None of these is called training. Each of them carries
information from the evaluation windows into the training window, and each of them makes the
val and test numbers better while making the production number worse.

The failure has two properties that decide how to catch it. It is silent: nothing raises, nothing
looks wrong, the metric simply improves. And it is committed at the moment of fitting, which is
usually deep inside a helper that the person writing the experiment is not reading. By the time
anyone looks at a number, the fit that produced it is several stages back.

There is a further reason a convention is not enough here. The entity key groups labels tightly:
0.9664 of multi-transaction entities are label-pure (ADR 0001), and 0.3334 of test-window entities
appear in training. A statistic that crosses the boundary does not leak a little bit of
distribution, it leaks the answer for a third of the evaluation set.

## Options considered

**A documented convention.** Write "fit on train only" in the contributing notes and rely on
review. Rejected. It fails in exactly the case that matters: a helper written in stage 4 and
called from stage 6 by someone who has forgotten it fits anything. A convention also leaves no
trace of having been followed, so a reviewer looking at a number a month later cannot tell whether
it held.

**A scikit-learn Pipeline and nothing else.** A Pipeline does prevent the common version of this,
because `fit` runs on whatever is passed to `fit` and the transformers see only that. It is a good
tool and later stages should use it. It was rejected as the answer here for two reasons. It only
covers what is inside the pipeline: the aggregate tables this platform maintains in a feature
store, the thresholds chosen against review capacity, and the drift baselines are all fitted
quantities that live outside any Pipeline. And it does not check what it was given. A Pipeline
handed the whole frame is a leak with good manners.

**A guard called at the point of fitting, which raises.** Chosen. It is the only option that
inspects the actual rows a fit is about to see, at the moment it sees them, and stops it.

**Guard by construction: hand every fitting site a frame that physically cannot contain later
rows.** The strongest version, and worth aiming at. It is not sufficient on its own, because
"train" is a slice of a frame the caller already holds, so producing the safe frame is one more
step someone can skip, and a guard is what catches the skip. The two compose: later stages should
pass the train slice, and the guard is there for when they do not.

## Decision

`fraud_platform.data_loader.assert_train_only(frame, what=...)` is a runtime check that raises
`TrainOnlyError` unless every row falls strictly before `config.TRAIN_END_DT`. Every fitted object
in every later stage is routed through it before it is fitted. `what` names the thing being fitted
and appears in the message, because the failing call site is usually three layers below the one a
person is looking at.

Four properties make it hard to bypass without meaning to.

- It raises rather than warns. A warning in a notebook is a line of scrollback.
- It raises rather than asserts, so `python -O` cannot remove it.
- A frame with no TransactionDT column is a failure, not a pass. The guard cannot see the
  timestamps, so it refuses to claim it checked them.
- An empty frame is a failure, not a pass. "No row is out of window" is vacuously true of zero
  rows, and in practice an empty frame is a filter that matched nothing far more often than a
  deliberate fit on nothing.

A null timestamp is also a failure, for the same reason as a missing column.

`assert_train_only` returns the frame, so a fit can be written as one expression. A decorator,
`@train_only`, wraps a fit function and checks its first positional frame or its `frame` keyword.
The decorator is the convenient path, not the enforcing one: it removes the excuse for skipping
the check, and it cannot do anything about a function that was never decorated.

## Evidence

Judgement, not measurement. That leakage of this kind inflates offline metrics is not measured in
this repo, and no number here is derived from the claim.

What is measured is that the guard fires. tests/test_temporal_contract.py asserts it accepts the
train split, rejects the val and test splits, rejects a train frame with a single val row appended
to it, and rejects the missing column, empty frame and null timestamp cases. Those tests run on
both a synthetic frame in CI and the real 590,540 rows locally. `make test`.

The tests were also seen to fail. The guard was replaced with a function that returns its
argument unchanged, and 9 tests failed; `time_based_split` was replaced with a shuffled 70 / 15 /
15 split, and 10 failed. Both were restored. Recorded in docs/notes/stage-01.md.

## Consequences

- Every later stage carries an import from the data layer at each fitting site. That is the cost,
  and it is deliberate: an import that has to be written is a decision that has to be made.
- The guard checks where rows sit in time. It says nothing about where their values came from. It
  will pass a frame whose columns were already computed with a statistic taken over the whole
  file, or a target encoding fitted earlier on val, because by then the leak is inside the numbers
  and the timestamps are innocent. Column provenance is a stage 4 problem and needs its own
  mechanism.
- It cannot see a fit that never calls it. An inline `df.mean()` in a later module is invisible
  here. This is the main hole, and the only thing that closes it is the rule that fitted objects
  are constructed through the guard, which is a convention again, one layer up.
- It cannot see label leakage inside the training window, where the rows are legitimate and the
  target is what leaked.
- It cannot see an estimator fitted once through the guard and refitted later without it.
- It trusts `config.TRAIN_END_DT`. If the boundary is wrong, the guard enforces the wrong boundary
  perfectly. tests/test_temporal_contract.py checks the constant against reports/split_summary.json
  so the two cannot drift apart silently.
- Val is where thresholds and calibration are chosen (ADR 0002), and fitting a calibrator on val
  is correct. Those calls must not use this guard. The distinction is between a quantity that
  feeds the model and a quantity chosen after the model exists, and it has to be made by whoever
  writes the call.
