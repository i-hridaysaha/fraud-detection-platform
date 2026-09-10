# 0007: The exploratory analysis is computed on the train split

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 2

## Context

ADR 0002 fixed a chronological split and ADR 0006 built a guard that stops a fit from seeing
rows outside the training window. Neither says anything about looking. An analyst who reads a
percentile off all 590,540 rows has not fitted anything, has not tripped the guard, and has
still moved information backwards across the boundary, because the choice made from that
percentile carries it: a winsorising cap, a rare-category floor, a bin edge, a decision that a
column is constant and can go.

The leak is worse than the usual framing suggests. It is not a small optimistic bias in a
number. It is a decision taken with knowledge nobody will have when the model runs, made at a
point in the pipeline where nothing records that it was taken at all.

Stage 2 also has to establish the drift baseline that stage 9 calibrates monitoring against, and
that work is about the difference between splits by definition. So the rule needs an exception,
and the exception needs to be narrow enough that it cannot be used as cover.

## Options considered

**Explore on everything, fit on train.** The common practice, and it is defended on the grounds
that looking is not fitting. Rejected. The output of exploration is not a number, it is a list of
decisions for stage 3, and every one of those decisions is fitted in the sense that matters.
Nothing downstream can tell that a cap came from a percentile of the whole file.

**Explore on train, then re-run on everything at the end to check nothing was missed.** Rejected
for a subtler reason: the check has no action attached to it. If the two disagree, either the
train answer stands, in which case the second run was decoration, or the all-data answer wins,
in which case the rule was never real.

**Explore on train only, with the difference between splits measured explicitly and separately.**
Chosen.

## Decision

Every section under reports/eda/ is computed on the train split, 413,378 rows at
TransactionDT < 10,437,998. scripts/eda.py cuts the frame with `data_loader.time_based_split` and
runs `assert_train_only(train, what="stage 2 exploratory analysis")` before any section builds, so
a section that reached for the whole frame would have to defeat the guard to do it. Every
artifact carries `"computed_on": "train split only, ..."` in its envelope, and a test asserts
that field is present in all nine.

Three places see later rows, and each says so in its own metadata:

- the per-split fraud rates and their intervals in target.json, which are a comparison of splits;
- the PSI block in temporal.json, which bins on train and scores val and test against those bins;
- train_only.json, which exists to measure the size of the mistake this ADR avoids.

Everything else, including all univariate statistics, all missingness work, the correlation
matrix, information value, the entity structure and the duplicate counts, sees train and nothing
else.

## Evidence

reports/eda/train_only.json, regenerated with `.venv/bin/python scripts/eda.py --sections
train_only`. It computes the same statistic twice, once on train and once on all 590,540 rows.

Of 393 numeric feature columns, 130 have a p99 or a median that moves by more than 10 percent
when the later windows are folded in. The largest is V324: its 99th percentile is 7.0 on the
train split and 457.0 over all rows, a relative move of 64.3. V322 moves from 5.0 to 231.0 and
V332 from 1,650.0 to 60,335.0. A cap or a bin edge chosen from the all-data figure would be sized
for a distribution the training window does not contain.

The categorical case is sharper because it is not a matter of degree. Four columns take levels in
val or test that they never take in train: DeviceInfo (240 such levels), id_33 (77), id_31 (22)
and id_30 (4). Those levels cover 8,075 later rows for id_31 alone, 0.0456 of the 177,162 rows in
val and test together. An EDA over the whole file lists them as ordinary categories and a
frequency floor computed there keeps them. A model trained on the training window has never seen
one of them and never will until it is in production.

The specific example worth carrying into stage 3 is id_31, the browser and version string. Its
PSI from train to test is 1.500 (reports/eda/temporal.json), the second highest of 432 features.
Read over the whole file it looks like a high-cardinality categorical with a long tail. Read on
train, with the later windows scored against the train bins, it is a column whose vocabulary
turns over: the versions that dominate the last month are not the versions that dominate the
first. Those are two different findings and they call for two different treatments.

## Consequences

- Every statistic in docs/eda.md is a statement about 413,378 rows and 120 days, not about the
  dataset. The document says so at the top and the artifacts repeat it in every envelope.
- The percentiles in univariate.json are not the percentiles of the file. A reader comparing them
  against a published figure computed on all rows will find they differ, and the difference is
  measured in train_only.json rather than being a discrepancy to explain away.
- Rare-level handling in stage 3 has to be built for levels that do not exist yet, because
  train_only.json shows they arrive. A frequency floor fitted on train is the mechanism; a list
  of known categories is not.
- The drift baseline in temporal.json is the only sanctioned view across the boundary, and its
  bins come from train. If stage 9 needs a different baseline it needs a different artifact, not
  a re-binning of this one.
- This ADR does not protect against the analyst who read a number, remembered it, and used it
  later. Nothing does. What it does is make the sanctioned path cheap and the unsanctioned one
  visible in a diff.
