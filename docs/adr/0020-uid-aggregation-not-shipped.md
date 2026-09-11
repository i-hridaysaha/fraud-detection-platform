# 0020: The published UID aggregation is not shipped

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 4

## Context

The best-known feature construction on this dataset builds a client identifier and then attaches
group statistics of other columns to it. The identifier and the aggregation are two separate
things and only the second is at issue here.

The identifier is `card1_addr1` pasted to `floor(day - D1)`. That is the same key stage 0 arrived
at independently and recorded in ADR 0001, chosen there on measured label purity rising from
0.8480 under card1 to 0.9664 under the three-part key. This repo already uses it and this ADR does
not revisit it.

The aggregation is the part that is not shipped. It takes a frame, groups it by that identifier,
and computes means and standard deviations of the C, M and normalised D columns per group,
attaching the group's statistic back to every row in it. Roughly 45 such columns in the published
version. The construction also drove a post-processing step that replaced a row's prediction with
its group's average prediction.

The measured claim about why it works is specific and worth stating properly, because it is a
clever observation rather than a brute-force one. The three-part identifier is coarser than a
single client, so a group can contain more than one. A finer normalised delta, D15 minus its day
index, is closer to one client each. So `groupby(uid).D15n.std()` is zero when the group holds a
single client and positive when it holds several, and a model reading that column learns whether
the identifier it is being handed is trustworthy on this row.

## What was fetched and read

Kaggle renders its competition pages client side. Stage 0 recorded that the data description
returned no text when fetched, and the same held when this was written for both the competition
discussion page and the first-place writeup page: each returned a title and no body.

What was fetched and read for this record is an NVIDIA Technical Blog post by Carol McDonald and
Chris Deotte, dated 26 January 2021, at
`https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/`.
It states the identifier construction as
`X_train['UID'] = X_train.card1_addr1.astype(str)+'_'+np.floor(X_train.day-X_train.D1).astype(str)`,
lists group statistics including `C1_UID_mean` through `C14_UID_mean`, `M1_UID_mean` through
`M9_UID_mean`, `TransactionAmt_UID_mean`, `TransactionAmt_UID_std`, `D4_UID_mean` and
`D4_UID_std`, and reports a final score of 0.9677 public and 0.9459 private, in first place.

**What that page does not say, and what is therefore not claimed here:** whether the frame the
groupby was applied to combined the training and test sets. That question is not settled by
anything read here and no claim about it appears in this repo. The argument below does not
need it, and that is worth being explicit about, because "they concatenated train and test" is the
version of this objection that gets repeated and it is not the version that matters.

## Options considered

**Ship it, because it is measured to work.** The strongest form of this argument is that a
first-place result is evidence and this repo has none of its own yet. Rejected on the grounds
below, and it is rejected as a **production** feature rather than as a technique.

**Ship it with a lag, the way ADR 0014 lagged the target encoding.** Rejected because a lag does
not address it. A lag protects against reading labels that had not arrived yet. Here the problem
is not the labels, it is that the statistic is taken over rows that had not happened yet. Lagging
the aggregate would turn it into a point-in-time aggregate, which is what stage 4 built instead,
and it would no longer be the published construction.

**Build the point-in-time version instead, and leave the published one out of the production path
entirely.** Chosen.

**Build it anyway, behind a switch, so stage 6 can measure it.** Rejected, and this was the close
call. A switch would make the measurement convenient and it would also put an undeployable
construction one boolean away from the serving path, in a repository whose entire argument is that
the distinction is structural. Stage 7 is where the gap is measured, and stage 7 can implement it
as an experiment that imports nothing from `fraud_platform.features`.

## Decision

The published UID aggregation is not implemented in `fraud_platform.features`, is not behind a
configuration switch, and is not reachable from the production path. `NOT_BUILT` records the
absence in the module and a test asserts that no public name in the module contains `uid` or
`agg`.

The reason is one sentence: **a group statistic taken over every row of a group is a function of
rows that have not happened yet, and an authorisation endpoint does not have them.**

That follows from the construction alone, with no assumption about which frame it was applied to.
`groupby('uid').C1.mean()` attaches the same number to every row of the group, including the
group's first row, and that number is computed from the group's later rows. Scoring the first
transaction of a card in production, the later ones do not exist. Restrict the aggregate to rows
before t and the column changes value on almost every row, and is no longer the feature that was
measured.

This is not a claim that the construction is illegitimate. In a competition, the test set is
handed over in full at the start; a transductive feature is available at scoring time because
scoring time is after the data arrived. The construction is a correct use of the information the
task provides. It is the task that differs from an authorisation endpoint, not the reasoning.

## Evidence

There is no artifact for this ADR, and the absence is the point: nothing was measured because
nothing was built. **This is judgement, not measurement.** The judgement is that a feature which
cannot be computed at authorisation does not belong in a repository about scoring at
authorisation, and the argument for it is the construction quoted above rather than a number from
this repo.

What was built instead is measured. `reports/feature_summary.json` records 36 point-in-time
features over three grains, and `tests/test_causality.py` asserts that truncating the stream at
any time leaves every one of them bit for bit unchanged on the rows before the cut, which is
exactly the property the published aggregation does not have.

One number from the stage bears on the trade directly.
`reports/feature_summary.json`, `coverage`, records that 0.3402 of test rows sit on an entity the
training window saw, against 0.7036 with an earlier transaction anywhere in the stream. A
train-fitted entity statistic reaches the first population. A point-in-time entity feature reaches
the second, which is roughly twice as many rows.

## Consequences

- Stage 5 and stage 6 train on point-in-time features only, so no headline number in this repo is
  inflated by a transductive aggregate. It also means no number here is comparable with a
  leaderboard position that used one, and the direction of that gap is known.
- **Stage 7 owes the measurement.** Fit one model with the published aggregation and one without,
  on the same split at the same operating point, and report the difference as the price of
  deployability. Without that number this ADR is an argument and not a result. The stage 7
  implementation stands alone and imports nothing from `fraud_platform.features`.
- Stage 8's online store holds what `reports/feature_summary.json` says it holds: per key, a
  bounded ring of recent transactions, three scalars for a running moment, a device set and an
  address counter. There is no entry for a group aggregate, because a group aggregate has no
  online form.
- The exclusion is enforced weakly, by a naming test and by absence. It is not enforced by a
  raise, the way ADR 0015 enforces the target-encoding exclusion, because there is no function to
  put a raise in. A later stage that wants this construction has to write it, which is the
  intended amount of friction.
- If stage 7 measures the gap and finds it small, that is a result worth having and it does not
  change this decision. If it finds it large, that is a more interesting result and it still does
  not change this decision, because the feature would remain uncomputable at authorisation.
