# 0022: The duplicate-content features are shipped, in their causal form

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 4

## Context

Stage 2 asked whether repeated identical transactions exist in this file and got two answers,
depending on how the question was asked.

Asked strictly, comparing whole rows, the answer is almost nothing. Identical on every column
except TransactionID and TransactionDT: 5 groups, 10 rows out of 413,378, and **no fraud among
them**. The reason is visible once stated: the C and V blocks are running counters, so two
transactions that are the same purchase are never the same row.

Asked the way a person would ask it, comparing only the fields that describe the purchase, the
answer is a large part of the split. Rows agreeing on card1, card2, addr1, TransactionAmt,
ProductCD and P_emaildomain: 60,310 groups covering 200,113 rows, a share of 0.4841, at a fraud
rate of 0.04157 inside against 0.02917 outside. Stage 2 wrote "that is a real signal and it is a
feature, not dirt" and left the decision to a later stage. Stage 3 kept the rows: ADR 0017 records
duplication as not an error condition and `schema.NOT_CAUGHT` says so.

So the rows are here and the question this ADR settles is whether the feature is.

The thing that makes it a question rather than a formality is that stage 2's measurement is not a
feature. A groupby marks every member of a group, including its first row, from a statistic that
includes the group's later rows. At authorisation, the later rows do not exist. The 1.425 lift
stage 2 measured is therefore an upper bound on what a deployable version can carry, and the
number that decides this is the causal one, which had not been measured.

## Options considered

**Drop the family.** The strict definition finds 10 rows and no fraud, so a reader who only saw
that number would conclude there is nothing here. Rejected: the strict definition is measuring
the running counters, not the purchases, and stage 2 said so.

**Ship stage 2's grouping as a feature.** A per-row flag for group membership, and a group size.
Rejected for the reason above. It is the construction ADR 0020 refuses, on a different key.

**Deduplicate the rows.** Rejected. It would delete a measured fraud lift and it would also delete
the rows a card-testing pattern consists of.

**Build the causal version and decide on its measurement.** Chosen, and the rule had to be
rewritten once, which is recorded here rather than tidied away.

The rule written first was: ship when the causal version separates the label on the **train**
split by more than the split resolves, taken as the two Wilson intervals not overlapping. That
rule is insufficient and it is insufficient for a reason this repo had already written down. ADR
0009 set the precedent that a relationship learned from one window is a candidate until a later
window has seen it, and stage 3's ADR 0011 acted on it. A train-only ship rule ignores the
precedent, and on this family it gives the wrong answer.

The rule applied is: ship when the separation has the **same sign on all three splits** and the
two Wilson intervals do not overlap on any of them. A column whose marginal relationship reverses
between the training window and the later ones is a candidate for stage 6 to decide with a model,
not a feature this stage can claim carries signal.

Adding a measurement after the first one looked good is the mirror image of the habit ADR 0013
flagged, and it deserves the same note: the out-of-sample comparison should have been in the plan
from the start, and it is now in the artifact for every count feature in the stage rather than
only for the one that failed.

## Decision

Three features ship, all keyed on the content group and all reading only strictly earlier rows.
Two of them ship as features. The third ships as a candidate and is labelled one.

| Feature | Definition | Status |
| --- | --- | --- |
| dup_seconds_since_prev | t minus the timestamp of the latest earlier identical row, null when there is none | ships, meets the rule |
| dup_prior_count_1h | the same count restricted to the last 3,600 seconds | ships, meets the rule |
| dup_prior_count | strictly earlier transactions agreeing on the six fields | candidate, fails the rule, kept for stage 6 |

**The finding of the section is that the window is what carries the signal, not the repetition.**
Counted over all history, an earlier identical purchase is riskier on train and **safer** on
validation and test. Counted over the last hour it is riskier on all three. The numbers are in
Evidence.

The stage brief asked for "whether any fell inside a short window". `dup_prior_count_1h` is that
fact plus the multiplicity, and `dup_prior_count_1h > 0` recovers the boolean exactly, so the
count is what is stored rather than a flag that throws information away.

The short window is 3,600 seconds and the number is not round by accident: stage 2 measured the
median inter-transaction gap of an entity that ever carries fraud at 3,177 seconds, against
165,394 seconds for one that never does. An hour is one gap at the scale that population runs at.

Nulls follow the stage rule: a count of history is zero when there is no history, a comparison
against history is null. So the two counts are never null and `dup_seconds_since_prev` is null on
0.6618 of train rows, meaning the stream holds no earlier identical purchase.

**This family reads a second keyspace.** Content is not the entity, so
`reports/feature_summary.json` records `state_key: content` for these three and stage 8 has two
stores to maintain rather than one wider row in the first.

## Evidence

`reports/features/duplicate_content.json`, regenerated with `.venv/bin/python scripts/features.py
--sections duplicates`. `figures/duplicate_content.png` draws both panels.

**The causal grouping reproduces stage 2's, through an unrelated implementation.** Stage 2 hashed
the six fields and compared rows value by value inside each hash bucket. Stage 4 factorizes each
field, codes the tuple, and counts strictly earlier matches per row with searchsorted.

| Quantity | Value |
| --- | --- |
| Stage 2, rows in a group beyond the first | 139,803 |
| Stage 4, rows with a strictly earlier identical row | 139,792 |
| Difference | 11 |
| Rows in a content group sharing a timestamp with a companion | 102 |

The 11 are accounted for and bounded: they are rows whose only identical companion shares their
timestamp, which stage 2's groupby counts and ADR 0018's tie exclusion does not. The difference is
0.008 percent of the count. Two implementations of the same definition agreeing to that degree is
the strongest cross-check in the stage, and a test asserts the difference stays inside the tied
count.

**On the train split the lift survives being made causal, and shrinks.**

| Grouping | Rows marked | Rate marked | Rate unmarked | Ratio |
| --- | --- | --- | --- | --- |
| Stage 2 groupby, whole group | 200,113 | 0.04157 | 0.02917 | 1.425 |
| Causal, any earlier identical | 139,792 | 0.04213 | 0.03161 | 1.333 |
| Causal, an earlier identical inside 3,600 s | 31,697 | 0.08815 | 0.03077 | 2.865 |

That is where the first version of this ADR stopped, and it is the wrong place to stop.

**On the later splits the all-history version reverses.** Same comparison, per split, from
`causal_grouping_by_split`:

| Split | Rows marked | Rate marked | 95 percent Wilson | Rate unmarked | 95 percent Wilson | Ratio |
| --- | --- | --- | --- | --- | --- | --- |
| train | 139,792 | 0.04213 | 0.04109 to 0.04320 | 0.03161 | 0.03096 to 0.03227 | 1.333 |
| val | 40,921 | 0.03057 | 0.02895 to 0.03228 | 0.03758 | 0.03591 to 0.03932 | 0.814 |
| test | 44,663 | 0.03267 | 0.03106 to 0.03436 | 0.03698 | 0.03525 to 0.03878 | 0.883 |

The sign flips and it flips outside the intervals. On train, having repeated an earlier purchase
is a risk factor; on validation and test it is a mild protective factor. This is not noise: the
intervals are disjoint on all three splits, in opposite directions on train against the other two.
`dup_prior_count` therefore fails the rule.

**The hour window holds on all three.** Same comparison for `dup_prior_count_1h`:

| Split | Rows marked | Rate marked | 95 percent Wilson | Rate unmarked | Ratio |
| --- | --- | --- | --- | --- | --- |
| train | 31,697 | 0.08815 | 0.08508 to 0.09132 | 0.03077 | 2.865 |
| val | 6,302 | 0.07521 | 0.06896 to 0.08199 | 0.03121 | 2.410 |
| test | 8,876 | 0.06377 | 0.05887 to 0.06904 | 0.03158 | 2.019 |

Same sign, disjoint intervals, on every split. The ratio decays from 2.865 to 2.019 across six
months, which is drift worth watching in stage 9 rather than a failure.

So the signal is not that a purchase repeats. It is that it repeats **soon**, and the all-history
count dilutes that to the point of reversing it: 139,792 marked rows against 31,697, most of them
repeats from days or months earlier, which the decile table below shows run below the base rate.

**And the strongest of the three is the recency, not either count.** `dup_seconds_since_prev`
carries an information value of 0.1802 against 0.0086 for `dup_prior_count`, and its decile table
is monotone across nine of ten bins:

| Seconds since the last identical purchase | Fraud rate |
| --- | --- |
| 1 to 186 | 0.10689 |
| 187 to 1,431 | 0.07399 |
| 1,432 to 63,534 | 0.06282 |
| 63,539 to 198,660 | 0.04292 |
| 198,670 to 450,308 | 0.02926 |
| 450,321 to 785,133 | 0.02418 |
| 785,204 to 1,291,560 | 0.02103 |
| 1,291,562 to 2,148,136 | 0.02060 |
| 2,148,153 to 3,467,542 | 0.01924 |
| 3,467,794 to 10,315,537 | 0.02039 |
| no earlier identical purchase, for comparison | 0.03161 |

An identical purchase within three minutes runs at 3.4 times the rate of a row with no identical
history. An identical purchase from the fifth decile up, 198,670 seconds or more, runs **below** it, so a repeat
purchase from months back is a mild signal of a legitimate returning customer. Both directions
are in one column and a tree can use both, which is why the recency ships as a continuous column
rather than as a flag.

That table is also the mechanism for the reversal above. Six of the ten deciles sit below the
0.03161 base rate, and they hold 0.6 of the marked population, so a flag that marks all of them
together is averaging a strong positive over a large mild negative. Which side wins depends on the
mix of recent against old repeats in the window, and that mix is not stationary.

**A note on the two count features' information values.** `dup_prior_count_1h` reads 0.0000 and
`dup_prior_count` reads 0.0086, and neither number means what it appears to. Both columns are zero
on most rows, so ten equal-frequency bins collapse to one and three respectively, and the
information value of a single bin is zero by construction. The artifact records the bin count
beside every information value and the zero-against-positive comparison beside that, which is why
the table above is the measurement this decision rests on. `dup_prior_count_1h` is the sharpest
duplicate feature in the stage and the naive reading of its information value would have dropped
it.

## Consequences

- Three columns ship, on a keyspace the entity features do not use. Stage 8's online store holds a
  second keyed structure: per content key, the prior count, the last prior timestamp, and the
  timestamps inside the last hour.
- The content key includes `TransactionAmt` as a float, compared for exact equality. That is the
  same comparison stage 2 made and it is deterministic on values parsed from the file, but it is
  a brittle definition of "the same purchase": 68.95 and 68.96 are different keys. Nothing here
  measures how much that costs.
- Nulls compare equal, so two rows both missing P_emaildomain and equal elsewhere are the same
  content. That is stage 2's rule, reproduced deliberately, and it is what makes the two counts
  comparable across the stages.
- **What is still not established** is what stage 2 already flagged: whether these groups are card
  testing or ordinary repeat business. The rate difference is measured and the mechanism is not,
  and the recency table above is suggestive of card testing without demonstrating it. Nothing in
  this repo names the mechanism.
- **`dup_prior_count` is kept as a candidate and is not claimed to carry signal.** It is kept for
  three reasons and none of them is that it works. It is the denominator the other two are read
  against. Its inversion is the measurement, and deleting the column would leave the measurement
  without the thing it is about. And ADR 0009's precedent is that a single-feature marginal does
  not drop a feature: a tree can use a column whose marginal reverses if it interacts with
  recency, and stage 6 is where that is settled with a model and a paired comparison. If stage 6
  finds nothing, the column goes, and that is a new ADR.
- The reversal is the stage's clearest case of the rule the whole repository runs on, applied to
  this repository's own work rather than to somebody else's. The train-split lift was real,
  causal, reproducible, cross-checked against stage 2 to within 11 rows, and still not a feature.
