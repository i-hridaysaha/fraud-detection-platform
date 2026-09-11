# 0031: Four leakage variants, one per rule the pipeline enforces

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 7

## Context

Stages 1 to 5 built a pipeline that fits nothing past the training boundary (ADR 0006), reads no
statistic over a row's future (ADR 0020), and is scored on a window after the one it was fitted on
(ADR 0002). Stage 6 measured that pipeline at a test PR-AUC of 0.5451. Until this stage, every
claim about why those rules matter was an assertion, and ADR 0020 says so in as many words:
"without that number this ADR is an argument and not a result."

A delta needs a leaky variant to stand against, and the set of leaks one could build is open.
Which ones to build, and in what combination, is the decision here. The stage brief named four;
this record says why those four are the right scope and not a longer or shorter list.

## Options considered

**Every leak that can be named.** Beyond the four below: a threshold tuned on the test window, a
calibrator fitted on it, feature selection against it, per-day statistics over the whole file,
row deduplication that reads later rows. Rejected. The list has no natural end, each entry costs
a fit and a paragraph, and most of them are not something this repo's rules forbid so much as
something no serious evaluation does. The claim the stage exists to price is the cost of the
specific discipline stages 1 to 5 built, so the variants should be the ones that discipline
refuses.

**One variant, the whole competition pipeline at once.** Cheapest, and the headline number a
reader wants. Rejected as the only measurement: one number cannot say which rule paid for what,
and the stage's brief asks for the shortcuts one at a time. It is kept as the last row.

**One variant per rule the pipeline enforces, then their combination.** Chosen. Each of the four
maps to one thing the repo made structural, so each delta prices one decision:

| Variant | The rule it breaks | Where the rule lives |
| --- | --- | --- |
| (a) encoders fitted on every row | fitted objects see train rows only | ADR 0006, `data_loader.assert_train_only` |
| (b) entity aggregates over every row | a feature reads no row after the one it describes | ADR 0018, ADR 0020, `tests/test_causality.py` |
| (c) each prediction replaced by the entity's mean over the scored window | a score at authorisation reads no later transaction | the serving contract; no module can build this |
| (d) a random split | evaluation rows come after training rows | ADR 0002, `data_loader.time_based_split` |

## Decision

`scripts/leakage_delta.py` builds the four variants and two combinations, holding the model
family (xgboost, no reweighting), the hyperparameters (`modelling.DEFAULT_HYPERPARAMETERS`, the
set stage 6 shipped), the seed (42) and the 186 column stack (`models/shipped_model.json`) fixed
across all of them. A delta between two differently tuned models measures tuning; these measure
the shortcut and nothing else.

**What each variant is, exactly.**

- **(a)** `prepare.fit_preparation` on all 590,540 rows instead of the 413,378 train rows, at the
  recorded lag (14 days) and smoothing (20), applied to the three chronological windows. The
  vocabulary, the frequency tables, the target-encoding tables and the V block means all read the
  validation and test windows, labels included.
- **(b)** The construction ADR 0020 kept out of `fraud_platform.features`: `groupby` the ADR 0001
  entity, take the mean and standard deviation of every C, M and D column (the D columns first
  normalised to an origin day, `transforms.to_d_origin`; the M levels as their codes), attach the
  entity's statistic to every row of the entity, over all three windows. 76 columns beside the
  shipped 186. The page ADR 0020 fetched, read again for this record, lists the published set as
  means of C1 to C14 and M1 to M9 and means and standard deviations of D4, D9, D10, D15 and the
  amount; this repo's version is in that style and wider, and the artifact lists every column.
- **(c)** The baseline's own predictions, each replaced by the mean prediction over the entity's
  rows in the same scored window. The same page describes the step as "replace all predictions
  from one UID value with that client's average prediction". No refit: the model is the baseline's.
- **(d)** A seeded permutation of the 590,540 rows cut at the 0.70 and 0.85 row fractions, the
  encoders fitted on the random train rows through the same pipeline, the model scored on the
  random test rows.
- **(a)+(b)+(c)** on the chronological windows, so the combined delta stays paired with the
  baseline on identical resamples.
- **(a)+(b)+(c)+(d)**, the competition-style pipeline end to end, scored on the random test rows.

**The guard.** (a), (d) and the combinations cannot be built through `fraud_platform`: the
preparation fit refuses a frame with a row at or after the training boundary. The script replaces
`assert_train_only` with a pass-through inside one context manager, by name, and restores it on
exit; a test asserts the restoration. The leak has to be switched off in the open, in scripts/,
and a diff that did the same thing under src/ would be visible for what it is.

**Why each is standard in a competition and a defect at an authorisation endpoint.** The
difference is the task, not the reasoning. A competition hands over the full test set at the
start; scoring happens after every row has arrived, so a statistic over all of them is available
at scoring time, and using it is the correct use of the information the task provides. An
authorisation endpoint scores one transaction at the moment it happens, with nothing after it in
existence yet.

- (a) is what `fit_transform` on the concatenated frame does, and on a competition it is the
  natural way to get one vocabulary for train and test. At authorisation, the table that would
  have been fitted on next month's rows does not exist.
- (b) is the construction the first-place solution on this dataset rests on, per the page fetched
  and read in stage 7 and in stage 4. It is legitimate there because the group's later rows are
  in hand. At authorisation, the first transaction of a card has no later rows to average.
- (c) is described on the same page as part of that solution. It is a correct use of the fact that
  labels are card-level and the whole card is visible. At authorisation, the transactions the mean
  would read have not happened.
- (d) is the default `train_test_split`. On a competition it is a reasonable way to hold out
  rows when the organisers already hold out the future. At authorisation, a model is asked about
  entities and weeks it has never seen, and a random split never asks it that.

The constraint changed, not the competence.

## Evidence

`reports/leakage_delta.json`, regenerated with `make leakage`. Test PR-AUC per variant with its
95 percent interval over 1,000 row resamples, and the difference against the causal baseline:
paired on identical resamples for the chronological variants, the difference of independent draws
for the two scored on the random test rows.

| Variant | Test PR-AUC | Interval | Difference against baseline | Test ROC-AUC | Verdict |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.5451 | 0.5274 to 0.5614 | | 0.9050 | |
| (a) encoders on every row | 0.5511 | 0.5336 to 0.5678 | +0.0060, +0.0013 to +0.0108 | 0.9118 | inflates, by less than the shipped model's own half-width |
| (b) entity aggregates over every row | 0.5822 | 0.5648 to 0.5981 | +0.0371, +0.0290 to +0.0458 | 0.9155 | inflates |
| (c) entity-mean post-processing | 0.4898 | 0.4711 to 0.5070 | -0.0554, -0.0656 to -0.0463 | 0.9053 | deflates |
| (d) random split | 0.8044 | 0.7926 to 0.8171 | +0.2593, +0.2378 to +0.2809 | 0.9633 | inflates |
| (a)+(b)+(c), chronological | 0.5190 | 0.5011 to 0.5360 | -0.0262, -0.0399 to -0.0141 | 0.9174 | deflates |
| (a)+(b)+(c)+(d), random | 0.8317 | 0.8203 to 0.8439 | +0.2866, +0.2653 to +0.3092 | 0.9759 | inflates |

The rule, in the artifact before the numbers were read: a variant inflates when the interval of
its test PR-AUC difference on the row resample excludes zero and the point is positive. Two of
the four do not inflate: (c) lowers the metric on this window, and (a) clears zero by row and not
by card (+0.0060, -0.0084 to +0.0214 over 1,000 card resamples). Both are results and both are
reported as such; the note for the stage has what was expected instead.

## Consequences

- ADR 0020's owed measurement is paid: the published aggregation is worth +0.0371 of test PR-AUC
  on this split, +0.0290 to +0.0458, and it stays out of the production path because the number
  does not change what an endpoint can compute.
- The random split is the shortcut that matters. Nine tenths of the gap between the causal
  baseline and the competition-style pipeline (+0.2593 of +0.2866) is the split alone, and the
  split is the one of the four that touches no feature.
- The post-processing result is a property of this window and this key, not a general finding
  about the step. `postprocessing.diagnostic` in the artifact records where it acts: 0.3536 of the
  test window's fraud rows sit in the 457 entities that mix labels, and leaving alone the 3,302
  entities whose key has a null component turns the step from -0.0554 into +0.0035.
- The scope is closed. A later stage that wants a fifth variant writes a new record that
  supersedes this one and says what rule the fifth one breaks.
