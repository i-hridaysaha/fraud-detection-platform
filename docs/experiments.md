# The leakage delta and the label latency

Stage 7 prices the discipline. Stages 1 to 5 built a pipeline that fits nothing past the training
boundary, reads no statistic over a row's future and is scored on a window after the one it was
fitted on; stage 6 measured it at a test PR-AUC of 0.5451. Every claim about why those rules
matter was an assertion until this stage, which turns two of them into measurements: what the
same model scores when the four standard competition shortcuts are switched on one at a time and
together, and what happens to it when the most recent weeks of training labels are not yet
mature.

Everything below is measured on the two artifacts the stage writes: `reports/leakage_delta.json`
from `make leakage` and `reports/label_latency.json` from `make latency`. Sections are named after
the artifact they read. The figures, `figures/leakage_delta.png` and `figures/label_latency.png`,
come from `make experiment-figures` and read the same two files.

The result of the stage, in one paragraph. The random split is the shortcut that matters:
+0.2593 of test PR-AUC on its own, nine tenths of the +0.2866 the competition-style pipeline
gains end to end. The published entity aggregation is worth +0.0371, resolvable and a tenth of
what the split is worth. Fitting the encoders on every row is worth +0.0060, which clears zero
by row and not by card. The entity-mean post-processing lowers the metric on this window by
0.0554, and the artifact says where. On the label side, a training set that treats the last 15
days of labels as final costs 0.0345 of test PR-AUC against dropping them, and by 60 days the
model has learned that recent rows are never fraud and implies a fraud rate of 0.0004 against an
observed 0.0348. The notes for the stage record what was expected before each number was read,
and most of it was wrong.

A word on numbering. Earlier documents that say "stage 7" for the serving path mean stage 8; ADR
0020's "stage 7 owes the measurement" is this stage.

---

## 0. What was held fixed

**The model.** XGBoost, no reweighting, `modelling.DEFAULT_HYPERPARAMETERS` (600 trees at depth
8, learning rate 0.05, `min_child_weight=5`, subsample 0.8, `colsample_bytree=0.6`), seed 42, and
the 186 column stack `models/shipped_model.json` names: stage 3's prepared frame with the C, D
and V groups and without M, no stage 4 or 5 feature. Both artifacts record the hyperparameters and
a test asserts they are the module defaults. A delta between two differently tuned models
measures tuning; these measure one change each.

**The baseline.** Rebuilt from the raw files inside `scripts/leakage_delta.py` through the same
code path every variant takes, and read from the stage 6 cache inside `scripts/label_latency.py`
with the target encoder refitted through the same path every treatment takes. Both reproduce the
shipped model's test PR-AUC of 0.5451 to four decimals (the refit reproduces the cached target
encodings to a largest difference of 0.0), and a test asserts both.

**The intervals.** 1,000 bootstrap resamples of the test window, every model scored on the
identical resample, so a difference between two models on the same rows is paired and carries
its own interval. PR-AUC by row and by card, ROC-AUC by row. The two variants scored on the
random test rows share no rows with the baseline; their difference is the difference of
independent draws and its interval is wider, which is the width there is. Stage 6's row-resample
half-width for the shipped model is 0.0170; a delta inside it is inside the noise.

**The rule, written into the artifact before the numbers were read.** A variant inflates when
the interval of its test PR-AUC difference on the row resample excludes zero and the point is
positive. Anything else is a null, and a null is reported as one.

---

## 1. Four shortcuts, and why each is standard in one place and a defect in the other

The four are ADR 0031's, one per rule the pipeline enforces:

| Variant | What it does | The rule it breaks |
| --- | --- | --- |
| (a) | every encoder fitted on all 590,540 rows instead of the 413,378 train rows, labels included | fitted objects see train only (ADR 0006) |
| (b) | mean and standard deviation of every C, M and origin-normalised D column per entity over all three windows, 76 columns beside the 186 | no feature reads a row after the one it describes (ADR 0020) |
| (c) | each of the baseline's predictions replaced by the mean over the entity's rows in the scored window | a score at authorisation reads no later transaction |
| (d) | a seeded permutation cut at the 0.70 and 0.85 row fractions, encoders fitted on the random train rows, scored on the random test rows | evaluation rows come after training rows (ADR 0002) |

Then (a)+(b)+(c) on the chronological windows, paired with the baseline, and (a)+(b)+(c)+(d),
the competition-style pipeline end to end.

(b) and (c) are load-bearing in the published first-place solution on this competition. The page
ADR 0020 fetched, read again this session, lists the UID aggregation (means of C1 to C14 and M1
to M9, means and standard deviations of D4, D9, D10, D15 and the amount) and describes the
post-processing as "replace all predictions from one UID value with that client's average
prediction". Both are legitimate there: a competition hands over the full test set at the start,
scoring happens after every row has arrived, and a statistic over all of them is a correct use of
the information the task provides. Neither can run at an authorisation endpoint, where the
transaction being scored is the latest one in existence and the group's later rows are not in
hand. The constraint changed, not the competence.

(a) and (d) are the defaults: `fit_transform` on the concatenated frame and `train_test_split`.
Nothing in `fraud_platform` can build either, because the preparation fit refuses a frame with a
row past the training boundary. The script switches the guard off by name inside one context
manager and restores it on exit; `tests/test_experiment_artifacts.py` asserts the restoration and
that no module under src/ does the same.

---

## 2. The leakage delta

All from `reports/leakage_delta.json`. Test PR-AUC with its 95 percent interval over 1,000 row
resamples, the difference against the causal baseline (paired on identical resamples where the
rows are shared, independent where they are not), ROC-AUC beside it deciding nothing.

| Variant | Columns | Test PR-AUC | Interval | Difference against baseline | Test ROC-AUC | Card-level PR-AUC | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 186 | 0.5451 | 0.5274 to 0.5614 | | 0.9050 | 0.6442 | |
| (a) encoders on every row | 186 | 0.5511 | 0.5336 to 0.5678 | +0.0060, +0.0013 to +0.0108 | 0.9118 | 0.6432 | inflates, inside the shipped half-width |
| (b) entity aggregates over every row | 262 | 0.5822 | 0.5648 to 0.5981 | +0.0371, +0.0290 to +0.0458 | 0.9155 | 0.6680 | inflates |
| (c) entity-mean post-processing | 186 | 0.4898 | 0.4711 to 0.5070 | -0.0554, -0.0656 to -0.0463 | 0.9053 | 0.5999 | deflates |
| (d) random split | 186 | 0.8044 | 0.7926 to 0.8171 | +0.2593, +0.2378 to +0.2809 | 0.9633 | 0.7923 | inflates |
| (a)+(b)+(c), chronological | 262 | 0.5190 | 0.5011 to 0.5360 | -0.0262, -0.0399 to -0.0141 | 0.9174 | 0.6157 | deflates |
| (a)+(b)+(c)+(d), random | 262 | 0.8317 | 0.8203 to 0.8439 | +0.2866, +0.2653 to +0.3092 | 0.9759 | 0.8333 | inflates |

**(a), the encoders.** +0.0060 by row with the interval just clear of zero and sign agreement
0.997; +0.0060 by card with an interval of -0.0084 to +0.0214 that is not. ROC-AUC moves +0.0068,
+0.0047 to +0.0090. The rule reads the row resample, so the artifact records it as inflating, and
the honest description is that it inflates by a third of the shipped model's own half-width.
Validation PR-AUC moves from 0.6184 to 0.6189. The target-encoding tables fitted on every label
at a 14 day lag hand a test row the fraud rate of its `card1` level through the validation
window and the first days of test; on this window that is worth less than a hundredth.

**(b), the aggregation.** +0.0371 by row, +0.0290 to +0.0458, and the same point by card with
+0.0197 to +0.0574. ROC-AUC +0.0105. The validation number moves more than the test one: 0.6184
to 0.7111 on validation against 0.5451 to 0.5822 on test, a gap the artifact records and this
document does not explain. This is ADR 0020's owed measurement, paid: the construction the
first-place solution rests on is worth about four hundredths of test PR-AUC on this split when
everything else is held at the shipped model, and it stays out of the production path because
an endpoint cannot compute it whatever it is worth.

**(c), the post-processing.** -0.0554, -0.0656 to -0.0463, with ROC-AUC unchanged at +0.0004,
-0.0022 to +0.0028. The step moves no ranking across entities that it does not also move within
them, and on this window the within-entity damage wins. `postprocessing.diagnostic` in the
artifact partitions the 88,581 test rows by the kind of entity they sit on:

| Entity kind in the test window | Entities | Rows | Fraud rows | Share of fraud rows |
| --- | --- | --- | --- | --- |
| one row | 28,960 | 28,960 | 585 | 0.1898 |
| several rows, all legitimate | 15,563 | 54,656 | 0 | 0.0000 |
| several rows, all fraud | 368 | 1,408 | 1,408 | 0.4567 |
| several rows, mixed | 457 | 3,557 | 1,090 | 0.3536 |

The mean changes nothing on a singleton and helps or is neutral on a pure entity; the 457 mixed
entities hold 0.3536 of the window's fraud rows in 3,557 rows, and there the mean pulls 1,090
fraud rows and 2,467 legitimate rows toward one number. Most of the mixing is the null-component
bucket: ADR 0001 keeps a null `addr1` or `D1` as its own key component, so `card1|NA|NA` pools
every row of a `card1` with a missing address, 0.1053 of the test rows at a fraud rate of 0.1366
across 3,302 entities. Leaving those entities alone and averaging the rest gives 0.5486, +0.0035
over the baseline and inside the noise. The published construction pools the same way (`nan`
pasted into the identifier), so the variant is faithful; the result is a property of this key on
this window and not a general finding about the step.

**(d), the split.** +0.2593, +0.2378 to +0.2809, independent draws; ROC-AUC 0.9633, +0.0583. The
random test rows share 0.7410 of their rows with an entity in the random train rows against
0.3402 for the chronological test window, and the fraud rate flips: 0.0387 on seen entities and
0.0227 on unseen under the random split, 0.0170 and 0.0440 under the chronological one. The
model is scored on cards it was fitted on, and the population it is scored on is one where the
cards it knows are the risky ones. Nothing about the features changed.

**Together.** (a)+(b)+(c) on the chronological windows lands at 0.5190, -0.0262 against the
baseline, because (c) costs more than (a) and (b) gain. With (d) as well, 0.8317 and a ROC-AUC of
0.9759, +0.2866 over the baseline, and the split is +0.2593 of it. The published first-place
private score on this competition is a ROC-AUC of 0.9459 on a different test set with a
different model; the number here is not comparable with it and is quoted only to say which
neighbourhood the shortcuts reach.

---

## 3. Simulated label latency

All from `reports/label_latency.json`. The simulation is ADR 0032's: at training time every
fraud in the final N days of the 119.81 day training window reads as legitimate. Two treatments
per N, each against the reference fitted on every train row with the file's labels: **immature**
fits on every row with the immature labels treated as final; **excluded** fits on the rows more
than N days before the boundary. The target encoder is refitted on the same rows and labels the
model sees. **The latency is simulated and the true chargeback timing is unknown here**; the
labels in the file are final, which is what makes the reference available.

| N, days | Recent rows, fraud among them | Immature test PR-AUC | Against reference | Excluded test PR-AUC | Against reference | Immature minus excluded |
| --- | --- | --- | --- | --- | --- | --- |
| 15 | 45,663, 2,112 | 0.5106 | -0.0345, -0.0428 to -0.0268 | 0.5327 | -0.0124, -0.0190 to -0.0059 | -0.0221, -0.0277 to -0.0171 |
| 30 | 98,496, 3,883 | 0.4593 | -0.0858, -0.0961 to -0.0754 | 0.4877 | -0.0574, -0.0662 to -0.0493 | -0.0284, -0.0380 to -0.0192 |
| 45 | 143,405, 5,551 | 0.4163 | -0.1289, -0.1413 to -0.1170 | 0.4839 | -0.0613, -0.0706 to -0.0528 | -0.0676, -0.0784 to -0.0572 |
| 60 | 190,607, 7,586 | 0.3435 | -0.2017, -0.2163 to -0.1881 | 0.4732 | -0.0719, -0.0825 to -0.0616 | -0.1297, -0.1407 to -0.1174 |
| 90 | 280,122, 11,168 | 0.1629 | -0.3823, -0.3964 to -0.3664 | 0.4493 | -0.0959, -0.1069 to -0.0855 | -0.2864, -0.2979 to -0.2732 |

The reference scores 0.5451. Every difference in the table is paired and every interval excludes
zero.

**The direction and size of the bias.** The fraud rate a model implies on test is its mean score;
the observed rate is 0.0348 and the reference implies 0.0327.

| N, days | Immature implies | Excluded implies | Immature recall at the shipped threshold | Excluded recall | Immature alerts per day | Excluded alerts per day |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (reference) | 0.0327 | | 0.4713 | | 78.2 | |
| 15 | 0.0299 | 0.0302 | 0.4327 | 0.4541 | 70.3 | 71.6 |
| 30 | 0.0195 | 0.0316 | 0.3364 | 0.4178 | 48.2 | 73.9 |
| 45 | 0.0115 | 0.0253 | 0.2267 | 0.3756 | 30.1 | 58.1 |
| 60 | 0.0004 | 0.0211 | 0.0032 | 0.3568 | 0.3 | 52.1 |
| 90 | 0.0000 | 0.0215 | 0.0000 | 0.3425 | 0.0 | 52.2 |

The bias is downward under both treatments and it is not the same bias. The excluded model's
implied rate drifts down with N because its fit set is older; at N=90 it fits on 133,256 rows
and implies 0.0215, six tenths of the observed rate, and still recalls 0.3425 at the shipped
threshold. The immature model's implied rate collapses: 0.0299 at 15 days, 0.0004 at 60, 0.0000
at 90, where the shipped threshold produces no alerts at all. That is not a gradual loss of
signal from fewer positives. The excluded model at N=60 fits on the same 6,952 positives and
scores 0.4732.

**What the immature model learned.** Each run's `regime` block records the mean score the fit
gives the older train rows, the recent train rows and the two later windows. The immature model
at N=60 scores the older train rows at 0.0307 and the recent ones at 0.0005, in sample, and the
test rows at 0.0004; at N=15 the split is already 0.0336 against 0.0013 in sample, with the test
rows at 0.0299. The recent window has become a regime of its own in which nothing is fraud, and
the further the boundary moves back, the more of the serving window looks like it. The
`regime_diagnostic` block refits the immature model at N=60 with a block of columns removed, to
ask what carries the window. Without the 16 target encodings, whose values move with the
cumulative count, the test mean score is 0.0038, ten times more and still under an eighth of
the reference's 0.0327, at a test PR-AUC of 0.2387; without the 15 D columns, 0.0003; without both,
0.0043. The gain shares say the same: the target encodings carry 0.0519 of the reference fit's
gain, 0.0917 of the immature fit's at N=60 and 0.1633 at N=90. No single block is the door, and
which other columns carry the window is not established.

**The sensitivity to N.** The cost of the correction is steep at the start and flat after: the
excluded model loses 0.0124 at 15 days and 0.0574 at 30, then 0.0613, 0.0719 and 0.0959 to 90.
Most of what a month of exclusion costs is the second fortnight. The cost of not correcting
grows without a flat part, and the gap between the two widens from 0.0221 to 0.2864. There is no
N in the sweep at which keeping the immature labels beats dropping the rows.

---

## 4. What the numbers are and are not

- Every delta in section 2 is one change against the shipped model at fixed family,
  hyperparameters, seed and stack, on one split of one file. A different dataset, a different
  key or a different model family gets different numbers; the direction of (d) is the only one
  this document would expect to survive that.
- The two random-split variants are scored on different rows from the baseline. Their intervals
  are independent-draw intervals and their population facts are in the artifact beside the
  chronological ones.
- The post-processing result is explained by a measured partition, not a mechanism this repo
  verified on another window. Leaving the null-key entities alone was scored because the
  partition suggested it; it is a diagnostic, not a fifth variant.
- The label latency is simulated with a deterministic cutoff. A real chargeback delay is a
  distribution, some of it shorter than 15 days and some longer than 90, and a real training
  set has partly matured labels rather than none. What the sweep establishes is the shape:
  downward bias under both treatments, collapse under one, and a widening gap with N.
- No value of N is recommended and no correction is derived. ADR 0032 says what would supersede
  it.

---

## What stage 8 inherits

| Finding | Number | Consequence |
| --- | --- | --- |
| The split is the shortcut that matters | +0.2593 of +0.2866 | a serving-time evaluation that is not chronological measures nothing this repo cares about |
| The published aggregation, priced | +0.0371, +0.0290 to +0.0458 | stays out of the production path (ADR 0020); the price of deployability is four hundredths |
| Full-fit encoders, priced | +0.0060 by row, not by card | the guard costs under a hundredth on this window and is kept for the cases where it would cost more |
| Entity-mean post-processing | -0.0554 on this window and key | not a serving step, and not a competition step on a key with a null bucket either |
| Immature labels | -0.0345 at 15 days, implied rate 0.0004 at 60 | the training pipeline needs a maturation window; the retraining cadence has to leave one |
| The excluded correction | -0.0124 at 15 days, -0.0574 at 30 | the second fortnight of exclusion costs more than three times the first |
| The reference implies 0.0327 against 0.0348 observed | ratio 0.938 | the shipped model itself under-predicts the rate by a sixteenth on test |

## What is not established

- The true chargeback delay on this data, and therefore the maturation window a real deployment
  would need. The sweep brackets it; nothing here locates it.
- Why (b) inflates validation by about 0.093 (0.6184 to 0.7111) and test by 0.0371. Both are leaks and both are
  reported; the difference between the two windows is not explained.
- Whether (c) helps on a key without a null bucket, on another window, or on the published
  model. The partition says where it hurts here; it does not say the step is worthless.
- What a partly matured training set does. The simulation is all-or-nothing inside the window,
  and a real one is neither.
- Everything stages 0 to 6 left open stays open. The serving path is stage 8's and the promotion
  margin stage 9's.
