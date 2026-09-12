# Monitoring and the lifecycle: two clocks, a gate, a rollback

Stage 8 put the model behind an endpoint. This stage watches it and replaces it. The design point
is that a model in production fails on two clocks: its inputs can move the day they move, and its
accuracy can only be read once the labels for a batch have arrived, which on chargeback-labelled
data is weeks later. The two failures are kept apart in two jobs with two artifacts, and they
meet once, in the retrain decision, where realised decay is the primary trigger and drift held
across consecutive cycles is the earlier, secondary one. A challenger goes through the same
pipeline under the same guard with its boundary moved, is gated against the champion on rows
neither model trained on, and is promoted only past a margin derived from the noise stage 6
measured. Promotion is an alias move in the stage 8 registry and rollback is the move back.

Everything below is measured on the five artifacts the stage writes:
`reports/monitoring/drift_reference.json` and `reports/monitoring/drift.json` from `make
monitor-drift`, `reports/monitoring/decay.json` from `make monitor-decay`,
`reports/adversarial_validation.json` from `make adversarial`, and `reports/lifecycle_demo.json`
from `make lifecycle-demo`. The figures come from `make monitoring-figures` and read the same
files.

The result of the stage, in one paragraph. The textbook PSI alert band puts 16 to 20 of the
shipped model's 186 input columns in alert on every in-control week of this file; calibrated on
the windows the model was accepted on, the drift flag fires on none of them held out. The
weekly PR-AUC of the accepted model on those same weeks runs 0.4942 to 0.7458 against the 0.5451
it was accepted at, and the largest in-control drop, 0.0510, sits 0.009 under the decay
tolerance. Train and test are separable at AUC 1.0000 on the model's inputs because the target
encodings carry the encoder's own clock, and at 0.9031 without them, on days-since counters, one
identity attribute and the frequency of card identifiers: population turnover and the calendar,
not a change in what fraud looks like. On the demo, a shift injected at day 141 raised the drift
flag at day 148 and the decay flag at day 183; of four challengers the gate refused two and
promoted two, one of them by 0.0021 over the margin; the rollback put the previous version back
and the served path reproduced its batch scores to 0.0.

A word on numbering. The brief listed these records as ADRs 0035 to 0037; 0035 was taken by
stage 8, so they are 0036, 0037 and 0038.

---

## 0. What was held fixed

- **The model.** The stage 6 booster, `models/shipped_model.ubj`, its 186 columns, its
  hyperparameters and seed; the stage 3 pipeline refitted on train where a script needs it,
  bit-identical to the cache (`drift.json`, `scores_against_stage_6_cache`, 0.0 on val and test).
- **The reference window.** The test split, where the model's acceptance number came from:
  PR-AUC 0.5451, 0.5274 to 0.5614 over 1,000 row resamples at seed 42, and the decay job's
  bootstrap reproduces `reports/metric_variance.json` to 0.0 (`decay.json`, `reference`).
- **The batch.** Seven days, `monitoring.BATCH_DAYS`, cut from the first timestamp's day on the
  stage 0 rule `start <= dt < end`. Chosen.
- **The maturity window.** `lifecycle.MATURITY_DAYS = 30`. An assumed input: ADR 0032 swept 15 to
  90 and found nothing in this file that says where the chargeback delay sits. The decay job
  refuses a batch until the clock has passed its close by this many days.
- **The bins.** Deciles of the training distribution with missing as its own level, floored at
  1e-6, the stage 2 functions called directly. On the 59 raw columns both stages hold, the
  whole-test-window PSI here equals `reports/eda/temporal.json` to 1e-9 on all 59
  (`drift.json`, `against_stage_2`).

---

## 1. The drift job

`monitoring.drift_report` runs when a batch closes, reads the model's prepared input columns and
its scores, and raises if handed a frame that carries the label. It compares each column and the
score distribution with `drift_reference.json`, built by `monitoring.build_reference` from the
training split: 186 histograms plus one for the score. The bands are the convention's, 0.10 and
0.20, reported per column. The alert is not.

**Calibration** (ADR 0037). Stage 2 measured what normal drift looks like on this file at the
window grain: 30 of 432 raw columns above PSI 0.10 and 22 above 0.25 between train and test,
with nothing wrong. At the monitoring grain the same fact reads as follows, on the ten weekly
batches of the val and test windows, which are the windows the model was accepted on.

| Quantity | Value |
| --- | --- |
| Columns above 0.20 on the whole test window, of 186 | 18 |
| Columns above 0.20 per in-control week, textbook | 16 to 20, mean 18.4 |
| Column edges that had to rise above 0.20 to hold the in-control weeks | 20: the 16 target encodings, `id_13` to 1.104, `D11` 0.427, `vblockmean_V1` 0.393, `id_30_cat` 0.265 |
| Score PSI per in-control week | 0.0059 to 0.0628; edge 0.20 |
| Held-out weeks with one alertable column past its edge / with two | 4 of 10 / 0 of 10 |
| Batch flag, held-out false alert rate | 0.0 |

A column's edge is the larger of 0.20 and the worst value it took on an in-control week; a
column alerts strictly above it. The batch flag is the score past its edge, or more alertable
columns past theirs than any held-out in-control week showed (`feature_count_edge`, 1). Either
path raises it, because each misses what the other sees: a scratch probe that tripled the seven
C columns moved the score PSI to 0.076 and seven columns past their edges; one that multiplied
the amount by 100 moved the score to 0.538 and two columns.

**The target encodings are reported and never alert.** Their in-control PSI against the
training window is 0.21 to 11.26 (`card6_te` at the top), because a training row is encoded at
its own lagged day and a served row against the frozen table (ADR 0014): the training histogram
is the encoder's clock, not the population's. The level mix they encode is watched through the
`_cat` and `_freq` columns of the same sources. Whether fitting on that path and serving on its
end point costs accuracy is not established here.

What the job can say: the inputs moved, which columns, by how much, and whether the model's
output distribution moved with them. What it cannot say: whether the model is worse. Figure:
`figures/drift_calibration.png`.

---

## 2. The decay job

`monitoring.performance_report` runs when a batch's labels have matured and refuses the batch
before that, to the second (`assert_mature`; `decay.json` is written as of day 217.0, when the
last batch's labels had matured, and `--as-of-day` runs it earlier with the refusals listed). It
computes precision and recall at the shipped threshold (0.2519, `reports/operating_points.json`)
and PR-AUC with a 1,000-draw row bootstrap on the batch, and the drop against the reference
window as the difference of two independent sets of draws, the interval there is for one model
on two windows.

On the ten in-control weekly batches:

| Batch | Rows, fraud | PR-AUC, interval | Drop against 0.5451 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| val 0 | 20,798, 714 | 0.7458, 0.7173 to 0.7758 | -0.2007 | 0.7628 | 0.6485 |
| val 1 | 19,258, 727 | 0.6348, 0.5971 to 0.6709 | -0.0897 | 0.7319 | 0.5296 |
| val 2 | 18,572, 611 | 0.5904, 0.5480 to 0.6300 | -0.0453 | 0.7181 | 0.4795 |
| val 3 | 18,549, 639 | 0.4984, 0.4584 to 0.5391 | +0.0467 | 0.5902 | 0.3944 |
| val 4 | 11,404, 351 | 0.5582, 0.5054 to 0.6077 | -0.0130 | 0.5739 | 0.4758 |
| test 0 | 21,273, 632 | 0.5794, 0.5419 to 0.6157 | -0.0343 | 0.6546 | 0.4858 |
| test 1 | 20,716, 642 | 0.5378, 0.4982 to 0.5762 | +0.0073 | 0.5740 | 0.4953 |
| test 2 | 20,454, 725 | 0.4942, 0.4580 to 0.5342 | +0.0510 | 0.6160 | 0.4028 |
| test 3 | 18,027, 745 | 0.5863, 0.5489 to 0.6248 | -0.0412 | 0.5902 | 0.5007 |
| test 4 | 8,111, 339 | 0.5275, 0.4736 to 0.5839 | +0.0176 | 0.5842 | 0.4808 |

No batch is a decay under the rule (a drop of at least the 0.06 tolerance with an interval
excluding zero). The largest drop is test batch 2's 0.0510, interval 0.0091 to 0.0904, so the
tolerance clears the in-control weeks by 0.009. The first val week, at 0.7458, is the week
after the training window ends; what follows is distance from the training window, not decay,
and it is a tenth of PR-AUC at the weekly grain against a monthly half-width of 0.0170.

What the job can say: the model is worse than it was accepted at, by how much, and whether
that clears the noise. What it cannot say: why, or anything before the labels arrive. Figure:
`figures/decay_in_control.png`.

---

## 3. The character of the drift: adversarial validation

`scripts/adversarial_validation.py` fits a LightGBM classifier to tell a training row from a
test row on the shipped model's 186 input columns, five folds out of fold, three times.

| Fit | Columns | Out-of-fold AUC | Top of the gain ranking |
| --- | --- | --- | --- |
| full | 186 | 1.0000 | `addr2_te` 0.695, `ProductCD_te` 0.114, `card3_te` 0.107, `card4_te` 0.052; the target encodings carry 0.9815 of the gain |
| without the target encodings | 170 | 0.9031 | `id_13` 0.186, `D11` 0.119, `D15` 0.112, `D10` 0.070, `vblockmean_V1` 0.059, `dist1` 0.053, `D4` 0.029, `C12` 0.026, `card1_freq` 0.022, `vblockmean_V322` 0.017 |
| without identifiers, their encodings and the timedeltas | 128 | 0.8600 | `id_13` 0.260, `vblockmean_V1` 0.103, `C13` 0.065, `dist1` 0.063, `missing_block_D11` 0.048 |

The full fit is the encoder's clock again: the encodings of the narrow columns take a handful
of frozen values on test that the training smear never lands on exactly, and the classifier
needs nothing else. The reading is the second fit. Its top ten split their gain 0.508 to
columns that name who or when (four timedeltas, `card1_freq`) and the rest to columns that
describe the transaction or the device (`id_13`, the V1 block, `dist1`, `C12`); by group over
all 170 columns, the timedeltas carry 0.365, the identity numerics 0.222 (mostly `id_13`), the
V block means 0.132, the identifier encodings 0.083, the counts 0.067, the raw identifiers
0.017, the amount 0.008.

**Against stage 2.** Over the 59 raw columns both hold, the Spearman correlation between this
gain share and stage 2's whole-window PSI is 0.663 (p 1.1e-8): the two rankings agree. Stage 2's
two loudest drifters that survive into the stack, `id_13` and `D11`, are ranks 1 and 2 here. Of
the 59 columns stage 2 flagged as time-inconsistent (a single-feature AUC that inverted inside
train), the three that survive into the stack by name are not in the top thirty; through their
sources, five top-thirty columns touch them, at ranks 10, 14, 15, 22 and 24 (block means
holding flagged V columns, and `id_38_cat`). Stage 2's finding was that the columns that drift
are not the columns whose label relationship inverted; that holds here by name and is softer by
source.

**What the README and the case study say, from this.** The shift between the training and test
windows on this file is population turnover and the calendar: days-since counters that grow for
every returning card, an identity attribute whose value mix changed, and the frequency of card
identifiers. It is not a change in what fraud looks like; the columns whose relationship with
the label inverted appear only through block averages at the tail of the ranking. Calling it
adversarial drift, or concept drift, would claim something this data does not show. The second
finding is about the pipeline: the target encodings separate train from test perfectly by
construction, which is a fact about how they are computed, not about the cards. Figure:
`figures/adversarial_validation.png`; the expectations written before the run are in
`docs/notes/stage-09.md`, and most of the ordering was wrong.

---

## 4. The lifecycle

`lifecycle.should_retrain` reads the two histories, oldest first. Decay on the most recent
matured batch triggers on its own. Failing that, the drift flag raised on `SUSTAINED_CYCLES`
consecutive batches triggers; one raised batch does not. Failing both, nothing.

**The constants** (ADR 0038), each with its derivation in the comment beside it.

- `PROMOTION_MARGIN = 0.06`: stage 6's widest paired 95 percent half-width on test PR-AUC is
  0.0172 by row and 0.0592 by card (`reports/metric_variance.json`, `noise_band`); the card
  resample is the wider and the one the gate runs; 0.0592 rounded up at the second decimal.
- `DECAY_TOLERANCE = PROMOTION_MARGIN`: a champion that has lost less than the margin cannot be
  replaced by a challenger that merely recovers the loss, so a trigger below the margin asks for
  work the gate discards. The in-control weeks clear it by 0.009.
- `SUSTAINED_CYCLES = 2`: the smallest run that is not a spike. The held-out false alert rate of
  the batch flag is 0.0 on the ten in-control weeks, so no in-control run exists to calibrate a
  larger number against.
- `MATURITY_DAYS = 30`: assumed, as above.

**The challenger.** `lifecycle.fit_challenger` takes the joined raw frame and three windows of
mature data, newest first: a gate window the fit never sees, a calibration window the isotonic
calibrator and the F1 threshold are read off, and a fit window. It fits the stage 3 pipeline on
the fit window inside `data_loader.training_boundary(fit_end)`, which moves the boundary the
train-only guard enforces and puts it back; nothing is switched off. Then the stage 6 family at
the shipped hyperparameters on the champion's columns, less any the window's fit did not produce
(the frequency encoder skips a column under 50 levels, and `addr2` has 67 over the training
split and about 29 over sixty days of it; the record names the dropped column and the bundle's
schema is trimmed to match). The result is a serving bundle in the stage 8 layout.

**The gate.** `lifecycle.gate` scores both models on the gate window, bootstraps them on
identical resamples by entity (1,000 draws), and promotes when the paired difference has a
point at or above the margin and a lower bound above zero. Each refusal says which condition
failed. `tests/test_lifecycle_units.py` runs it in both directions on generated scores: a
better challenger promoted, a worse one refused, a better-by-less-than-the-margin one refused,
and one whose point clears a zero margin while the interval reaches zero refused for the
interval.

**Promotion and rollback.** `lifecycle.Registry` moves aliases on the stage 8 registry.
`promote(version)` sets `previous` to the version `production` pointed at and `production` to
the new one; `rollback()` sets `production` back to `previous`, clears `previous`, and marks
the version it moved from as `rolled-back`. A rollback with nothing to roll back to raises.
The unit test registers two versions into a temporary registry and checks every alias after
each move.

---

## 5. The demo, and what it does not establish

`scripts/lifecycle_demo.py`, seed 42, 306 seconds. The live stream is the val and test windows
in order, 177,162 rows in nine weekly batches from day 120.0 to day 183.0, real consecutive
weeks with no row replayed. From batch 3, day 141.0, every raw row carries an upstream change
the model never saw: the amount times 100 and every C column to 3c + 7, applied before
preparation (118,534 rows). One cycle per batch; the clock keeps ticking after the stream ends
until the last batch's labels have matured. Challenger windows 60, 14 and 14 days; the registry
is `mlruns/lifecycle_demo/`, rebuilt every run, with the shipped model registered as version 1.

| Cycle, day | Drift job | Decay job (batch: PR-AUC against the reference) | Decision | Challenger and gate |
| --- | --- | --- | --- | --- |
| 0 to 2, 127 to 141 | score PSI 0.0059 to 0.0178, 0 columns, no flag | nothing mature | none | |
| 3, 148 | score PSI 1.8274, 15 columns past their edges, flag | nothing mature | one cycle, not sustained | |
| 4, 155 | flag | nothing mature | sustained drift | gate window would overlap the champion's training window: not trained |
| 5, 162 | flag | batch 0: 0.7458 against 0.5451 | sustained drift | same, not trained |
| 6, 169 | flag | batch 1: 0.6348 | sustained drift | challenger 1, fit days 51 to 111, gate 125 to 139 (clean): 0.6520 against 0.6082, -0.0438 (-0.0644 to -0.0250), refused: not better |
| 7, 176 | flag | batch 2: 0.5904 | sustained drift | challenger 2, fit 58 to 118, gate 132 to 146 (12,841 of 36,852 rows drifted): 0.2875 against 0.3829, +0.0954 (+0.0736 to +0.1185), promoted; production 1 to 3 |
| 8, 183 | flag, against the new champion's reference | batch 3: 0.2012 against 0.3829, drop 0.1817, decay | decay | challenger 3, fit 65 to 125, gate 139 to 153: 0.2557 against 0.2980, +0.0424 (+0.0200 to +0.0644), refused: better, but by less than the margin |
| 9, 190 | stream ended | batch 4: 0.3025, drop 0.0803, decay | decay | challenger 4, fit 72 to 132, gate 146 to 160: 0.2792 against 0.3413, +0.0621 (+0.0403 to +0.0867), promoted; production 3 to 5 |
| 10 to 13, 197 to 218 | | batches 5 to 8: 0.3082, 0.3625, 0.3613, 0.3077 against 0.3413, no decay | none | |
| drill | | | | rollback: production 5 to 3, `rolled-back` 5; loaded version 3 through the registry, 200 clean test rows through the serving path against the batch path, largest difference 0.0 |

What the run establishes. The two clocks are different clocks: the drift job saw the shift on
the batch it started in and the decay job five cycles later. One drifted batch did not trigger
a retrain; two did, and the early trigger had a challenger at the gate at cycle 6 and one
promoted at cycle 7, before any decayed batch had matured. The gate refused twice and promoted
twice, with the reasons recorded. The rollback moved the alias back and the version it moved
back to served as it scored in batch.

What the run does not establish, stated plainly.

- **Whether the gate was exercised near its margin.** Once, and not firmly: the cycle 9
  promotion cleared 0.06 by 0.0021, inside that pair's own half-width of 0.0232. The other three
  verdicts were 0.10 or more from the margin on either side. One near-margin decision on one
  seed says nothing about the gate's behaviour at the margin.
- **Whether sustained drift was observed across real consecutive cycles or synthesised.** The
  cycles are real consecutive weeks of the val and test windows and no row was replayed; the
  drift itself is synthetic, a transformation applied to real rows from a chosen day. Sustained
  drift of the file's own making was not observed: on the in-control weeks the flag never rose.
- **Whether a challenger learned the drift.** None did. Every fit window ended before day 141
  (`drifted_rows_in_fit_window` is 0 for all four), because thirty days of maturity plus
  twenty-eight days of calibration and gate windows consumed every mature drifted row. The two
  promoted models won on drifted gate windows by being less damaged than the champion, not by
  having seen the change, and why a sixty-day model is less damaged by this shift than the
  120-day one is not established.
- **Whether the lifecycle holds an absolute standard.** It does not. The reference moves to the
  promoted model's gate window on promotion, so after cycle 9 the bar is 0.3413 and batches at
  0.31 to 0.36 are "no decay" while the shipped model was accepted at 0.5451. The design as
  built ratchets; a floor is a design not written.
- **A promoted model's drift reference.** It is calibrated on the four batches of its
  calibration and gate windows, which hold drifted rows, so its edges absorb the shift it was
  promoted under: the second promoted model's score edge is 0.7429 and its count edge 5
  (`promotion.new_drift_reference`), against 0.20 and 1 for the shipped model on ten clean
  weeks. Under the first promoted model's reference the cycle 8 flag was raised on the score
  path alone, with 15 columns past their edges under its count edge.
- **Anything about the true chargeback delay, the batch length, or the window lengths.** All
  inputs; the artifact labels them.

Figure: `figures/lifecycle_demo.png`. An interactive replay of the same artifact, cycle by
cycle, is `docs/demo/index.html` (`make demo-page`, ADR 0039); it is one self-contained file
for a static host and quotes nothing the artifacts do not hold.

---

## What stage 10 inherits

- Two jobs, two schedules and two artifacts, and a lifecycle that reads both. A deployment runs
  `monitor-drift` on the batch close and `monitor-decay` on the batch close plus the maturity
  window; the Makefile says so beside the targets.
- A statement about the drift on this file for the README and the case study: population
  turnover and the calendar, not concept drift; and the target encodings' clock as a pipeline
  finding.
- The promotion margin, derived and tested, and a demo artifact whose every verdict a test
  recomputes from its own numbers.

## What is not established

- The true chargeback delay, and therefore the maturity window. Thirty days is an input.
- Whether the target encodings' train-time distribution costs accuracy at serving time, and
  what encoding training rows from a table frozen at a fixed lag would do to the stage 6
  numbers.
- Why a sixty-day challenger fitted before the shift is less damaged by it than the shipped
  model, and whether a challenger fitted on drifted rows would recover the accepted level.
- The gate's behaviour at the margin, beyond one verdict 0.0021 over it.
- An absolute floor for the decay reference, and a decay batch longer than a week.
- The batch flag's false alert rate on more than ten in-control weeks.
