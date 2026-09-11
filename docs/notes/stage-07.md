# Stage 7: the leakage delta and the label latency

Stages 1 to 5 built a strictly causal, train-only-fitted pipeline and stage 6 measured it at a
test PR-AUC of 0.5451 (0.5274 to 0.5614). Every claim about why that discipline matters has been
an assertion so far. This stage prices two of them: what the same model scores when the four
standard competition shortcuts are switched on one at a time and together, and what happens to it
when the most recent weeks of training labels are not yet mature.

## Expectations, written before the runs

This section was committed before either script ran. The point of writing it down first is that
the comparison between what I expected and what the artifacts say is the material for the case
study, and a prediction written after the number is not a prediction.

The reference is the shipped model: xgboost at the default hyperparameters, seed 42, the 186
column stack, test PR-AUC 0.5451 with a row-resample half-width of 0.0170. A delta inside that is
not a finding.

### Experiment 1, the four shortcuts

**(a) Encoders fitted on all 590,540 rows instead of the 413,378 train rows.** I expect +0.02 to
+0.04 test PR-AUC. The vocabulary and frequency encoders leak only the shape of the test window.
The target encodings are the leak: fitted on every label at the same 14 day lag, a test row on
day 170 reads the fraud rate of its own `card1` level through day 156, which includes the val and
test windows. `card1` is the coarsest component of a key with 0.9664 label purity, so that is the
entity's own later labels under another name. Above the noise band, below (b).

**(b) Entity aggregates over the whole dataset.** The largest of the four on its own: +0.06 to
+0.12. Mean and standard deviation of the C, M and normalised D columns per entity, computed over
all of the entity's rows including the ones after each row being scored. It reaches every test
row where a train-fitted entity statistic reaches 0.3402 of them (ADR 0020), and it is the
construction the first-place solution rests on.

**(c) Replacing each prediction with the entity's mean prediction over the test window.** +0.03 to
+0.06. With label purity at 0.9664, a fraud entity's rows are almost all fraud, so averaging
carries a high score from the rows the model catches to the rows it misses. Bounded above by the
share of test rows on an entity with more than one test row, which the script measures.

**(d) A random 70/15/15 split instead of the chronological one.** The largest single delta, +0.15
or more: PR-AUC above 0.70, ROC-AUC above 0.95. Nearly every test entity is then seen in training
with its labels, and the model memorises the entity. The test population changes with the split,
so this delta is unpaired and I expect it to be too large for that to matter.

**All together.** On the chronological split, (a) plus (b) plus (c): +0.12 to +0.20, less than
the sum because (b) and (c) are both entity-level information. With (d) as well: PR-AUC above
0.85 and ROC-AUC above 0.97, which is the neighbourhood of a leaderboard score.

### Experiment 2, simulated label latency

The simulation: a fraud transaction in the final N days of the training window has not been
charged back at training time, so its label reads 0. The immature variant fits on every train row
with those labels; the corrected variant drops the final N days from the fit. The target encoder
is refitted on the same labels the model sees in both cases, so nothing at training time knows a
label that has not arrived.

**Immature.** Test PR-AUC falls monotonically with N: inside or near the noise band at N=15,
material by 45, large at 90 where about three quarters of the training positives read 0. The
predicted fraud rate on test falls below the observed 0.0348 by a margin that grows with N; at
N=90 I expect it under half the observed rate.

**Corrected.** Test PR-AUC falls slowly with N, from a smaller and staler fit set; at N=90 the fit
set is 30 days and I expect a drop of a few hundredths. The predicted fraud rate stays within a
few thousandths of the observed rate at every N.

**The comparison.** Immature is worse than corrected at every N and the gap widens with N. If
dropping the data costs more than keeping the immature labels at N=15, that is the number I am
wrong about, and it is the one a team deciding a maturation window would need.

## What the artifacts say, against what I expected

Everything below is from `reports/leakage_delta.json` and `reports/label_latency.json`. The
expectations are the section above, unedited. Ten items: six wrong, one half right, three right,
and two of the six wrong in direction.

| Item | Expected | Measured | Right? |
| --- | --- | --- | --- |
| (a) encoders on every row | +0.02 to +0.04 | +0.0060, +0.0013 to +0.0108 by row; -0.0084 to +0.0214 by card | no: a fifth of the low end |
| (b) entity aggregates over every row | +0.06 to +0.12 | +0.0371, +0.0290 to +0.0458 | no: below the low end |
| (c) entity-mean post-processing | +0.03 to +0.06 | -0.0554, -0.0656 to -0.0463 | no: wrong sign |
| (d) random split | +0.15 or more, PR-AUC above 0.70, ROC-AUC above 0.95 | +0.2593, 0.8044, 0.9633 | yes |
| (a)+(b)+(c) chronological | +0.12 to +0.20 | -0.0262, -0.0399 to -0.0141 | no: wrong sign |
| (a)+(b)+(c)+(d) random | PR-AUC above 0.85, ROC-AUC above 0.97 | 0.8317, 0.9759 | half |
| immature at N=15 | inside or near the noise band | -0.0345, twice the band | no |
| immature implied rate at N=90 | under half the observed | 0.0000 | yes, by more than I meant |
| excluded implied rate | within a few thousandths of observed at every N | 0.0302 at 15, 0.0211 at 60, against 0.0348 | no |
| immature below excluded at every N, gap widening | yes | -0.0221 at 15 to -0.2864 at 90 | yes |

## The number that changed my mind

-0.0554. The entity-mean post-processing, the step I was most confident would inflate the metric,
lowers test PR-AUC by more than the published aggregation raises it, with an interval nowhere
near zero and ROC-AUC unmoved (+0.0004, -0.0022 to +0.0028).

I had reasoned from the label purity: 0.9664 of multi-transaction entities are label-pure, so
averaging within an entity should carry a caught fraud row's score to its uncaught siblings. The
purity is real and it is the wrong statistic. Purity counts entities; PR-AUC counts rows. The
diagnostic I added after the number came in partitions the test window's 3,083 fraud rows by
the kind of entity they sit on: 0.1898 on singletons, where the mean does nothing; 0.4567 on
entities that are all fraud, where it helps or is neutral; 0.3536 on the 457 entities that mix
labels, where it pulls 1,090 fraud rows and 2,467 legitimate rows toward one number. The 457
impure entities are 0.0279 of the multi-row entities and hold a third of the fraud, because
the impure ones are the big ones.

And most of the big impure ones are the null bucket. ADR 0001 keeps a null `addr1` or `D1` as
its own key component, so `card1|NA|NA` is one entity for every row of a `card1` with no
address, 0.1053 of the test rows at a fraud rate of 0.1366, across 3,302 such entities. Leaving
those alone and averaging the rest gives +0.0035, inside the noise. The published construction
pastes `nan` into its identifier and pools the same way, so the variant is faithful to it; what
I do not know is what the step does on the competition's own test set, where a much larger
model and a different window might put the fraud rows somewhere else. The note for the case study
is narrower than I wanted: on this key and this window, the step hurts, and the reason is
measured.

## (b), and the number I was closest to

+0.0371 for the published aggregation, against an expectation of +0.06 to +0.12. ADR 0020 owed
this measurement and had set it up as the price of deployability. The price is four hundredths,
resolvable and a tenth of what the split is worth. I expected more because the construction
reaches every test row where a train-fitted entity statistic reaches 0.3402 of them, and because
it is what the first-place solution is built around. Both are still true. What I had not
weighed is that the shipped model already reads the 12 C columns, which ADR 0021 measured as the
strongest block in the frame and which are themselves counts over the card's history; an
entity mean of C1 to C14 is a smoothed version of something the tree already cuts on. The
validation number is stranger: about +0.093 on validation (0.6184 to 0.7111) against +0.0371 on test, and I have no
measured account of why the leak is worth more on the nearer window. It is in the artifact and
in the "not established" list.

## (a), and the guard's price

+0.0060 by row, clearing zero at the low end of the interval with sign agreement 0.997, and
not clearing it by card. The guard that ADR 0006 made structural, that every fitted object goes
through, that the leaky script has to switch off by name, is worth under a hundredth on this
window with this model. I expected +0.02 to +0.04 because the target encodings fitted on every
label hand a test row the fraud rate of its own `card1` level through the validation window,
and `card1` is a component of a key with 0.9664 purity. The lag is part of why it is small: at 14
days a test row on day 170 reads labels through day 156, so for most of the test window the
gain is the validation window's labels and little else. How many test rows share a `card1`
level with a validation fraud is not something I measured, so that is as far as the account
goes. The honest reading is that the guard is
cheap insurance against a leak that on this file happens to be small, and that a file with a
shorter lag or a stickier entity would price it higher. I would not write "the guard is worth
0.006" in a README without the second half of that sentence.

## (d), the shortcut that matters

The one I got right, and the one that carries the stage. +0.2593 on its own, nine tenths of the
+0.2866 the competition-style pipeline gains end to end, and it touches no feature. The random
test rows share an entity with the random train rows 0.7410 of the time against 0.3402 for the
chronological window, and the fraud rate flips: under the random split the entities the model
has seen are the risky ones (0.0387 against 0.0227 unseen); under the chronological split they
are the safe ones (0.0170 against 0.0440). A random split does not just leak; it changes which
population the model is asked about, to the one it has memorised.

The combined chronological variant, (a)+(b)+(c), is below the baseline at -0.0262, because (c)
costs more than (a) and (b) gain. I had expected +0.12 to +0.20. The competition-style number
without the split is worse than the disciplined one on this window, which is the sentence I
least expected to write.

## The immature labels, and the collapse

I expected a gradual loss: a few hundredths at 15 days, growing with N. The loss at 15 days is
-0.0345, twice the shipped model's row half-width, and the immature model's implied fraud rate
on test goes 0.0299, 0.0195, 0.0115, 0.0004, 0.0000 across the sweep. At 60 days it recalls
0.0032 of the test fraud at the shipped threshold and raises 0.3 alerts a day against the
reference's 78.2. The queue goes silent.

The mechanism is in the artifact's `regime` blocks. The immature model at N=60 scores the older
train rows at a mean of 0.0307 and the recent train rows at 0.0005, in sample; the test rows get
0.0004. It has not lost signal, it has learned that the newest rows are a regime of their own in
which nothing is fraud, and every row at serving time is a newest row. I assumed the clock was
the target encodings, whose values move with the cumulative count. The ablation at N=60 says
partly: dropping the 16 target encodings raises the test mean score to 0.0038, ten times, and
still under an eighth of the reference's; dropping the 15 D columns changes nothing; dropping both
gives 0.0043. The frame carries the recent window in more columns than the two I would have
named, and the counts and the V block means are the next suspects, unmeasured. What the
diagnostic does establish is that the failure is a regime, not a dilution, and that no single
block is the door.

The excluded correction is not free of bias either. Its implied rate is 0.0302 at 15 days and
0.0211 at 60, against an observed 0.0348 and a reference that itself implies 0.0327; the fit
set is older and the window has drifted. I expected it within a few thousandths. It is within
a few thousandths of the reference at 15 days and a hundredth below by 45. And the cost of the
correction is not linear: -0.0124 at 15 days and -0.0574 at 30, then -0.0613, -0.0719, -0.0959.
The second fortnight costs more than three times the first, which says the most recent
fortnight before the boundary is the one the model most needs. At no N in the sweep does
keeping the immature labels beat dropping the rows.

## Dead ends and things that cost a rerun

- The first smoke run of the leakage script, on an 8 percent row sample, fell over on
  `addr2_freq`: the sample dropped `addr2` below the frequency encoder's cardinality floor, so
  the prepared frame had 246 columns and the shipped list names 247. The smoke mode now reads
  whatever survived and says so; the committed artifact is from the full run.
- The post-processing diagnostic and the regime diagnostic were both added after reading the
  first run's numbers, and both scripts were rerun in full so the committed artifact carries
  them. Each rerun reproduced every score to four decimals.
- I wrote "three quarters" for 0.2593 of 0.2866 in the first draft of the ADR. It is nine
  tenths. The cross-check caught it because the ratio was not in any artifact.
- The guard is switched off with a context manager that patches two names,
  `data_loader.assert_train_only` and `prepare.assert_train_only`, because `prepare` binds the
  name at import and the decorator looks it up at call time. A test asserts both are restored.
  The first draft patched one and the preparation fit raised on the full frame, which is the
  guard doing its job against the script that exists to defeat it.
- Stage numbering: this is stage 7 and the serving path is stage 8. Earlier documents that say
  "stage 7" for the serving path were written before the plan drifted, and ADRs are immutable.
  The stage document says so once.

## Smaller things

- The random split's own row-resample half-width is 0.0122, narrower than the chronological
  0.0170. A model that memorises its entities is also more stable under resampling of them.
- The leakage script's baseline reproduces the shipped model's 0.5451 through a rebuild from the
  raw files, and the latency script's reference reproduces it through a refit of the target
  encoder that matches the cached columns to 0.0. Two independent routes to one number, both
  asserted by tests.
- ROC-AUC moves +0.0709 under the full competition pipeline where PR-AUC moves +0.2866. The
  metric that leaderboards use is the one that shows the least of this.
