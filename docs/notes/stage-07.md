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
