# 0009: Time consistency as a feature screen

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 2

## Context

The split is chronological (ADR 0002), so the model is always asked to work on a window later
than the one it learned from. A feature can be strongly associated with the label over the whole
training span and still be useless, or worse than useless, in that arrangement: if it separated
the classes one way in the first month and the other way in the last, a measurement taken over
all 120 days averages the two and reports whatever is left over.

Every screen this repo has so far is blind to that. Information value is computed over the whole
train split and has no time argument. Correlation is a single number per pair with no time
argument either. PSI does have a time argument, and it answers a different question: it asks
whether a distribution moved, not whether the relationship between that distribution and the
label moved. A column can be perfectly stable under PSI and have inverted, and a column can drift
heavily while its relationship with the label holds.

## Options considered

**Rely on the validation split.** Train the model, look at the val metric, and drop features if
it disappoints. Rejected as the only mechanism. It is a single scalar over 432 features, it
arrives after the model exists, and it cannot say which feature caused the shortfall. It is also
one use of val per attempt, and val is where thresholds and calibration are chosen.

**Split the training window and compare information value between the halves.** Cheap and
usable, and it was the runner-up. Rejected because IV over quantile bins fixed on one half does
not distinguish a feature whose relationship inverted from one whose distribution moved into a
different bin. The two need different actions.

**Train a single-feature model on an early window and score it on a later one.** Chosen. With one
feature the trees can only cut that column, so the model is close to a supervised binning of it,
and the AUC on the later window is a direct answer to the question asked: is the ordering this
column induced on the early window still the right ordering later.

## Decision

For each of the 432 feature columns, fit a LightGBM classifier on that column alone over the
first 30 days of the training window (day index 1 to 30, 134,339 rows, 3,401 fraud) and score it
on the last 30 days of the same window (day index 91 to 120, 97,451 rows, 3,836 fraud). Report
both AUCs.

The model is `lightgbm.LGBMClassifier` with n_estimators 50, num_leaves 15, min_child_samples 200,
learning_rate 0.1 and random_state 42 from config.SEED. A categorical column is recoded onto the
early window's category set, so a level the early window never saw becomes missing rather than
inheriting another level's code.

A column is flagged when the early AUC is at least 0.51 and the late AUC is below 0.50. Both
numbers are choices and are recorded in the artifact next to every column's pair, so a reader can
move the line and re-read the list without re-running anything.

The early AUC is in-sample and is not a performance estimate. It is there to separate a column
that never had any signal from a column whose signal inverted, which is the whole point of the
screen and is not visible from the late number alone.

## Evidence

reports/eda/temporal.json, block `time_consistency`, regenerated with `.venv/bin/python
scripts/eda.py --sections temporal`. 432 models, about 95 seconds, every one returning status
"ok". Wall-clock time moves between
runs and the artifact records what the run that produced it measured.

59 columns are flagged. 62 have a late AUC below 0.50 in total, so the flag rule excludes three
whose early AUC did not reach 0.51: id_22 at 0.504 early and 0.500 late, id_07 at 0.508 and
0.499, id_08 at 0.510 and 0.498. Those three never separated the classes to begin with, which is
a different finding from inverting.

The flagged set is not scattered. It is three coherent groups.

| Group | Flagged members | Early AUC range | Late AUC range |
| --- | --- | --- | --- |
| V138 to V166 | 15: V138, V141, V142, V144, V148, V151, V153, V154, V155, V159, V160, V161, V162, V163, V166 | 0.576 to 0.618 | 0.468 to 0.498 |
| V322 to V339 | 18: every column in that range | 0.569 to 0.588 | 0.467 to 0.489 |
| The rest | 26: V14, V19, V20, V25 to V28, V41, V46, V55, V56, V61, V62, V65 to V68, V88, V89, V285, card4, id_14, id_26, id_32, id_34, id_38 | 0.511 to 0.604 | 0.363 to 0.498 |

The strongest single case is id_38: 0.6037 early, 0.3634 late. A column that ranks the classes
better than chance in the first month and worse than chance in the last, by that margin, is
carrying a pattern that reversed inside four months.

**What it catches that a correlation screen does not.** Take the V138 to V166 block. The
correlation work groups them because they are copies of each other: V141 and V161 correlate at
0.9998, V142 and V163 at 0.9998. That grouping says keep one of them. It says nothing about
whether the one you keep is any good, and this screen says the one you keep inverts. The two
screens are not substitutes and neither subsumes the other.

The same point against PSI. Of the 59 flagged columns, none appears in the 22 features whose PSI
from train to test exceeds 0.25. The intersection of the two lists is empty. The features that
drift most (card_start_day at 1.841, id_31 at
1.500, id_13 at 0.563, D11, M7 to M9, V1 to V11) and the features that invert are two different
lists. A pipeline that screened on drift alone would keep all 59.

**What the screen is not.** It is one seed and one hyperparameter setting, and the AUCs are not
carried forward as measurements of anything. Run-to-run variance on this screen is not measured
in this repo, so a column at 0.499 late and a column at 0.501 late are not meaningfully different
and the flag list should be read as a block, not as a ranking. The 59 are candidates for removal
in stage 3, not a decision taken here.

## Consequences

- Stage 3 inherits a list of 59 columns with a reason attached to each. It is a candidate list.
  Removing a feature is a modelling decision and belongs in the stage that can measure the effect
  of removing it.
- The screen has to be re-run whenever the training window moves, because it is defined relative
  to that window's first and last 30 days. `make eda` does that, and the window boundaries are in
  the artifact so a stale run is visible.
- It cannot see a feature that inverts inside the first 30 days or after the last 30, and it
  cannot see an interaction that inverts while both of its parts hold. Single-feature models see
  single features.
- Running 432 gradient boosting fits to screen features is a real cost, and it is under two
  minutes, which settles the objection.
- The early AUC being in-sample means the pairs cannot be read as a generalisation gap. Anyone
  reading the scatter in figures/time_consistency.png should read the vertical axis and use the
  horizontal one only to tell inversion from noise.
