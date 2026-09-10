# 0002: Split strategy and boundaries

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 0

## Context

The system is asked to score transactions that happen after the ones it was trained on. Any
evaluation that does not respect that ordering measures interpolation instead. Two distinct
leaks are possible here and they need separating.

**Temporal leakage:** training on rows that occur after the evaluation rows. A model that has
seen the future of a fraud campaign will look better than it will behave.

**Entity leakage:** training and evaluating on transactions from the same entity. Because labels
cluster at entity level (0.9664 purity at the chosen key, ADR 0001), a model can memorise the
entity and score its other transactions correctly without learning anything transferable.

TransactionDT is a relative offset in seconds, not wall-clock time. It runs from 86,400 to
15,811,131, spanning 181.999 days, integer valued, with a measured granularity of 1 unit, and it
is non-decreasing in file order. The unit is a second and 86,400 units is a day: folding
TransactionDT into a position within an 86,400 unit cycle produces the diurnal shape of card
activity, with a trough at hour position 9 (2,479 transactions) and a peak at hour position 19
(42,115 transactions), a peak to trough ratio of 16.99. A unit other than a second, or a cycle
length other than a day, would not produce that. What the offset is measured from is not
established, so a "day" in this repo means a 86,400 unit block of TransactionDT and day indices
are relative, running from day 1 to day 182.

## Options considered

**Random split.** Rejected. It breaks both properties at once. It trains on the future, and it
scatters an entity's transactions across train and test, so the 0.9664 label purity becomes a
memorisation shortcut. Reported numbers would be high and would not survive deployment.

**GroupKFold on the entity key.** This fixes entity leakage and nothing else. Folds still mix
time, so the model trains on later transactions and is scored on earlier ones. It also destroys
the thing the platform most needs to measure, which is how performance decays as the gap between
training and scoring grows. A grouped random split additionally guarantees that no evaluation
entity has history, which overstates the "cold entity" share relative to what production sees:
in the chronological test window here, 0.3334 of entities are ones the training window had
already seen. Rejected as the primary protocol. It remains a reasonable diagnostic later for
isolating the entity-memorisation component, which would be its own ADR.

**Chronological split with a held-out final window.** Chosen. It reproduces the production
ordering, it leaves entity overlap at whatever the data actually produces rather than at 0 or at
an arbitrary value, and it makes drift measurable because each split is a contiguous time window.

## Decision

Split chronologically 70 / 15 / 15 on TransactionDT. Boundaries are TransactionDT thresholds and
assignment is `dt < boundary`, so no timestamp is split across two windows.

| Split | Boundary rule | Rows | Row fraction | Day index range | Span (days) | Fraud rate |
| --- | --- | --- | --- | --- | --- | --- |
| train | dt < 10,437,998 | 413,378 | 0.7000 | 1 to 120 | 119.810 | 0.03517 |
| val | 10,437,998 <= dt < 13,151,845 | 88,581 | 0.1500 | 120 to 152 | 31.410 | 0.03434 |
| test | dt >= 13,151,845 | 88,581 | 0.1500 | 152 to 182 | 30.778 | 0.03480 |

The boundaries are the 0.70 and 0.85 quantiles of TransactionDT, which is why the realised row
fractions land on the requested ones. Choosing quantiles of the row distribution rather than
round day numbers keeps the training window large enough while leaving two evaluation windows of
roughly a month each, which is the shortest window over which a monthly retrain cadence could be
judged.

The test window is held out from model selection entirely. Threshold choice, calibration and
model comparison run on val.

## Evidence

- **Artifact:** reports/audit.json, sections `chronological_split`, `entity_overlap`,
  `transaction_dt`.
- **Command:** `make audit`
- Overall fraud rate is 0.03499 on 590,540 rows (20,663 fraud). The three split fraud rates
  (0.03517, 0.03434, 0.03480) sit within 0.00083 of each other, so the split does not by itself
  introduce a class-balance shift that would confound later drift measurement.
- Entity overlap at the chosen key: the training window holds 164,702 entities. The test window
  holds 45,348 entities of which 30,227 (0.6666) never appear in training, carrying 0.6598 of
  test rows. The val window holds 46,884 entities of which 26,894 (0.5736) are unseen, carrying
  0.5311 of val rows.
- Fraud rate on test rows belonging to unseen entities is 0.0440 against 0.0170 on rows whose
  entity appeared in training.

## Consequences

- Evaluation is dominated by cold entities: two thirds of test rows belong to entities with no
  training history. A model that leans on entity aggregates will show a larger train to test gap
  than one that does not, and that gap is a real property of the problem, not a bug in the split.
- The val to test gap is one month of relative time, so a val result is a forecast one month out.
  Promotion rules in later stages have to account for that horizon.
- Because the boundaries are fixed TransactionDT integers, every later stage can reproduce the
  exact same rows without re-running quantiles on a filtered frame.
- The seen and unseen entity fraud rates differ by a factor of about 2.6, so any future decision
  to drop or impute entity history changes the evaluation population and is not a neutral choice.
