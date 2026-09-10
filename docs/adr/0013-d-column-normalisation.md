# 0013: D-column normalisation is measured and not applied

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

The 15 D columns are time deltas measured backwards from the transaction. The argument for
normalising them is clean and it is worth stating properly before the measurement contradicts it.
A card observed on day 30 and again on day 110 carries a D that has grown by 80 in between, so the
column drifts with the calendar by construction. Subtracting the delta from the day index gives
the origin the delta counts from, `D_n = TransactionDT / 86400 - D`, and an origin does not move.
A column that has stopped drifting is a column a model trained on one window can carry into the
next.

Stage 2 supplies the way to check it. PSI against a later split, binned on train with missing as
its own level, is exactly the question "did this distribution move", and stage 2 already measured
it for all 432 features so the before number can be reproduced rather than recomputed with a
different convention.

## Options considered

**Apply the normalisation to every D column.** What the argument says to do. Rejected by the
measurement below.

**Apply it to none, on the grounds that it might not help.** Rejected as the starting position:
that is a guess in the other direction and it forgoes a transformation that is cheap, invertible,
and well argued.

**Measure it per column and let the number decide, with the rule stated first.** Chosen. The rule
is: the normalisation is applied to a column when it lowers that column's PSI against test, and
not otherwise. That rule was written into `scripts/prepare.py` before the numbers were read, and
the `applied_to` list in the artifact is whatever the rule produced.

A second measurement was added after the PSI result came back, and it is worth being explicit
about why. PSI answers whether a distribution moved, not whether the column got better at the job.
A transformation could raise PSI and still improve a model, so a single-feature LightGBM fitted on
the whole training window and scored on validation was run for each column twice, once on the raw
delta and once on its origin. That is a second question, not a second chance at the first.

## Decision

The D-column normalisation is implemented in `fraud_platform.transforms.to_d_origin`, is
invertible through `from_d_origin`, and is applied to no column. The prepared frame carries the
raw D columns.

The implementation stays because the measurement is the deliverable and has to be reproducible,
and because a later window could change the answer. `reports/prep/schema.json` reads the applied
list out of `reports/prep/transforms.json`, so if the rule ever selects a column the pipeline
picks it up without an edit.

## Evidence

`reports/prep/transforms.json`, block `d_normalisation`, regenerated with `.venv/bin/python
scripts/prepare.py --sections transforms`. `figures/d_column_psi.png` draws both panels.

The before column reproduces `reports/eda/temporal.json` exactly for all 15 columns, which the
artifact records field by field and a test asserts. That is what makes the after column
comparable.

| Column | PSI raw | PSI normalised | Validation AUC raw | Validation AUC normalised |
| --- | --- | --- | --- | --- |
| D1 | 0.0226 | 1.8304 | 0.6426 | 0.6197 |
| D2 | 0.0374 | 0.5138 | 0.6469 | 0.6374 |
| D3 | 0.0352 | 2.2125 | 0.6602 | 0.5337 |
| D4 | 0.0414 | 0.9894 | 0.6208 | 0.5974 |
| D5 | 0.0206 | 1.6019 | 0.6688 | 0.5549 |
| D6 | 0.0207 | 0.3653 | 0.6304 | 0.6299 |
| D7 | 0.0081 | 0.2732 | 0.6072 | 0.6071 |
| D8 | 0.0152 | 0.2127 | 0.6386 | 0.6456 |
| D9 | 0.0120 | 1.3919 | 0.6415 | 0.6382 |
| D10 | 0.0746 | 1.2908 | 0.6497 | 0.6282 |
| D11 | 0.3313 | 0.9536 | 0.6890 | 0.6902 |
| D12 | 0.0054 | 0.3584 | 0.6284 | 0.6282 |
| D13 | 0.0100 | 0.4887 | 0.6028 | 0.6021 |
| D14 | 0.0124 | 0.3505 | 0.6174 | 0.6176 |
| D15 | 0.0950 | 0.9977 | 0.6499 | 0.6092 |

**The normalisation raises PSI on all 15 columns.** D3 goes from 0.0352 to 2.2125, a column that
was among the most stable in the frame turned into one of the least. D1 goes from 0.0226 to
1.8304, which is a result the repo has seen before: stage 2 measured `card_start_day`, which is
`floor(TransactionDT / 86400 - D1)`, at PSI 1.841 against test and called it the loudest and least
interesting number in the document. D1's origin is card_start_day without the floor, and it lands
in the same place.

**And it lowers validation AUC on 12 of 15.** The three that rise do so by 0.0013 (D11), 0.0002
(D14) and 0.0070 (D8), all far inside the 0.0149 that stage 2's class balance says an unpaired
comparison on validation can resolve. The two largest falls are D3 at 0.1265 and D5 at 0.1139,
which are not inside anything.

The mechanism is visible once the numbers are in front of you and it inverts the argument. The
raw D columns are already stationary, because the population being observed is a mixture of cards
at every age and new cards keep arriving at D near zero. The mixture is what holds still. The
origin is a calendar date, and a later window necessarily contains later origins, so normalising
takes a stationary column and writes the calendar into it. The transformation does exactly what it
was designed to do, to the wrong column: it removes a drift that these columns did not have and
introduces one they did not have either.

## Consequences

- The prepared frame carries the raw D columns, and stage 4's entity-history features build on
  them unchanged.
- `figures/d_column_psi.png` is a case-study figure and it says the opposite of what the section
  was expected to say. The PSI delta is a real number and its sign is the finding.
- The normalisation cannot be quietly reinstated. The rule reads the measurement out of the
  artifact and the pipeline reads the rule's output, so reinstating it means the numbers changed.
- This does not say the transformation is wrong in general. It says these 15 columns on this
  window do not have the problem it solves. A dataset where the observed population is a fixed
  cohort rather than a stream of new cards could look entirely different, and nothing here has
  measured that case.
- The AUC probe is one seed and one hyperparameter setting, exactly as ADR 0009 noted about the
  screen it borrows. Run-to-run variance on it is not measured, and the three columns that
  improved should be read as "did not change" rather than as a ranking.
