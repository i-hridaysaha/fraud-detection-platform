# 0017: The schema contract, and what it deliberately does not catch

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

Stage 8 puts a serving path in front of this pipeline. The failure a serving path has is not a
crash: it is a row arriving with a column renamed, a counter arriving as a string, or a negative
amount, and the model scoring it anyway and returning a number nobody can tell is wrong. A batch
pipeline has the same failure at a different cadence.

Stage 3 produces a 247 column frame from a 434 column input through a column plan, four encoders,
a set of derived columns and a block reduction. That is a lot of surface for a shape to change
across, and "it worked when I ran it" is not a contract.

## Options considered

**No schema; rely on the pipeline being deterministic.** It is deterministic, and that is a
different property. Determinism says the same input gives the same output. It says nothing about
what happens when the input is not the same.

**Infer the schema from the train frame including value ranges, and reject anything outside
them.** The tempting version, and the one this ADR exists to argue against. Stage 2's train-only
section measured what it would cost: 130 of the 393 numeric columns move a median or a 99th
percentile by more than 10 percent when the later splits are folded in, and V332 goes from 1,650
on train to 60,335 over all rows. A cap fitted on the training window rejects legitimate later
rows for the offence of being later, and it does it at exactly the moment stage 9's drift work
wants to see them.

**Declare structure, not distribution.** Chosen. The schema constrains the set of columns, the
dtype family of each, whether a null is allowed, and the sign and finiteness bounds stage 2
measured. Everything distributional is left to monitoring.

## Decision

`fraud_platform.schema` declares a `Schema` of `ColumnSpec` entries and `validate` raises
`SchemaError` carrying every violation, not just the first. Raising on the first would turn a
broken producer into a queue of one-line fixes.

What is constrained:

- **The column set.** Exact. A missing declared column and an unexpected extra column are both
  violations, and `allow_extra_columns` is false.
- **The dtype family**, not the exact width. Stage 1's dtype map narrows an integer column to the
  smallest type its measured range fits, so a later frame with a wider range legitimately arrives
  as int32 where train was int16. Four families: integer, float, categorical, boolean.
- **Nullability**, on exactly four columns: TransactionID, isFraud, TransactionDT and
  TransactionAmt. Everything else is declared nullable. That line is drawn on a measurement and on
  a distinction. Stage 2 measured all four null-free on train, and everything downstream indexes,
  splits or scores on them, so a null in one is a broken row rather than a sparse one. Other
  columns that happened to be complete on train are still nullable, because completeness on one
  window is a fact about that window and not a property of the column.
- **Lower bounds**, from stage 2's `quality.json.impossible_values`. That section went looking for
  values the data should not be able to take and found none. A numeric column not on its list of
  17 columns holding negative values gets a lower bound of zero: stage 2 looked for a negative
  there and found none, so a negative arriving later is a change in the data rather than a wide
  tail. A column on the list gets no lower bound. TransactionAmt gets a strict positive bound,
  since stage 2 measured 0 negative amounts, 0 zero amounts and a minimum of 0.251.
- **Uniqueness**, on TransactionID alone.

Derived columns are a separate case and getting this wrong was a real bug the validator caught.
"No negative was observed" is a statement about a measurement, and there is no measurement of a
column that did not exist when the measuring was done. So a derived column's bounds come from the
arithmetic that produces it: a frequency encoding is a count and cannot be negative, a target
encoding is a smoothed mean of a zero-or-one label and lies in the unit interval, an indicator is
an indicator, a cents fraction is 0 to 99. A derivation whose range is genuinely unbounded, such
as a standardised block mean or a principal component score, is declared with no bounds rather
than with a guess. `schema.DERIVED_BOUNDS` holds the table with the reason per entry.

## What it deliberately does not catch

Recorded in `schema.NOT_CAUGHT` and in the artifact, so it is a stated limit rather than a gap
somebody discovers later.

- A value that is the right type, in range, and wrong. The schema knows nothing about what a
  correct amount is for a given card.
- **Distribution drift.** No upper bounds, no distributional limits, for the reason in the options
  above. Drift is a monitoring question and stage 9 owns it; putting it in a validator turns every
  distribution shift into an outage.
- **A category level the training window never saw.** Stage 2 measured 240 such levels on
  DeviceInfo and 22 on id_31, the id_31 ones covering 0.0456 of val and test. Rejecting an unseen
  level would reject those rows. The encoders have an explicit `(unseen)` token and that is where
  the case is handled.
- A row that is internally inconsistent across columns, such as a D column implying an origin
  later than its own timestamp. Nothing here compares two columns to each other.
- A label that is wrong. Stage 0 did not establish the host's labelling rule and a validator
  cannot check what has not been established.
- A duplicated row. Stage 2 measured 10 near-duplicate rows in 413,378 and found the core-field
  repeat grouping to be signal rather than dirt, so duplication is not an error condition here.
- Column order. It is not part of the contract.

## Evidence

`reports/prep/schema.json`, regenerated with `.venv/bin/python scripts/prepare.py --sections
schema`.

The prepared train frame is 247 columns: 92 raw columns that survived the plan and the V
reduction, and 155 derived. By family: 89 integer, 93 float, 65 categorical. 151 columns carry a
lower bound, 89 carry an upper bound, and 31 numeric columns carry neither, all of them either on
stage 2's negative-value list or derivations with no bounded range.

All three splits validate with zero violations. The pipeline is applied to the train split twice
from the same fitted objects and the two frames agree column by column and value by value, with a
SHA-256 fingerprint recorded in the artifact; a `needs_data` test refits the whole pipeline from
the CSVs and reproduces that fingerprint.

The three rejections the contract exists for are each proved in `tests/test_prep_units.py` by
constructing the bad frame: a column recast to a string dtype, an amount set to -1.0, and an
extra column appended. A missing declared column, a null in a non-nullable column and a frame with
two faults at once are tested the same way, the last one to assert that both violations come back
rather than the first.

The bug the validator caught during this stage is the best evidence it works. Every derived column
initially inherited the "no negative observed on train" rule, and all 14 block means failed
validation on all three splits, each with a large count of values below a declared minimum of
zero. A standardised mean is signed by construction. The rule was wrong, the validator said so on
the first run, and the fix was to derive those bounds from the arithmetic instead. The failing
counts are not quoted here because the bug is fixed and nothing regenerates them.

## Consequences

- Stage 8's serving path validates each incoming row against the same declared object the training
  frame was checked against, and a shape change is an exception at the boundary rather than a
  score.
- The schema is regenerated with the pipeline, so a deliberate change to the prepared frame shows
  up as a diff in `reports/prep/schema.json` and has to be committed. An accidental one shows up
  as a validation failure.
- Drift is not this module's problem and stage 9 inherits it with a baseline already measured:
  stage 2's PSI table over all 432 raw features.
- A frame that satisfies this schema can still be nonsense. The contract is a floor, and the list
  above is what is under the floor.
