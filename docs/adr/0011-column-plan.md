# 0011: The column plan and the evidence each drop category requires

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

Stage 2 measured the frame and handed stage 3 four candidate removal lists: 30 effectively
constant columns, 12 columns missing on more than 0.90 of rows, 59 columns flagged by the time
consistency screen, and 132 V columns removable by a correlation reduction. None of those lists
is a decision. Two of them come with an explicit warning attached: stage 2 said C3 is 0.9958 one
value and may still separate the classes on the other 0.42 percent, and ADR 0009 said the 59
flagged columns are candidates because removing a feature has to be measured.

The failure this ADR is written against is the ordinary one. A column gets dropped because it
looked unpromising, the count goes into a case study, and nobody can say afterwards which number
justified it. A frame of 434 columns offers a lot of opportunities to make that mistake quietly.

## Options considered

**Drop on a single ranked score.** Sort every column by information value, keep the top N. Cheap
and defensible-sounding. Rejected on stage 2's own decomposition: information value on this frame
is dominated by the missing bin on the columns where missingness is the signal, so the ranking
would keep addr1 for a reason that belongs to an indicator and would rank DeviceInfo above C4 for
having 1,546 levels rather than for separating anything. One number cannot carry four different
arguments.

**Drop nothing and let the model sort it out.** Also defensible: gradient boosting tolerates
irrelevant columns well, and stage 5 could measure the difference. Rejected because it defers every
question to a stage that will measure them jointly and be unable to attribute the result, and
because 434 columns is a serving surface as well as a modelling one. Stage 8 has to receive and
validate every column that reaches the model.

**Four named categories, each with its own evidence standard.** Chosen. A column is kept unless
one of four measured conditions holds, each condition names the artifact that established it, and
there is no fifth category. The categories are not interchangeable and neither are the numbers
they need.

## Decision

`fraud_platform.cleaning` builds `reports/column_plan.json` with one row per raw column. A column
is dropped only under one of these, applied in this order:

| Rule | Condition | Evidence required |
| --- | --- | --- |
| effectively_constant | one value covers at least 0.99 of non-null rows, and information value is below 0.02 | the top-value share and the information value, both quoted |
| missing_and_not_mnar | missing rate at least 0.95 and the column is not on the label-linked list | the missing rate and the missingness-against-target result |
| time_inconsistent | a single-feature model fitted on the whole training window scores below 0.50 on validation | the train AUC and the validation AUC |
| redundant | the column sits in a correlation cluster at Spearman 0.95 and is not the representative | the cluster, the representative kept, and the minimum correlation to it |

The order is fixed and it matters. Redundancy runs last because it is the only rule whose answer
depends on which columns are still standing: representatives are chosen from the members that
survived the other three, so a cluster can never trade a column that holds up over time for one
that does not. A column matching more than one condition is recorded under the first and carries
the rest in `also_matched`, so a drop is never counted twice.

Two of the four thresholds deserve their own line.

**The rescue at 0.02.** An effectively constant column is kept when its information value reaches
0.02. That number is not new: it is the bottom band boundary in stage 2's own
`bivariate.json.iv_bands`, so the rescue line is a line stage 2 already drew.

**The redundancy rule outside the V block.** Stage 2 listed six correlated pairs involving no V
column. Only pairs whose correlation was measured over the whole split are actionable. That rule
selects C8 and C10, whose 0.9710 is measured on all 413,378 rows, and rejects D4 with D12 at
rho 1.0000, whose correlation rests on 46,966 complete cases while the two columns differ in
missingness on 244,197 rows. The rule is the coverage, not the pair.

Two derived columns are decided separately, under `derived_columns`, because they are not raw
columns and have no row in the plan. `card_start_day` is excluded from the feature set on its PSI
of 1.841 against test, the highest in the frame. `entity_id` is excluded on ADR 0015.

## Evidence

`reports/column_plan.json`, regenerated with `.venv/bin/python scripts/prepare.py --sections
column_plan`. 434 raw columns, 288 kept, 146 dropped.

| Rule | Columns dropped |
| --- | --- |
| effectively_constant | 14 |
| missing_and_not_mnar | 0 |
| time_inconsistent | 0 |
| redundant | 132 |

Three of those four rows are more interesting than the fourth.

**The constant rule drops 14 of the 30 flagged columns and rescues 16.** The rescued ones are
addr2 at 0.9901 one value and an information value of 0.4200, id_04 at 0.9913 and 0.3215, V240
and V241 at 0.3122 each, M1 at 0.1841, V1 at 0.1794, and nine more between 0.0213 and 0.1197.
Every one of those would have gone under a rule that read only the top-value share.

C3 is the case stage 2 named, and the measurement does not support the warning. C3 is 0.9958
zero, and its information value against the label is 0.0000. It separates nothing on the 0.42
percent where it is not zero, and it is dropped. V107, which takes the value 1.0 on every
non-null row, has an information value of 0.0000 for the reason a constant column must.

**The missing rule drops nothing.** Nine columns are missing on at least 0.95 of train rows:
id_07, id_08, id_21, id_22, id_23, id_24, id_25, id_26 and id_27. All nine are on stage 2's
label-linked list. Moving the threshold to 0.90 adds three columns and changes nothing, because
those three are label-linked too. The category only starts removing columns at 0.80, where it
would take 51, and lowering the line until a rule does something is not a way to choose a line.
An empty category is a result: on this frame, a column being nearly empty is not evidence that it
is uninformative, and stage 2 is why we can say that rather than assume it.

**The time consistency rule drops nothing either, and this is the finding that changed the
stage.** ADR 0009 flagged 59 columns that separated the classes in the first 30 days of train and
inverted in the last 30, and asked stage 3 to measure them against a held-out window before
acting. Measured: a single-feature LightGBM fitted on the whole 120 day training window and
scored on validation gives a validation AUC of at least 0.50 for all 59. The lowest is id_32 at
0.5022. The highest is id_38 at 0.6704, which is the column stage 2 called the strongest case for
inversion at 0.6037 early and 0.3634 late.

The two measurements do not contradict each other and the difference is the point. Stage 2 fitted
on one month and scored on a later month, which asks whether a relationship learned from a narrow
early window survives. Stage 3 fits on the whole window, which is the arrangement a real model
uses, and a model fitted on all 120 days has seen the late behaviour as well as the early. The
inversion is real inside the training span and it does not survive being averaged over. The
candidate list was right to be a candidate list.

**The redundancy rule drops 132 columns across 88 correlation groups**, which reproduces the
number stage 2 predicted, plus C10 in favour of C8 outside the V block. A test asserts that no
group whose members include an unflagged column chose a flagged one as its representative.

## Consequences

- The case study can quote 146 dropped columns with a reason and a number per column, and the
  breakdown by category is itself a result: two of the four categories are empty, and the empty
  ones are the two that a pipeline written on intuition would have leaned on hardest.
- Three thresholds are choices and the plan records all three next to the numbers they were
  applied to. A reader who disagrees with the rescue line at 0.02 can read the 16 rescued columns
  and their values and move it.
- The plan is built from `reports/eda/` plus one measurement, so it is regenerable without the
  CSVs except for the validation probe. That probe is the only part of the plan that needs data,
  and it is the only part that turns a stage 2 candidate list into a decision.
- The redundancy rule is not final. It is one of four strategies compared in ADR 0016, and the
  reduction chosen there overrides it for the V block.
- Nothing is dropped for drift. PSI is a monitoring signal, not a removal criterion, and stage 2
  measured that the drift list and the inversion list do not intersect. ADR 0017 makes the same
  argument about the schema.
