# 0010: What the measured class balance allows and forbids

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 2

## Context

The primary metric, the operating point and the promotion margin have been open since stage 0.
None of them can be fixed here: the metric needs a review-capacity assumption that has not been
made, and the margin needs run-to-run variance that has not been measured. What can be fixed here
is the arithmetic that constrains all three, because the positive counts are now measured.

Train holds 14,538 fraud rows in 413,378, a rate of 0.03517. Val holds 3,042 in 88,581, rate
0.03434. Test holds 3,083 in 88,581, rate 0.03480. Those three numbers, and not any modelling
choice, decide how small a difference a later comparison can resolve.

## Options considered

**Leave the metric question to stage 5 and write nothing now.** Rejected. Stage 6 will compare
models on val and stage 8 will report a test number, and if nobody has written down what those
splits can resolve, the first comparison that comes back at a difference of 0.004 AUC will be
read as a result.

**Fix the primary metric now.** Rejected, and deliberately. The choice between average precision,
recall at a fixed alert budget and a cost-weighted objective depends on how many alerts a review
team can absorb per day, which is an assumption this repo has not made. Making it here to fill in
a section would be inventing an input.

**Record what the counts allow, and leave the metric open with the constraint attached.** Chosen.

## Decision

Three statements, all derived from the measured counts, stand as constraints on every later
comparison.

**Accuracy is not a metric here, and neither is a raw AUC difference below 0.015 on one split.**
The standard error of an AUC estimate at these class counts, by the Hanley and McNeil formula, is
0.00376 on val and 0.00373 on test at an assumed true AUC of 0.90, falling to 0.00276 and 0.00274
at an assumed 0.95. Two independent evaluations at this size, at alpha 0.05 and power 0.80, can
resolve a difference of 0.0149 on val or 0.0148 on test at an assumed 0.90, and 0.0110 or 0.0109
at an assumed 0.95. Those are upper bounds: two models scored on the same rows give correlated
estimates and the paired standard error is smaller by a factor that depends on a correlation
nobody here has measured. The paired number is a stage 6 measurement, not a stage 2 one.

**A recall figure read off val or test carries about plus or minus 0.017.** At 3,042 val
positives the binomial standard error of a recall of 0.50 is 0.00907, giving a 95 percent half
width of 0.0178; at a recall of 0.70 it is 0.00831 and 0.0163. Test at 3,083 positives is the
same to three decimals. A model that recalls 0.68 and a model that recalls 0.70 have not been
distinguished by these splits.

**The primary metric carries a bootstrap interval, always.** Given the above, a point estimate
reported without one is not a result. The repo already has the machinery:
`eda.bootstrap_rate_interval`
does it for a rate, seeded from config.SEED, and target.json carries a Wilson interval and a
20,000 draw bootstrap interval side by side for every split, agreeing to within 2e-4, which
tests/test_eda_artifacts.py asserts.

What stays open: the primary metric itself, the operating point, the review capacity assumption
behind it, and the promotion margin. The margin cannot be set before run-to-run variance is
measured, and this ADR is the reason it must be set at least as wide as the numbers above.

## Evidence

reports/eda/target.json, block `class_balance`, regenerated with `.venv/bin/python scripts/eda.py
--sections target`. Positive and negative counts per split are measured. The standard errors and
detectable differences are derived from them by `eda.hanley_mcneil_se` and
`eda.detectable_auc_difference`, and the AUC values they are evaluated at (0.90, 0.93, 0.95) and
the recall values (0.50, 0.70) are assumed inputs, chosen to span the range this work is aimed at
and labelled as choices in the artifact.

The Hanley and McNeil formula carries its own assumption: Q1 and Q2 are exponential
approximations to the probabilities that two positives, or two negatives, both rank on the wrong
side of a case from the other class. That is an assumption about the shape of the score
distribution, not a measurement of one, and it is stated in the function's docstring rather than
hidden in it.

The class balance itself is measured three ways that agree: reports/audit.json from stage 0,
reports/split_summary.json from stage 1, and target.json here, which checks itself against the
split summary field by field and records the result.

## Consequences

- Stage 6 cannot promote a model on a val AUC improvement smaller than the interval around it.
  The comparison has to be paired, the pairing has to be measured, and the margin has to be set
  from that measurement.
- Stage 8's test number is a single draw with about 3,083 positives behind it. It gets an
  interval and it gets reported once.
- Any sliced metric is smaller still. A fraud rate quoted for one ProductCD level, one card type
  or one hour of the day rests on the count in that cell, and every rate table in reports/eda/
  carries a Wilson interval per row for exactly that reason. The smallest levels measured here
  are card6 "charge card" at 15 rows and "debit or credit" at 29, and no rate computed on those
  means anything.
- Cross-validation would buy more positives per evaluation, and ADR 0002 already ruled it out as
  the primary protocol for a chronological problem. This ADR is the cost of that decision, stated
  in the units it is paid in.
- Nothing here fixes the metric. A later ADR does that, and it inherits these constraints.
