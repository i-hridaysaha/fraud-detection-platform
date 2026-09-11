# 0029: Two band edges derived from an assumed cost matrix

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 6

## Context

A single threshold turns a score into approve or decline. The stage brief asked for two: an
auto-block edge above which the transaction is declined without a human, an approve edge below
which it passes, and a review band between them that goes to an analyst. Two edges need a reason
to be where they are, and "tuned for F1" is not a reason for a band.

The dataset carries no chargeback amount, no margin, no analyst cost and no review capacity, so
none of the inputs to that reason can be measured here. They can be stated. The costs are inputs
this stage chose; the derivation from them is arithmetic; and the rule is that the two are kept
apart in the code, in the artifact and in this record.

## Options considered

**One threshold, tuned for F1 on validation.** Kept as the tuned vertex in
`reports/operating_points.json` because model comparison needs a single operating point, and
rejected as the decisioning policy because it has no review band and F1 weights a missed fraud
and a false alarm equally, which no fraud operation does.

**Two thresholds set by alert volume.** Block the top so many per day, review the next so many.
Rejected because the volumes would be assumptions with no derivation behind them, and because a
volume policy moves with traffic while a cost policy moves with risk.

**Two thresholds derived from a cost matrix.** Chosen. Three costs, stated as assumptions in
`fraud_platform.config` and labelled there and in the artifact; two edges derived from them by
`fraud_platform.evaluation.cost_bands`; the volumes and the realised cost reported afterwards.

## Decision

The assumed costs, in units of one false decline:

| Outcome | Cost | Status |
| --- | --- | --- |
| a fraudulent transaction approved | 10 | assumption |
| a legitimate transaction blocked | 1 | assumption |
| one analyst review, whatever the outcome | 0.5 | assumption |

With a fourth assumption that an analyst review resolves the transaction correctly, the expected
cost of each action at a calibrated fraud probability p is `p * 10` to approve, `0.5` to review,
`(1 - p) * 1` to block. Review beats approve once `p > 0.5 / 10 = 0.05`; block beats review once
`(1 - p) * 1 < 0.5`, so `p > 0.5`. The band edges are 0.05 and 0.5 on the calibrated probability,
and the single threshold the same matrix would set with no review band is
`1 / (1 + 10) = 0.0909`. The middle band exists because `0.05 < 0.5`, which needs the review to be
cheaper than `10 * 1 / 11`; a review costing more than that makes the band empty and the code says
so rather than producing two edges in the wrong order.

The probability is the shipped model's score through an isotonic regression fitted on the
validation split. The edges are on that probability; the equivalent raw scores on validation,
0.0396 and 0.4773, are in the artifact for a caller that holds the raw score.

## Evidence

`reports/operating_points.json`, regenerated with `make train` or
`.venv/bin/python scripts/train.py --sections thresholds`, block `bands`. The costs are
`config.COST_MISSED_FRAUD`, `config.COST_FALSE_DECLINE` and `config.COST_REVIEW`, read by the
driver and asserted equal to the artifact by `tests/test_training_artifacts.py`. The derivation
is `evaluation.cost_bands`, checked against a brute-force minimum over the three actions at 201
probabilities by `tests/test_evaluation_units.py`.

What the edges produce on test, 88,581 rows over 30.78 days:

| Band | Rows | Share | Per day | Fraud in band | Fraud rate | Share of all fraud |
| --- | --- | --- | --- | --- | --- | --- |
| approve, below 0.05 | 76,949 | 0.8687 | 2,500.1 | 780 | 0.0101 | 0.2530 |
| review, 0.05 to 0.5 | 10,188 | 0.1150 | 331.0 | 1,181 | 0.1159 | 0.3831 |
| block, 0.5 and above | 1,444 | 0.0163 | 46.9 | 1,122 | 0.7770 | 0.3639 |

Realised cost under the matrix, test: 780 missed at 10, 10,188 reviews at 0.5, 322 false declines
at 1, total 13,216, or 0.1492 per row. Approving everything: 0.3480 per row. The single threshold
at 0.0909 with no review band: 0.1729 per row, catching 2,091 with 5,398 false declines and 992
missed. On validation the same edges realise 0.1181 per row against 0.3434 and 0.1385.

The calibration the edges rest on is measured in the same artifact: on test the raw score has a
Brier of 0.02186 and an expected calibration error of 0.00246, the calibrated score 0.02200 and
0.00508. The isotonic step changed nothing material for this model; it stays because the edges
are defined on a probability and a reweighted or non-probabilistic model would need it.

## Consequences

- The costs are inputs. Change one in `config.py` and the edges, the volumes and the realised cost
  move with it, and the artifact records which costs produced which edges. Nothing in this repo
  claims the numbers 10, 1 and 0.5 are right; it claims the edges follow from them.
- The matrix has no capacity term. The review band it produces is 331 rows per day on test and
  261 on validation, and whether that is a queue an operation can work is not a question this
  repo can answer. A capacity constraint would be a fourth input and would move the edges up; that
  is the first thing a real deployment would add.
- The review assumption is strong: an analyst who catches everything in the band. A review
  accuracy below one raises the effective cost of review and moves both edges; the code takes the
  costs as given and the artifact's `assumption_about_review` names the assumption.
- The single tuned threshold in the same artifact is for comparing models, not for deciding
  transactions. The two are kept in one file so that nobody quotes the F1 vertex as a policy.
- Segment edges are the same mechanic on a different partition key, ADR 0030.
