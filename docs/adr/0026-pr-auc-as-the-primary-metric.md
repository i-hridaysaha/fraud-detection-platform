# 0026: PR-AUC is the primary metric, and ROC-AUC is reported beside it

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 6

## Context

ADR 0010 left the primary metric open and attached the constraint that would bind it: at the
measured positive counts (14,538 of 413,378 on train, 3,042 of 88,581 on validation, 3,083 of
88,581 on test; rates 0.03517, 0.03434 and 0.03480) no comparison resolves a small difference, and
any metric chosen has to carry a bootstrap interval. This stage fits nine models on the same split
and is the first point at which the metric question can be answered with numbers from this data
rather than with the general argument that ROC-AUC flatters an imbalanced problem.

The general argument is true and it is not evidence. What is evidence is what the two metrics do
on the nine models in `reports/model_comparison.json`, scored on identical rows.

## Options considered

**ROC-AUC as the primary metric.** The conventional headline, threshold-free, and the number most
published work on this dataset reports. Rejected on the measurement below: across the nine models
it spans 0.0825 while PR-AUC spans 0.2336, it orders five of the thirty-six model pairs the other
way round from PR-AUC, and the whole operating region every tuned threshold lands in occupies the
first 0.0081 to 0.0332 of its false-positive axis.

**Recall at a fixed alert budget.** The number an operations team would actually ask for.
Rejected as the primary metric because the budget is an assumption this repo has not been given;
`evaluation.alert_budget_vertex` computes it for any budget and `reports/threshold_curve.json`
reports alerts per day at every threshold, so the number is available without being the headline.

**A cost-weighted objective.** The cost matrix in `config.py` is an assumption (ADR 0029), and a
primary metric that moves when an assumed input moves is a primary metric nobody can compare
across stages. The realised cost under the matrix is reported per band in
`reports/operating_points.json`; it is not the selection criterion.

**PR-AUC, computed as average precision, with ROC-AUC reported beside it.** Chosen. It reads the
ranking on the side of the curve where the alert queue lives, its differences between models are
of the size the bootstrap can resolve, and it needs no assumed input.

## Decision

The primary metric is PR-AUC as `sklearn.metrics.average_precision_score` computes it: the
precision at each recall step, weighted by the recall gained at that step, with no interpolation.
Every headline claim, the family selection in `reports/model_comparison.json`, every ablation
verdict in `reports/ablations.json`, the hyperparameter selection in
`reports/hyperparameter_search.json` and the promotion margin stage 9 derives from
`reports/metric_variance.json` key off it. ROC-AUC is computed on the same scores and reported in
every one of those artifacts, and nothing is decided on it.

## Evidence

`reports/model_comparison.json`, regenerated with `make train` (or
`.venv/bin/python scripts/train.py --sections comparison`), block `metric_agreement` and the
per-model `val`, `test` and `tuned_threshold` blocks.

**The two metrics disagree on which of two models is better, on this data.** On the validation
split, over the thirty-six pairs of the nine models, five pairs are ordered one way by ROC-AUC and
the other by PR-AUC:

| First | Second | Val ROC-AUC | Val PR-AUC |
| --- | --- | --- | --- |
| random forest, no reweighting | random forest, class weight | 0.9120 against 0.9198 | 0.5530 against 0.5466 |
| logistic regression, no reweighting | logistic regression, class weight | 0.8484 against 0.8520 | 0.4094 against 0.3862 |
| logistic regression, no reweighting | logistic regression, SMOTE | 0.8484 against 0.8492 | 0.4094 against 0.3892 |
| random forest, SMOTE | xgboost, SMOTE | 0.8968 against 0.8923 | 0.5134 against 0.5373 |
| logistic regression, SMOTE | logistic regression, class weight | 0.8492 against 0.8520 | 0.3892 against 0.3862 |

The first row is the one that matters for ADR 0027: ROC-AUC says class weighting helps the
random forest and PR-AUC says it does not. The two metrics are computed from the same score
vectors, so the choice of metric is the decision on that pair.

**The spread the metrics can resolve differs by a factor of about three.** On validation, ROC-AUC
runs from 0.8484 to 0.9308 across the nine models, a spread of 0.0825; PR-AUC runs from 0.3862 to
0.6198, a spread of 0.2336. ADR 0010 computed that two independent ROC-AUC evaluations at these
counts resolve about 0.015; the paired bootstrap in `reports/metric_variance.json` gives the
PR-AUC resolution directly.

**The region of the ROC curve any operating point sits in is a sliver.** At each model's tuned
threshold on test, the false-positive rate is `fp / (fp + tn)`:

| Model | fp | tn | False-positive rate | Precision | False alarms per catch |
| --- | --- | --- | --- | --- | --- |
| xgboost, class weight | 694 | 84,804 | 0.0081 | 0.6516 | 0.535 |
| xgboost, SMOTE | 811 | 84,687 | 0.0095 | 0.6187 | 0.616 |
| xgboost, no reweighting | 1,084 | 84,414 | 0.0127 | 0.5828 | 0.716 |
| random forest, SMOTE | 1,252 | 84,246 | 0.0146 | 0.5288 | 0.891 |
| logistic regression, no reweighting | 1,352 | 84,146 | 0.0158 | 0.4395 | 1.275 |
| random forest, class weight | 1,542 | 83,956 | 0.0180 | 0.4964 | 1.014 |
| logistic regression, class weight | 2,437 | 83,061 | 0.0285 | 0.3369 | 1.968 |
| logistic regression, SMOTE | 2,732 | 82,766 | 0.0320 | 0.3207 | 2.118 |
| random forest, no reweighting | 2,837 | 82,661 | 0.0332 | 0.3459 | 1.891 |

Every tuned operating point sits below a false-positive rate of 0.034. The 82,661 to 84,804 true
negatives are what ROC-AUC integrates over, and they are the same rows for every model; the
difference between a queue that is 0.65 fraud and one that is 0.32 fraud is a difference in
1,084 against 2,732 false alarms, which the x axis of a ROC curve draws as 0.0127 against 0.0320.
PR-AUC draws it as 0.5490 against 0.3480.

**What the base rate does to precision is arithmetic, not a model property.** At a positive rate
of 0.0348 and a recall of 0.4911 (the xgboost tuned point, 1,514 of 3,083), a false-positive rate
of 0.0127 puts 1,084 legitimate rows in the queue beside the 1,514 fraud rows. The same recall and
false-positive rate on the same 88,581 rows at a positive rate of 0.5 would put about 560
legitimate rows beside about 21,750 fraud rows (44,290 positives times 0.4911, 44,291 negatives
times 0.0127). ROC-AUC is the same number in both worlds; the queue is not.

## Consequences

- Every artifact this stage writes carries both metrics; every decision reads one.
- The bootstrap interval ADR 0010 required is on PR-AUC, paired across models, and its widest
  paired half-width is the noise band stage 9's promotion margin must clear (ADR 0028 records it).
- A reader comparing this repo to published ROC-AUC figures on the same dataset is comparing a
  different quantity, and the ROC-AUC beside every PR-AUC is there so that comparison can still
  be made without changing what is decided on.
- The metric is threshold-free and the operating point is not. `reports/operating_points.json`
  is the only source of any precision, recall, alert-volume or false-alarm number the repo quotes
  at a threshold.
- Average precision is one of several PR-AUC definitions. Trapezoidal integration of the same
  curve gives a different number, and any comparison against a figure from elsewhere has to
  check which was computed.
