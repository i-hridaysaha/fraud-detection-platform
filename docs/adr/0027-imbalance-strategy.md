# 0027: No reweighting ships; class weighting and SMOTE were measured on the same split and lost

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 6

## Context

The positive rate is 0.03517 on train (ADR 0010). The conventional response is to do something
about it before fitting: weight the positive class up, or synthesise positive rows until the
classes balance. Both are in this repo as configurable strategies, `fraud_platform.modelling.
IMBALANCE_STRATEGIES`, beside the third option of doing nothing, and the stage brief asked for all
three to be measured on one split rather than for one to be asserted.

The measurement is nine fits: three families, three strategies, identical train split, identical
default feature stack, identical hyperparameters per family, seed 42, each scored on the
validation and test splits. SMOTE is applied strictly after the split: the neighbours are found
among the train positives in the complete-data standardised numeric space, the synthetic rows are
appended to the train matrix only, and no other split is touched. The implementation is this
repo's own (`modelling.smote_plan`, `modelling.materialise_smote`): each synthetic row is a
uniformly random point on the segment between a positive row and one of its five nearest positive
neighbours, numeric columns interpolated, level codes and indicators copied from the base row, and
enough rows are drawn to balance the classes 1:1, which is the balance class weighting reaches by
another route. The two strategies are therefore compared at the same effective balance.

## Options considered

**Class weighting.** `class_weight="balanced"` for the sklearn families, `scale_pos_weight` at the
same negatives-per-positive ratio (27.43 on train, 398,840 to 14,538) for xgboost. Rejected on the
measurement: it lowers validation PR-AUC for all three families.

**SMOTE after the split.** Rejected on the same measurement: it lowers validation PR-AUC for all
three families, by more than weighting does for the two tree families, and it costs between
two and six times the fit time.

**Nothing.** Chosen. The fit rows as they are. Every family reaches its highest validation PR-AUC
this way, and the shipped family's margin over its own reweighted fits is the largest of the three.

## Decision

The shipped configuration uses no reweighting and no resampling. The strategy stays configurable
and the two rejected options stay implemented and tested, because the measurement is about this
data, these features and these families, and a later stage that changes any of the three has the
switch and the harness to measure again.

## Evidence

`reports/model_comparison.json`, regenerated with `make train` (or
`.venv/bin/python scripts/train.py --sections comparison`), blocks `models`, `selection.by_family`
and `imbalance`.

| Family | Strategy | Val PR-AUC | Test PR-AUC | Val ROC-AUC | Fit seconds | Tuned threshold |
| --- | --- | --- | --- | --- | --- | --- |
| logistic regression | none | 0.4094 | 0.3655 | 0.8484 | 11.8 | 0.2340 |
| logistic regression | class weight | 0.3862 | 0.3414 | 0.8520 | 32.3 | 0.8409 |
| logistic regression | SMOTE | 0.3892 | 0.3480 | 0.8492 | 71.8 | 0.8250 |
| random forest | none | 0.5530 | 0.4744 | 0.9120 | 51.4 | 0.2412 |
| random forest | class weight | 0.5466 | 0.5123 | 0.9198 | 26.3 | 0.4490 |
| random forest | SMOTE | 0.5134 | 0.4804 | 0.8968 | 107.6 | 0.4617 |
| xgboost | none | 0.6198 | 0.5490 | 0.9308 | 16.8 | 0.2340 |
| xgboost | class weight | 0.5745 | 0.5221 | 0.9225 | 18.6 | 0.7166 |
| xgboost | SMOTE | 0.5373 | 0.4998 | 0.8923 | 35.3 | 0.5584 |

**On validation, doing nothing wins in every family.** Logistic regression loses 0.0232 to
weighting and 0.0202 to SMOTE; the random forest loses 0.0064 and 0.0396; xgboost loses 0.0453 and
0.0825. SMOTE adds 384,302 synthetic rows to the 413,378 real ones (`n_fit_rows` 797,680).

**On test, one pair reverses.** The random forest with class weighting scores 0.5123 against
0.4744 without, the one place in the table where a reweighted fit beats its plain fit. It is a
test number and the selection does not read test; the paired bootstrap in
`reports/metric_variance.json` reports the interval on that difference, and the random forest is
not the shipped family in any case. It is recorded here because the pattern is the one ADR 0010
warned about: a single split, a single seed, and a difference that would be read as a result if
the other split were not beside it.

**The rejected strategies move the threshold, not the ranking they were meant to improve.**
Weighting pushes the validation-tuned threshold from 0.2340 to 0.7166 on xgboost and from 0.2340
to 0.8409 on logistic regression. That is what reweighting does to a probability estimate: it
shifts the scores up, and a threshold tuned on validation shifts with them. A ranking metric does
not benefit from the shift, and the tables show it did not. The calibration this stage applies to
the shipped model (`reports/operating_points.json`, block `calibration`) is the other route to a
probability that means something, and it does not move the ranking either.

**What was not measured.** One seed per fit in this table; the ablations in
`reports/ablations.json` measure three seeds for the shipped family and strategy only. Other
sampling ratios for SMOTE, other neighbour counts, and undersampling of the majority were not
run. The result is about the three named strategies at the balance stated.

## Consequences

- `modelling.DEFAULT_HYPERPARAMETERS` carries no weighting and the driver's shipped configuration
  records `"imbalance": "none"` in every artifact that names the model.
- The shipped model's raw score turned out to be close to a probability already (ADR 0029 has the
  reliability numbers); the isotonic step on validation stays for a model whose score is not, and
  the band edges are on the calibrated output either way.
- Stage 8's retraining harness inherits the switch. A future data mix at a different positive
  rate is a reason to run the three again, not a reason to assume this table holds.
- The SMOTE implementation is in-repo and tested on generated data (`tests/test_modelling_units.py`);
  no external resampling library was added, and none of its behaviour is claimed from memory.
