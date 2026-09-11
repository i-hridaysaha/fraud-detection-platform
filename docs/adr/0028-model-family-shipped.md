# 0028: XGBoost ships, on the stage 3 frame with the C, D and V blocks and nothing from stages 4 or 5

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 6

## Context

Three families were fitted on the identical chronological split, each through its stage 3
preparation path, with fixed hyperparameters chosen before any number was read and the imbalance
strategy ADR 0027 settled. The stage brief said gradient boosting was expected to win and said not
to assume it. This record is what was measured: which family, on which feature stack, with which
hyperparameters, and whether the lead over the alternatives is larger than the resampling noise.

Two of the three decisions here are made by rules written into `scripts/train.py` before the
fits ran; the third, the rule for retiring a provided native block, was rewritten after the
numbers were read, and both versions and both outcomes are in the artifact.

## Options considered

**The family.** Logistic regression on the complete-data path, the random forest and xgboost on
the tree path. Selection rule: the largest validation PR-AUC across the nine family-by-strategy
fits; test is not consulted. XGBoost without reweighting wins at 0.6198 on validation, the random
forest without reweighting is next at 0.5530, the logistic regression at 0.4094. Rejecting the
random forest: 0.0668 behind on validation and 0.0746 behind on test with a paired interval of
0.0670 to 0.0824 that excludes zero on every resample. Rejecting the logistic regression: 0.2104
behind on validation, 0.1835 on test, 0.1697 to 0.1995.

**The feature stack.** Twelve stacks, xgboost, three seeds each, each stack differing from a
named reference by exactly one block. Two rules:

- A block this repo builds (stage 4's causal features, stage 5's graph candidates) ships when
  adding it raises validation PR-AUC on every seed with the paired interval excluding zero on every
  seed. This is ADR 0022's rule applied to a model. Neither block passes: the causal block moves
  validation by -0.0052, +0.0029 and -0.0076 over the three seeds; the graph block by -0.0049,
  -0.0015 and -0.0079 over the causal stack.
- A block the frame already carries (the native C, D, M and V groups) is retired only when
  removing it lowers validation PR-AUC on no seed with the interval excluding zero. Removing C
  costs 0.0590, 0.0663 and 0.0658 across the seeds; removing D costs 0.0107 on one seed outside
  its interval; removing V costs 0.0167 on one seed outside its interval; removing M moves the
  score by +0.0020, +0.0026 and -0.0027 with every interval straddling zero. C, D and V ship, M
  does not.

The rejected alternative for the second rule was its first draft, the mirror of the first: retire
a native block unless every seed shows a loss outside its interval. It kept only C and would have
shipped a 132 column stack no ablation had measured, composing three single-block removals as if
their costs added. Rejected because a rule that decides on a stack it never measured is not a
measurement, and because the two errors are not symmetric: a wrong drop costs detection and a
wrong keep costs width. The artifact records the first draft's outcome
(`decision.superseded_first_draft`) and the note for the stage records the change.

**The hyperparameters.** A random search of twenty configurations plus the default, under a
three-fold expanding-window cross-validation on the train split that never fits on a row later
than its validation rows. Selection rule for the search: the largest mean fold PR-AUC. Selection
rule for shipping: the search's best ships only if its validation PR-AUC on the full train fit is
at least the default's. The best configuration beat the default by 0.0031 across the folds and
lost to it on the full fit, 0.6048 against 0.6184. The default ships.

## Decision

The shipped model is xgboost with `n_estimators=600`, `learning_rate=0.05`, `max_depth=8`,
`min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.6`, `reg_lambda=1.0`,
`tree_method="hist"`, seed 42, no reweighting, on the 186 column stack
`prepared|native=count,timedelta,vesta`: the stage 3 prepared frame with the C, D and V blocks,
without the M block, and with no stage 4 or stage 5 feature. `models/shipped_model.json` names all
of it; the booster is rebuilt by `make train`.

## Evidence

`reports/model_comparison.json`, `reports/ablations.json`, `reports/hyperparameter_search.json`
and `reports/metric_variance.json`, all regenerated with `make train`.

**The headline, with its interval.** Test PR-AUC 0.5451, 95 percent interval 0.5274 to 0.5614
over 1,000 row resamples and 0.5100 to 0.5805 over 1,000 card resamples; ROC-AUC 0.9050. At card
level, 0.6442, 0.6200 to 0.6684. Validation PR-AUC 0.6184.

**The lead survives resampling.** Shipped minus each alternative on test, paired on identical
resamples, row resample, with the interval and the share of resamples on which the sign agrees:

| Against | Difference | 95 percent interval | Sign agreement |
| --- | --- | --- | --- |
| random forest, none | +0.0707 | +0.0626 to +0.0796 | 1.000 |
| random forest, class weight | +0.0328 | +0.0241 to +0.0418 | 1.000 |
| random forest, SMOTE | +0.0647 | +0.0543 to +0.0750 | 1.000 |
| logistic regression, none | +0.1796 | +0.1658 to +0.1949 | 1.000 |
| logistic regression, class weight | +0.2037 | +0.1883 to +0.2211 | 1.000 |
| logistic regression, SMOTE | +0.1972 | +0.1822 to +0.2143 | 1.000 |
| xgboost, class weight | +0.0230 | +0.0159 to +0.0298 | 1.000 |
| xgboost, SMOTE | +0.0453 | +0.0353 to +0.0541 | 1.000 |
| xgboost, none, default stack | -0.0039 | -0.0084 to +0.0008 | 0.951 |

Every lead over another family or strategy excludes zero. On the card resample the closest of
them, the random forest with class weighting, still excludes zero (the random forest without
reweighting against xgboost is 0.0219 to 0.1207). The one difference that includes zero is between
the shipped model and the 237 column default-stack xgboost it was cut from, on both resamples,
which is the measurement that the 51 removed columns cost nothing the test split can see.

**The noise band.** The widest paired half-width on transaction-level PR-AUC is 0.0172 over the
row resample and 0.0592 over the card resample; the artifact's `noise_band` is the larger. The
band is what a promotion margin has to clear, and the artifact carries every pair, because the
pair stage 9 compares (a candidate against the incumbent, same family, same stack) is paired far
more tightly: 0.0046 by row and 0.0121 by card for the shipped model against its default-stack
parent.

**What the ablation says about the earlier stages.** Stage 4's 36 causal features and stage 5's 4
graph candidates do not raise validation PR-AUC over the stage 3 frame for this family. Stage 3's
frame carries the C block, which is worth 0.0637 of validation PR-AUC on its own and takes 0.2672
of the shipped model's summed mean absolute SHAP on 12 columns. The velocity block ADR 0021 built
to compare against it recovers 0.0062 of the C block's 0.0751 when C is absent and nothing when it
is present, and carries 6.35 of gain per column against 103.67 for C in the seed-42 fit.

## Consequences

- The serving path in stage 7 computes stage 3's pipeline and reads 186 columns. It computes no
  velocity, recency, amount, device, geography, duplicate or graph feature, and the online store
  those were designed for holds nothing this model reads. That is a smaller stage 7 than the
  design document planned and the reason is measured.
- Stage 8's problem is sharper, not gone: the most valuable block in the shipped model is one
  whose construction is not established (ADR 0021), and the block this repo built to stand in for
  it does not.
- The stage 4 and 5 code stays, tested, switchable, and measured to add nothing here. A different
  family, a different frame, or a future without the C columns is a reason to run the ablation
  again, and the harness runs it in one command.
- The hyperparameter search is recorded as having found nothing, with the fold scores beside the
  means. A future search needs a resolution the three folds did not have.
- Stage 9's promotion margin is derived from `reports/metric_variance.json`, from the pair it
  actually compares, and sits above that pair's half-width. This ADR fixes the band; the margin is
  stage 9's decision.
