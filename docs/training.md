# Training and evaluation

Stage 6 fits the models. Three families on the identical chronological split, each through its
stage 3 preparation path; three imbalance strategies measured rather than chosen; twelve feature
stacks that decide which of stages 3, 4 and 5's work ships; a hyperparameter search under a
time-aware cross-validation; 1,000 paired bootstrap resamples of the test split; a threshold sweep
and a set of operating points from one command; two band edges derived from an explicit cost
matrix; per-segment thresholds on `ProductCD`; and SHAP on the shipped model, with the card-level
caveat measured rather than stated.

The primary metric is PR-AUC, computed as average precision. ADR 0026 has the measurement behind
the choice: on this data, ROC-AUC orders five of the thirty-six model pairs the other way round,
and every tuned operating point sits below a false-positive rate of 0.034, which is the sliver of
the ROC axis the alert queue lives in. ROC-AUC is reported beside every PR-AUC and decides nothing.

Everything below is measured on the eight artifacts `make train` writes:
`reports/model_comparison.json`, `reports/ablations.json`, `reports/hyperparameter_search.json`,
`reports/metric_variance.json`, `reports/threshold_curve.json`, `reports/operating_points.json`,
`reports/shap/shap_global.json` and `reports/shap/shap_example.json`. Sections are named after
the artifact they read.

The result of the stage, in one paragraph. XGBoost wins PR-AUC on validation and on test, with no
reweighting, and the lead over the random forest and the logistic regression survives resampling
(section 5). Stage 4's causal block and stage 5's graph block do not raise validation PR-AUC over
the stage 3 frame for that family and do not ship; the native C block is the single most valuable
block in the frame and this repo's own velocity carries about a sixteenth of its gain per column.
The hyperparameter search found a configuration that beat the default across the folds by 0.0031
and lost to it on the full train fit, so the default ships and the search is recorded as having
found nothing. The shipped model scores a test PR-AUC of 0.5451 at transaction level and 0.6442 at
card level, and its 95 percent interval is in section 5.

---

## 0. What was fitted, and how

**The split** is ADR 0002's: `TransactionDT < 10,437,998` is train (413,378 rows, 14,538 fraud),
below 13,151,845 is validation (88,581 rows, 3,042 fraud), the rest is test (88,581 rows, 3,083
fraud). Validation spans 31.41 days and test 30.78, which is what "alerts per day" divides by.
Every model is fitted on train, every threshold and calibrator is fitted on validation, and test
is scored once with everything fixed.

**The two paths** are stage 3's, not new ones. The tree path receives the prepared frame as it is:
nulls left missing (ADR 0012), no clipping (ADR 0013), every categorical as the integer code of its
vocabulary level. The complete-data path receives what a linear model needs: stage 3's imputer (a
train median, or the sentinel below the train minimum for the D columns), stage 3's clip at the
0.999 train quantile on the amount and the C columns, one column per vocabulary level, a train
mean and standard deviation per numeric column, and an indicator beside every stage 4 or 5 feature
that is null on train. The random forest and xgboost take the tree path; the logistic regression
takes the complete-data path. `fraud_platform.modelling.PATH_BY_FAMILY` is the assignment and
`fit_complete_data_path` goes through the train-only guard.

**What a model reads.** The prepared frame's 247 columns minus the label, the identifier, the
clock columns (`TransactionDT`, `day_index`, `week_index`, `dow_position`; `hour_position` stays,
ADR 0013's clock decision), the entity key components, every raw categorical whose `_cat` column
carries it, and the encodings of the two raw free-text columns whose normalised versions replace
them. That last exclusion is stage 3's intent (`fraud_platform.prepare` says the normalisation
replaces rather than supplements) and it was applied here after a measurement: the raw
`DeviceInfo` vocabulary holds 913 levels against 67 for the normalised one, and the complete-data
matrix was 1,584 columns wide with it and is 606 without. The default stack is 237 columns on the
tree path: 201 from stage 3 and 36 from stage 4. Stage 5's four candidate features are off by
default, as `graph_features.DEFAULT_ENABLED` records, and the ablations turn them on.

**Hyperparameters** are fixed per family in `modelling.DEFAULT_HYPERPARAMETERS`, chosen before any
number was read and held across the comparison and the ablations: logistic regression with lbfgs,
`C=1.0`, `max_iter=5000`; the random forest with 200 trees, `min_samples_leaf=5`, square-root
feature sampling; xgboost with 600 trees at depth 8, learning rate 0.05, `min_child_weight=5`,
row subsample 0.8, column subsample 0.6, `tree_method="hist"`. The search in section 4 moves the
shipped family's values and records both sets.

**Seeds.** The comparison runs at the repo seed, 42. The ablations run at 42, 7 and 1,337, the
three seeds stage 3 used for its own paired probe, so the seed set is one set across stages.

**The cache.** `scripts/train.py --sections matrices` rebuilds the prepared frame from the raw
files through `prepare.fit_preparation` on train, reads the four stage 3 decisions from the
artifacts that made them (`prepare.read_decisions`), builds stage 4's features and stage 5's
candidate set over the whole stream, and writes the three splits under `reports/cache/`, which is
gitignored. Every later section reads the cache; nothing reads the CSVs twice.

---

## 1. The comparison: nine fits on one split

`reports/model_comparison.json`, regenerated with `make train` or
`.venv/bin/python scripts/train.py --sections comparison`. Three families by three strategies,
default stack, seed 42. The tuned threshold is the vertex of the validation precision-recall
curve with the largest F1, applied unchanged to test.

| Family | Strategy | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Test precision | Test recall | Test F1 | Alerts per day | False alarms per catch | Fit seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic regression | none | 0.4094 | 0.8484 | 0.3655 | 0.8464 | 0.4395 | 0.3438 | 0.3858 | 78.4 | 1.275 | 11.8 |
| logistic regression | class weight | 0.3862 | 0.8520 | 0.3414 | 0.8506 | 0.3369 | 0.4016 | 0.3664 | 119.4 | 1.968 | 32.3 |
| logistic regression | SMOTE | 0.3892 | 0.8492 | 0.3480 | 0.8476 | 0.3207 | 0.4184 | 0.3631 | 130.7 | 2.118 | 71.8 |
| random forest | none | 0.5530 | 0.9120 | 0.4744 | 0.8907 | 0.3459 | 0.4865 | 0.4043 | 140.9 | 1.891 | 51.4 |
| random forest | class weight | 0.5466 | 0.9198 | 0.5123 | 0.9027 | 0.4964 | 0.4930 | 0.4947 | 99.5 | 1.014 | 26.3 |
| random forest | SMOTE | 0.5134 | 0.8968 | 0.4804 | 0.8778 | 0.5288 | 0.4557 | 0.4895 | 86.3 | 0.891 | 107.6 |
| xgboost | none | 0.6198 | 0.9308 | 0.5490 | 0.9111 | 0.5828 | 0.4911 | 0.5330 | 84.4 | 0.716 | 16.8 |
| xgboost | class weight | 0.5745 | 0.9225 | 0.5221 | 0.8975 | 0.6516 | 0.4210 | 0.5115 | 64.7 | 0.535 | 18.6 |
| xgboost | SMOTE | 0.5373 | 0.8923 | 0.4998 | 0.8748 | 0.6187 | 0.4269 | 0.5052 | 69.1 | 0.616 | 35.3 |

The confusion matrices are in the artifact under `tuned_threshold.test`; the xgboost row without
reweighting is 1,514 true positives, 1,084 false positives, 1,569 false negatives and 84,414 true
negatives on the 88,581 test rows.

**XGBoost wins, and it was expected to.** The stage brief said so and said not to assume it. The
measurement is that xgboost without reweighting has the largest validation PR-AUC of the nine,
0.6198, and the largest test PR-AUC, 0.5490; the next family down is the random forest at 0.5530
on validation, 0.0668 behind. Whether that lead survives resampling is section 5's question, and it
does.

**The logistic regression converged, and the first version of it did not.** The artifact records
`n_iter_` against `max_iter` for every logistic fit: 333, 840 and 971 iterations against a cap of
5,000 for the plain, weighted and SMOTE fits. The cap was 1,000 when the stage started and the
weighted fit hit it (the solver's own warning: `lbfgs failed to converge after 1000 iteration(s)`),
which would have made the baseline a crippled one; the cap was raised before any number was kept.
A baseline that silently hit the iteration cap is not a baseline, and the artifact carries the
check so a reader does not have to take that on trust.

**Card level is higher than transaction level for every model.** Labels in this file are card
level (ADR 0001 measured the entity's label purity), so the artifact also collapses each split to
one row per entity key, with the maximum score and any-fraud label per key, and scores that:
46,884 cards with 1,408 fraud cards on validation, 45,348 with 1,410 on test. XGBoost without
reweighting scores 0.6644 at card level on validation against 0.6198 at transaction level, and
0.6515 against 0.5490 on test. The gap is the number section 7's caveat is about: a share of what
a transaction score explains is which card the transaction is on.

---

## 2. Imbalance: the strategy that wins is none

ADR 0027. On validation, doing nothing beats class weighting and SMOTE in every family: by 0.0232
and 0.0202 for the logistic regression, 0.0064 and 0.0396 for the random forest, 0.0453 and
0.0825 for xgboost. SMOTE, this repo's own implementation applied strictly after the split, adds
384,302 synthetic rows to the 413,378 real ones and costs two to six times the fit time. The one
reversal is on test: the weighted random forest scores 0.5123 against 0.4744 unweighted, a test
number the selection does not read and section 5 puts an interval on.

What the rejected strategies do move is the threshold: weighting pushes the validation-tuned
vertex from 0.2340 to 0.7166 on xgboost. A reweighted score is a shifted score, and a ranking
metric does not benefit from a shift. The calibration in section 6 is the route to a probability
that means something, and it moves no ranking either.

---

## 3. Ablations: what each stage's work is worth to the shipped family

`reports/ablations.json`, regenerated with `make train` or
`.venv/bin/python scripts/train.py --sections ablations`. XGBoost without reweighting, default
hyperparameters, three seeds per stack, twelve stacks. Every stack has a reference stack it
differs from by exactly one block, and the difference against the reference is computed per seed,
at the same seed, with a 200-resample paired bootstrap interval on the validation split.

| Stack | Columns | Val PR-AUC, mean over seeds | Val, seed range | Test PR-AUC, mean | Against | Mean val gain | Val gain per seed, with interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| prepared | 201 | 0.6242 | 0.6181 to 0.6294 | 0.5446 | | | |
| prepared + causal | 237 | 0.6209 | 0.6198 to 0.6218 | 0.5502 | prepared | -0.0033 | -0.0052 [-0.0098, -0.0005]; +0.0029 [-0.0021, +0.0081]; -0.0076 [-0.0121, -0.0027] |
| prepared + causal + graph | 241 | 0.6161 | 0.6139 to 0.6194 | 0.5398 | prepared + causal | -0.0048 | -0.0049 [-0.0099, +0.0001]; -0.0015 [-0.0064, +0.0034]; -0.0079 [-0.0128, -0.0033] |
| prepared, native off | 120 | 0.3962 | 0.3867 to 0.4011 | 0.2593 | prepared | -0.2280 | all three below -0.22, intervals exclude zero |
| prepared + causal, native off | 156 | 0.4357 | 0.4342 to 0.4375 | 0.3037 | prepared + causal | -0.1851 | all three below -0.18, intervals exclude zero |
| prepared + causal + graph, native off | 160 | 0.4279 | 0.4223 to 0.4317 | 0.2901 | prepared + causal + graph | -0.1882 | all three below -0.18, intervals exclude zero |
| full stack without C | 229 | 0.5524 | 0.5481 to 0.5559 | 0.4719 | prepared + causal + graph | -0.0637 | -0.0590 [-0.0692, -0.0500]; -0.0663 [-0.0763, -0.0582]; -0.0658 [-0.0743, -0.0561] |
| full stack without D | 212 | 0.6103 | 0.6088 to 0.6122 | 0.5274 | prepared + causal + graph | -0.0058 | -0.0026 [-0.0082, +0.0018]; -0.0107 [-0.0155, -0.0061]; -0.0040 [-0.0084, +0.0008] |
| full stack without M | 226 | 0.6167 | 0.6112 to 0.6220 | 0.5419 | prepared + causal + graph | +0.0006 | +0.0020 [-0.0027, +0.0058]; +0.0026 [-0.0023, +0.0063]; -0.0027 [-0.0073, +0.0017] |
| full stack without V | 216 | 0.6091 | 0.6027 to 0.6124 | 0.5230 | prepared + causal + graph | -0.0070 | -0.0025 [-0.0075, +0.0026]; -0.0167 [-0.0215, -0.0114]; -0.0018 [-0.0080, +0.0035] |
| prepared without C | 189 | 0.5491 | 0.5487 to 0.5495 | 0.4775 | prepared | -0.0751 | all three below -0.068, intervals exclude zero |
| prepared + causal without C | 225 | 0.5553 | 0.5532 to 0.5586 | 0.4782 | prepared + causal | -0.0655 | all three below -0.063, intervals exclude zero |

"Full stack" is prepared + causal + graph with every native group on. The per-seed intervals
elided as "all three below" are in the artifact.

**Stage 4's causal block does not ship.** The rule, stage 4's own (ADR 0022) applied to a model
instead of a marginal: a built block ships when switching it on raises validation PR-AUC on every
seed with the paired interval excluding zero on every seed. Adding the 36 causal features to the
stage 3 frame moves validation PR-AUC by -0.0052, +0.0029 and -0.0076 across the three seeds, two
of the three intervals below zero. It moves test by +0.0056 on average, which the rule does not
read and which sits inside the noise band of section 5. The block is built, tested, causal and
measured, and for this family on this frame it adds nothing the frame did not already carry.

**Stage 5's graph block does not ship**, which is what stage 5 expected: -0.0049, -0.0015 and
-0.0079 on validation over the causal stack, -0.0104 on test, and the four features take 0.0101
of the model's total gain when present.

**The native C block is the most valuable block in the frame.** Removing the 12 C columns the
plan kept from the full stack costs 0.0637 of validation PR-AUC, every seed's interval well below
zero; removing all four native groups costs 0.1882. The D block's removal costs 0.0058 on average,
outside its interval on one seed of three; the V block's 0.0070, outside on one seed; the M block's
removal changes nothing measurable, +0.0006 with every interval straddling zero.

**Which native groups ship, and the rule that was replaced.** The rule as first written retired a
provided block unless every seed showed a loss outside its interval, the mirror of the rule for a
built block. Applied to the table it kept only C and would have shipped a 132 column stack no
ablation had measured, composing three single-block removals as if they were additive. The rule
was replaced after the numbers were read, and the artifact records both the first draft and the
stack it would have shipped (`decision.superseded_first_draft`). The rule that stands: a provided
block is retired only when its removal lowers validation PR-AUC on no seed with the interval
excluding zero. The burden is on the removal for a block the frame already carries, and on the
addition for a block this repo builds, because a wrong drop costs detection and a wrong keep costs
width. Under it C, D and V ship, M does not, and the shipped stack is
`prepared|native=count,timedelta,vesta` at 186 columns. Section 4 fits it on the whole train
split and reports what it scores; that stack was not among the twelve either, and the number is
beside the default stack's so the cost of the rule is visible. `docs/notes/stage-06.md` has the
account of the change.

**Own velocity against the native counters, the comparison ADR 0021 asked for.** Two readings,
same model, same seed. The 2 by 2 on validation PR-AUC (means over three seeds): the stage 3
frame with C scores 0.6242, without C 0.5491; adding the causal block to each gives 0.6209 and
0.5553. So the 18 velocity columns recover 0.0062 of the 0.0751 the C block is worth when C is
absent, and nothing when it is present. And the gain importances from the seed-42 fit of prepared
+ causal: the 12 C columns take 0.2742 of the model's summed gain, 103.67 per column, with C7 the
single largest feature in the model at 527.9; the 18 velocity columns take 0.0252, 6.35 per
column, the largest of them `vel_amount_sum_1h_entity` at 9.8. Per column, the native counter
block carries about sixteen times the gain of the built velocity. ADR 0021 recorded that the C
block was the stronger on single-feature signal and left the model comparison to this stage; this
is the answer, and it leaves stage 8's problem exactly where ADR 0021 put it: the serving path
needs a block it cannot define.

---

## 4. The search, and why the default ships

`reports/hyperparameter_search.json`, regenerated with `make train` or
`.venv/bin/python scripts/train.py --sections search`. XGBoost without reweighting on the shipped
stack, 186 columns.

**The cross-validation never fits on a row later than its validation rows.** The train split is
cut into four contiguous blocks at timestamp quantiles, on the stage 0 rule that assignment is
`ts < boundary` so a shared timestamp never straddles two blocks. Fold k fits on blocks 0 to k
and validates on block k + 1: 103,345 rows fitting and 103,344 validating (3,634 fraud) on fold 0,
206,689 and 103,344 (4,227) on fold 1, 310,033 and 103,345 (4,062) on fold 2. Every fold's
`fit_end_ts` is below its `validate_start_ts` and the artifact test asserts it. Three folds, not
more, because the first fold already fits on a quarter of the window.

**Twenty random draws from a grid, plus the default as draw zero.** The space: 300, 600 or 1,000
trees; learning rate 0.02, 0.05 or 0.1; depth 4, 6, 8 or 10; `min_child_weight` 1, 5 or 20; row
subsample 0.6, 0.8 or 1.0; column subsample 0.4, 0.6 or 0.8; `reg_lambda` 1, 5 or 10. Sixty-three
fits, 542 seconds of fitting. Mean fold PR-AUC ran from 0.5458 to 0.5754 across the 21
configurations; the default scored 0.5724 (folds 0.5409, 0.5905, 0.5857) and the best, trial 15
(600 trees, learning rate 0.02, depth 10, `min_child_weight` 1, no row subsampling, column
subsample 0.4, `reg_lambda` 10), scored 0.5754 (folds 0.5516, 0.5868, 0.5880), a gain of 0.0031.

**Both were refitted on the whole train split, and the default won.** The rule, written before
the numbers: the tuned configuration ships when its validation PR-AUC on the full fit is at least
the default's, otherwise the default ships and the search is recorded as having found nothing.
The default scores 0.6184 on validation and 0.5451 on test; the tuned configuration 0.6048 and
0.5359. The search found nothing. A gain of 0.0031 across three folds whose scores differ by up to
0.05 from one another was never going to be a result, and the artifact says so with the fold
scores beside the means.

**The shipped model.** XGBoost, no reweighting, default hyperparameters, the 186 column stack,
seed 42: validation PR-AUC 0.6184 and ROC-AUC 0.9288, test PR-AUC 0.5451 and ROC-AUC 0.9050,
card-level PR-AUC 0.6631 on validation and 0.6442 on test. The default stack's xgboost from
section 1, 237 columns, scores 0.6198 and 0.5490 on the same splits; the 51 columns the
ablation rule removed cost 0.0014 on validation and 0.0039 on test, both inside the noise band
section 5 measures. `models/shipped_model.json` records the family, the stack, the column list
and the hyperparameters; the booster itself, 3.7 MB, is rebuilt by `make train` and not
committed.

---

## 5. Variance: what a difference has to clear

`reports/metric_variance.json`, regenerated with `make train` or
`.venv/bin/python scripts/train.py --sections variance`. The nine comparison models and the
shipped model, held fixed, every one scored on the identical resample of the test split, 1,000
resamples, seed 42. Three resamples: rows drawn with replacement; whole cards drawn with
replacement (the entity key of ADR 0001), which is the resample that respects card-level labels;
and the card-level metric of section 1 under the card resample.

| Model | Test PR-AUC | Row resample, 95 percent | Half-width | Card resample, 95 percent | Half-width | Card-level PR-AUC, 95 percent |
| --- | --- | --- | --- | --- | --- | --- |
| logistic regression, none | 0.3655 | 0.3476 to 0.3818 | 0.0171 | 0.3291 to 0.4064 | 0.0387 | 0.4726, 0.4478 to 0.4986 |
| logistic regression, class weight | 0.3414 | 0.3233 to 0.3577 | 0.0172 | 0.3050 to 0.3833 | 0.0391 | 0.4467, 0.4204 to 0.4727 |
| logistic regression, SMOTE | 0.3480 | 0.3295 to 0.3646 | 0.0176 | 0.3129 to 0.3894 | 0.0383 | 0.4478, 0.4221 to 0.4741 |
| random forest, none | 0.4744 | 0.4573 to 0.4914 | 0.0171 | 0.4188 to 0.5388 | 0.0600 | 0.6277, 0.6041 to 0.6519 |
| random forest, class weight | 0.5123 | 0.4940 to 0.5299 | 0.0180 | 0.4818 to 0.5446 | 0.0314 | 0.6272, 0.6026 to 0.6516 |
| random forest, SMOTE | 0.4804 | 0.4617 to 0.4983 | 0.0183 | 0.4504 to 0.5126 | 0.0311 | 0.5922, 0.5674 to 0.6180 |
| xgboost, none | 0.5490 | 0.5316 to 0.5662 | 0.0173 | 0.5170 to 0.5807 | 0.0318 | 0.6515, 0.6277 to 0.6742 |
| xgboost, class weight | 0.5221 | 0.5044 to 0.5398 | 0.0177 | 0.4884 to 0.5543 | 0.0330 | 0.6251, 0.6004 to 0.6509 |
| xgboost, SMOTE | 0.4998 | 0.4822 to 0.5180 | 0.0179 | 0.4679 to 0.5339 | 0.0330 | 0.5842, 0.5596 to 0.6094 |
| shipped | 0.5451 | 0.5274 to 0.5614 | 0.0170 | 0.5100 to 0.5805 | 0.0352 | 0.6442, 0.6200 to 0.6684 |

**The headline number with its interval.** The shipped model's test PR-AUC is 0.5451, 95 percent
interval 0.5274 to 0.5614 over row resamples and 0.5100 to 0.5805 over card resamples; at card
level it is 0.6442, 0.6200 to 0.6684. The card resample roughly doubles every interval, which is
what correlated labels inside a card do to a row-level estimate and why both are reported.

**The lead survives resampling.** Paired differences, shipped minus the other, on the row
resample: 0.0707 over the random forest without reweighting (0.0626 to 0.0796), 0.0328 over the
weighted random forest (0.0241 to 0.0418), 0.1796 over the logistic regression (0.1658 to 0.1949),
0.0230 over weighted xgboost (0.0159 to 0.0298). Every one of those intervals excludes zero and
the sign agrees on all 1,000 resamples. On the card resample the random forest pair widens to
0.0219 to 0.1207 and still excludes zero. The one pair that does not is the shipped model against
the default-stack xgboost it was cut from: -0.0039, -0.0084 to +0.0008 by row, -0.0161 to +0.0082
by card, sign agreement 0.951 and 0.704. The 51 columns the ablation rule removed cost nothing the
test split can see.

**The reversal of section 2, with its interval.** The weighted random forest beats the plain one
on test by 0.0379, 0.0319 to 0.0442 on the row resample; on the card resample the interval is
-0.0068 to 0.0754 and includes zero, sign agreement 0.882. The row resample calls it a result and
the card resample does not, on a pair validation ordered the other way. That is the case ADR 0010
described and it is why no decision in this stage reads a test number.

**The noise band.** The widest paired half-width on transaction-level PR-AUC is 0.0172 over the
row resample (logistic regression with SMOTE against xgboost with SMOTE) and 0.0592 over the card
resample (weighted logistic regression against the plain random forest, whose card interval is the
widest in the table at 0.0600). The artifact's `noise_band` takes the larger, 0.0592. A promotion
margin inside it promotes on noise. Stage 9 has to set its margin above the band that applies to
the pair it compares: two models of one family on one stack are paired far more tightly than the
widest pair in this table (the shipped model against its own default-stack parent has a half-width
of 0.0046 by row and 0.0121 by card), and the artifact carries every pair so the margin can be
derived from the right one rather than from the widest.

---

## 6. Thresholds, bands and segments

`reports/threshold_curve.json` and `reports/operating_points.json`, both written by
`.venv/bin/python scripts/train.py --sections thresholds` and by nothing else. Every precision,
recall, alert volume or false-alarm figure the repo quotes at a threshold comes from one of these
two files.

**Calibration.** An isotonic regression from the shipped model's score to a probability, fitted on
the validation split, applied to test. It bought nothing measurable: the raw score already has a
Brier of 0.02186 and an expected calibration error of 0.00246 on test, with a mean prediction of
0.03266 against an observed rate of 0.03480, and the calibrated score has 0.02200 and 0.00508. An
unweighted gradient-boosted classifier trained on the log loss produces something close to a
probability already, which is the other side of ADR 0027's finding that reweighting shifts the
score. The step stays because the band edges are defined on a probability and the mechanism has
to survive a model whose score is not one. The reliability tables, ten equal-count bins, are in
the artifact and `figures/calibration.png`.

**The tuned vertex.** The largest-F1 vertex of the validation precision-recall curve is at a raw
score of 0.2519: on validation, precision 0.6906, recall 0.5128, F1 0.5886, 71.9 alerts per day,
0.448 false alarms per catch. Applied unchanged to test: precision 0.6034, recall 0.4713, F1
0.5292, 78.2 alerts per day, 0.657 false alarms per catch; 1,453 true positives, 955 false
positives, 1,630 false negatives, 84,543 true negatives. The validation-to-test drop is the drift
stage 2 measured, arriving at the operating point.

**The sweep.** `reports/threshold_curve.json` carries precision, recall, F1, alerts per day and
false alarms per catch at every hundredth from 0.01 to 0.99, on the calibrated probability and on
the raw score, on validation and on test. `figures/threshold_curve.png` draws the validation
sweep with the band edges marked.

**Two thresholds from three costs.** ADR 0029. The cost matrix in `config.py` is an assumption
and says so: a missed fraud costs 10, a false decline 1, an analyst review 0.5, in units of one
false decline, under the further assumption that a review resolves the transaction correctly.
From those, and from nothing else: review beats approve once `p > review / missed = 0.05`; block
beats review once `(1 - p) * decline < review`, so `p > 1 - review / decline = 0.5`; and with no
review band at all, block beats approve at `p > decline / (decline + missed) = 0.0909`. The edges
are on the calibrated probability and map to raw scores of 0.0396 and 0.4773 on validation.

| Band | Val rows | Val share | Val per day | Val fraud in band | Test rows | Test share | Test per day | Test fraud in band | Test share of all fraud |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| approve, below 0.05 | 78,900 | 0.8907 | 2,511.9 | 613 | 76,949 | 0.8687 | 2,500.1 | 780 | 0.2530 |
| review, 0.05 to 0.5 | 8,206 | 0.0926 | 261.3 | 1,184 | 10,188 | 0.1150 | 331.0 | 1,181 | 0.3831 |
| block, 0.5 and above | 1,475 | 0.0167 | 47.0 | 1,245 | 1,444 | 0.0163 | 46.9 | 1,122 | 0.3639 |

The block band runs at a fraud rate of 0.8441 on validation and 0.7770 on test; the approve band
at 0.0078 and 0.0101. Under the matrix, the realised cost on test is 0.1492 per row: 780 missed
fraud at 10, 10,188 reviews at 0.5, 322 false declines at 1. Approving everything costs 0.3480 per
row and the single cost-optimal threshold without a review band costs 0.1729 (2,091 caught, 5,398
false declines, 992 missed). The review band is doing the work, and it is doing it at 331 reviews
per day on test, a volume the matrix does not constrain and an operations team would. Review
capacity is the fourth input this derivation does not have; with it, the band edges would move
and the artifact records which costs produced which edges so that they can.

**Segments.** ADR 0030. The stage brief asked for per-market calibration on `addr2` if its values
allowed it. Stage 2 measured `addr2` at a top-value share of 0.9901 on train, so there is one
market and nothing to calibrate against. `ProductCD` is the partition with five levels none of
which is in the rare tail (0.7207, 0.1210, 0.0729, 0.0680 and 0.0174 of train rows for W, C, R, H
and S), and the same mechanic runs on it: the largest-F1 vertex and an isotonic calibrator fitted
on the segment's validation rows, test scored with both fixed.

| Segment | Val rows | Val base rate, Wilson 95 | Test rows | Test base rate, Wilson 95 | Val PR-AUC | Test PR-AUC | Tuned threshold | Test precision | Test recall | Test alerts per day |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 72,291 | 0.0205, 0.0195 to 0.0216 | 69,468 | 0.0186, 0.0176 to 0.0196 | 0.4027 | 0.2504 | 0.1344 | 0.2772 | 0.3248 | 49.2 |
| C | 9,125 | 0.1260, 0.1194 to 0.1330 | 9,388 | 0.1353, 0.1285 to 0.1423 | 0.8019 | 0.7664 | 0.2177 | 0.6980 | 0.7244 | 42.8 |
| R | 3,365 | 0.0475, 0.0409 to 0.0553 | 4,190 | 0.0506, 0.0444 to 0.0577 | 0.8487 | 0.8009 | 0.3833 | 0.7895 | 0.7075 | 6.2 |
| H | 2,350 | 0.0596, 0.0507 to 0.0699 | 2,562 | 0.0640, 0.0552 to 0.0742 | 0.7370 | 0.4982 | 0.4339 | 0.5772 | 0.4329 | 4.0 |
| S | 1,450 | 0.0738, 0.0614 to 0.0884 | 2,973 | 0.0484, 0.0413 to 0.0568 | 0.7312 | 0.5021 | 0.1750 | 0.5448 | 0.5069 | 4.4 |

The segments are not one population. W is 0.7207 of train, has a base rate of 0.0186 on test and
is the segment the model is worst on, PR-AUC 0.2504; C has a base rate seven times higher and a
PR-AUC of 0.7664. The tuned thresholds run from 0.1344 to 0.4339. One threshold for the file is a
threshold tuned for W with everything else riding on it.

Per-segment calibration matters most where the base rate moved: S runs at 0.0738 on validation and
0.0484 on test, Wilson intervals disjoint, and the global calibrator puts 1,297 of its 2,973 test
rows into review, for a realised cost of 0.4035 per row; the segment's own calibrator puts 207
there, at 0.2619. On W the two agree to within 0.0005 per row. The per-segment bands, base rates
and realised costs for every segment on both splits are in the artifact under `segments`.

---

## 7. SHAP, and the caveat measured rather than stated

`reports/shap/shap_global.json` and `reports/shap/shap_example.json`, regenerated with
`.venv/bin/python scripts/train.py --sections shap`. `shap.TreeExplainer` on the shipped booster,
tree-path-dependent perturbation, log-odds output, on a seeded uniform sample of 5,000 test rows.

**Global.** The twenty largest mean absolute contributions are led by `C13` (0.325), `C5`
(0.318), `card1_te` (0.293), `C1` (0.226), `vblockmean_V279` (0.204) and `C14` (0.176). By block,
the 12 native C columns take 0.2672 of the summed mean absolute SHAP at 0.1259 per column, against
0.0269 per column for the 29 D columns and 0.0309 for the 25 V columns; the 16 target encodings
take 0.1597 at 0.0564 per column; the 36 missing-block indicators take 0.0016 in total, which is
what an indicator is worth to a model that reads the null itself. `figures/shap_global.png` draws
both panels.

**One transaction.** The fraud transaction with the largest shipped score in the sample,
`TransactionID` 3556851, scored 0.9966. Its contributions sum to 9.05 log-odds over the base
value of -3.369, led by `C14 = 0` at +1.15, `C1 = 6` at +1.02, `vblockmean_V138` at +0.78 and
`C13 = 0` at +0.74. The entity it sits on has one row in the test split.

**The caveat, and the number that makes it visible.** Labels in this file are card level, so a
transaction's attribution partly explains which card it is on rather than what the transaction
did. Section 1 measured the size of that: the shipped model scores 0.6442 at card level against
0.5451 at transaction level on test, and every model in the comparison is higher at card level.
The card1 target encoding is the third feature in the global ranking, and it is a feature about
the card. A per-transaction explanation from this model is an explanation of the card and the
transaction together, and the artifact says so beside the numbers rather than in a footnote.

---

## What stage 7 inherits

| Finding | Number | Consequence |
| --- | --- | --- |
| Shipped family | xgboost, no reweighting, default hyperparameters | `models/shipped_model.json` names the family, stack, columns and hyperparameters; the booster is rebuilt by `make train` |
| Shipped stack | 186 columns, stage 3 with C, D and V, no M, no causal, no graph | the serving path computes stage 3's pipeline and no stage 4 or 5 feature |
| Primary metric | PR-AUC 0.5451 on test, 0.5274 to 0.5614 | ROC-AUC 0.9050 beside it, deciding nothing |
| Card level | 0.6442, 0.6200 to 0.6684 | the per-transaction explanation is partly a card explanation |
| Noise band | 0.0172 by row, 0.0592 by card, every pair in the artifact | stage 9's margin sits above the band for the pair it compares |
| Tuned vertex | raw score 0.2519 | test precision 0.6034, recall 0.4713, 78.2 alerts per day |
| Band edges | 0.05 and 0.5 on the calibrated probability | from an assumed cost matrix; 331 reviews per day on test |
| Segment key | ProductCD | thresholds 0.1344 to 0.4339 across five levels |
| The C block | 0.2672 of the model's attribution on 12 columns | stage 8's serving path needs a block whose construction is not established |

## What is not established

- Whether a different family would want stage 4's features. The ablation is for xgboost; the
  logistic regression and the random forest were fitted on the default stack only, and a block
  that adds nothing to a tree that already reads the C columns might add something to a model that
  cannot cut on them.
- What the C columns are. The single most valuable block in the model and the one the serving
  path cannot compute, exactly as ADR 0021 left it.
- The review capacity. The band derivation has three costs and no volume constraint, and the
  331 reviews per day it produces on test are reported, not targeted.
- Whether the isotonic step should stay. It moved test calibration by less than the gap between
  the two splits' base rates. It is kept for the model that needs it and measured for the one that
  does not.
- Why the search found nothing. Twenty-one configurations and three folds is a small search, and
  the fold scores differ by up to 0.05 within one configuration; a search that could resolve 0.003
  needs more folds than a 120 day window has room for.
- Everything earlier stages left open stays open, with one closed: the primary metric and the
  operating point are now fixed. The promotion margin is stage 9's.
