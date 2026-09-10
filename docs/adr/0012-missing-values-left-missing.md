# 0012: Missing values are left missing for the tree path

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

Stage 2 measured that missingness on this frame carries the label. Of the 372 columns with nulls,
251 have a fraud rate that differs by at least 0.010 in absolute terms between their present and
missing rows, at p below 1e-4. The largest single gap in the whole document belongs to a null and
not to a value: D7 present runs at a fraud rate of 0.1478 against 0.0274 when missing, a gap of
0.1204. addr1 runs the other way, 0.1125 missing against 0.0251 present. And the sharpest plain
fact stage 2 found is a join failing: a row with an identity record is 3.41 times as likely to be
fraud, 0.07312 against 0.02142.

Filling those nulls with a median deletes all of it. The model then sees a column where 0.7341 of
rows carry the same fabricated value and no indication that they are not real.

## Options considered

**Impute everything with a median or a mode.** The default, and the reason this ADR exists.
Rejected on the numbers above: it destroys measured signal on 251 columns and replaces it with a
value the data never contained.

**Impute everything, and add a was-missing indicator per column.** Correct, and it was the
runner-up. It preserves the information. Rejected as the primary path because it costs 251
columns to reconstruct something the model can already read, and because stage 2's block
structure says most of those indicators would be copies of each other: columns inside a
missingness block are null on identical rows.

**Leave nulls alone and let the model split on them.** Chosen for the tree path. A gradient
boosted tree learns a default direction per split, so the null is a side of a cut rather than a
hole to be filled, and the information stays exactly where stage 2 found it.

That last option is a claim about libraries, not about statistics, and the repository rule is
that library behaviour is run and checked rather than recalled. So it is verified rather than
asserted, and verified twice, because tolerating a null and learning from one are different
properties and only the second justifies the decision.

## Decision

Three policies, and which one a column gets depends on what the model reading it can do.

1. **A model that reads nulls gets nulls.** Nothing is filled anywhere on the tree path.
2. **A model that cannot gets an imputed value and an indicator**, and the indicator is per
   missingness block rather than per column, because stage 2 defined a block as columns whose null
   mask is identical row for row, so the per-column indicators inside one block are the same
   vector.
3. **Where the null has a known meaning it gets a sentinel that says so, and never zero.** Zero is
   a value these columns take: stage 2 measured 229 zero-inflated columns and D1 exactly zero on
   0.5013 of its non-null rows, so a zero fill merges two populations that stage 2 spent a section
   separating.

`fraud_platform.missing_policy.GROUP_POLICY` records the policy per column group with the stage 2
measurement behind each row. The identity join gets its own indicator, `has_identity_record`,
separate from any block indicator, computed from id_01, which stage 2 named as one of exactly two
columns null on precisely the rows where the join fails.

The sentinel applies to the 15 D columns and is placed one unit below that column's train
minimum, recorded per column. It is not a semantic value and it is not doing the work: the
indicator beside it is. It exists so a complete-data pipeline has something to put there that
orders correctly and cannot be mistaken for an observation. What it means is a choice this stage
made, not a measurement: a D column is a delta backwards from the transaction, so a null is read
as the observation window holding no earlier event of that kind for this card. Nothing in this
repo establishes that reading.

## Evidence

`reports/prep/missing_policy.json`, regenerated with `.venv/bin/python scripts/prepare.py
--sections missing`. The library verification runs on generated data, so it runs in CI and re-runs
whenever a pinned version changes.

Two probes per estimator. `accepts_nan` fits three columns with nulls scattered at 0.3 and records
whether the fit raises. `informative_null_auc` fits one column that is pure noise wherever it is
present and null on exactly the positive rows: an estimator that sends nulls to a learned side of
a split scores 1.0, and one that merely tolerates them scores 0.5.

lightgbm 4.7.0, xgboost 3.4.1, scikit-learn 1.9.0:

| Estimator | Accepts nulls | Informative-null AUC |
| --- | --- | --- |
| lightgbm.LGBMClassifier | yes | 1.0 |
| xgboost.XGBClassifier | yes | 1.0 |
| sklearn.HistGradientBoostingClassifier | yes | 1.0 |
| sklearn.RandomForestClassifier | yes | 1.0 |
| sklearn.ExtraTreesClassifier | yes | 1.0 |
| sklearn.DecisionTreeClassifier | yes | 1.0 |
| sklearn.GradientBoostingClassifier | no, ValueError | not applicable |
| sklearn.LogisticRegression | no, ValueError | not applicable |

Six of the eight accept nulls and all six of those learn a direction for them. The two that refuse
raise `ValueError: Input X contains NaN`, and the artifact records the message.

The result worth naming is `sklearn.RandomForestClassifier`. It is in the group that accepts
nulls and scores 1.0 on the second probe, which is not what a reader of older scikit-learn
documentation would predict, and it is exactly the kind of claim this repo will not make from
memory. `sklearn.GradientBoostingClassifier` is the counterexample that keeps the check honest: it
sits in the same module and refuses.

The indicator spec covers the kept columns: 66 indicators over 244 columns, of which 23 are
multi-column indicators covering 201 columns and 43 are single-column. Stage 2's own count was 24
blocks over 392 columns; the difference is the 146 columns the column plan removed, which shrinks
several blocks to a single member, and a one-member block still needs its own indicator because
its null mask is its own.

The complete-data pipeline fits 240 medians and 15 sentinels. `apply_imputer` is idempotent and a
test asserts that running `add_missing_indicators` after it gives an all-zero indicator, which is
the failure mode the ordering exists to prevent.

## Consequences

- The tree path in stage 5 receives the frame with nulls intact and no indicator columns for
  them beyond the block indicators, which are there for the models that need them and for stage 7,
  where a reviewer asking why a score is high should be able to see "no identity record" as a
  named thing rather than as forty columns going quiet.
- A model in stage 5 that cannot read a null has a supported path and it is a different frame, not
  the same frame with a fill applied. The difference is recorded rather than hidden.
- The verification is a test, not a one-off. `tests/test_prep_units.py` re-measures LightGBM and
  XGBoost on every run, so an upgrade that changes the behaviour fails the suite instead of
  silently changing what the pipeline means.
- The sentinel's meaning is labelled a choice. If a later stage establishes what a null D actually
  is, that is a new ADR and the sentinel may need to change with it.
- Nothing here helps a model that needs complete data and also needs the 251 label-linked columns
  to keep their meaning without 251 indicators. The block indicators are an approximation
  justified by the identical-mask definition, and if a later stage finds a block whose members
  have drifted apart in a later window, the approximation is where to look.
