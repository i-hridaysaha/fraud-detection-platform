# 0016: V-column reduction by block mean, chosen on a tie-break

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

The V block is 339 columns and stage 2 measured two structures in it. It goes missing in blocks:
14 multi-column blocks with nulls, and columns inside one block are null on identical rows. And it
is redundant inside those blocks: at Spearman 0.95, 339 columns across 14 blocks reduce to 207
representatives, so a representative-per-group reduction removes 132.

Structure is not a strategy. There are at least three ways to spend it and they are different
trades, so picking one and reporting the column count it produces is not a decision, it is a
preference with a number attached.

## Options considered

Three reductions and the unreduced baseline, all measured rather than argued.

**Keep a representative per correlated group.** The kept column is a real column, so a later
explanation names something a reviewer can look up. It discards whatever the other members held
that the representative does not.

**PCA inside each missingness block.** Keeps more variance per column kept. The components are not
columns anyone can name, which stage 7 will feel.

**One standardised mean per block.** The smallest frame by a wide margin and the crudest: it
assumes a block is one thing measured several ways.

**No reduction.** The baseline, and it is not optional. A reduction that costs validation AUC
against the unreduced frame is buying width with signal, and without the baseline that trade is
invisible.

PCA and the block mean both run inside a missingness block rather than across the frame, and that
is not a convenience. Columns in a block are null on identical rows, so within a block the present
rows are the same rows for every column and there is a complete matrix to decompose. Across blocks
there is not, and filling one to make one would put invented values into the decomposition. Both
standardise first: stage 2 measured 212 columns with an absolute skew above 10 and the V columns
run over wildly different scales, so an unstandardised mean over a block is that block's
largest-scale column plus noise.

## Decision

Ship the block mean: 14 columns in place of 327.

The rule, stated before the numbers were read: the strategy with the best validation AUC, and
among the strategies inside the resolvable band of it, the one with the fewest columns.

The comparison runs on the V columns that survived the constant, missing and time-inconsistency
rules of ADR 0011, whether or not the redundancy rule then removed them, because the redundancy
rule is the `representative` strategy and cannot be applied before the comparison that includes
it. `cleaning.v_reduction_pool` builds that pool and `apply_column_plan` spares it from the plan's
drop when a strategy other than `representative` ships.

## Evidence

`reports/prep/v_reduction.json`, regenerated with `.venv/bin/python scripts/prepare.py --sections
v_reduction`. `figures/v_reduction.png` draws it.

One small LightGBM per strategy, fitted on the V columns alone on train and scored on validation,
identical hyperparameters across strategies because the comparison is between column sets and
anything that varied between them would confound it. It is a probe. No number here is a
performance estimate and stage 5 is where models are built.

| Strategy | Columns | Train AUC | Validation AUC | Validation AUC across three seeds |
| --- | --- | --- | --- | --- |
| all | 327 | 0.8927 | 0.8406 | 0.8372 to 0.8414 |
| representative | 196 | 0.8904 | 0.8378 | 0.8362 to 0.8414 |
| pca | 122 | 0.9026 | 0.8466 | 0.8440 to 0.8466 |
| block_mean | 14 | 0.8791 | 0.8375 | 0.8354 to 0.8380 |

The PCA keeps components until they explain 0.95 of a block's variance, capped at 12 per block:
122 components over the 14 blocks, from 6 on the 16-column block at missing rate 0.7384 to 12 on
three of the larger ones.

The four strategies span 0.0090 in validation AUC, from 0.8375 to 0.8466. One strategy spans
0.0051 across three LightGBM seeds on identical rows and identical columns. The probe therefore
does not separate them: `resolved: false` in the artifact, and the stated band, 0.0176, contains
all four. The seed variance was measured for this reason, and because ADR 0009 left run-to-run
variance open on its own screen and it should not stay open twice.

So the tie-break is what chose, and the tie-break is column count. 327 columns to 14 for a
validation AUC difference of 0.0031 against the unreduced baseline, which is a difference this
split cannot resolve.

The result the tie-break gives up is worth stating plainly rather than burying: the PCA scored
highest on every seed, and its 0.0091 over the block mean is larger than either strategy's own
seed spread, though not larger than the band the rule was written against. A rule with a different
tie-break would have chosen it.

## Consequences

- The prepared frame carries 14 columns where the raw frame carried 339, and the V block stops
  being the bulk of the feature count. That is most of why the prepared frame is 247 columns.
- Nothing in the V block is nameable any more. Stage 7 explaining a score cannot say "V258 was
  high", only "the mean of the block V217 belongs to was high". Stage 2 measured V258's upper bin
  at a fraud rate of 0.6076 on 4,348 rows, and that specific fact is now invisible to an
  explanation. This is the cost, it was foreseeable, and it was not enough to overturn a rule
  stated in advance.
- The decision is provisional in a stated way. `prepare.fit_preparation` takes `v_strategy` as an
  argument and reads it from this artifact, so stage 5 can re-run the comparison with a real model
  and a paired test on the same rows, which resolves more than an unpaired probe does. If it
  separates them, that is a new ADR superseding this one.
- Nothing here says the V columns are unimportant. All four strategies reach a validation AUC
  around 0.84 on the V block alone, which is the strongest single-block number this repo has
  measured.
