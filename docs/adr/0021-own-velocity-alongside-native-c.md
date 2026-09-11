# 0021: Own velocity features are kept alongside the native C columns

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 4

## Context

Stage 4 built point-in-time velocity: transaction count, amount sum and amount mean over trailing
10 minute, 1 hour and 24 hour windows, at two grains, with the current row and its
same-timestamp ties excluded by construction (ADR 0018).

The dataset already contains a block that appears to be doing something similar. C1 to C14 are
integer valued, null-free over all 590,540 rows, and stage 2 measured C8 with C10 at Spearman
0.9710 over all 413,378 train rows and described the block as running counters. Their meaning is
widely repeated as counts associated with the payment card. **This repo has not established
that.** Stage 0 could not fetch the competition's data description, which is rendered client
side, and the same fetch failed again in stage 4. So what follows treats the C block as fourteen
unlabelled counters whose structure is measured and whose semantics are not.

That leaves a real question rather than a rhetorical one. If this repo's own velocity is
substantially the same quantity as the native counters, then one of the two blocks is redundant
width and the honest thing is to drop one. If it is not, both belong in stage 6 and the
comparison is worth reporting, because it is a comparison nobody else on this dataset has
reported: a hand-built point-in-time velocity block whose causality is verified, against a
provided counter block whose construction is unknown, on the same rows.

## Options considered

**Keep only the native C columns.** They are free, they are already in the frame, and stage 2
measured four of them among the strongest single features in the file. Rejected: their
construction is unknown, so nothing can be said about whether they respect a point-in-time rule,
and a serving path cannot reproduce a column it cannot define. A model resting on them would be a
model resting on an input the platform cannot compute.

**Keep only this repo's velocity, and drop the C block as an unverifiable input.** Rejected on the
measurement below, and it was tempting because it makes the causal story clean. The C columns
carry more single-feature signal than the built velocity does, by a wide margin, and dropping them
because they are inconvenient to explain would be choosing the story over the number.

**Keep both, measure the correlation, and let stage 6 measure the contribution of each.** Chosen.
The rule was written before the numbers were read: keep both when the correlation between them is
below the redundancy threshold stage 2 used for its own reduction, and drop one when it is not.

Stage 2's redundancy threshold is 0.95 (`eda.REDUNDANT_RHO`), which is where the V-block reduction
groups two columns as the same column. That is the line this decision is measured against, and
the measured maximum comes nowhere near it.

**Which grain the comparison happens at** is part of this decision. The native counters are
described as card-associated, and this repo's entity key is finer than a card by construction:
stage 0 measured 13,553 card1 values against 217,850 entities at the ADR 0001 key. An entity-grain
velocity and a card-associated counter are therefore two different quantities, and comparing them
would not be like for like. So the velocity family is built at both grains, `_entity` and
`_card`, the card grain is the comparison, and the entity grain is the feature the stage brief
asked for.

## Decision

Both blocks are kept into stage 6. The velocity family ships at both the entity and the card
grain, 18 columns, and the native C block stays available through
`fraud_platform.features.FeatureConfig.native_groups`, which is the switch item 5 of the stage
brief asked for. Stage 6 measures the contribution of each by switching a block off and refitting,
which is like for like because the only thing that changes is which columns are present.

## Evidence

`reports/features/velocity_vs_c.json`, regenerated with `.venv/bin/python scripts/features.py
--sections velocity_c`. `figures/velocity_vs_native_c.png` draws both panels.

**They are not the same measurement.** Spearman correlation on the train split, all 84 pairs of
the six velocity counts against the fourteen C columns, computed exactly with pandas rather than
screened:

| Quantity | Value |
| --- | --- |
| Largest absolute rho over 84 pairs | 0.2773 |
| Median absolute rho | 0.0658 |
| Pairs at or above 0.50 | 0 |
| Pairs at or above 0.80 | 0 |
| Stage 2's redundancy threshold, for reference | 0.95 |

The largest pair is `vel_count_24h_entity` with C7 at 0.2773. Nothing reaches half of the line at
which stage 2 would call two columns the same column, so the rule keeps both.

**And the native block is stronger.** Reported as the fraud rate on rows where the count is
positive against rows where it is zero, on train, because six of these columns are zero-inflated
enough that ten equal-frequency bins collapse and the information value understates them. The
artifact records the bin count next to every information value so the collapse is visible rather
than implicit.

| Feature | Share of rows positive | Rate positive | Rate zero | Ratio | Information value | Bins |
| --- | --- | --- | --- | --- | --- | --- |
| vel_count_10m_entity | 0.0925 | 0.07748 | 0.03086 | 2.511 | 0.0000 | 1 |
| vel_count_1h_entity | 0.1615 | 0.06821 | 0.02881 | 2.368 | 0.0738 | 2 |
| vel_count_24h_entity | 0.2983 | 0.06140 | 0.02402 | 2.556 | 0.1747 | 3 |
| vel_count_10m_card | 0.1958 | 0.05199 | 0.03107 | 1.673 | 0.0165 | 2 |
| vel_count_1h_card | 0.4279 | 0.04227 | 0.02986 | 1.415 | 0.0174 | 4 |
| vel_count_24h_card | 0.7873 | 0.03815 | 0.02414 | 1.580 | 0.0386 | 8 |
| C7 | 0.1183 | 0.11232 | 0.02482 | 4.525 | 0.4899 | 10 |
| C12 | 0.1521 | 0.09938 | 0.02365 | 4.201 | 0.5013 | 10 |
| C4 | 0.2522 | 0.07800 | 0.02072 | 3.764 | 0.5629 | 10 |
| C8 | 0.2637 | 0.07386 | 0.02131 | 3.466 | 0.5532 | 10 |
| C10 | 0.2536 | 0.07502 | 0.02163 | 3.469 | 0.5226 | 10 |

The strongest native counter separates the label about 1.8 times as sharply as the strongest built
velocity, and its information value is more than three times as large. That is the result of this
section and it is not the result the section was expected to produce.

Two readings are available and this repo cannot choose between them, so both are recorded.

The C block may simply be a better feature: a longer history, a wider notion of what to count, or
a definition with information this file does not otherwise contain. If so, a production system
should want the quantity the C block holds and stage 8 has a problem, because it cannot compute a
column whose definition is unknown.

Or the C block may not be point-in-time. A counter built over a whole observation window, or one
that counts forward as well as backward, would separate the label better than a strictly trailing
one and would not be reproducible at authorisation. Nothing in this repo can distinguish the two
cases: the necessary evidence is the construction, and the construction is not published in
anything fetched here. The one adjacent measurement, stage 2's time-consistency screen, does not
resolve it either, since it flagged 59 columns and none of C1 to C14 is among them.

**Both blocks hold their direction out of sample, which had to be checked rather than assumed.**
The same zero-against-positive comparison on each split, reported as the ratio of the two rates.
A train-only version of this table would have been the wrong measurement: the duplicate-content
family in ADR 0022 passes it on train and reverses on validation and test.

| Feature | train | val | test | Same sign on all three, intervals disjoint |
| --- | --- | --- | --- | --- |
| vel_count_10m_entity | 2.511 | 2.431 | 1.961 | yes |
| vel_count_1h_entity | 2.368 | 2.291 | 1.989 | yes |
| vel_count_24h_entity | 2.556 | 2.625 | 2.482 | yes |
| vel_count_10m_card | 1.673 | 1.645 | 1.537 | yes |
| vel_count_1h_card | 1.415 | 1.328 | 1.353 | yes |
| vel_count_24h_card | 1.580 | 1.535 | 1.549 | yes |
| C7 | 4.525 | 5.294 | 5.909 | yes |
| C12 | 4.201 | 3.919 | 4.050 | yes |
| C4 | 3.764 | 4.786 | 5.060 | yes |
| C8 | 3.466 | 4.525 | 5.128 | yes |
| C10 | 3.469 | 4.546 | 5.021 | yes |

Every velocity feature holds, and the entity-grain 24 hour count is the steadiest thing in the
stage at 2.556, 2.625 and 2.482. The five strongest C columns hold too and four of them
strengthen: C7 runs 4.525, 5.294 and 5.909 across the three splits.

Of the fourteen C columns, 11 keep the sign of their separation on all
three splits and 3 do not: C2, C6, C11. Those three
are a measurement artifact rather than a finding, because each is positive on almost every row
(C2 is zero on 404 of 413,378 train rows, a share of 0.00098) so the rate on their zero side is computed from a few hundred
transactions. Six of the eleven stable columns are stably **inverse**, with the zero side
carrying the higher rate, which is an ordinary relationship a tree reads without difficulty and
not an instability. The artifact reports `positive_is_riskier` and `direction_stable` separately
for exactly that reason.

**The comparison stays available rather than being decided here.** Both blocks reach stage 6, the
information value and the per-split separation of each are recorded, and the model importances are
stage 6's to measure. This ADR records that they are weakly correlated, that the provided block is
the stronger of the two on single-feature signal, and that both hold their direction across the
six months the file covers.

## Consequences

- 18 velocity columns and 14 C columns both reach stage 5, at a cost of width that stage 6 is
  expected to justify or refute per block.
- **Stage 6 owes two things**: model importances for each block reported separately, and a
  like-for-like model comparison with each block switched off through
  `FeatureConfig.native_groups`. The comparison is only meaningful with the paired-seed variance
  ADR 0016 said the stage 3 probe lacked.
- **Stage 8 has a real problem to solve and it is now stated.** If stage 6 finds the C block
  carries contribution the built velocity does not, then the serving path needs a column it
  cannot define. The options are to serve the C columns as provided inputs, which assumes an
  upstream system computes them, or to accept the loss. That is a decision for stage 8 with a
  number from stage 6 behind it.
- The card-grain velocity is weaker than the entity-grain velocity on every window measured, which
  is a point in favour of the ADR 0001 key doing work. It is kept because it is the like-for-like
  comparator here, not only as a feature.
- Nothing in this ADR asserts what the C columns count. If a later stage establishes their
  construction from a source it has read, that changes the reading of this table and needs a new
  ADR rather than an edit to this one.
