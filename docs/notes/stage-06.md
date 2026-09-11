# Stage 6: training and evaluation

What the stage built: two modules (`modelling`, `evaluation`), one driver with seven sections,
one figure script, eight artifacts, five figures, five decision records, and a shipped model whose
description is committed and whose booster is not. What it found is that xgboost wins as expected,
that the imbalance handling everyone reaches for loses, and that two stages of this repo's own
feature work add nothing a tree could measure on top of the frame it already had. Most of what
follows is about the second and third of those, and about a rule I wrote wrong.

## The number that changed my mind

-0.0033. The mean change in validation PR-AUC from adding stage 4's 36 causal features to the
stage 3 frame, over three seeds, for xgboost.

I expected the causal block to carry the stage. It is the most carefully built thing in the repo:
point-in-time by construction, brute-force checked against a networkx-free recomputation, verified
across 84 correlation pairs against the C block that it was not the same measurement. ADR 0021 had
already recorded that the C block was stronger on single-feature signal, and I read that as "the C
block is stronger, the velocity adds to it". The model says the velocity adds to it when the C
block is absent (0.5491 to 0.5553) and adds nothing when it is present (0.6242 to 0.6209, two of
three seeds down). A gradient-boosted tree that can cut on C7 at gain 527.9 has no use for
`vel_amount_sum_1h_entity` at gain 9.8.

The test split says +0.0056 for the same block, which is the number I would have led with if I
were choosing what to believe. It is inside the noise band, the rule reads validation, and the
rule was written before the fits ran. So the block does not ship, the code stays, and stage 7 is
smaller than the design document drew it: no online store of entity aggregates, because the
shipped model reads none.

## The rule I wrote wrong, and what replaced it

The ablation rule for a block this repo builds is stage 4's: ship it when switching it on raises
validation PR-AUC on every seed with the paired interval excluding zero on every seed. I wrote its
mirror for the native blocks without thinking about it: retire a native block unless switching it
off lowers the score on every seed with every interval excluding zero. Symmetric, tidy, and wrong.

Applied to the numbers it kept C and retired D, V and M, because the D and V removals each cost
about 0.006 on average with only one seed of three outside its interval. The stack it produced,
132 columns, was one no ablation had fitted; it composed three single-block removals as if their
costs added, and a rule that decides on a configuration it never measured is a guess wearing a
rule's clothes. It also treated the two errors as equal when they are not: a wrong drop costs
detection at every transaction and a wrong keep costs a few columns of width.

The rule that stands puts the burden on the removal for a block the frame already carries, and on
the addition for a block this repo builds: a native block is retired only when no seed shows a
loss outside its interval. C, D and V ship; M does not, with every one of its three intervals
straddling zero. The artifact records both rules and both outcomes under
`decision.superseded_first_draft`, and I changed it after reading the numbers, which is the thing
the rule-before-numbers discipline exists to prevent. The mitigation is that both are in the
artifact, the shipped stack was then fitted and reported beside the default stack it was cut
from, and the difference between them, -0.0039 on test with an interval of -0.0084 to +0.0008,
is what the stack rule cost. If the second rule had produced a stack that beat the default I would
trust this paragraph less.

## The baseline that did not converge

The logistic regression's iteration cap was 1,000 and the weighted fit hit it. The warning is
one line in a wall of solver output and the score it produced was plausible; it would have gone
into the table as a baseline. The brief said to check
`n_iter_` against `max_iter` and say so, and the check is now a field in the artifact with a test
asserting `converged is (n_iter < max_iter)` for every logistic fit. At a cap of 5,000 the three
fits stop at 333, 840 and 971 iterations. The first thing the raised cap cost was a second
discovery: the complete-data matrix was 1,584 columns wide, and the weighted SMOTE fit on 797,680
rows of it had not finished after twenty minutes.

## 913 levels of DeviceInfo

The width was the raw `DeviceInfo` vocabulary: 913 levels after stage 3's rare-tail collapse,
against 67 for the normalised column stage 3 built to replace it. Both were in the prepared frame
and both had `_cat`, `_freq` and `_te` encodings, and my column rule read all of them. Stage 3's
module docstring says the normalisation replaces the raw column rather than supplementing it,
and the raw string is kept for a reviewer to see. So the rule now excludes the raw free-text
encodings (`modelling.REPLACED_BY_NORMALISED`), the complete-data matrix is 606 columns, the tree
path is 237 on the default stack, and the weighted logistic fit takes 32 seconds. I applied that
exclusion after a measurement of its cost and before any comparison number was kept; it is
recorded in the module as a stage 3 intent applied late rather than as a stage 6 finding.

## Calibration that bought nothing

I built the isotonic step because the band edges are on a probability and a model trained with
`scale_pos_weight` does not produce one. The shipped model is trained without it, and its raw
score on test has a Brier of 0.02186 and an expected calibration error of 0.00246 against 0.02200
and 0.00508 after calibration on validation. The step made the top bin worse: in the top test
bin the calibrated prediction averages 0.281 against an observed rate of 0.2446. It stays, because ADR
0027's rejected strategies would have needed it and the mechanism should not depend on which
strategy won, and the artifact carries both reliability tables so the cost is visible. The lesson I
take is that "calibrate on the held-out slice" is a default that costs something when the split
drifts, and the drift here is stage 2's, not the calibrator's.

## The card resample

The row bootstrap gives the shipped model a half-width of 0.0170; the card bootstrap, whole
entities drawn with replacement, gives 0.0352. Every interval roughly doubles, and one model's
does more: the unweighted random forest's card interval is 0.0600 wide on each side, half again as
wide as any other model's, which is why the widest paired half-width in the artifact, 0.0592, comes from a pair
that includes it. I do not know why the random forest's scores are that much more concentrated on
a few cards than xgboost's; it is recorded, not explained.

The brief defines the noise band as the widest paired half-width and stage 9's margin has to
clear it. Taken literally that is 0.0592, which would let nothing this repo produces ever be
promoted over anything else. The artifact carries every pair so stage 9 can read the half-width
of the pair it compares, and the same-family same-stack pair is 0.0046 by row and 0.0121 by card.
I have written that in the ADR rather than deciding the margin, because deciding it is stage 9's.

## The search that found nothing, on purpose

Twenty-one configurations, three expanding-window folds, sixty-three fits, 542 seconds. The best
beat the default by 0.0031 across the folds, whose scores within one configuration differ by up
to 0.05, and lost to it by 0.0136 on the full train fit. The rule to ship the default when the
tuned configuration does not beat it on the full fit was written before the search ran, and it is
the rule that keeps this from being "tune one model until it wins". The search is small. A larger
one would want more folds than a 120 day window can hold without each fold fitting on a sliver,
and it would want a resolution the fold scores do not have.

## Smaller things

- SMOTE is in-repo and tested against a brute-force neighbour search. I did not want a
  dependency whose null and categorical handling I would have had to state from memory.
- The tree path reads a vocabulary level as its integer code, which is an ordinal a tree splits
  on. XGBoost's native categorical support was not used, so the random forest and xgboost read
  the same matrix. That is the like-for-like the comparison needs and it is a choice, recorded
  here.
- The search process is deterministic: the ablations were run twice with different decision
  code and produced identical per-seed scores to four decimals. Worth knowing when re-running a
  section costs fourteen minutes.
- `addr2_freq` is the second most important feature in the seed-42 fit of the default stack, at
  gain 404. A column that is one value on 0.9901 of rows is doing that work through its other
  0.0099, and ADR 0030 records why it could not be a segmentation key anyway.
- The 36 missing-block indicators take 0.0016 of the shipped model's summed mean absolute SHAP.
  A tree reads the null itself; the indicators exist for the complete-data path and cost the tree
  path nothing but width.
