# Preparation

Stage 2 measured the frame. This stage acts on it, and the rule the whole stage runs on is that
every decision here cites a stage 2 artifact. A column dropped without a number behind it is a
column dropped on a hunch, and there is nowhere in `fraud_platform.cleaning` to put one.

The second rule is that every fitted object is fitted on the train split, and not by convention.
Each fit function is wrapped in `data_loader.train_only`, so the guard runs at the point of
fitting and a fit on a frame that reaches past the training boundary raises. `tests/
test_prep_units.py` calls all seven of them with a straddling frame and requires the raise.

Every number below is read from an artifact. Regenerate the lot with:

```bash
make prep
```

which writes six JSON files and then `make prep-figures`, which draws three PNGs from those files
and from nothing else.

| Section | Artifact |
| --- | --- |
| 1. The column plan | `reports/column_plan.json` |
| 2. Missing values | `reports/prep/missing_policy.json` |
| 3. Transformations | `reports/prep/transforms.json` |
| 4. Encoding | `reports/encoding_spec.json` |
| 5. The V block | `reports/prep/v_reduction.json` |
| 6. The contract | `reports/prep/schema.json` |

Seven decision records were written this stage, ADR 0011 to ADR 0017, and each section names the
one that argues it.

---

## 1. The column plan

434 raw columns in, 288 kept, 146 dropped. Every drop falls under one of four categories and each
category demands its own evidence. ADR 0011 has the argument.

| Rule | Condition | Dropped |
| --- | --- | --- |
| effectively_constant | one value covers at least 0.99 of non-null rows and information value is below 0.02 | 14 |
| missing_and_not_mnar | missing rate at least 0.95 and the column is not label-linked | 0 |
| time_inconsistent | a single-feature model fitted on the whole train window scores below 0.50 on validation | 0 |
| redundant | in a correlation cluster at Spearman 0.95 and not the representative | 132 |

Two of the four are empty, and the empty ones are the two a pipeline written on intuition would
have leaned on hardest.

### The constant rule keeps 16 of the 30 columns it was pointed at

Stage 2 flagged 30 columns as effectively constant and warned that a column which is 0.9958 one
value can still separate the classes on the rest. The rule reads both numbers, and 16 of the 30
clear the rescue line at an information value of 0.02.

| Column | Top value share | Information value |
| --- | --- | --- |
| addr2 | 0.9901 | 0.4200 |
| id_04 | 0.9913 | 0.3215 |
| V240 | 0.9992 | 0.3122 |
| V241 | 0.9998 | 0.3122 |
| M1 | 0.9999 | 0.1841 |
| V1 | 1.0000 | 0.1794 |

The remaining ten rescued columns sit between 0.0213 and 0.1197. All 16 would have gone under a
rule that read only the share.

C3 is the column stage 2 named for the rescue, and it does not clear it. C3 is 0.9958 zero and its
information value against the label is 0.0000. It separates nothing on the 0.42 percent where it
is not zero. The warning was right to be a warning and the measurement settles it the other way.

### The missing rule is empty, at any threshold that means anything

Nine columns are missing on at least 0.95 of train rows: id_07, id_08, id_21 to id_27. All nine
are on stage 2's list of 251 label-linked columns, so all nine stay. At 0.90 the category picks up
three more and they are label-linked too. It only starts removing columns at 0.80, where it would
take 51, and moving a line until a rule does something is not how a line gets chosen.

That is a result rather than a failure. On this frame, a column being nearly empty is not evidence
that it is uninformative, and stage 2 is why that can be said rather than assumed.

### The inversion candidates all survive, and that is the surprise of the stage

ADR 0009 flagged 59 columns that separated the classes over the first 30 days of train and
inverted over the last 30, and said explicitly that turning the list into a decision needed a
measurement against a held-out window. Here it is: a single-feature LightGBM fitted on the whole
120 day training window, scored on validation.

All 59 score at least 0.50. The lowest is id_32 at 0.5022. The highest is id_38 at 0.6704, which
is the column stage 2 called the strongest case for inversion at 0.6037 early against 0.3634 late.

The two measurements ask different questions and both answers are correct. Stage 2 fitted on one
month and scored on a later one, which asks whether a relationship learned from a narrow early
window survives. This stage fits on the whole window, which is what a real model does, and a model
that has seen all 120 days has seen the late behaviour too. The inversion is real inside the
training span and does not survive being averaged over it. Nothing is dropped.

### Redundancy takes 132 columns, and one outside the V block

88 correlation groups, 132 columns removed, which reproduces the number stage 2 predicted.
Representatives are chosen from the members that survived the other three rules and prefer a
column the time consistency screen did not flag, so a cluster never trades a stable column for an
inverted one. A test asserts that.

Outside the V block, stage 2 listed six correlated pairs and only one is actionable. The rule is
coverage: a pair is actionable when its correlation was measured over the whole split. That
selects C8 and C10, whose 0.9710 rests on all 413,378 rows, and C10 goes. It rejects D4 with D12
at a Spearman of exactly 1.0000, because that number rests on 46,966 complete cases while the two
columns differ in missingness on 244,197 rows, and the missingness is most of what either column
carries.

### Two derived columns, decided separately

`card_start_day` and `entity_id` are not raw columns and have no row in the plan. Both are
excluded from the feature set. card_start_day carries the highest PSI in the frame, 1.841 against
test, and is a key component rather than a feature. entity_id is ADR 0015.

**For stage 4:** the frame it inherits is 288 raw columns and the plan is a lookup, not a
suggestion. `cleaning.apply_column_plan` is the only thing that drops a column.

---

## 2. Missing values

There is no global rule and there could not be one. Stage 2 measured 251 of the 372 columns with
nulls to be label-linked, with D7 running at 0.1478 fraud when present against 0.0274 when
missing. Imputing that away is deleting a measurement. ADR 0012 has the argument.

Three policies, and which one a column gets depends on what the model reading it can do.

| Policy | Applies to | What happens |
| --- | --- | --- |
| Leave missing | the tree path | nothing is filled anywhere |
| Impute and indicate | any model needing complete data | train median or sentinel, plus a block indicator |
| Sentinel | the 15 D columns | one unit below the column's train minimum, recorded per column |

`fraud_platform.missing_policy.GROUP_POLICY` records the policy per column group with the stage 2
number behind each row.

### The libraries were checked, not remembered

Leaving nulls alone is a claim about libraries, so it is run rather than recalled, and run twice:
tolerating a null and learning from one are different properties and only the second justifies the
decision. The second probe fits one column that is pure noise where present and null on exactly
the positive rows, so an estimator that learns a direction for the null scores 1.0 and one that
merely tolerates it scores 0.5.

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

Six of eight accept a null and all six learn a direction for it. `RandomForestClassifier` doing so
is the reason the check exists rather than a recollection of it, and `GradientBoostingClassifier`
refusing from the same module is what keeps the check honest. The probe runs on generated data, so
it runs in CI and re-runs whenever a pinned version changes.

### Indicators are per block, not per column

66 indicators cover 244 kept columns: 23 multi-column indicators over 201 columns and 43
single-column. Stage 2's own count was 24 blocks over 392 columns, and the difference is the 146
columns the plan removed, which shrinks several blocks to a single member. A one-member block
still needs its own indicator because its null mask is its own.

The identity join gets a separate one, `has_identity_record`, computed from id_01, which stage 2
named as one of exactly two columns null on precisely the rows where the join fails. Stage 2
measured that join at a fraud rate of 0.07312 against 0.02142, a factor of 3.41, and called it the
largest plain fact in the analysis. It should be a column, not an inference from forty columns
going quiet.

### Never zero

229 columns are zero-inflated and D1 is exactly zero on 0.5013 of its non-null rows, so a zero
fill merges two populations stage 2 spent a section separating. The sentinel sits one unit below
each column's train minimum and is recorded per column. It is not doing the work. The indicator
beside it is, and what a missing D means is labelled a choice in ADR 0012, because nothing in this
repo establishes it.

---

## 3. Transformations

`fraud_platform.transforms` holds the changes to a raw column that need no history to compute. The
line between this module and stage 4 is whether a row's value depends on any other row.

### The D-column normalisation does not work, and the measurement is the deliverable

The argument is good. The D columns are deltas measured backwards from the transaction, so a card
seen on day 30 and again on day 110 carries a D that has grown by 80, and subtracting the delta
from the day index gives an origin that does not move with the calendar.

The rule was written before the numbers were read: apply the normalisation to a column when it
lowers that column's PSI against test, and not otherwise.

It lowers PSI on none of the 15 columns and raises it on all 15. See `figures/d_column_psi.png`.

| Column | PSI raw | PSI normalised | Validation AUC raw | Validation AUC normalised |
| --- | --- | --- | --- | --- |
| D1 | 0.0226 | 1.8304 | 0.6426 | 0.6197 |
| D3 | 0.0352 | 2.2125 | 0.6602 | 0.5337 |
| D5 | 0.0206 | 1.6019 | 0.6688 | 0.5549 |
| D9 | 0.0120 | 1.3919 | 0.6415 | 0.6382 |
| D10 | 0.0746 | 1.2908 | 0.6497 | 0.6282 |
| D15 | 0.0950 | 0.9977 | 0.6499 | 0.6092 |

The full table for all 15 is in the artifact. The before column reproduces stage 2's
`temporal.json` exactly, which is what makes the after column comparable, and a test asserts the
reproduction field by field.

A second measurement was run because PSI answers whether a distribution moved and not whether a
column got better at the job: a single-feature model per column, on the raw delta and on its
origin. Validation AUC falls on 12 of 15. The three that rise do so by 0.0013, 0.0002 and 0.0070,
all inside the 0.0149 an unpaired comparison on this validation split can resolve. The two largest
falls, D3 at 0.1265 and D5 at 0.1139, are not inside anything.

The mechanism inverts the argument. The raw D columns are already stationary, because the observed
population is a mixture of cards at every age and new cards keep arriving at D near zero, and the
mixture is what holds still. An origin is a calendar date, and a later window necessarily contains
later origins. Normalising removes a drift these columns did not have and introduces one they did
not have either. Stage 2 had already seen the endpoint without naming it: `card_start_day`, which
is D1's origin with a floor applied, carries the highest PSI in the frame at 1.841.

ADR 0013 records it. The prepared frame carries the raw D columns.

### Amount

`log1p` plus the cents fraction, on stage 2's numbers: skew 17.639 to 0.499 and excess kurtosis
1,603.6 to 0.765. The raw column stays, because a monetary threshold is expressed in currency and
stage 7 has to show a reviewer the amount that was charged rather than its logarithm.

The cents definition is stage 2's, `round(amount * 100) mod 100`, so the distribution the pipeline
produces is the one `univariate.json` already reports: 0.5255 of rows at 0 and 0.2712 at 95. Two
columns come out of it, the fraction itself and a whole-units indicator.

### Clipping at 0.999, for the models that read a value rather than its rank

Trees get nothing. A tree cuts on order, so a clipped maximum and an unclipped one produce the
same split and the clip only costs information. The bound exists for the stage 5 models that
cannot see a tail.

The quantile is a choice and the cost table is why it is 0.999 and not 0.99. Across TransactionAmt
and the 14 C columns:

| Quantile | Values touched | Fraud values touched | Share of fraud rows touched |
| --- | --- | --- | --- |
| 0.99 | 64,377 | 10,004 | 0.6881 |
| 0.995 | 34,993 | 3,932 | 0.2705 |
| 0.999 | 7,093 | 746 | 0.0513 |
| 0.9995 | 3,252 | 356 | 0.0245 |

A bound at the 99th percentile would touch a value on more than two thirds of the fraud rows in
the split. Stage 2 measured TransactionAmt U-shaped against the label, top decile at a fraud rate
of 0.0509 against a trough of 0.0192, so the tail is not noise to be flattened. At 0.999 the
fitted amount bound is 4.109 to 2,734.99 against a measured maximum of 31,937.391.

### Clock columns

Four derived, one in the feature set. Stage 2 measured the fraud rate running 0.0217 to 0.1001
across hour positions, a ratio of 4.62, and 0.0322 to 0.0379 across day-of-week positions, a ratio
of 1.18. The hour is a feature on that evidence and the day of week is not, though it is derived
because the reports want it. All four are positions in a relative cycle: stage 0 did not establish
what TransactionDT counts from.

---

## 4. Encoding

Four encoders, every one fitted on train alone and every fit routed through the guard.

### The label lag is the decision that matters

A target encoding built from labels that would not have arrived by scoring time is a leak, and a
chargeback matures over weeks. A row on day `d` is encoded from training rows on day `d - LAG` or
earlier and from nothing later. ADR 0014 has the argument.

The lag is not chosen on validation AUC. A longer lag can only remove information, so the best
validation AUC sits at or near lag 0 by construction and choosing on it would choose the leak. The
sweep's useful column is the gap between the in-sample train AUC and the validation AUC: an
encoding summarising a row's own label separates the rows it was fitted on much better than the
rows it was not.

Six lags by eight smoothing values over 18 identifier columns, one fit, 864 points, no model
trained. Averaged over the smoothing grid and the columns:

| Lag, days | Train AUC | Validation AUC | Gap | 95 percent half width | Validation cost against lag 0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.68911 | 0.60794 | +0.08117 | 0.02961 | 0.00000 |
| 1 | 0.65913 | 0.60595 | +0.05319 | 0.02429 | 0.00199 |
| 3 | 0.64902 | 0.60594 | +0.04308 | 0.02307 | 0.00200 |
| 7 | 0.63452 | 0.60309 | +0.03143 | 0.02243 | 0.00486 |
| 14 | 0.60882 | 0.60113 | +0.00768 | 0.01729 | 0.00681 |
| 28 | 0.58946 | 0.59627 | -0.00681 | 0.01362 | 0.01167 |

The gap falls monotonically from 0.081 to below zero. See `figures/target_encoding_lag.png`. Two
lags clear the gap test, 14 and 28, and the stated rule takes the smaller: a lag is a delay on
information and the shortest defensible one keeps the most. Its validation cost is 0.00681,
inside the 0.02185 this split resolves.

Smoothing at lag 14 peaks at 20 with a mean validation AUC of 0.60282, and the sweep does not
resolve it: every value on the grid sits inside the paired interval of the best, which the
artifact records as `resolved: false`. What the sweep does establish is that the ends of the grid
cost something, 500 giving up 0.0064 against 20.

Three states get three different answers. A level counted zero times before the boundary falls to
the prior at that boundary, which is the formula's own answer. A null encodes to null. A row whose
boundary precedes every training row encodes to null, because at that point the window knows
nothing at all. There is no global fallback anywhere.

### The entity key is excluded, and this is the sharpest number in the stage

Stage 2 measured 0.9675 of multi-transaction entities label-pure at the ADR 0001 key. A target
encoding of that key is close to a lookup table from entity to label, and no lag repairs it,
because the labels it reuses are not stale, they are the same entity's.

At the chosen lag the composite validation AUC is 0.7835, which looks like a strong feature. Split
on whether the training window ever saw the row's entity:

| Quantity | Value | Rows |
| --- | --- | --- |
| Validation AUC over the whole split | 0.7835 | 88,581 |
| Validation AUC on rows with a seen entity | 0.9877 | 41,536 |
| Validation AUC on rows with an unseen entity | 0.4915 | 47,045 |
| Validation AUC of the seen-or-unseen indicator alone | 0.4204 | 88,581 |

0.9877 and stage 2's 0.9675 label purity are the same fact. On the rows where an entity history
exists the column recovers the label; on the rows where it does not, the column is a constant and
carries nothing. 0.7835 is the average of a lookup and a constant over two groups sitting at
different base rates, and the seen-or-unseen indicator scores 0.4204 on its own, so the
concentration is the entity's labels rather than the join.

The column is excluded. `encoders.NEVER_TARGET_ENCODED` holds it and `fit_target_encoding` raises,
so the exclusion is enforced rather than remembered. ADR 0015.

Entity history is not banned. Summarising the entity's **labels** is. Stage 4 is free to build
behavioural aggregates, and stage 2 named the strongest candidate already: the median gap between
an entity's consecutive transactions is 3,177 seconds for an entity that ever sees fraud against
165,394 seconds for one that never does, a factor of 52, with no label in the feature at all.

### Frequency encoding

13 identifier columns wide enough to need one, at the 50-level line stage 2 used to flag a column
high-cardinality. An unseen level encodes to 0, which is the measurement and not a fallback: the
training window contains it zero times. A null stays null. Columns narrower than 50 are skipped:
ProductCD takes five levels and every one is a level a model can split on directly.

### The vocabulary, and three tokens that stay apart

32 categorical columns get a bounded level set, with the rare tail collapsed on stage 2's own
rule: sort levels by frequency ascending and collapse while the rows they hold total no more than
0.01 of the column's non-null rows. That makes the threshold a count read off the train
distribution, and it reproduces stage 2's rare-tail table exactly.

| Column | Levels on train | Kept | Collapsed | Rows collapsed | Share |
| --- | --- | --- | --- | --- | --- |
| DeviceInfo | 1,546 | 910 | 636 | 912 | 0.00998 |
| id_33 | 183 | 48 | 135 | 575 | 0.01000 |
| id_31 | 108 | 58 | 50 | 1,035 | 0.00965 |
| id_30 | 71 | 53 | 18 | 604 | 0.00978 |
| R_emaildomain | 60 | 30 | 30 | 993 | 0.00964 |
| P_emaildomain | 59 | 29 | 30 | 3,414 | 0.00978 |

Three tokens are reserved and they mean three different things: a level below the rare line, a
level the training window never saw, and a null. Stage 2 measured all three separately and merging
any two would erase one of its findings.

### DeviceInfo and id_31 normalise structurally

Stage 2 found DeviceInfo to be free text at 1,546 levels with three naming schemes in the top
seven values, and id_31 at PSI 1.500 against test with 19 levels test holds that train does not.

The normalisation uses no brand table. This repo has not established what any of these strings
names, and asserting it would be a fact from outside the repo. So: lowercase, drop the `Build/...`
suffix, collapse the `rv:` and `Trident/` engine tokens, and otherwise take the first token before
a separator with its digits stripped. `SM-J700M Build/MMB29K` and `SM-G610M Build/MMB29K` land
together on `sm`, and this module does not claim to know whose devices those are. id_31 loses its
version and keeps family and platform: `chrome 63.0 for android` becomes `chrome|android`.

| Column | Levels on train | After normalisation | After the rare collapse |
| --- | --- | --- | --- |
| DeviceInfo | 1,546 | 281 | 64 |
| id_31 | 108 | 43 | 11 |

The coverage that buys, on test:

| Column | Present rows | Rows on an unseen level, raw | Rows on an unseen level, normalised |
| --- | --- | --- | --- |
| DeviceInfo | 15,091 | 500 | 30 |
| id_31 | 17,842 | 8,068 | 149 |

id_31 is the case ADR 0007 turned on and the normalisation closes it: 8,068 test rows that carry a
browser string the training window never saw become 149. The raw column is kept alongside, so
nothing is lost and stage 7 can still show a reviewer the string that arrived.

---

## 5. The V block

Stage 2 clustered the V columns two ways: 14 missingness blocks whose members are null on
identical rows, and correlation groups inside them that reduce 339 columns to 207 representatives.
Structure is not a strategy, so three reductions and the unreduced baseline were measured rather
than one being picked. ADR 0016.

One small LightGBM per strategy, on the V columns alone, fitted on train and scored on validation.
It is a probe and no number here is a performance estimate.

| Strategy | Columns | Train AUC | Validation AUC | Across three seeds |
| --- | --- | --- | --- | --- |
| all | 327 | 0.8927 | 0.8406 | 0.8372 to 0.8414 |
| representative | 196 | 0.8904 | 0.8378 | 0.8362 to 0.8414 |
| pca | 122 | 0.9026 | 0.8466 | 0.8440 to 0.8466 |
| block_mean | 14 | 0.8791 | 0.8375 | 0.8354 to 0.8380 |

The four strategies span 0.0090 in validation AUC. One strategy spans 0.0051 across three seeds on
identical rows and identical columns. The probe does not separate them, which is why the seed
variance was measured at all, and the stated rule then chose on its tie-break: among strategies
inside the resolvable band, the fewest columns.

The block mean ships. 327 columns become 14, for a validation AUC difference of 0.0031 against the
unreduced baseline that this split cannot resolve. See `figures/v_reduction.png`.

What it costs is worth saying plainly. The PCA scored highest on every seed, and a rule with a
different tie-break would have taken it. And nothing in the V block is nameable any more: stage 7
explaining a score can say that the mean of a block was high, not that V258 was, and stage 2
measured V258's upper bin at a fraud rate of 0.6076 on 4,348 rows. That specific fact is now
invisible to an explanation. The decision is provisional in a stated way: `fit_preparation` reads
the strategy from the artifact, so stage 5 can re-run the comparison with a real model and a
paired test, which resolves more than an unpaired probe does.

---

## 6. The contract

A declared schema for the prepared frame, validated on every load, raising with every violation
rather than the first. ADR 0017.

The prepared train frame is 247 columns: 92 raw columns that survived the plan and the V
reduction, and 155 derived. 89 integer, 93 float, 65 categorical. All three splits validate with
zero violations, and the pipeline applied twice from the same fitted objects produces the same
frame value by value, checked by a SHA-256 fingerprint that a `needs_data` test reproduces from a
fresh refit.

What is constrained is structure: the exact column set, the dtype family, nullability on four
columns, and the sign bounds stage 2 measured. What is deliberately not constrained is
distribution. Stage 2 measured 130 of the 393 numeric columns moving a median or a 99th percentile
by more than 10 percent when the later splits are folded in, with V332 going from 1,650 to 60,335,
so a train-fitted cap would reject legitimate later rows for being later, and it would do it at
exactly the moment stage 9 wants to see them. No category allow-list either, for the mirror-image
reason: stage 2 measured 240 DeviceInfo levels and 22 id_31 levels appearing only after the
training window, and the encoders have an explicit unseen token for that.

The bug the validator caught on its first run is the best evidence it works. Every derived column
initially inherited the raw-column rule that stage 2 observed no negative, and all 14 block means
failed on all three splits, each reporting a large count of values below the declared minimum of
zero. A standardised mean is signed by construction. Bounds for derived columns now come from the
arithmetic that produces them, and `schema.DERIVED_BOUNDS` records the reason per entry.

`schema.NOT_CAUGHT` lists seven things the contract does not check, including drift, unseen
category levels, cross-column consistency and a wrong label. It is a stated limit rather than a
gap somebody finds later.

---

## 7. Figures

All three are drawn from the artifacts above and from nothing else, at 200 DPI, by
`make prep-figures`.

| Figure | Source | Shows |
| --- | --- | --- |
| `figures/d_column_psi.png` | transforms.json | D column PSI and validation AUC, raw against normalised |
| `figures/target_encoding_lag.png` | encoding_spec.json | what the label lag costs and what it removes |
| `figures/v_reduction.png` | v_reduction.json | column count against validation AUC for four strategies |

---

## What stage 4 inherits

| Finding | Number | Consequence |
| --- | --- | --- |
| Column plan | 434 raw, 288 kept, 146 dropped | `apply_column_plan` is the only thing that drops a column |
| Two drop categories are empty | 0 missing, 0 time-inconsistent | the candidate lists did not survive the measurement they asked for |
| Nulls stay nulls | 6 of 8 estimators learn a direction | the tree path receives the frame unimputed |
| D normalisation | PSI worse on 15 of 15 | raw D columns, transformation implemented and unused |
| Amount | skew 17.639 to 0.499 | log1p plus cents, raw column kept |
| Clipping | 0.99 touches 0.6881 of fraud rows | bound at 0.999, trees exempt |
| Label lag | gap 0.08117 at lag 0, 0.00768 at lag 14 | 14 days, and stage 8 has to honour it |
| Entity target encoding | 0.9877 on seen rows, 0.4915 on unseen | excluded, enforced by a raise |
| V reduction | four strategies within 0.0090, seeds within 0.0051 | block mean, 14 columns, provisional |
| Free text | id_31 unseen test rows 8,068 to 149 | normalise before encoding |
| Prepared frame | 247 columns, deterministic | schema validated on every load |
| Cold entities | 0.6598 of test rows | entity-history features need a defined cold behaviour |

## What is not established

- How long a chargeback actually takes to mature. The lag sweep measures the price of each choice,
  not the right answer, and stage 0 did not establish the host's labelling rule.
- What a missing D column means. The sentinel's meaning is a reading this stage chose and labelled
  as a choice.
- Whether the V-reduction strategies differ at all. The probe could not separate them and the
  tie-break chose. A paired comparison with a real model in stage 5 would resolve more.
- Run-to-run variance on the single-feature probes used for the column plan and the D columns. One
  seed, one setting, exactly as ADR 0009 noted about the screen they borrow.
- Whether the 0.02 information value rescue line is in the right place. 16 columns depend on it and
  the plan records all 16 with their values.
