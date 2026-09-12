# Exploratory analysis

Everything below is measured on the **train split**: 413,378 rows, day index 1 to 120, every
TransactionDT below 10,437,998. It is not a description of the dataset. Three places look past
that boundary and each says so where it appears: the per-split rates in section 1, the drift
measurements in section 6, and section 0, which exists to show what reading the whole file would
have cost. ADR 0007 has the reasoning.

Every number here is read from an artifact under `reports/eda/`. Nothing is retyped from a
notebook. Regenerate the lot with:

```bash
make eda
```

which writes nine JSON files and then `make eda-figures`, which draws six PNGs from those files
and from nothing else.

| Section | Artifact |
| --- | --- |
| 0. What train-only costs | `reports/eda/train_only.json` |
| 1. The target | `reports/eda/target.json` |
| 2. Univariate structure | `reports/eda/univariate.json` |
| 3. Missingness | `reports/eda/missingness.json` |
| 4. Redundancy and correlation | `reports/eda/correlation.json` |
| 5. Bivariate signal | `reports/eda/bivariate.json` |
| 6. Temporal behaviour | `reports/eda/temporal.json` |
| 7. Entity structure | `reports/eda/entities.json` |
| 8. Duplicates and data quality | `reports/eda/quality.json` |

Each section ends with what it hands to stage 3. Nothing here changes a value: this stage
measures and recommends, and preparation is the next stage's job.

---

## 0. What computing this on train only costs, and what it saves

The rule has a price and it is worth stating first, because every number in the rest of this
document is smaller or larger than the same number computed over all 590,540 rows.

130 of the 393 numeric feature columns have a median or a 99th percentile that moves by more than
10 percent when val and test are folded in. The extreme case is V324, whose 99th percentile is
7.0 on train and 457.0 over all rows. V322 moves from 5.0 to 231.0, V323 from 15.0 to 594.5, and
V332 from 1,650.0 to 60,335.0.

The categorical case is not a matter of degree. Four columns take levels after the training
window that they never take inside it: DeviceInfo takes 240 such levels, id_33 takes 77, id_31
takes 22, id_30 takes 4. For id_31 those levels cover 8,075 rows, 0.0456 of the 177,162 rows in
val and test together.

That last one is the example worth carrying forward. Read over the whole file, id_31 is a
high-cardinality string with a long tail, and a frequency floor fitted there would keep the tail
levels because they are present. Read on train, it is a column whose vocabulary turns over
between the first month and the last, and any handling of it has to work for a level that does
not exist yet.

**For stage 3:** rare-level handling is a fitted rule, not a list. Any cap, floor or bin edge
comes from the train distribution and is applied unchanged to later windows.

---

## 1. The target

### Rate

| Split | Rows | Fraud | Rate | 95 percent bootstrap |
| --- | --- | --- | --- | --- |
| train | 413,378 | 14,538 | 0.03517 | 0.03461 to 0.03573 |
| val | 88,581 | 3,042 | 0.03434 | 0.03314 to 0.03554 |
| test | 88,581 | 3,083 | 0.03480 | 0.03361 to 0.03601 |

The point estimates reproduce `reports/split_summary.json` exactly, which the artifact checks
field by field. The intervals are new here. Wilson and a 20,000 draw bootstrap agree on the train
rate to within 2e-4, and a test asserts that they do.

### Is the rate stationary across the train span? No.

Three tests on the 18 weekly rates, because they ask three different questions.

| Test | What it asks | Result |
| --- | --- | --- |
| Mann-Kendall (Kendall tau) | is the series monotone | tau 0.399, p 0.0214 |
| Least squares on the week index | how big is a linear trend | slope 0.000955 per week, p 0.00468, r squared 0.403 |
| Chi-square homogeneity | do the weeks share one rate at all | chi-square 847.5 on 17 degrees of freedom, p 3.4e-169 |

The weekly rate runs from 0.0207 to 0.0507, a range of 0.0300 and a ratio of 2.45 between the
highest week and the lowest. The last week of the train window holds one day, so the same three
tests are reported again over the 17 complete weeks: tau 0.382 at p 0.0341, slope 0.00100 per
week at p 0.00779. Dropping the partial week does not change the finding.

Daily, the rate runs from 0.0110 to 0.0699 across 120 days, Kendall tau 0.302 at p 1.03e-06.

The honest reading is that the trend test and the homogeneity test disagree about how interesting
this is. The upward drift is real but modest: an r squared of 0.403 means a straight line through
the weekly rates leaves most of the variation unexplained. What the chi-square is reacting to is
that a 2.45 ratio between the best and worst week is far outside sampling noise at 20,000 rows a
week. The train window does not have one fraud rate. It has a rising floor with weeks that bounce
around it.

See `figures/fraud_rate_over_time.png`.

### Rate by category

| ProductCD | Rows | Fraud rate |
| --- | --- | --- |
| W | 297,911 | 0.02078 |
| C | 50,006 | 0.11175 |
| R | 30,144 | 0.03497 |
| H | 28,112 | 0.04518 |
| S | 7,205 | 0.06037 |

ProductCD C runs at 5.4 times the rate of ProductCD W and carries 0.121 of the split.

| Column | Level | Rows | Fraud rate |
| --- | --- | --- | --- |
| card4 | visa | 270,078 | 0.0346 |
| card4 | mastercard | 130,967 | 0.0356 |
| card4 | discover | 4,763 | 0.0657 |
| card4 | american express | 6,741 | 0.0280 |
| card6 | debit | 303,147 | 0.0241 |
| card6 | credit | 109,361 | 0.0659 |
| DeviceType | mobile | 41,434 | 0.0986 |
| DeviceType | desktop | 65,963 | 0.0593 |
| DeviceType | (missing) | 305,981 | 0.0214 |

card6 also holds two levels that are too small to say anything about: "debit or credit" at 29
rows and "charge card" at 15, both with zero fraud. Their Wilson intervals reach 0.117 and 0.204
respectively, which is the point.

DeviceType missing is not a level of a device: it is the identity join failing, and it is covered
properly in section 3.

### Rate by position in the day and the week

TransactionDT is an offset from an origin this repo has not established, so "hour" here means
position inside an 86,400 unit cycle and "day of week" means position inside a seven day cycle.
Stage 0 established that the 86,400 cycle carries the diurnal shape of card activity.

Hour position matters and day-of-week position does not. Fraud rate by hour position runs from
0.0217 at position 13 to 0.1001 at position 8, a ratio of 4.62. Positions 8, 9 and 10 are the
three thinnest hours of the day at 1,969, 1,580 and 1,967 transactions, and they carry the three
highest fraud rates at 0.1001, 0.0956 and 0.0747. The rate rises exactly where legitimate volume
collapses. By day-of-week position the rate runs from 0.0322 to 0.0379, a ratio of 1.18,
and the largest cell is 72,198 rows.

### Class balance and what it allows

14,538 positives in train, 3,042 in val, 3,083 in test. ADR 0010 works through the consequences.
In short: at these counts two independent evaluations can resolve an AUC difference of about
0.0148 on test at an assumed true AUC of 0.90, or 0.0109 at an assumed 0.95, and a recall read
off either later split carries a 95 percent half width of about 0.017. Any comparison closer than
that has not been made.

**For stage 3:**

- The label is not stationary within train. Anything fitted on the train split as a whole
  averages a rate that moves by a factor of 2.45 across it. This does not forbid it, but it is
  the reason a time-aware validation inside train is worth more than a random one.
- ProductCD, card6 and DeviceType are strong low-cardinality splits and need no rare-level work.
- Hour position is worth deriving as a feature. Day-of-week position is not, on this evidence.
- card6 needs its two microscopic levels folded into a rare bucket or left as their own level and
  never quoted alone.

---

## 2. Univariate structure

435 columns profiled on train: 404 numeric, 31 categorical. Full per-column tables are in
`reports/eda/univariate.json` under `numeric` and `categorical`, each carrying count, missing
rate, min, max, mean, median, standard deviation, skew, excess kurtosis and the 1st, 5th, 25th,
50th, 75th, 95th and 99th percentiles.

### The frame is extremely skewed

Over the 401 numeric feature columns the median absolute skew is 10.73 and the 90th percentile is
56.29. 359 columns have |skew| above 1, 268 above 5, 212 above 10 and 22 above 100. The largest
is V305 at 371.2 with excess kurtosis 137,785. Only 26 of the 401 are close to symmetric at
|skew| below 0.5.

This is measured, not assumed, and it decides the correlation method in section 4 (ADR 0008).

### Degenerate columns

| Flag | Rule | Count |
| --- | --- | --- |
| Effectively constant | one value covers more than 0.99 of non-null rows | 30 |
| Single value dominated | one value covers more than 0.95, below 0.99 | 50 |
| Zero-inflated | zero covers more than 0.50 of non-null rows | 229 |
| All null | no non-null value on train | 0 |

The thresholds are choices and the artifact records them. The 30 effectively constant columns are
C3, M1, V1, V14, V27, V28, V41, V65, V68, V88, V89, V107, V108, V110 to V114, V117 to V122, V240,
V241, V305, addr2, id_04 and id_27.

Three of them are worth naming individually. V107 takes one value, 1.0, on every non-null row:
it is constant, not merely dominated. addr2 is 0.9901 the single value 87.0 across 67 distinct
values. C3 is 0.9958 zero with a maximum of 24.

229 zero-inflated columns is not a defect, it is what a frame of counters looks like. It does
mean that a mean or a standard deviation over most of these columns describes the few non-zero
rows and says nothing about the majority.

### TransactionAmt

| Quantity | Value |
| --- | --- |
| Skew, raw | 17.639 |
| Skew, log1p | 0.499 |
| Excess kurtosis, raw | 1,603.6 |
| Excess kurtosis, log1p | 0.765 |
| Minimum | 0.251 |
| Median | 68.95 |
| 99th percentile | 1,104.0 |
| Maximum | 31,937.391 |

log1p takes the skew down by a factor of 35 and the excess kurtosis from 1,604 to 0.77. See
`figures/amount_distribution.png`.

The cents fraction takes all 100 possible values. It is not uniform and not close to it: 0.5255
of rows end in .00, 0.2712 end in .95, 0.0383 in .50, 0.0134 in .97 and 0.0102 in .99. The
remaining 95 values share 0.1414 of the rows between them. A cents fraction of .95 covering more
than a quarter of transactions is a currency conversion artefact or a pricing convention, and
which it is has not been established here.

Round and not round behave differently, but less than the split suggests:

| Amount | Rows | Share | Median amount | Fraud rate | 95 percent Wilson |
| --- | --- | --- | --- | --- | --- |
| Whole units | 217,245 | 0.5255 | 92.00 | 0.03631 | 0.03554 to 0.03711 |
| Fractional | 196,133 | 0.4745 | 57.95 | 0.03390 | 0.03311 to 0.03471 |

The intervals do not overlap, so the difference is real, and it is 0.0024 in absolute terms
against a base rate of 0.0352. The larger difference is in the amounts themselves: a median of
92.00 against 57.95.

### Categorical columns

| Column | Cardinality | Top value | Top share | Categories in the rare tail | Appearing once | Missing rate |
| --- | --- | --- | --- | --- | --- | --- |
| DeviceInfo | 1,546 | Windows | 0.4013 | 636 | 399 | 0.7790 |
| id_33 | 183 | (screen size) | 0.2329 | 135 | 58 | 0.8609 |
| id_31 | 108 | chrome 63.0 | 0.2034 | 50 | 10 | 0.7406 |
| id_30 | 71 | (operating system) | 0.2590 | 18 | 0 | 0.8505 |
| R_emaildomain | 60 | gmail.com | 0.4141 | 30 | 0 | 0.7507 |
| P_emaildomain | 59 | gmail.com | 0.4561 | 30 | 0 | 0.1555 |
| ProductCD | 5 | W | 0.7207 | 0 | 0 | 0.0 |
| card6 | 4 | debit | 0.7348 | 2 | 0 | 0.0020 |
| M1 | 2 | T | 0.9999 | 1 | 0 | 0.5380 |
| id_27 | 2 | Found | 0.9977 | 1 | 0 | 0.9905 |

"Categories in the rare tail" counts how many levels it takes, from the least frequent upwards,
to cover 1 percent of the non-null rows. For DeviceInfo that is 636 of 1,546 levels. For
P_emaildomain and R_emaildomain it is 30 of about 60, which is half the vocabulary carrying one
row in a hundred.

Cardinality is counted on levels that appear in train. The categorical dtype carries the level
set from the whole file, so the empty levels are dropped before counting; not doing so overstates
DeviceInfo at 1,786 instead of 1,546.

**For stage 3:**

- Model TransactionAmt on the log1p scale, and keep the raw column alongside it, since the raw
  scale is what a monetary threshold is expressed in.
- Derive a cents feature. Whether it is the fraction itself, an is-round indicator, or a
  three-level split for .00, .95 and everything else, the distribution above is what the choice
  should be made from.
- The 30 effectively constant columns are candidates for removal. Check them against section 5
  before dropping: a column that is 0.9958 one value can still separate the classes on the other
  0.42 percent, and C3 is exactly that shape.
- The rare tail is large in six columns. A frequency floor fitted on train is the mechanism, and
  section 0 says why a list of known levels is not.
- Do not standardise on mean and standard deviation without checking skew per column. On 212
  columns those two numbers do not describe the distribution.

---

## 3. Missingness

435 columns on train: 372 carry at least one null, 63 carry none. 229 columns are missing on more
than half their rows and 12 on more than 0.9 of them.

### Columns go missing in blocks

Grouping columns whose null mask is identical row for row gives 67 distinct patterns. 24 of those
patterns are shared by more than one column, and those 24 cover 392 of the 435 columns. The
largest single block is the 63 columns with no nulls at all.

The V block does cluster the way it was expected to, and the expectation is now measured rather
than assumed. The multi-column blocks with nulls, largest first:

Every block of eight columns or more:

| Representative | Columns | Missing rate |
| --- | --- | --- |
| C1 | 63 | 0.0 |
| V217 | 46 | 0.7560 |
| V279 | 32 | 0.000029 |
| V167 | 31 | 0.7422 |
| V138 | 29 | 0.8423 |
| V12 | 23 | 0.1481 |
| V53 | 22 | 0.1513 |
| V75 | 20 | 0.1654 |
| V169 | 19 | 0.7422 |
| V35 | 18 | 0.2957 |
| V322 | 18 | 0.8416 |
| V220 | 16 | 0.7384 |
| D1 | 13 | 0.00066 |
| D11 | 12 | 0.5462 |
| id_11 | 8 | 0.7398 |

The C1 block is the 63 columns with no nulls anywhere. The V279 block is 32 columns missing on
the same 12 rows, and the D1 block is 13 columns missing on the same 273: both are blocks by the
exact-mask rule and neither is missingness in any useful sense.

Two blocks share a missing rate of 0.7422 (V167 with 31 columns and V169 with 19) and are still
two blocks: they are missing at the same rate on different rows. That is why the grouping is done
on the exact mask and not on the rate.

`figures/missingness_blocks.png` shows, for each pair of blocks, the share of train rows missing
in both.

### The identity table is not one block

The natural reading is that a row joins `train_identity.csv` or it does not, so its 40 columns
arrive together. Measured, that is false. Those 40 columns take 26 distinct masks, and exactly
two of them, id_01 and id_12, are null on precisely the rows where the join fails. The rest carry
additional nulls inside joined rows: id_03 and id_04 are missing on 0.8811 of the split against a
join failure rate of 0.7341, and id_24 on 0.9913.

The join itself, measured from TransactionID membership rather than from any column:

| | Rows | Fraud rate | 95 percent Wilson |
| --- | --- | --- | --- |
| Identity row present | 109,911 (0.2659) | 0.07312 | 0.07160 to 0.07468 |
| Identity row absent | 303,467 (0.7341) | 0.02142 | 0.02091 to 0.02194 |

A row with an identity record is 3.41 times as likely to be fraud. That is the single largest
plain fact in this document, and it is a property of a join, not of a feature.

### Missingness carries the label

For each of the 372 columns with nulls, the fraud rate when present against the fraud rate when
missing. A column is called label-linked when the absolute gap is at least 0.010 and the
two-proportion test clears p < 1e-4. Both gates are needed: at 413,378 rows the test alone flags
gaps of 0.002.

**251 of 372 columns are label-linked.** The largest gaps:

| Column | Rows missing | Fraud rate missing | Fraud rate present | Gap |
| --- | --- | --- | --- | --- |
| D7 | 386,632 | 0.0274 | 0.1478 | -0.1204 |
| D14 | 368,410 | 0.0257 | 0.1131 | -0.0874 |
| addr1 | 47,518 | 0.1125 | 0.0251 | +0.0873 |
| addr2 | 47,518 | 0.1125 | 0.0251 | +0.0873 |
| D12 | 366,089 | 0.0252 | 0.1122 | -0.0870 |
| D13 | 369,305 | 0.0264 | 0.1086 | -0.0822 |
| D6 | 360,808 | 0.0255 | 0.1015 | -0.0760 |
| id_03 | 364,211 | 0.0264 | 0.1001 | -0.0737 |
| D8 | 357,191 | 0.0254 | 0.0975 | -0.0722 |

The two directions are both present and they mean different things. For the D columns, having a
value at all is the risky state: D7 present carries a fraud rate of 0.1478 against 0.0274 when
missing. For addr1, the missing state is the risky one: 0.1125 against 0.0251.

The list itself is the deliverable. `reports/eda/missingness.json` under
`against_target.label_linked_columns` holds all 251.

**For stage 3:**

- Imputation without a was-missing indicator destroys measured signal on 251 columns. Either
  keep an indicator per column, or per missingness block, which is cheaper: 24 indicators cover
  392 columns and the block structure says they would carry the same information.
- Prefer a model that handles nulls natively. LightGBM and XGBoost both do, and on this frame the
  alternative is 251 indicator columns.
- The identity join deserves its own explicit indicator, separate from any per-column one. It is
  a 3.41 times risk factor and it is currently only visible through 40 columns going quiet.
- Do not treat DeviceType missing as a device type. It is the join.

---

## 4. Redundancy and correlation

Spearman throughout, on the skew measured in section 2. ADR 0008 has the argument and the
implementation, which screens all 80,200 pairs of the 401 numeric columns cheaply and then
recomputes the 1,762 that clear |rho| = 0.80 exactly with pandas.

### The V block collapses by more than a third

Grouping runs inside each missingness block, so two V columns never observed on the same rows
cannot be called redundant on a handful of them. At |rho| >= 0.95, 339 V columns across 14 blocks
reduce to 207 representatives. **A representative-per-group reduction would remove 132 columns**,
0.389 of the V block.

The blocks that collapse hardest are not the largest. The 29-column block at missing rate 0.8423
goes to 12 groups (17 removed); the 18-column block at 0.8416 goes to 8 (10 removed); the
46-column block at 0.7560 goes to 32 (14 removed).

### Near-duplicate pairs

127 pairs sit at |rho| >= 0.99 and 508 at |rho| >= 0.95. Inside the V block the strongest are
V286 with V311 at 0.9998, V141 with V161 at 0.9998, V142 with V163 at 0.9998 and V92 with V93 at
0.9998.

Six pairs above 0.95 involve no V column at all:

| Pair | rho | Complete cases |
| --- | --- | --- |
| D4, D12 | 1.0000 | 46,966 |
| D5, D7 | 0.9985 | 24,571 |
| D6, D12 | 0.9939 | 46,962 |
| D1, D2 | 0.9813 | 207,044 |
| D4, D6 | 0.9803 | 49,849 |
| C8, C10 | 0.9710 | 413,378 |

D4 and D12 rank identically on every one of their 46,966 complete cases. That is 0.114 of the
split: D4 is present on 291,163 rows and D12 on 47,289, so on 244,197 rows D4 has a value and D12
does not. C8 and C10 at 0.9710 is the only one of the six measured on the whole split.

**For stage 3:**

- The 132-column V reduction is available and should be taken, with the group memberships from
  `reports/eda/correlation.json` recorded so the choice of representative is reviewable. Check the
  list against section 6 first: dropping a column that survived the time consistency screen in
  favour of one that failed it would be worse than keeping both.
- Do not drop D12 for D4 on a correlation measured over a ninth of the split. Their missingness
  differs by 244,197 rows and that is most of what either column contributes.
- C8 and C10 at 0.9710 over all rows is the one clean case for dropping one of the pair.

---

## 5. Bivariate signal

Information value per feature against the label, computed on train, with missing as its own level
and a 0.5 event smoothing. Numeric columns are binned into deciles of the train distribution;
categorical columns bin on their own levels.

Because IV sums over bins, the missing bin's term comes out exactly, so every column carries
three numbers: total IV, the missing bin's contribution, and the rest. That decomposition turns
out to matter more than the ranking.

### The distribution of information value

| Band | Features |
| --- | --- |
| IV above 0.5 | 55 |
| 0.1 to 0.5 | 198 |
| 0.02 to 0.1 | 88 |
| Below 0.02 | 91 |

### The top of the table is mostly the identity join

| Column | IV | Missing bin | Present bins | Missing rate |
| --- | --- | --- | --- | --- |
| V258 | 0.806 | 0.112 | 0.693 | 0.756 |
| V257 | 0.768 | 0.112 | 0.656 | 0.756 |
| V189 | 0.754 | 0.146 | 0.608 | 0.742 |
| DeviceInfo | 0.750 | 0.088 | 0.662 | 0.779 |
| V201 | 0.749 | 0.146 | 0.603 | 0.742 |
| V200 | 0.736 | 0.146 | 0.589 | 0.742 |

V258 takes three bins: 312,530 rows missing at a fraud rate of 0.0233, 96,500 rows in the lower
present bin at 0.0478, and 4,348 rows in the upper bin at **0.6076**. A bin of 4,348 rows running
at 60 percent fraud is the single sharpest cut in the frame, and it sits on 0.0105 of the split.

DeviceInfo at 0.750 is a different story and should not be read the same way. It bins into 1,547
levels, and information value rewards granularity as much as separation. Six columns are flagged
`high_cardinality` for this reason.

Four columns with no missingness at all reach the top by value alone: C4 at 0.563, C8 at 0.553,
C10 at 0.523 and ProductCD at 0.516. Those four are the cleanest signal in the frame in the sense
that nothing about their availability is doing the work.

### Entity-identifying columns are reported separately

Information value on a column that identifies the entity is inflated by the label clustering
stage 0 measured: 0.9664 of multi-transaction entities are label-pure at the chosen key. Six
columns carry the flag, being the ADR 0001 key, its sources, and the row identifier.

| Column | IV | Missing bin | Present bins |
| --- | --- | --- | --- |
| addr1 | 0.4257 | 0.3260 | 0.0996 |
| card_start_day | 0.2501 | 0.0011 | 0.2490 |
| D1 | 0.2046 | 0.0011 | 0.2035 |
| card1 | 0.0573 | 0.0000 | 0.0573 |

addr1 is the clearest case for the decomposition. Its total IV of 0.4257 ranks it 81st of 432.
0.3260 of that is the missing bin, and recomputed on present rows alone it falls to 0.0118. addr1
is not a strong predictor: the absence of addr1 is.

card_start_day at 0.2501 with almost nothing from missingness is a real ordering, and it is also
the single most drifted feature in section 6, which is what a derived time offset should be
expected to do.

### Shape, not just strength

`figures/fraud_rate_by_decile.png` shows eight columns chosen by name rather than by rank, so the
shapes are readable. The tables are in `reports/eda/bivariate.json` under `deciles_reference`;
the top-ranked columns get theirs under `deciles`.

The shapes differ enough to matter:

- **TransactionAmt** is U-shaped. The lowest decile runs 0.0551 and the highest 0.0509, with a
  trough of 0.0192 in the fourth. A monotone treatment of the amount would miss both ends.
- **C1** is monotone increasing, 0.0245 in the lowest bin to 0.0862 in the highest.
- **C13** is monotone decreasing, 0.0438 to 0.0205.
- **D15** is neither: 0.0454 missing, then 0.0442, 0.1036, 0.0438, and a decline to 0.0129. The
  0.1036 bin holds 11,988 rows.
- **dist1** is missing on 255,775 rows at 0.0444, then dips to 0.0128 and climbs back to 0.0313.

**For stage 3:**

- Do not rank features on raw information value. Rank on the present-bins term, and treat the
  missing-bin term as belonging to the missingness indicator that section 3 already asks for.
- addr1 is a missingness feature dressed as a value feature. Whatever encoding it gets, the
  indicator is the part that carries the signal.
- The amount needs a treatment that survives a U shape: a binned or monotone-free model, or
  explicit low and high tail features.
- Six high-cardinality columns need their information value re-read after whatever rare-level
  handling stage 3 applies, because the current number partly measures the number of levels.

---

## 6. Temporal behaviour

### Drift baseline

PSI for all 432 features, train against val and train against test. Bins are deciles of the train
distribution with missing as its own level, and the later split is scored against those bins and
never rebinned. This is the baseline stage 9 calibrates monitoring against.

The frame is mostly stable and the exceptions are concentrated. Median PSI against test is 0.0207.
30 features exceed 0.10 and 22 exceed 0.25. Against val, 35 exceed 0.10 and 9 exceed 0.25.

| Feature | PSI val | PSI test | Note |
| --- | --- | --- | --- |
| card_start_day | 1.448 | 1.841 | derived from TransactionDT, so it moves by construction |
| id_31 | 0.938 | 1.500 | 19 levels in test that train never saw |
| id_13 | 0.707 | 0.563 | |
| D11 | 0.244 | 0.331 | |
| M7, M8, M9 | 0.299 to 0.300 | 0.330 to 0.331 | |
| V1 to V11 | 0.211 to 0.212 | 0.292 to 0.297 | |
| M1, M2, M3 | 0.313 to 0.314 | 0.279 to 0.280 | |
| id_30 | 0.196 | 0.266 | |

card_start_day at 1.841 is the loudest and the least interesting: it is `floor(TransactionDT /
86400 - D1)`, so a later window necessarily contains later start days. It is a key component, not
a feature, and the drift is a statement about the calendar.

id_31 at 1.500 is the interesting one, and it is the case ADR 0007 turns on. Browser version
strings turn over: 19 levels present in test do not exist in train.

`figures/psi_train_vs_test.png` shows the top 25 and the distribution over all 432.

### The time consistency screen

For each feature, a LightGBM model on that column alone, fitted on the first 30 days of train
(134,339 rows, 3,401 fraud) and scored on the last 30 days of train (97,451 rows, 3,836 fraud). A
column is flagged when it separates the classes early (AUC at least 0.51, in sample) and inverts
late (AUC below 0.50). ADR 0009 has the reasoning and the full flag list.

**59 of 432 columns are flagged.** They fall into three groups: 15 in the V138 to V166 range
(early 0.576 to 0.618, late 0.468 to 0.498), all 18 of V322 to V339 (early 0.569 to 0.588, late
0.467 to 0.489), and 26 others including V14, V19, V20, V25 to V28, V41, V46, V55, V56, V61, V62,
V65 to V68, V88, V89, V285, card4, id_14, id_26, id_32, id_34 and id_38.

The strongest is id_38 at 0.6037 early and 0.3634 late.

**The intersection between the 59 flagged features and the 22 features whose PSI exceeds 0.25 is
empty.** Drift and inversion are different failures and neither screen finds the other's. See
`figures/time_consistency.png`.

### Cold entities point two different ways

This is the finding in this stage that most contradicts what was expected.

Inside the train window, an entity's first transaction is **safer** than its later ones: 164,702
cold rows at a fraud rate of 0.0263, against 248,676 warm rows at 0.0410.

Across the boundary, the direction reverses. Stage 1 measured that test rows on an entity the
training window never saw run at 0.0440, against 0.0170 for rows on entities it did see
(`reports/split_summary.json`, `entity_overlap.test`), with 0.6598 of test rows on unseen
entities.

Both are true and they are not the same measurement. "First time we have seen this entity, in a
window that also contains its later transactions" and "an entity that appears only after the
model was trained" describe different populations. The first is the ordinary case of a new
customer whose history the data does contain. The second is what a deployed model actually faces,
and it is the risky one.

**For stage 3:**

- card_start_day is a key component. Do not pass it to a model as a feature without an explicit
  decision, and if it goes in, it goes in knowing its PSI is 1.841.
- id_31 needs normalising before encoding: the browser family is stable and the version is not.
  Splitting it is the obvious move and the artifact quantifies what it would buy.
- The 59 flagged columns are candidates for removal. It is a candidate list and not a decision:
  removing a feature has to be measured.
- Entity-history features are missing for 0.66 of test rows by construction. Whatever stage 4
  builds on entity history needs a defined behaviour for a cold entity, and that behaviour will
  be exercised on two thirds of the evaluation set.

---

## 7. Entity structure

At the ADR 0001 key, `card1 + addr1 + floor(TransactionDT / 86400 - D1)`, the train split holds
164,702 entities over 413,378 rows.

| Quantity | Value |
| --- | --- |
| Transactions per entity, mean | 2.510 |
| Median | 1.0 |
| 75th percentile | 2.0 |
| 95th percentile | 8.0 |
| 99th percentile | 18.0 |
| Maximum | 319 |
| Entities with exactly one transaction | 0.5850 |
| Share of rows on single-transaction entities | 0.2331 |

Most entities are a single transaction and most rows are not. That is the shape that makes
entity-level aggregation worth doing and also makes it unavailable for a quarter of rows.

### Label purity

Of the 68,359 entities with more than one transaction: 0.9466 are entirely non-fraud, 0.0208 are
entirely fraud, and 0.0325 are mixed. That puts label purity at 0.9675 on the train split against
the 0.9664 stage 0 measured over the whole file, which is the closest thing to a replication this
key has had. 5,824 entities carry at least one fraud row.

Inside the 2,225 mixed entities the within-entity fraud rate is spread wide: 25th percentile
0.167, median 0.333, 75th percentile 0.500, 95th percentile 0.800. The median mixed entity has a
third of its transactions labelled fraud. Mixed does not mean one bad transaction among many; it
means the entity is a coin flip.

### Time between an entity's consecutive transactions

248,676 gaps measured. Entities that carry at least one fraud row behave completely differently
from those that do not.

| Percentile | All gaps | Entities with fraud | Entities without |
| --- | --- | --- | --- |
| 1st | 35 s | 18 s | 37 s |
| 5th | 105 s | 61 s | 123 s |
| 25th | 2,737 s | 460 s | 4,243 s |
| 50th | 89,296.5 s | 3,177 s | 165,394 s |
| 75th | 886,340 s | 41,511 s | 1,033,668 s |
| 95th | 3,028,002 s | 1,079,306 s | 3,176,420 s |

The median gap for an entity that ever sees fraud is 3,177 seconds, about 53 minutes. For an
entity that never does it is 165,394 seconds, about 46 hours. A factor of 52.

That is velocity, and it is measured here without a single feature having been built.

**For stage 4:** inter-transaction gap and its aggregates are the highest-value derived features
this stage found. The distribution above is the evidence, and the 0.2331 of rows on
single-transaction entities is the population where they will be undefined.

**For stage 3:** nothing. This section builds no columns and cleans nothing. It is here because
stage 4 needs it and because it is the argument for the entity key surviving contact with more
data.

---

## 8. Duplicates and data quality

### Exact duplicates barely exist, and that was not the expectation

| Definition | Groups | Rows in groups | Share |
| --- | --- | --- | --- |
| Identical on all 435 columns | 0 | 0 | 0 |
| Identical except TransactionID | 1 | 2 | 5e-06 |
| Identical except TransactionID and TransactionDT | 5 | 10 | 2.4e-05 |

TransactionID is unique, so the first row is zero by construction. The second and third are the
real measurements and they find almost nothing: 10 rows in 413,378.

The expectation going in was that repeated identical transactions would show up as a card-testing
pattern. They do not, and the reason is visible once stated: the C counters and most of the V
block are running totals that change on every transaction of an entity, so two transactions that
are the same purchase are never the same row.

Asking the question the way a person would ask it gives a completely different answer. Rows
agreeing on card1, card2, addr1, TransactionAmt, ProductCD and P_emaildomain, ignoring the
counters:

| | Value |
| --- | --- |
| Groups | 60,310 |
| Rows in a group | 200,113 (0.4841) |
| Rows beyond the first of each group | 139,803 |
| Fraud rate inside groups | 0.04157 |
| Fraud rate outside groups | 0.02917 |

Nearly half the split is a repeat of some earlier purchase on the same card, and those rows carry
1.43 times the fraud rate of the rest. That is a real signal and it is a feature, not dirt.

### Impossible values: none found

No negative amounts, no zero amounts, no non-finite amounts, minimum 0.251 and maximum 31,937.391.
No null target, no null timestamp. Every level of ProductCD, card4, card6 and DeviceType is in
range.

17 columns contain negative values and all 17 are explicable: D4, D6, D11, D12, D14 and D15 hold
a handful each (1 to 9 rows), the identity columns id_01, id_03 to id_10 and id_14 use negatives
as part of their encoding (id_01 on 92,934 rows, id_14 on 63,042), and card_start_day is negative
on 135,519 rows because a card whose D1 exceeds its day index started before the observation
window opens. None of these is a data error. All of them break a naive "counts are non-negative"
assumption, which is why they are listed.

### Email domains

P_emaildomain has 59 levels on train and is missing on 0.1555 of rows. R_emaildomain has 60 and
is missing on 0.7507.

Fraud rate by P_emaildomain, levels with at least 100 rows, highest first:

| Domain | Rows | Fraud rate |
| --- | --- | --- |
| mail.com | 398 | 0.1985 |
| aim.com | 239 | 0.1674 |
| outlook.es | 340 | 0.1324 |
| outlook.com | 3,550 | 0.0890 |
| hotmail.es | 201 | 0.0796 |
| hotmail.com | 32,817 | 0.0534 |
| gmail.com | 159,214 | 0.0440 |
| icloud.com | 4,319 | 0.0310 |
| anonymous.com | 26,745 | 0.0236 |
| yahoo.com | 70,404 | 0.0236 |
| aol.com | 19,798 | 0.0211 |
| att.net | 2,712 | 0.0044 |

A 45-fold spread between att.net at 0.0044 and mail.com at 0.1985, though the small levels carry
wide intervals and are in the artifact with them.

Whether the two domains agree carries more than either domain does:

| Case | Rows | Fraud rate |
| --- | --- | --- |
| Both present and equal | 77,595 | 0.09051 |
| Both present and different | 18,670 | 0.02689 |
| Purchaser present, recipient missing | 252,817 | 0.02011 |

Agreement runs at 3.37 times the rate of disagreement. This is the opposite of the intuition that
a mismatched recipient is the suspicious case, and it is a large effect on 96,265 rows.

### DeviceInfo needs normalising

1,546 distinct strings on the 0.2210 of rows where it is present. The top 10 cover 0.7965 of
present rows and the top 50 cover 0.8552, so the tail is long and thin: 399 strings appear exactly
once and together they cover 0.0044 of present rows.

The top values are a mix of families and build strings: Windows (36,668), iOS Device (16,084),
MacOS (10,191), Trident/7.0 (6,048), rv:11.0 (1,550), rv:57.0 (948), then device model strings
like "SM-J700M Build/MMB29K" (373). Three different naming schemes are present in the top seven
values.

**For stage 3:**

- Do not deduplicate. Exact duplicates do not exist in any quantity, and the near-duplicate
  definition that does find something is finding signal, not dirt.
- Build a repeat-purchase feature from the core-field grouping. 0.4841 of rows at 1.43 times the
  base rate is worth a column.
- Nothing to clean in the value ranges. Mark the negative-value columns as known and expected so
  a later validation rule does not reject them.
- Normalise DeviceInfo to a device family before encoding. The tail is 636 levels covering 1
  percent of present rows and the top of the distribution already mixes three naming schemes.
- Add an email-agreement feature. It carries more than either domain column alone.
- R_emaildomain is missing on 0.7507 of rows and that missingness is itself a level with a
  measured rate of 0.02011. Section 3 applies.

---

## 9. Figures

All six are drawn from the artifacts above and from nothing else, at 200 DPI, by
`make eda-figures`.

| Figure | Source | Shows |
| --- | --- | --- |
| `figures/fraud_rate_over_time.png` | target.json | daily and weekly fraud rate over the train span |
| `figures/amount_distribution.png` | univariate.json | TransactionAmt before and after log1p |
| `figures/missingness_blocks.png` | missingness.json | which column blocks go missing together |
| `figures/fraud_rate_by_decile.png` | bivariate.json | shape of the relationship for eight columns |
| `figures/psi_train_vs_test.png` | temporal.json | drift per feature against val and test |
| `figures/time_consistency.png` | temporal.json | early window AUC against late window AUC |

---

## What stage 3 inherits

| Finding | Number | Action |
| --- | --- | --- |
| Missingness carries the label | 251 of 372 columns | Indicator per column or per block, never silent imputation |
| Identity join is a risk factor | 0.0731 against 0.0214 | Explicit join indicator |
| V block is redundant | 132 of 339 columns removable | Take the reduction, cross-check against the inversion list |
| Features invert over time | 59 of 432 | Candidate removal list, decision needs a measurement |
| Distribution drift | 22 features above PSI 0.25 | Baseline for stage 9; id_31 needs splitting |
| Amount is extremely skewed | skew 17.6 to 0.499 under log1p | log1p, keep the raw scale alongside |
| Amount relationship is U-shaped | 0.0551, trough 0.0192, 0.0509 | No monotone assumption |
| Cents fraction is structured | 0.5255 at .00, 0.2712 at .95 | Derive a cents feature |
| Repeat purchases are common | 0.4841 of rows, 1.43 times the rate | Build the feature, do not deduplicate |
| Email agreement carries signal | 0.0905 against 0.0269 | Derive an agreement feature |
| DeviceInfo is free text | 1,546 levels, 636 in the 1 percent tail | Normalise to a family |
| Degenerate columns | 30 effectively constant | Candidate removal, check against C3 first |
| Rate is not stationary in train | weekly range 2.45 times | Time-aware validation inside train |
| Cold entities dominate later splits | 0.66 of test rows | Defined cold-start behaviour in stage 4 |
| Class balance is fixed | 3,042 and 3,083 positives | No comparison below about 0.015 AUC |

## What is not established

- What TransactionDT is measured from. Day, hour and week indices here are relative.
- Why the cents fraction concentrates at .95 on 0.2712 of rows.
- What the V columns are. Their block structure, correlations and time behaviour are measured;
  their meaning is not, and nothing here guesses.
- Whether the repeat-purchase grouping in section 8 is card testing or ordinary repeat business.
  The rate difference is measured; the mechanism is not.
- Run-to-run variance on the time consistency screen, which is one seed and one setting.
