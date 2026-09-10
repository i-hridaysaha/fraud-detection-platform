# 0008: Rank correlation, and a screen-then-recompute implementation

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 2

## Context

Stage 2 has to answer two questions about redundancy. Which of the 339 V columns are copies of
each other, so stage 3 can propose a reduction. And are there near-duplicate pairs anywhere else
in the frame. Both need a correlation over 401 numeric columns and 413,378 train rows.

Two things about this data decide the answer. The columns are not close to normal, and the
missingness is heavy and structured.

## Options considered

**Pearson.** The default, and cheap: one pass to get means and cross-products. Rejected on the
measured shape of the columns. Pearson measures a linear relationship and its estimate is
dominated by the tails, which is where this frame keeps most of its mass.

**Spearman.** Rank correlation: invariant under any monotone transform, so it gives the same
answer on TransactionAmt and on log1p(TransactionAmt), and one extreme row cannot carry a
coefficient. Chosen.

**Kendall tau.** Also rank based and better behaved on ties, and O(n log n) at best per pair.
Rejected on cost: 80,200 pairs over 413,378 rows is not a computation worth two days for a
redundancy screen, and the quantity being estimated is close enough to Spearman that it would not
change which columns get grouped.

**Distance correlation or mutual information.** Would catch a non-monotone dependence that
Spearman misses. Rejected for this stage: the question here is "is this column a copy of that
one", which is a monotone question, and the non-monotone version costs an order of magnitude
more. Stage 4 can revisit it if feature construction needs it.

## Decision

Spearman, computed in two passes.

A screen matrix ranks each column once over the train split and correlates each pair over the
rows where both are present, accumulating four cross-product matrices over 50,000 row chunks.
That takes about 5 seconds for all 80,200 pairs. Every pair the screen puts at or above |rho| =
0.80 is
then recomputed with `pandas.DataFrame.corr(method="spearman")`, which re-ranks inside the pair's
own complete cases. 1,762 pairs were recomputed, taking about 36 seconds. Every correlation quoted
in
docs/eda.md or used to group columns is an exact value from the second pass, except inside a
missingness block, where the first pass is exact and measured to be so.

## Evidence

**The skew that rules out Pearson.** reports/eda/univariate.json, `make eda`. Over the 401 numeric
feature columns on the train split, the median absolute skew is 10.73 and the 90th percentile is
56.29. 359 columns have |skew| above 1, 212 above 10 and 22 above 100. The largest is V305 at
371.2 with an excess kurtosis of 137,785. TransactionAmt alone has skew 17.64 and excess kurtosis
1,603.6, falling to 0.499 and 0.765 under log1p. Only 26 of the 401 columns are close to
symmetric at |skew| below 0.5. A Pearson coefficient over columns shaped like this is a statement
about their extreme values, and every one of those columns has a different extreme.

**Why the screen is not the answer on its own.** reports/eda/correlation.json, block `screen`.
The screen ranks each column once over the whole split; pandas ranks inside each pair. The two are
the same computation when both columns of a pair are missing on the same rows, and are not
otherwise. Measured on 82 sampled within-block pairs, the largest gap is 1.95e-08, which is
float32 rank storage and nothing else, and all 82 fall inside a 1e-6 tolerance. Measured on 175
random pairs drawn from anywhere in the frame, the median gap is 1.73e-04 and the largest is
0.1327, on V72 against V145, where the screen says -0.6226 and pandas says -0.4899. That is far
too large to report as a correlation.

**Why 0.80 is the screen threshold.** The redundancy threshold is 0.95, so a screen at 0.80
leaves a margin of 0.15 against a measured worst case of 0.1327. A pair whose exact correlation
clears 0.95 would have to be understated by the screen by more than the largest error measured
anywhere in the frame to be missed. tests/test_eda_artifacts.py asserts that margin holds against
the number in the artifact, so a future run where the screen behaves worse fails the suite rather
than quietly dropping pairs.

**What it found.** 508 pairs at |rho| >= 0.95 and 127 at |rho| >= 0.99. Inside the V block, the
reduction takes 339 columns in 14 missingness blocks down to 207 representatives, removing 132.
Outside the V block there are six pairs above 0.95, and the strongest is exact: D4 and D12
correlate at 1.000 over their 46,966 complete cases. D5 and D7 sit at 0.9985, D6 and D12 at
0.9939, D1 and D2 at 0.9813, D4 and D6 at 0.9803, and C8 and C10 at 0.9710 over all 413,378 rows.

Wall-clock times here are quoted to the nearest few seconds. They move between runs and the
artifact records what the run that produced it measured.

## Consequences

- Every correlation in this repo is a rank correlation and should be read as one. A pair at 0.99
  agrees on ordering, not on scale, and two columns that are exactly proportional and two columns
  that are one a rank-preserving transform of the other are indistinguishable here. For a
  redundancy decision that is the right indifference.
- The screen matrix is not published as a correlation matrix. correlation.json reports the pairs
  that were recomputed exactly and the group memberships, not 80,200 approximate numbers that a
  later reader could quote.
- The 0.80 screen has a blind spot by construction: a pair whose exact correlation clears 0.95
  while the screen puts it below 0.80. The margin argument above bounds it against the worst
  error measured, not against the worst possible error, and that distinction is the honest
  statement of what this buys.
- D4 at 1.000 against D12 is a fact about their 46,966 complete cases, 0.114 of the split.
  D4 is present on 291,163 rows and D12 on 47,289, so on 244,197 rows D4 is there and D12 is not.
  Whether one of them can go is a stage 3 decision that has to weigh those rows, and a
  correlation measured on a ninth of the split cannot settle it.
