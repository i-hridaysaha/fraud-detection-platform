# Stage 4: features

What the stage built: 36 point-in-time features over three keyspaces, a test suite that recomputes
every one of them by brute force, four artifacts, three figures and six decision records. What
follows is the parts that did not go the way they were supposed to.

## The number that changed my mind

The duplicate-content lift, from 1.333 on train to 0.814 on validation.

Stage 2 had measured that rows agreeing on the six fields that describe a purchase run at 0.04157
against 0.02917 outside, and wrote that it was a feature and not dirt. Stage 4's job was to build
the causal version of it, because stage 2's measurement came from a groupby and a groupby reads a
group's later rows. I built it, it reproduced stage 2's row count to within 11 rows through a
completely different implementation, the lift came back at 1.333 with disjoint Wilson intervals,
and I wrote the ADR saying it ships.

Then I looked at the per-split coverage table for a different reason and noticed the content-grain
rows had a *higher* fraud rate on the cold side in val and test. Marked rows run at 0.03057 on val
against 0.03758 unmarked, and 0.03267 against 0.03698 on test. The sign is the other way round
from train and the intervals are disjoint on all three splits, so it is a sign change and not
noise.

The ship rule I had written was "the causal version separates the label on the train split by more
than the split resolves". That rule is wrong and it is wrong for a reason this repo had already
written down twice. ADR 0009 says a relationship learned from one window is a candidate until a
later window has seen it. Stage 3 acted on exactly that and refused to drop 59 columns on a
single-window screen. I wrote a train-only rule anyway, two stages later, on the section whose
whole point was that stage 2's measurement was not a feature.

The restricted version does hold: an identical purchase inside 3,600 seconds runs at 2.865, 2.410
and 2.019 across the three splits, same sign, disjoint everywhere. So the signal is not that a
purchase repeats, it is that it repeats soon, and the all-history count dilutes it until it
reverses. Six of the ten recency deciles sit below the base rate and they hold 0.6000 of the
marked population, which is the arithmetic of the reversal.

Every count feature in the stage now carries a per-split comparison whether or not it survives
one. That block should have been in the plan from the start rather than added after a number
looked good, which is the same habit ADR 0013 flagged from the other direction: there I added a
measurement after a result looked bad. Both are the same mistake about when to decide what to
measure.

## The feature I built that could not possibly work

The brief asked for whether addr1 differs from the entity's modal addr1 so far. I wrote it, ran it
on a generated frame, and the column came back all zeros. My first thought was a bug in the mode.

It is not a bug. `addr1` is a component of the ADR 0001 entity key, so an entity holds exactly one
addr1 value by construction. Measured: maximum 1 distinct value across all 164,702 train entities,
0.000000 of entities with more than one. The feature is identically zero wherever it is defined.

Two things came out of that. The obvious one is that the family moved to the card grain, where
0.3320 of cards have used more than one addr1 and one card has used 63. The less obvious one is
that addr2 is *not* in the key and is nearly pinned by it anyway, 19 of 164,702 entities holding
two values, so at the entity grain it would have been effectively constant too. Reading the key
definition would have caught addr1 and would not have caught addr2. That needed measuring.

The general version, which ADR 0023 records because nothing in the repo had said it: a key chosen
for label purity pins the columns it is built from, so any within-key deviation feature over a key
component is constant by construction. That applies to card1 and D1 as well and nobody has
proposed a feature over either.

## The 2e-4 that changed the arithmetic

`tests/test_causality.py` asserts that truncating the stream at time T leaves every feature
unchanged on the rows before T. On the generated frame it passed. On the real file it failed, on
`amt_zscore_within_entity`, by about 2e-4 on entities that repeat one amount.

The cause is that the window sums came off a global `np.cumsum`. A cumulative sum carries a
rounding error proportional to what it has accumulated, so the last bits of a ten-minute window
total depended on every amount earlier in the file, and once the difference of two prefix values
was taken, on rows later than the one being computed. The running variance made it visible because
it came from `sum(x^2) - n * mean^2`, which cancels badly when a key's amounts are large and
close together, which is exactly what a repeated amount is.

The tempting fix was a tolerance on the truncation test. The values were immaterial and the
comparison would still have been meaningful. I changed the arithmetic instead: window sums restart
at every key, and the running mean and spread use Welford. The truncation tests now compare bit for
bit with no tolerance at all.

That turned out to be the better engineering rather than only the stricter test. Welford is the
three scalars an online store holds, so the offline path now does the same arithmetic the serving
path will, and the numerically bad form is gone rather than tolerated. The per-key restart costs
about three seconds on 590,540 rows.

The only tolerance left in the suite, 1e-6, is against the brute-force reference, which adds a
Python list with `sum()`. Adding the same numbers in a different order gives a different last bit:
662.8500000000001 against 662.85.

## The information value that nearly dropped the best feature in the family

`dup_prior_count_1h` has an information value of 0.0000 and it is the sharpest duplicate feature
in the stage. `vel_count_10m_entity` has an information value of 0.0000 and it separates the label
2.5 to 1.

Both are zero on most rows, so ten equal-frequency bins collapse into one and the information
value of a single bin is zero by construction. I had the table of information values in front of
me and was about to write that the hour window carried nothing, which would have been a sentence
with a measured number behind it and completely wrong.

The artifacts now record the bin count next to every information value, and a zero-against-positive
rate comparison next to that. A test asserts that an information value of 0.0000 is accompanied by
a bin count of 1 and a ratio above 1, so the artifact cannot report the collapsed number without
the thing that explains it.

Decile information value is a fine instrument for the columns stage 2 pointed it at and a bad one
for a zero-inflated count. Stage 2 was not wrong to use it; stage 4 was wrong to reuse it without
checking whether the column shape had changed.

## The result I did not want

My own velocity is weaker than the native C columns, by a wide margin, on every measure taken.

C7 separates the label 4.525 to 1 on train and 5.909 on test. The best built velocity feature does
2.556 and 2.482. On information value it is 0.5629 for C4 against 0.1747 for the strongest
velocity column, more than three times.

The correlation result is the one that makes this a finding rather than a defeat. The largest
absolute Spearman correlation across all 84 pairs is 0.2773 and the median is 0.0658, against the
0.95 at which stage 2 calls two columns the same column. So the C block is not a better version of
what I built; it is measuring something else, and something else that works better.

Two readings fit and I cannot separate them. Either the C block is a genuinely better feature,
built over a longer history or a wider notion of what to count, in which case stage 8 has a
problem because it cannot compute a column whose definition is unknown. Or the C block is not
point-in-time, in which case it separates better for the reason a non-causal feature always does.
The evidence that would settle it is the construction, and the construction is not published
anywhere I could fetch. Stage 0 could not load the data description and neither could I.

What I can say is that my block is verifiably causal and theirs is not verifiably anything, and
that both go into stage 6.

## The four breaks

Each applied, the suite run, then reverted.

| Break | Change | Tests failed | First failure |
| --- | --- | --- | --- |
| Trailing upper bound | `side="left"` to `side="right"` in `grouped_trailing_bounds` | 3 | brute-force agreement |
| Expanding upper bound | the same in `grouped_prior_bounds` | 5 | brute-force agreement, and the tie test |
| Transductive mean | the amount mean over the whole entity rather than the prefix | 2 | brute-force agreement, and truncation invariance |
| Mode over the block | the addr counter updated before the block is read instead of after | 1 | brute-force agreement |

The third is the one worth reading, because it is the shape ADR 0020 refuses and it is the shape a
brute-force test alone can miss. A feature that reads a whole-group aggregate agrees with a
reference that reads the same whole-group aggregate. What it cannot survive is having the future
removed, which is why truncation invariance is a separate check and not a nicer way of asserting
the first one.

The second break is the one that exercised the tie test, and it is why the tie test reads
`rec_prior_count` rather than a velocity count: the trailing and expanding bounds are different
functions and a break in one does not fail a test that only reads the other.

## Coverage, and a number stage 3 got the wrong way round

Stage 3 told stage 4 that entity-history features are missing for 0.66 of test rows by
construction, and pointed at the 0.6598 of test rows that sit on an entity the training window
never saw.

That number is about a train-fitted statistic. It does not apply to a point-in-time one, and the
difference is large: 0.3402 of test rows sit on an entity the training window saw, and 0.7036 have
an earlier transaction somewhere in the stream. An entity nobody had seen at training time can
still have transacted an hour ago, and at authorisation the online store holds that history
whatever window the model was fitted on.

So the coverage cost of ADR 0015's exclusion is roughly half what stage 3 expected, and the
direction is the useful one. It is also the clearest single argument in the repo for building
behaviour rather than encoding labels, and it took four stages to become visible.

## Smaller things

The tie exclusion shows up in a row count. On train, 164,711 rows have no earlier transaction on
their entity, against the 164,702 entities stage 2 counted. The 9 extra are tied first rows, where
two transactions on one entity share a timestamp and neither is prior to the other.

`geo_addr1_differs_from_card_mode` and `geo_addr2_differs_from_card_mode` have identical null
rates, 0.1415 on train, which looked like a copy-paste bug. The two source columns are their own
two-column missingness block at a missing rate of 0.11495, so they are null on exactly the same
rows and so are the features derived from them.

The brute-force reference started out calling `.iterrows()` to build a content key per row, which
made the real-data test take minutes. Building the keys once per frame with a plain loop over
columns is the same definition and about fifty times faster. The reference is meant to be slow and
obvious, not gratuitously slow.

`dist1` asked for ten quantile buckets and got ten. That is worth noting only because the code
expects not to: the artifact records the requested count next to the achieved one because a column
concentrated on a few values cannot be cut into ten equal parts, and a test builds exactly that
case.
