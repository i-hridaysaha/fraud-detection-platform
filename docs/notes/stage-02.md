# Stage 2: the exploratory analysis

What the stage built: nine JSON artifacts, six figures, a written analysis, and the four decision
records the brief asked for. What follows is the parts that did not go the way they were supposed
to.

## The duplicate count that was supposed to be a finding

The brief said, correctly, that repeated identical transactions are a known card-testing pattern,
and asked for the count of exact duplicates and content duplicates with the fraud rate inside
duplicate groups against outside them. The expectation on the way in was a number worth acting on.

The measured answer is ten rows. Five groups, ten rows, out of 413,378, on the definition the
brief gave: identical on every column except TransactionID and TransactionDT. Widen it to include
TransactionDT and it is one group of two.

The reason is obvious once it is written down and was not obvious before. The C block is running
counters and most of the V block moves with them. Two transactions that are the same purchase on
the same card, minutes apart, differ in every counter that ticked between them. A whole-row
comparison cannot see a repeated purchase in this frame, and there was never any chance it could.

So the question was asked a second way, on the six fields a person would use to say two rows are
the same purchase: card1, card2, addr1, TransactionAmt, ProductCD, P_emaildomain. That found
60,310 groups covering 200,113 rows, 0.4841 of the split, at a fraud rate of 0.04157 against
0.02917 outside. Both numbers are in quality.json under different keys, with their definitions
attached, because they answer different questions and only one of them is useful.

The lesson worth keeping: "duplicate" is not a property of rows, it is a property of a column
subset, and the subset is the decision.

## The correlation that was not the correlation

The plan was one Spearman matrix over 401 numeric columns. pandas will not do it in a useful time
because it re-ranks inside each pair's complete cases, which is a quadratic number of ranking
passes over 413,378 rows. So the matrix was built by hand: rank each column once, accumulate four
cross-product matrices over row chunks, divide. About five seconds for all 80,200 pairs, and the
docstring cheerfully said it produced "the same pairwise-complete answer".

It does not. Checking it against pandas on a random sample of 175 pairs gave a median gap of
1.73e-04 and a worst case of 0.1327, on V72 against V145, where the fast version says -0.6226 and
pandas says -0.4899. Ranking once over the column and ranking inside the pair are the same thing
only when the two columns are missing on the same rows, and across missingness blocks they are
not.

The interesting part is how close that came to being shipped. The number was plausible, the code
was correct for what it computed, and the docstring was wrong about which quantity that was.
Nothing but the check would have caught it, and the check was written because the docstring made a
claim specific enough to test.

The fix keeps the fast version as a screen and recomputes every pair above 0.80 exactly, 1,762 of
them. It also measures both cases and puts them in the artifact: within a missingness block the
screen agrees with pandas to 1.95e-08, which is float32 rank storage, and across blocks it does
not. And a test now asserts that the measured screen error stays inside the margin between the
0.80 screen and the 0.95 threshold, so a future run where it gets worse fails rather than quietly
losing pairs.

## Cold entities point two ways at once

Stage 0 and stage 1 both measured that test rows sitting on an entity the training window never
saw run at a fraud rate of 0.0440 against 0.0170 for rows on entities it did see. Unseen entities
are riskier. That has been in the build log since stage 0.

This stage measured the same idea inside the train window, where cold means an entity's first
transaction, and got the opposite: 164,702 cold rows at 0.0263 against 248,676 warm rows at
0.0410. A first transaction is safer than a repeat.

Both are correct and they are not the same measurement. Inside a window, a first transaction is
the first of several the data contains: the entity has a history, and the history is what makes
the later rows risky. Across the boundary, an unseen entity is one whose history the model never
had. The word "cold" was doing two jobs and the artifact now says which one it means in the
definition field, along with a pointer to split_summary.json for the other.

The version that matters for deployment is the second one, and it applies to 0.6598 of test rows.

## Information value is mostly measuring the join

The top of the information value table is a block of V columns at 0.7 to 0.8, all of them missing
on about three quarters of the split. Ranked on raw IV they look like the strongest features in
the frame.

Because IV is a sum over bins, the missing bin's term comes out exactly, and once it does the
picture changes. addr1 is the clearest: total IV 0.4257, of which 0.3260 is the missing bin, and
recomputed on present rows alone it falls to 0.0118. addr1 is 81st of 432 on the raw number and it
is not a predictor at all. What predicts is whether addr1 is there.

Every feature now carries three numbers instead of one. It is a small change to the artifact and
it moves the ranking enough that stage 3 should read the decomposition rather than the total.

## Two screens that do not overlap at all

The time consistency screen flags 59 columns whose single-feature model separates the classes in
the first month of train and inverts in the last. PSI flags 22 features whose distribution moves
by more than 0.25 from train to test.

The intersection is empty. Not small, empty.

That was not the expectation. Drift and inversion both sound like "the feature stopped working",
and it would have been easy to run one screen and consider the question covered. The features that
drift hardest are card_start_day, which moves by construction, and id_31, whose browser-version
vocabulary turns over. The features that invert are three coherent blocks of V columns and a
handful of identity flags, and their distributions barely move at all. A pipeline that screened on
drift alone would keep all 59 of them.

## The categorical column that was 240 columns too wide

DeviceInfo came back with cardinality 1,786 on a split of 413,378 rows, which matched the number
stage 1 measured over all 590,540. That coincidence was the tell. `value_counts` on a categorical
column returns every declared level, including the ones no row in the slice uses, and the level
set was fixed when the whole file was read. The true train cardinality is 1,546, and the rare-tail
count was overstated by 240 in the same way.

A one-line fix, and a reminder that a number agreeing with an earlier number is not a check unless
the two were computed on the same rows.

## Things that behaved

The train-only rule cost nothing to enforce and produced its own evidence. `assert_train_only`
runs once at the top of the driver and every section reads the frame it returns. train_only.json
measures what the rule saved: 130 of 393 numeric columns have a quantile that moves by more than
10 percent when the later windows are folded in, and V324's 99th percentile goes from 7.0 to
457.0.

The figures read from the artifacts and never from the data. That constraint sounded like
bookkeeping and turned out to be the reason the missingness figure exists at all: a heatmap of
413,378 rows by 435 columns is not a figure, and having to store something a figure could be drawn
from forced the question of what the picture was actually meant to show. It shows blocks, not
rows, because blocks are the finding.

Committing the artifacts means the artifact tests run in CI without the data. 38 checks on
internal consistency, on every push, from a checkout that has no CSVs in it.
