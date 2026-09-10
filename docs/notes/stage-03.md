# Stage 3: preparation

What the stage built: a column plan, four encoders, a schema, six artifacts, three figures and
seven decision records. What follows is the parts that did not go the way they were supposed to.

## The number that changed my mind

The D-column normalisation. The brief said to convert the D columns to their origin so they stop
drifting with time, and to report the before-and-after PSI to show it worked. The argument is
sound and I wrote the code expecting to be reporting a win: a delta measured backwards from the
transaction grows as a card ages, an origin does not, so subtracting the delta from the day index
should turn a drifting column into a still one.

It raises PSI on all fifteen. D3 goes from 0.0352 to 2.2125. D1 goes from 0.0226 to 1.8304, and I
should have seen that one coming, because stage 2 had already measured D1's origin under another
name: `card_start_day` is `floor(TransactionDT / 86400 - D1)` and carries the highest PSI in the
frame at 1.841. The whole finding was sitting in temporal.json for a month.

The reason is that the D columns were never drifting. The population being observed is a mixture
of cards at every age with new ones arriving at D near zero, and the mixture is what holds still.
An origin is a calendar date, and a later window contains later dates by definition. The
transformation works exactly as designed on a column that does not have the problem it solves,
and hands it one it did not have.

I added a second measurement afterwards, which is the part I would do differently next time: a
single-feature validation AUC per column, raw against normalised. PSI answers whether a
distribution moved, not whether a column got better at its job, and a transformation could
plausibly raise PSI and still help. It does not, here: validation AUC falls on twelve of fifteen,
D3 by 0.1265 and D5 by 0.1139. But that second question should have been in the plan from the
start rather than added once the first answer looked bad. Adding a measurement after seeing a
result you dislike is a habit worth watching, even when the measurement is the right one.

The figure is in the case study and it says the opposite of what the section was expected to say.
That is fine. The PSI delta was asked for and the sign of it is the finding.

## Two drop categories that caught nothing

The plan has four drop rules and two of them removed zero columns.

The missing-and-not-MNAR rule is empty because every column at or above 0.95 missing on train is
on stage 2's label-linked list, and so are the three more that a 0.90 threshold would add. It only
starts biting at 0.80, where it would take 51 columns. There was a real temptation to move the
threshold to 0.80 so the rule would have something to show for itself. Lowering a line until a
rule fires is the exact failure the evidence discipline is written against, so the line stayed and
the ADR records the sensitivity instead.

The time-inconsistency rule is more interesting, because it is empty for a reason that took a
while to see. ADR 0009 flagged 59 columns that separated the classes over the first 30 days of the
training window and inverted over the last 30, and said the list was candidates because removing a
feature has to be measured. Measured against validation with the model fitted on the whole 120 day
window, all 59 score at least 0.50. id_38, which stage 2 called the strongest case at 0.6037 early
and 0.3634 late, comes back at 0.6704.

Both measurements are right and they ask different questions. Stage 2 asked whether a relationship
learned from one narrow early month survives into a later one. Stage 3 asked what a model actually
trained on the whole window does on the window it will be judged on, and a model that has seen all
120 days has seen the late behaviour too. The inversion is real and it does not survive being
averaged over. ADR 0009 was right to refuse to make it a decision, and if it had made one, 59
columns would have gone for nothing.

## The entity encoding that looked like a great feature

This is the one I would have got wrong without the decomposition.

Target-encoding the entity key gives a validation AUC of 0.7835 at the chosen lag. That is a
strong number, it holds up across every lag on the grid, and the lag machinery cleanly removes the
lag-0 memorisation that shows as a train AUC of 0.9922. Everything a normal sweep reports says
ship it.

Splitting validation on whether the training window had ever seen the row's entity says something
else entirely. On the 41,536 rows where an entity history exists, the encoded rate recovers the
validation label at an AUC of 0.9877. On the 47,045 where it does not, the AUC is 0.4915, which is
chance. Stage 2 measured entity label purity at 0.9675 and those are the same fact seen twice. The
composite is the average of a lookup table and a constant over two groups sitting at different
base rates, and stage 1 already measured that the seen group shrinks over time: 0.4689 of val rows
and 0.3402 of test.

What made this findable was writing the decision rule before running the measurement. The rule
asks for a shape, power concentrated on the seen subset and nothing off it, rather than for a
threshold on the headline number, and the shape is what a label lookup looks like from the
outside. A rule of the form "ship it if validation AUC beats chance by more than the split
resolves" would have shipped it, and my first draft of the rule was exactly that.

## The V reduction that a tie-break decided

Three strategies and a baseline, measured. Column counts of 327, 196, 122 and 14, and validation
AUCs of 0.8406, 0.8378, 0.8466 and 0.8375. A spread of 0.0090 across four strategies.

Then I measured what one strategy does across three LightGBM seeds on identical rows and identical
columns: 0.0051. The between-strategy spread is 1.76 times the within-strategy spread, which is
not a comparison, it is four numbers in a cloud.

So the stated rule's tie-break chose, and the tie-break is column count, so 327 V columns became
14 block means. That is a large consequence for a rule I wrote in about a minute, and I do not
fully like it: the PCA scored highest on every seed, and after the reduction nothing in the V
block has a name any more. Stage 7 explaining a score can say the mean of a block was high, not
that V258 was, and stage 2 measured V258's top bin at a fraud rate of 0.6076 on 4,348 rows.

I kept the answer because the alternative is worse. Rewriting a rule after seeing which strategy
it selected is how a preference gets laundered into a measurement, and the whole stage is built
against that. What I did instead was record the cost in the ADR in the same voice as the benefit,
and leave `v_strategy` as an argument the pipeline reads from the artifact, so stage 5 can rerun
the comparison with a real model and a paired test and supersede this.

The seed measurement is the thing to carry forward. Stage 2 left run-to-run variance open on its
time consistency screen and said so. Leaving it open twice would have been a habit rather than an
oversight, and it cost seventy seconds to close.

## Things that behaved

The schema validator earned itself on its first run. Every derived column had inherited the rule
that stage 2 observed no negative values, and all fourteen block means failed on all three splits,
each with a large fraction of its rows below a declared minimum of zero. A standardised mean is
signed by construction, and "no negative was observed" is a statement about a measurement of columns that
existed when the measuring was done. That distinction is now a table in the module with a reason
per entry. A validator that has never rejected anything is not evidence of anything, and this one
rejected the code that wrote it.

The lagged target encoder was the piece I expected to fight. A dense level-by-day table would be
316 megabytes for the entity column alone, so it stores cumulative counts as one array the length
of the split, sorted so that a single `searchsorted` on a packed key answers "how many rows of
this level fell on or before this day" for every row at once. The payoff is that the tables do not
depend on the lag or the smoothing, so the 864 point sweep is one fit and 864 reads, and it runs
in under forty seconds. Before writing any of it I checked the arithmetic against a brute-force
recomputation on a 300 row frame, and that test is still in the suite.

Committing the artifacts means 60 of the 61 artifact tests run in CI without the data, and the
ones that matter are the cross-checks: every number the plan quotes per column, re-read from the
stage 2 artifact it claims to come from, over more than 400 columns. That caught the one place I
had retyped a number, entity label purity as 0.9675 in the driver, and it now reads
`share_all_clean` plus `share_all_fraud` out of entities.json.

The last one is small and worth writing down. The free-text normalisers return a pandas string
column, not an object column, and my predicate for "is this categorical" tested `dtype == object`.
So the two columns the stage went to the most trouble to normalise were silently skipped by the
vocabulary encoder. Nothing failed. The artifact just had two fewer rows in it than it should
have, and I found it while reading the output rather than from a test. There is now a named
predicate with the reason in its docstring, and a test for it.
