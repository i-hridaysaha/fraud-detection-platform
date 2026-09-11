# Stage 5: graph features

What the stage built: an incremental union-find over the six link columns, eight point-in-time
graph features behind a switch, a brute-force graph rebuild in the causality suite, one artifact,
one figure and two decision records. What it found is that there is no graph in this file worth
the name, and most of what follows is about how that became a finding rather than a shrug.

## The number that changed my mind

23. The number of components the six-column graph has at the end of the training window, with
413,349 of 413,378 transactions in one of them.

I had expected hubs. `gmail.com` is on 0.3852 of train rows and `addr1 = 299.0` on 0.0797, and
a graph that links on either is going to have a large component. What I had not expected was that
there would be nothing else. I ran the leave-one-out variants expecting to find the one column
responsible and drop it: without `card1` the largest component holds 0.9977, without `addr1`
0.9996, and without any of the other four it is unchanged at 0.9999. Then the pairs: `card1` alone
gives 12,242 components, which is the number of cards, and `card1` with any single other column
gives a largest component of at least 0.8528. There is no subset of these columns with structure
in it. None of the six is an identifier, and a graph over shared non-identifiers is one blob.

That reframed the stage. The brief says that if component size carries no signal, that is a
finding to report. It carries no signal because there is no component; the size of the component
a transaction sits in is the number of transactions before it. Spearman with `TransactionDT` on
train is 0.9768, PSI against test is 11.50, and the fraud rate by decile on train is the weekly
fraud rate stage 2 measured, trough in the third decile and rising after, because the deciles are
the twelfths of the window in order. I had the decile table in front of me before I had the
correlation, and for a minute it looked like a U shape worth writing about.

## The hub sweep, and why it did not rescue anything

The obvious next move was to exclude hubs. I wanted a derived line for it and the repo has one
that resembles it: stage 3's rare-tail share of 0.01, below which a categorical level is collapsed.
At that share the largest component still holds 0.8871 of train. A decade below it, 0.4104, with
0.5066 of transactions alone. Two decades below, the largest component has 2,377 transactions in
it and 0.7890 of the split is alone. The graph goes from one blob to dust with no middle where it
has both structure and coverage, and no point on that grid is derived from anything: the first is
borrowed from a decision about vocabulary size and the other two are the first one divided by ten
and a hundred.

So the sweep is in the artifact as a measurement and not as a knob. The hub set is a fitted object
through the guard, because which values are hubs is learned on train, and the reason the code
supports it at all is the transductive comparison below, which needs a graph with components in
it to say anything.

## The feature that measured ADR 0015 by subtraction

I built the component fraud rate twice: once over every matured train label in the component and
once with the row's own entity's labels removed from both numerator and denominator. The second was
the one I thought would survive, because the entity-label problem ADR 0015 recorded is about the
entity's own labels and subtracting them should leave the neighbourhood.

On the six-column graph it does the opposite. The component is the whole training window, so the
others rate is `(Y - y_own) / (N - n_own)` with `Y` and `N` fixed, which is a decreasing function
of the entity's own fraud count. On test rows whose entity the training window saw, its AUC is
0.0244. That is 0.9756 the other way round, and ADR 0015 measured 0.9877 for the direct encoding.
I had reproduced the exclusion's finding with the sign flipped, through a feature designed to
avoid it. On unseen entities there is nothing to subtract and the number is 0.4720, the same as
the version that subtracts nothing.

The general version is worth writing down because it is not about graphs. Any statistic of the
form `(total - own) / (count - own count)` over a neighbourhood that is mostly the training window
carries the entity's labels with the opposite sign. The decomposition on seen and unseen entities
is what shows it, and it is now the test I would run on any label-derived neighbourhood feature
before reading its AUC.

## The signal that exists and does not ship

On the graph fragmented at 0.0001 the component fraud rate scores 0.6415 on validation rows whose
entity the training window never saw and 0.6061 on the same rows of test. Those entities
contributed no label to anything, so the number is other entities' matured labels in a component
the row was linked into, with the same sign on both later splits. That is a real neighbourhood
signal and I did not expect the file to have one.

It stays out for two reasons I can point at. The share that produces the graph is a grid point,
and the rule against guessed thresholds is the whole reason this repo is worth reading. And the
feature is defined on 0.1597 of test rows, because at that share most rows have no labelled
neighbour. ADR 0025 records the numbers so that a later stage starts from them. Writing a rule
that selects 0.0001 after seeing 0.6415 is the mistake ADR 0013 already described from the other
side, and I came close enough to it to write this paragraph.

## The leak, quantified

ADR 0024 asks what a full-dataset component computation would have leaked. On the six-column graph
the answer has an unusual shape: the transductive leave-one-out rate has an AUC of exactly 0.0000
on validation and test. One component, so the rate is `(Y - y_i) / (N - 1)`, and a fraud row
always reads a lower rate than a non-fraud row. The leak is not a boost to the AUC; it is the
row's own label, negated, and any model would find it in one split. On the fragmented graph the
transductive rate scores 0.8462 on validation against 0.7087 for the causal one, and 0.8415
against 0.6638 on test. The 0.14 to 0.18 is what reading the future adds on top of a number that
already includes the entity's own labels.

## The break that only the invariance tests catch

Four deliberate breaks, each applied and reverted. Reading a block after adding it fails four
tests; ignoring the lag fails two; walking the stream row by row instead of by timestamp block
fails three, including the row-order test, which I had not predicted. Adding every edge before any
read, which is the full-dataset construction, fails six, and two of the six are the truncation and
append invariance tests that the read-after-add break does not touch.

That last pair is the point stage 4 made and I would not have believed it without seeing the
counts: a brute-force reference over `TransactionDT < t` catches a wrong read order, but a feature
that was right about the past and also read the future agrees with the reference on rows where the
future happens not to matter. Only removing the future and checking that nothing moves catches it
in general.

## Smaller things

The first draft nulled the anchor column to exclude hub cards, and the module refused it, because
I had just written the check that the anchor is null-free. The check was right and the draft was
wrong: an excluded hub is a value with no node, not a missing value, so it is coded as absent after
the null check and reads as a degree of zero and an empty component. The distinction is the same
one stage 4 made between a count of nothing and a comparison against nothing.

`graph_degree_addr1` has 318 rows on its zero side on train, and that is the number of `addr1`
values. The zero side of an all-history degree is the first appearance of each value, which is
why the stage 4 rule cannot see anything in it: the cut is at zero and the interesting cut is
somewhere in the first decile. On rows that carry a device, the lowest decile of the device degree
runs at roughly twice the present-row rate on all three splits, 0.1210, 0.1342 and 0.1974 against
0.0679, 0.0839 and 0.0917. I have not written a rule for it, and the reason is in the paragraph
about 0.0001.

The two helpers stage 4's driver used to compare a count against the label moved into the package,
because this stage needed the same comparison and a second copy would have drifted. The stage 4
duplicates artifact regenerates identically through the moved code, checked field by field before
the move was committed.

The brute-force reference builds a networkx graph per sampled row. On the real file it does that
over a twelve-day prefix rather than stage 4's twenty, because every sampled row rebuilds the
whole prefix, and the two `needs_data` tests take about twelve seconds together.

The `train` target in the Makefile said "not implemented until stage 5" since stage 0, when the
plan had a model here. The graph stage moved it; it now says stage 6.
