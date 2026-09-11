# Graph features

Stage 5 adds structural features over the graph that shared attribute values draw between
transactions. There is no merchant node in this dataset and no customer identifier either; what
there is, is six columns that many transactions share: `card1`, `addr1`, `P_emaildomain`,
`R_emaildomain`, the normalised `DeviceInfo` and `dist1`. Every transaction is a node joined to
the value nodes it carries, and two transactions are connected when they share a value or a chain
of them.

The stage 4 rule binds this stage too, and the graph is where it is hardest to keep: **a feature
for a transaction at time t reads only rows strictly before t.** A connected component computed
over the whole file joins a transaction to every later transaction it will ever share a value
with, which is the future written into a feature. So the graph is a union-find that grows one
timestamp block at a time, and the features for a block are read before the block is added.

Everything below is measured on `reports/graph_summary.json`. Regenerate it with `make graph`.

The result of the stage is a finding rather than a feature set. On this file, the six-column
graph is one component: by the end of the training window, 413,349 of 413,378 train transactions
sit in the largest component, a share of 0.9999, and 17 sit alone. The size of the component a
transaction is in is therefore a count of the transactions before it, and the fraud rate inside
it is the lagged training rate. Neither carries anything structural, and the artifact says so with
numbers. Eight features are computed, four are the candidate set the switch turns on, none passes
the stage 4 ship rule on its marginal, and the switch defaults to off. Stage 6 measures the
candidate set in a model, which is the instrument the marginal comparisons are not.

---

## 0. What the graph is, and what is read before a transaction is added

Nodes are transactions and typed values: `("card1", 7919)` is one node, `("addr1", 299.0)` is
another. A transaction is joined to each non-null value it carries in the six link columns.
`DeviceInfo` is the normalised column stage 3 built, because the raw string carries a build number
and would split one device model into 1,546 nodes where the normalised one has 281.

The stream is walked in blocks of equal `TransactionDT`. For each block, in this order: the labels
a read at this instant may see are matured into their components; every feature is read for every
row of the block; the block's edges go in. So a transaction never sees a simultaneous one,
whichever nodes they share, which is the same tie rule stage 4 applies. Stage 0 measured 573,349
distinct timestamps across 590,540 rows, so the rule moves real rows.

**The entity's component is anchored on its card1 node.** The ADR 0001 entity key is
`card1 + addr1 + card_start_day` and `card1` is its null-free component, so every one of an
entity's transactions attaches to the same `card1` node and they are always one component. The
component the entity currently belongs to is the component of that node, and it is defined before
the entity's first transaction, because the card is in the graph before the entity is. When the
card has never been seen the component is empty and the count is zero.

The eight features, every one read before the row's own edges are added:

| Feature | Definition | Null |
| --- | --- | --- |
| graph_component_size | transactions in the card1 node's component | never; zero for an unseen card |
| graph_degree_card1 | strictly earlier transactions carrying this card1 | never |
| graph_degree_addr1 | the same for addr1 | addr1 is null |
| graph_degree_device | the same for the normalised DeviceInfo | DeviceInfo is null |
| graph_n_entities_on_device | distinct entities among those earlier device transactions | DeviceInfo is null |
| graph_component_fraud_rate | fraud rate over the matured train labels in the component | no matured label |
| graph_component_fraud_rate_others | the same with the row's own entity's labels removed | no other entity's label |
| graph_component_n_labelled | the count the rate is over | never |

The brief also asks for the distinct `addr1` values seen for this `card1` so far. That is the
typed degree of the card1 node towards addr1 nodes, and stage 4 already ships it as
`geo_n_distinct_addr1_on_card`. One number, one source: the module does not emit it a second
time, and `tests/test_causality.py` asserts that a brute-force graph reading of it equals stage 4's
column on every sampled row.

### How the tests enforce the rule

`tests/test_causality.py` rebuilds the graph from scratch with networkx over the rows with
`TransactionDT < t` and reads every feature off that graph, with the lag applied as a plain filter
on the day index. The reference shares nothing with the module: no incremental state, no node id
arithmetic, no maturation pointer. Three properties are checked, as in stage 4: brute-force
agreement on a sample built to contain cold cards, tied timestamps, rows with a device and rows on
both sides of the first matured label; truncation invariance, compared exactly, since every count
is an integer and every rate is a ratio of two integers computed the same way on both sides; and
the tie rule, on rows sharing a timestamp on one card. Two more are specific to the graph: a
label matures only after the lag, and the label source refuses rows outside the training window.
Two of the tests run on the real file under `needs_data`.

Four deliberate breaks were applied, the suite run, and each reverted:

| Break | Change | Tests failed | First failure |
| --- | --- | --- | --- |
| Read after add | the block's edges go in before the block is read | 4 | brute-force rebuild, tie test |
| Lag ignored | a label matures as soon as its transaction is strictly earlier | 2 | brute-force rebuild, lag test |
| Row by row | blocks of one row instead of one timestamp | 3 | brute-force rebuild, tie test, row-order test |
| Final component | every edge in the stream added before any read | 6 | brute-force rebuild, truncation, append, tie |

The last is the shape ADR 0024 refuses, and it fails six tests, two of which the first break does
not fail: truncation invariance and append invariance. A brute-force check against a reference
that reads `TransactionDT < t` catches a wrong read order; only the invariance checks catch a
feature that was right about the past and wrong because it also read the future.

---

## 1. The graph is one component

All from `reports/graph_summary.json`, block `end_of_train_graph`, the union-find over every train
row read after the last one was added.

| Quantity | Value |
| --- | --- |
| Train transactions | 413,378 |
| Components | 23 |
| Transactions in the largest | 413,349, a share of 0.9999 |
| Nodes in the largest, value nodes included | 428,847 |
| Transactions alone in a component of size 1 | 17 |
| The ten largest, in transactions | 413,349, 3, 3, 2, 2, 2, 1, 1, 1, 1 |
| Value nodes present | 15,521 |

Why, from block `link_columns`, measured on train:

| Column | Values | Null rate | Heaviest value | Its share of train rows |
| --- | --- | --- | --- | --- |
| card1 | 12,242 | 0.0000 | 7919 | 0.0245 |
| addr1 | 318 | 0.1150 | 299.0 | 0.0797 |
| P_emaildomain | 59 | 0.1555 | gmail.com | 0.3852 |
| R_emaildomain | 60 | 0.7507 | gmail.com | 0.1032 |
| DeviceInfo_norm | 281 | 0.7790 | windows | 0.0888 |
| dist1 | 2,561 | 0.6187 | 0.0 | 0.0310 |

None of the six columns identifies anyone. A single `P_emaildomain` value covers 0.3852 of train
rows, a single `addr1` value covers 0.0797, and the degree distribution of the value nodes says
the same thing from the other side: the median `P_emaildomain` node has 232 transactions on it
and the largest has 159,214; the median `addr1` node has 3 and the largest 32,938. Every column
has a few values that a large share of transactions pass through, and a component is transitive.

**Which columns the collapse depends on.** Block `link_variants` rebuilds the end-of-train graph
with one link column removed, with one alone, and with each paired with card1:

| Graph | Components | Share in the largest |
| --- | --- | --- |
| all six | 23 | 0.9999 |
| without card1 | 962 | 0.9977 |
| without addr1 | 140 | 0.9996 |
| without P_emaildomain | 23 | 0.9999 |
| without R_emaildomain | 23 | 0.9999 |
| without DeviceInfo_norm | 34 | 0.9999 |
| without dist1 | 24 | 0.9999 |
| card1 alone | 12,242 | 0.0245 |
| card1 with addr1 | 1,306 | 0.9887 |
| card1 with P_emaildomain | 495 | 0.9973 |
| card1 with R_emaildomain | 5,162 | 0.9102 |
| card1 with DeviceInfo_norm | 5,161 | 0.9200 |
| card1 with dist1 | 5,718 | 0.8528 |

Removing any one column leaves the largest component above 0.99 of the split. Adding any one
column to card1 takes the largest component from 0.0245 to at least 0.8528. There is no pair of
these columns that gives a graph with structure in it, so this is a fact about the columns and not
about which ones were picked.

**What excluding hubs does.** Block `hub_sweep` refits the graph with values that at least a share
of train rows carry linking nothing. The hub set is a fitted object: `fit_hub_values` learns it on
the train split through `assert_train_only`. The first share is stage 3's rare-tail line,
`encoders.RARE_TAIL_SHARE`, below which a categorical level is collapsed; the other two are a
decade and two decades below it. A grid, and no point on it is a decision.

| Hub share | Hub values, all columns | Components | Share in the largest | Share alone | Largest three |
| --- | --- | --- | --- | --- | --- |
| none | 0 | 23 | 0.9999 | 0.0000 | 413,349, 3, 3 |
| 0.01 | 60 | 40,760 | 0.8871 | 0.0942 | 366,725, 63, 29 |
| 0.001 | 308 | 214,622 | 0.4104 | 0.5066 | 169,658, 149, 146 |
| 0.0001 | 1,462 | 334,137 | 0.0058 | 0.7890 | 2,377, 831, 688 |

The hub value counts per column at 0.0001 are 1,089 for card1, 70 for addr1, 54 for
P_emaildomain, 41 for R_emaildomain, 49 for DeviceInfo_norm and 159 for dist1, and the rows that
pass through them are 0.8255, 0.8825, 0.8441, 0.2482, 0.2178 and 0.3342 of train. There is no
share on the grid at which the graph has both structure and coverage: at the rare-tail line the
largest component still holds 0.8871, one decade down it holds 0.4104 while 0.5066 of transactions
sit alone, and two decades down the largest component has 2,377 transactions in it and 0.7890 of
the split is alone. `figures/graph_components.png` draws the left panel of this from the artifact.

---

## 2. Component size is a clock

All from block `per_feature`, the point-in-time features on the whole stream.

`graph_component_size` has a Spearman correlation of 0.9768 with `TransactionDT` on the train
split. Its PSI against train, binned on train, is 11.4954 on val and 11.5034 on test, against the
0.25 that stage 2 adopted as a large shift. Its fraud rate by decile on train runs 0.0281, 0.0263,
0.0191, 0.0379, 0.0401, 0.0443, 0.0396, 0.0353, 0.0351, 0.0459: the deciles are the ten twelfths of
the training window in order, and the curve is the weekly fraud rate stage 2 measured, trough in
the third and rising after it. On val and test the deciles are the tenths of that split's own
month, and the curves have nothing in common with the train curve or with each other. The right
panel of `figures/graph_components.png` draws the three.

`graph_component_n_labelled` is the same clock with a 14 day delay: Spearman 0.9804 with time,
PSI 10.6828 and 10.6975.

`graph_component_fraud_rate` is the lagged training rate. Its maximum over 590,540 rows is
0.035166, which is the train split's rate of 0.03517 to four places, and its single-feature AUC is
0.5147 on train, 0.4935 on val and 0.4682 on test. Section 4 has the rest of that story.

The one non-trivial thing the component size does say is about its zero side. A row whose card1
has never been seen has a component of size 0, and there are 12,242 such rows on train (one per
card1 value), 675 on val and 636 on test. On train they run at 0.0230 against 0.0355 for the rest,
Wilson intervals 0.0205 to 0.0259 against 0.0350 to 0.0361, disjoint. On val they run at 0.0519
against 0.0342, intervals 0.0375 to 0.0713 against 0.0330 to 0.0354, disjoint the other way. A new
card inside the training window is safer than average and a new card in a later window is riskier:
the same two populations stage 2 and stage 4 called cold, seen again at the card grain. It is a
sign change and not noise, and it fails the stage 4 rule for the same reason `dup_prior_count`
did.

---

## 3. The degree features, and the rule none of them passes

The four features in the candidate set are node-local counts, not component counts, so they are
not clocks in the same way: Spearman with time on train is 0.3466 for `graph_degree_card1`,
0.6314 for `graph_degree_addr1`, 0.4631 for `graph_degree_device` and 0.4514 for
`graph_n_entities_on_device`. They do grow with the stream, and their drift says so: PSI against
test is 0.3362, 1.1796, 0.1816 and 0.2219, which is the same behaviour stage 4 measured for its
all-history counts against its windowed ones.

The ship rule is stage 4's, written in ADR 0022 before this stage existed: a count feature ships
when its zero-against-positive separation has the same sign on all three splits with disjoint
Wilson intervals on each. Applied to the graph, from block `per_feature`:

| Feature | Zero side, train | Zero side, val | Zero side, test | Direction stable | Disjoint everywhere |
| --- | --- | --- | --- | --- | --- |
| graph_component_size | 12,242 rows at 0.0230 | 675 at 0.0519 | 636 at 0.0487 | no | no |
| graph_degree_card1 | 12,242 at 0.0230 | 675 at 0.0519 | 636 at 0.0487 | no | no |
| graph_degree_addr1 | 318 at 0.0189 | 12 at 0.0000 | 2 at 0.0000 | yes | no |
| graph_degree_device | 281 at 0.0747 | 16 at 0.0625 | 17 at 0.0000 | no | no |
| graph_n_entities_on_device | 281 at 0.0747 | 16 at 0.0625 | 17 at 0.0000 | no | no |

None passes. For the card, the zero side reverses between windows as above. For the other three,
the zero side is the first appearance of a value, 318 rows on train for addr1 and 281 for the
device, and those cells are too thin to be disjoint from anything. So the switch defaults to off,
and block `decision` records the rule, the outcome and the list it produced, which is empty.

The single-feature AUCs agree that the marginals are flat: 0.5364, 0.5166 and 0.5153 across the
splits for `graph_degree_card1`; 0.4984, 0.5022 and 0.5003 for `graph_degree_addr1`; 0.5010,
0.4554 and 0.4358 for `graph_degree_device`; 0.4940, 0.4493 and 0.4297 for
`graph_n_entities_on_device`.

**What the rule does not see, recorded and not acted on.** The zero-against-positive comparison
cuts at zero, and for an all-history degree the interesting cut is not at zero. The lowest decile
of `graph_degree_device` on rows that carry a device runs at 0.1210 on train (degree 0 to 300),
0.1342 on val (0 to 561) and 0.1974 on test (0 to 701), against 0.0679, 0.0839 and 0.0917 for
all rows carrying a device on each split. A rarely seen device is riskier on all three splits by
a factor of roughly two, and the shape above that decile is not monotone. That is a real shape in
the tables and no rule in this repo selects on it; writing one after seeing the number is the
mistake ADR 0013 and the stage 4 notes both record. It stays a candidate, the four features stay
in the candidate set, and stage 6 measures them with a model, which reads a shape a marginal cut
cannot.

---

## 4. The component fraud rate, measured and dropped

All from block `component_fraud_rate`. ADR 0025 is the decision.

The mechanism is leakage-safe in the sense the brief asks for, and the tests are what say so.
Labels enter through `label_source`, which is decorated with `train_only` and so runs
`assert_train_only` before a label is copied; 413,378 labels with 14,538 fraud, the train split
exactly, and the artifact asserts the counts against `reports/split_summary.json`. A label matures
into its component only when its day is at least `lag_days` before the reading row's day, and the
lag is read from `reports/encoding_spec.json` rather than restated: 14, ADR 0014. `label_source`
has no default for it. All 413,378 labels have matured by the end of the stream, and a test
asserts that a label one day short of maturing is not read.

What the feature is worth, given the graph, is nothing, and the reason is section 1: the component
is the training window. Over the rows where it is defined its maximum is 0.035166, and the
single-feature AUC is 0.5147, 0.4935 and 0.4682 across the splits. On val and test it is one number
per timestamp block, the lagged rate as of the last matured day.

The variant that removes the row's own entity's labels is worse, and worse in an instructive way:

| Feature | Val AUC | Val, seen entities | Val, unseen | Test AUC | Test, seen | Test, unseen |
| --- | --- | --- | --- | --- | --- | --- |
| graph_component_fraud_rate | 0.4935 | 0.4694 on 41,536 | 0.4962 on 46,370 | 0.4682 | 0.4900 on 30,139 | 0.4720 on 57,806 |
| graph_component_fraud_rate_others | 0.4098 | 0.3457 on 41,533 | 0.4962 on 46,370 | 0.2896 | 0.0244 on 30,139 | 0.4720 on 57,806 |

The decomposition is ADR 0015's, on whether the training window ever saw the row's entity. With
the component being everything, the subtraction `(Y - y_own) / (N - n_own)` is the global rate
minus the entity's own labels, so on a seen entity the feature is that entity's own training
labels with the sign flipped. An AUC of 0.0244 on test seen rows is 0.9756 the other way, which is
the 0.9877 ADR 0015 measured for a direct entity target encoding, reached by subtraction. On
unseen entities there is nothing to subtract and the number is 0.4720. Removing the entity's own
labels from a rate does not remove the entity's labels from the feature; it changes their sign.

**On a graph with components in it.** Block `hub_excluded_rates` rebuilds the same three features
on each graph of the hub sweep, same lag, same label source, same decomposition, because the
six-column graph cannot say what a component rate would be worth where a component is smaller
than the file. At the two decades point, share 0.0001:

| Feature | Defined on | Val AUC | Val, seen | Val, unseen | Test AUC | Test, seen | Test, unseen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_component_fraud_rate | 0.1597 of test rows | 0.7087 | 0.9379 | 0.6415 | 0.6638 | 0.9858 | 0.6061 |
| graph_component_fraud_rate_others | 0.1527 of test rows | 0.6270 | 0.5250 | 0.6415 | 0.5987 | 0.5169 | 0.6061 |

The seen-side numbers of the first row, 0.9379 and 0.9858, are ADR 0015's finding again: a small
component that contains the entity's own earlier transactions is an entity label lookup. The
unseen-side numbers are not that. 0.6415 on 7,347 val rows and 0.6061 on 9,025 test rows come
from other entities' matured labels in a component the row's entity has never contributed a label
to, with the same sign on both later splits, and the others variant reproduces them exactly
because on an unseen entity there is nothing to subtract. That is a neighbourhood signal that is
not the entity's own labels, and it exists on this file.

It does not ship, for two reasons that are both measured. The graph it lives on is defined by a
hub share that is a grid point, two decades below the one number in the repo that resembles it,
and nothing derives it. And it is defined on 0.1597 of test rows, because at that share 0.7890
of train transactions sit alone and most rows have no labelled neighbour at all. ADR 0025 records
the number so a later stage that wants a component rate starts from it rather than from the brief.

---

## 5. What a full-dataset component computation would have handed the model

All from block `transductive_leak`. ADR 0024 is the decision. The comparison is the construction
this stage refuses: one graph over all 590,540 rows, every row reading the size of the component
it finally sits in and the fraud rate of every other transaction in it, val and test labels
included, no lag. It is computed in `scripts/graph.py` as a measurement and is not importable from
the package.

The graph over every row is 17 components with 590,521 of 590,540 transactions in the largest and
14 alone. On it the transductive component size exceeds the causal one on every row, by a median
factor of 2.02 over the causal size plus one, and its AUC is 0.5 on every split to three places
because it is one number for every row bar the 19 outside the largest component. The transductive leave-one-out
fraud rate has an AUC of 0.0000 on val and test and 0.0000125 on train. One component, so the
leave-one-out rate is `(Y - y_i) / (N - 1)`, which is the row's own label with the sign flipped:
a fraud row reads a lower rate than a non-fraud row, always. That is the purest form the leak
takes, and it is what a naive groupby over connected components on this file would have produced.

On the hub-excluded graph at share 0.0001, where components are small enough for the comparison
to mean something (473,480 components over every row, 19,620 in the largest, since a hub set
fitted on train does not bound values that become common later):

| Split | Transductive leave-one-out rate | Causal lagged rate | Gap |
| --- | --- | --- | --- |
| train | 0.8091 | 0.6359 | 0.1732 |
| val | 0.8462 | 0.7087 | 0.1375 |
| test | 0.8415 | 0.6638 | 0.1777 |

The causal number already carries the entity's own labels on seen entities, per section 4. The
extra 0.14 to 0.18 of AUC is what reading the component's future and the later splits' labels
adds on top, and none of it exists at authorisation.

---

## 6. The serving footprint, and why the component features would be expensive even if they worked

The union-find over every value node and every transaction node seen so far is unbounded. A
component is a fact about every edge ever added, so it cannot be windowed the way stage 4's
velocity ring is, and by the end of the training window it holds 15,521 value nodes and 413,378
transaction nodes with their per-root counters. The build over the whole stream takes about 2.4
seconds offline, which is not the concern; the concern is that a serving path would hold the
whole structure and update it on every authorisation.

The degree features need one counter per value node of their column, 12,242 for card1, 318 for
addr1 and 281 for the device on train, and `graph_n_entities_on_device` needs one set of entity
ids per device node. Those are the state a serving path holds if stage 6 finds the candidate set
worth its width. The catalogue in the artifact records the state per feature.

---

## What is not established

- Whether any graph feature contributes to a model. The marginal instruments say no; the device
  degree's lowest decile says something a marginal cut does not measure; stage 6 measures the
  block with a model and the switch is there for it.
- Whether a component fraud rate on a fragmented graph is worth building. 0.6415 and 0.6061 on
  unseen entities is measured; the hub share that produces the graph is a grid point, and no rule
  in this repo derives one.
- What a link column means. None of the six is an identifier and the artifact measures that; what
  `dist1` is a distance between, and what `card1` is, stay as stage 0 left them.
- Why a new card is safer inside the training window and riskier after it. The same reversal
  stage 4 recorded for a new entity, now at the card grain, with the same open explanation.
