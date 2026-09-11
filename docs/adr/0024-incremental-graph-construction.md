# 0024: The graph is built incrementally over the stream, and a full-dataset component computation is refused

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 5

## Context

Stage 5 adds structural features over the graph that shared values draw between transactions:
`card1`, `addr1`, `P_emaildomain`, `R_emaildomain`, the normalised `DeviceInfo` and `dist1` each
join a transaction to every other transaction carrying the same value. The features the brief
asks for are the size of the component an entity sits in, the degree of its value nodes, the
number of entities sharing its device, and a fraud rate inside the component.

Graph features are the easiest place in the project to leak, and the reason is structural rather
than careless. A connected component is transitive over every edge in the graph, so a component
computed over the whole file joins a transaction to every later transaction it will ever share a
value with, and to every transaction those share values with, and so on. The component a row
"belongs to" at the end of the file is not the component it belonged to when it was scored, and
on this file the two are not close: the end-of-train graph over the six link columns is one
component holding 0.9999 of the training window, and the graph over all 590,540 rows is 17
components with 590,521 transactions in the largest, so a full-file computation would tell almost
every transaction, whenever it happened, that it sits with 590,520 others.

Stage 4 already settled the rule for a feature at time t, strictly before t on the whole stream,
and settled how to enforce it, by brute-force recomputation and truncation invariance rather than
by a window guard. This ADR is about applying that rule to a structure whose whole point is
transitivity, and about what the alternative would have handed the model.

## Options considered

**Connected components over the whole frame, once, with each row reading its final component.**
The construction a groupby or a `networkx.connected_components` call gives, and the one a first
draft of any graph feature reaches for. Rejected, and the rejection is quantified below rather
than argued: on the six-column graph the row's leave-one-out component fraud rate has a
single-feature AUC of 0.0000 on validation and test, because with one component the rate is the
row's own label with the sign flipped. On a graph with components in it, the same construction
adds 0.14 to 0.18 of AUC on top of the causal reading, and none of it exists at authorisation.

**Connected components recomputed per row over the rows before it.** Correct and unaffordable:
one component computation per transaction over a growing graph is quadratic in the stream, and it
is what `tests/test_causality.py` does on a sample to check the chosen option. Rejected as the
implementation, kept as the reference.

**Components computed once per day, or per fold, with each row reading the latest complete
snapshot.** Cheaper than per-row and still wrong in the same direction, since a row at the start
of a day would read the whole day's edges. Rejected because it replaces a leak with a smaller leak
and the brute-force test would still fail on it.

**An incremental union-find over the time-ordered stream, read before each block is added.**
Chosen. The graph is a disjoint-set structure with per-root counters, grown one timestamp block
at a time. For every block, in this order: labels that a read at this instant may see are
matured into their components; every feature is read for every row of the block; the block's
edges go in. A transaction never sees its own edges and never sees a simultaneous one, whichever
nodes they share. The build over 590,540 rows takes about 2.4 seconds.

## Decision

`fraud_platform.graph_features` builds the graph incrementally and nothing in the package computes
a component over a whole frame and hands it back per row. `static_graph` builds the same
incremental structure over whatever rows it is given, read after the last row, and is used on the
train split for the artifact's component-size distribution. The transductive construction is
computed in `scripts/graph.py` as a measurement of a wrong construction and is not importable.

The entity's component is anchored on its `card1` node. `card1` is the null-free component of the
ADR 0001 key, so every transaction of an entity attaches to the same node and the entity's
transactions are always one component; the component the entity belongs to is that node's, and
it is defined before the entity's first transaction because the card is in the graph before the
entity is. When the card has never been seen the component is empty and the size is zero.

Same-timestamp ties are excluded by construction: a block is read before it is added. The brief's
"number of distinct addr1 values seen for this card1 so far" is stage 4's
`geo_n_distinct_addr1_on_card` and is not emitted a second time; the graph reading of it is
asserted equal to stage 4's column in the causality tests.

## Evidence

`reports/graph_summary.json`, regenerated with `make graph`.

**The structure.** Block `end_of_train_graph`: 23 components over 413,378 train transactions, the
largest holding 413,349 (a share of 0.9999) with 428,847 nodes counting the 15,521 value nodes,
and 17 transactions alone in a component of size 1. Block `link_variants`: removing any one link
column leaves the largest component above 0.9977 of the split; `card1` alone gives 12,242
components and a largest of 0.0245, and pairing it with any one other column takes the largest to
at least 0.8528. Block `hub_sweep`: with values carried by at least 0.01 of train rows linking
nothing (stage 3's rare-tail line), the largest component holds 0.8871; at 0.001, 0.4104 with
0.5066 of transactions alone; at 0.0001, 0.0058 with 0.7890 alone.

**What the full-dataset computation leaks.** Block `transductive_leak`, the graph over all
590,540 rows with each row reading its final component and the labels of every other transaction
in it, no lag, val and test labels included, against the point-in-time features:

| Graph | Split | Transductive leave-one-out rate, AUC | Causal lagged rate, AUC |
| --- | --- | --- | --- |
| six columns | val | 0.0000 | 0.4935 |
| six columns | test | 0.0000 | 0.4682 |
| hub-excluded at 0.0001 | train | 0.8091 | 0.6359 |
| hub-excluded at 0.0001 | val | 0.8462 | 0.7087 |
| hub-excluded at 0.0001 | test | 0.8415 | 0.6638 |

The six-column graph over every row (block `transductive_leak`, `final_graph_over_every_row`)
is 17 components with 590,521 of 590,540 transactions in the largest and 14 alone. On it the
transductive component size exceeds the causal one on every row, and its AUC is 0.5 to three
places on every split because it is one number for every row but the 19 outside the largest
component. The transductive rate is `(Y - y_i) / (N - 1)` over one component, a monotone function
of the row's own label, hence 0.0000.

On the fragmented graph, where the comparison means something, the full-dataset rate is 0.14 to
0.18 of AUC above the causal one on the later splits. The causal number there already carries the
entity's own labels on seen entities (ADR 0025); the gap is what reading the future adds on top.
That graph over every row has 473,480 components with 19,620 transactions in the largest, against
2,377 on the training window alone: a hub set fitted on train does not bound values that become
common later, which is one more reason the hub line would need a derivation before anything was
built on it.

**That the incremental build is the point-in-time reading.** `tests/test_causality.py` rebuilds
the graph with networkx over `TransactionDT < t` for a sample of rows built to contain cold
cards, tied timestamps, rows with a device and rows on both sides of the first matured label, and
asserts every feature agrees; it asserts truncation and append invariance exactly; it asserts the
tie rule on one card; and two tests run on the real file under `needs_data`. Four deliberate
breaks were applied and reverted: reading after adding fails 4 tests, ignoring the lag fails 2,
row-by-row blocks fail 3, and adding every edge before any read, which is the full-dataset
construction, fails 6, including the two invariance tests that the read-order break does not.

## Consequences

- No graph feature in this repo reads a component computed over rows after the row. The tests are
  the enforcement, as in stage 4, and the artifact is the record.
- The component features are correct and useless on this file, and the two facts are separate.
  `graph_component_size` has a Spearman correlation of 0.9768 with `TransactionDT` on train and a
  PSI of 11.50 against test; it is a count of the transactions before the row. That is a
  property of the columns, measured in `link_variants` and `hub_sweep`, and not of the
  construction, and it is why the component features are outside the candidate set whatever the
  ship rule says.
- The union-find is unbounded serving state: a component is a fact about every edge ever added
  and cannot be windowed. The degree features need one counter per value node and the device
  entity count one set per device node, which is what a serving path would hold if stage 6 finds
  the candidate set worth its width.
- The anchor choice is a choice. A different anchor, the entity's own latest transaction for
  instance, gives the same component on this graph because an entity's transactions share their
  card node; on a graph where card1 hubs are excluded the two anchors would differ, and the
  artifact's hub-excluded measurements read the card anchor with an excluded card giving an empty
  component.
- A later stage that wants a component fraud rate on a fragmented graph starts from block
  `hub_excluded_rates` and from ADR 0025, not from the brief.
