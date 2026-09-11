# 0025: The component fraud rate is built with a lag, measured, and dropped

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 5

## Context

The stage 5 brief asks for the fraud rate within the component an entity belongs to, computed from
train rows only and subject to the stage 3 label lag, and calls it the strongest and most dangerous
feature in the stage. The instruction is to route it through `assert_train_only`, apply the lag,
and leave it out with the reason if it cannot be made leakage-safe, rather than ship it flagged.

Two earlier decisions bear on it. ADR 0014 set the label lag at 14 days on a sweep whose useful
column was the train-minus-validation gap, and recorded that a serving path must read labels as
of `d - 14` and not as of `d`. ADR 0015 excluded the entity key from target encoding on a
measurement: at lag 14 the encoding scored 0.9877 on validation rows whose entity the training
window saw and 0.4915 on rows it did not, and the reason is that 0.9675 of multi-transaction
entities are label-pure, so a summary of an entity's labels is a copy of its label. A component
contains the entity's own earlier transactions, so a component rate is at least an entity rate,
and ADR 0015 applies to it before a single number is measured.

## Options considered

**Leave it out on ADR 0015 alone.** Defensible and rejected, because the brief asks whether it
can be made leakage-safe and that is a question with a measurable answer, and because a
component rate is not only an entity rate: it also reads the labels of other entities in the
component, which ADR 0015 said nothing about.

**Build it with train-only labels and the lag, and measure it as ADR 0015 measured the entity
encoding.** Chosen. Labels enter through `label_source`, decorated with `train_only`, so the frame
they are read from has been through `assert_train_only`. A transaction on day `j` contributes its
label to a read on day `d` only when `j <= d - lag_days`, and the lag is an argument read from
`reports/encoding_spec.json` by the driver; the module holds no copy of the number. Validation and
test are then split on whether the training window saw the row's entity, and the rate is scored
on each side.

**Build a second rate with the row's own entity's labels removed.** Also chosen, as a measurement.
The hypothesis was that subtracting the entity's own matured labels from both numerator and
denominator would leave the neighbourhood signal without the entity-label copy.

**Smooth the rate toward the prior, as the stage 3 encoder does.** Rejected for this stage. A
smoothed rate would rank rows differently only through the count, and the count is emitted
beside the raw rate as `graph_component_n_labelled`, so a model has both. Smoothing would also
make the brute-force reference an approximation, since the prior is a running quantity, where the
raw ratio of two integers is checked exactly.

## Decision

Neither rate ships. Both are computed by the module when named, neither is in the candidate set,
and the switch that stage 6 flips does not turn them on: they are reachable only by naming them.
`graph_component_n_labelled` goes with them.

The mechanism is leakage-safe in the sense the brief asks for. The label source is train-only by
the guard, the lag is stage 3's and is enforced by the maturation pointer, a label one day short of
maturing is not read (asserted by a test), and the brute-force reference applies the lag as a
plain filter on the day index and agrees on every sampled row. What the mechanism produces on
this graph is the decision, and it is measured below.

## Evidence

`reports/graph_summary.json`, blocks `component_fraud_rate` and `hub_excluded_rates`, regenerated
with `make graph`. The lag is 14 days, read from `reports/encoding_spec.json`; the label source is
413,378 rows with 14,538 fraud, the train split, and all of them have matured by the end of the
stream.

**On the six-column graph the rate is the lagged training rate.** The component is the training
window (ADR 0024), so the rate over its matured labels is one number per timestamp block. Its
maximum over 590,540 rows is 0.035166, the train split's rate to four places. Single-feature AUC:
0.5147 on train, 0.4935 on val, 0.4682 on test. There is no component to have a rate of.

**Removing the entity's own labels does not remove them; it flips their sign.** The ADR 0015
decomposition:

| Feature | Val AUC | Val, seen entities | Val, unseen | Test AUC | Test, seen | Test, unseen |
| --- | --- | --- | --- | --- | --- | --- |
| graph_component_fraud_rate | 0.4935 | 0.4694 | 0.4962 | 0.4682 | 0.4900 | 0.4720 |
| graph_component_fraud_rate_others | 0.4098 | 0.3457 | 0.4962 | 0.2896 | 0.0244 | 0.4720 |

Seen rows are 41,536 of validation and 30,139 of test. With the component being everything, the
others rate is `(Y - y_own) / (N - n_own)` with `Y` and `N` the whole training window's matured
totals, so on a seen entity it is a monotone function of that entity's own training labels, sign
reversed. 0.0244 on test seen rows is 0.9756 the other way, which is ADR 0015's 0.9877 reached by
subtraction. On unseen rows there is nothing to subtract and the two features are the same
number, 0.4962 and 0.4720.

**On a graph with components in it the neighbourhood signal exists.** Block `hub_excluded_rates`
rebuilds both rates on the graphs of the hub sweep, hub set fitted on train through the guard,
same lag and label source. At share 0.0001, the most fragmented point on the grid:

| Feature | Defined on, test | Val AUC | Val, seen | Val, unseen | Test AUC | Test, seen | Test, unseen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| graph_component_fraud_rate | 0.1597 of rows | 0.7087 | 0.9379 | 0.6415 | 0.6638 | 0.9858 | 0.6061 |
| graph_component_fraud_rate_others | 0.1527 of rows | 0.6270 | 0.5250 | 0.6415 | 0.5987 | 0.5169 | 0.6061 |

Unseen rows are 7,347 of the defined validation rows and 9,025 of the defined test rows. The seen
side of the first row is ADR 0015 again, 0.9379 and 0.9858. The unseen side is not: those rows'
entities contributed no label to any component, so 0.6415 and 0.6061 are other entities' matured
labels in a component the row was linked into, with the same sign on both later splits, and the
others rate reproduces them exactly. At the two coarser grid points the unseen-side numbers are
0.5026 and 0.4839 at 0.01, and 0.5433 and 0.5030 at 0.001.

**What the same construction without the lag and without the train restriction would give** is
in block `transductive_leak` and ADR 0024: 0.8462 on val and 0.8415 on test at the same share,
against 0.7087 and 0.6638 for the causal rate.

## Consequences

- The candidate set stage 6 measures holds no label-derived feature. If stage 6 wants one, it
  names it, and it inherits this measurement.
- A rate that subtracts the entity's own labels is not a rate without them. This generalises
  beyond graphs: any statistic of the form `(total - own) / (count - own count)` over a
  neighbourhood that is mostly the training window carries the entity's labels with the opposite
  sign, and the ADR 0015 decomposition is the test that shows it.
- There is a neighbourhood signal on this file that is not the entity's own labels: 0.6415 and
  0.6061 on unseen entities, on 0.1597 of test rows, on a graph fragmented at a hub share of
  0.0001. It does not ship because that share is a grid point two decades below the one line in
  the repo that resembles it and nothing derives it, and because the graph that produces it
  leaves 0.7890 of train transactions alone. A later stage that wants it has the number and the
  block to start from, and would need a derivation for the hub line first.
- The lag is stage 3's and is not a claim about how long a chargeback takes to mature, which
  stage 0 did not establish and this stage does not either. If ADR 0014 is superseded, the driver
  reads the new lag from the same artifact field and this ADR's numbers are regenerated, not
  edited.
