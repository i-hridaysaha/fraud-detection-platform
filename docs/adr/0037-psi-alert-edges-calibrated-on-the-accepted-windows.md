# 0037: PSI bands are reported as the convention says; the alert is calibrated on this file

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 9

## Context

The population stability index has a textbook reading: below 0.10 stable, 0.10 to 0.20 (some
say 0.25) watch, above that alert. Stage 2 measured what this file does to that reading with
nothing wrong: between the training window and the test window, 30 of 432 raw columns sit above
0.10 and 22 above 0.25, with a median of 0.0207, and its artifact says in so many words that
stage 9 sets the monitoring thresholds against that baseline rather than against the convention.
The question this record answers is how, at the grain the monitor runs at, which is a week and
not a window.

## Options considered

**The textbook edges, per column, as the alert.** Rejected on the measurement below: on the
shipped model's 186 input columns, every one of the ten in-control weekly batches of the val and
test windows puts between 16 and 20 columns above 0.20. An alert that fires every week on a
model that is fine is not an alert.

**The stage 2 window-level PSI as each column's edge.** Rejected. It is the right baseline at the
wrong grain: a weekly batch is a fifth of the window and its PSI against train moves week to
week, so the target encodings that sit at 8 to 11 on the whole window swing by units between
weeks, and a column at 0.3 on the window is at 0.25 one week and 0.43 the next. Held out, a
window-level edge fires on the weeks above the window's own value, which is about half of them.

**Per column, the larger of 0.20 and the worst value the column took on an in-control weekly
batch of the accepted windows; and a batch-level flag that needs more columns past their edges
than any held-out in-control week showed.** Chosen. The edge is the control chart's idea of a
limit: the range the process showed while it was known to be in control, and the accepted
windows are exactly the period the model was accepted on. The count edge exists because a
single column past a max-of-nine edge is a one-in-ten event per column and there are 170 of
them; measured by leaving each batch out, one such column appears on 4 of the 10 weeks and two
never do, so the flag needs two.

**The score PSI alone as the batch flag.** Rejected, on the probe recorded in ADR 0036: a
tripling of the C columns cost the model a third of its PR-AUC and moved the score PSI to 0.076.
The score path is kept alongside the column path, because a unit change on the amount moved the
score PSI to 0.54 while raising only two columns, so each path sees what the other misses.

**The target encodings as alertable columns.** Rejected. A training row is encoded at its own
lagged day (ADR 0014), so the training histogram of a `_te` column is a smear across the window
while every served row reads the table frozen at the window's end. Against that reference the
in-control weekly PSI of `card6_te` is 11.26, of `ProductCD_te` 10.27, of `addr2_te` 9.89, and
they swing by units between weeks. That is the encoder's clock, not the population's; the level
mix these columns encode is watched through the `_cat` and `_freq` columns of the same sources.
They are reported with their bands and never alert.

## Decision

`monitoring.build_reference` fits decile histograms on the training window for the model's
columns and its scores (the stage 2 binning: `eda._bin_edges` and `eda.bin_series`, floor
`eda.PSI_FLOOR`, so the whole-window numbers reproduce stage 2's), cuts the accepted windows into
seven-day batches, and stores per column the worst batch PSI, the edge `max(0.20, worst)`, the
score edge the same way, and `feature_count_edge`, the largest number of alertable columns any
batch put past edges calibrated on the other nine. A column alerts strictly above its edge. The
batch flag is the score past its edge or more alertable columns past theirs than
`feature_count_edge`. The bands 0.10 and 0.20 are reported per column unchanged.

## Evidence

`reports/monitoring/drift_reference.json` and `reports/monitoring/drift.json`, regenerated with
`make monitor-drift`; 186 columns, 170 alertable, ten batches of 8,111 to 21,273 rows.

| Quantity | Value | Field |
| --- | --- | --- |
| Whole test window, columns above 0.10 / 0.20 / 0.25, of 186 | 20 / 18 / 18 | `calibration.n_columns_whole_test_above_*` |
| Median whole-test PSI | 0.0079 | `calibration.median_whole_test_psi` |
| Stage 2, raw columns above 0.10 / 0.25, of 432 | 30 / 22 | `against_stage_2.stage_2_summary` |
| Raw columns held by both and their whole-test PSI here against stage 2 | 59, all equal to 1e-9 | `against_stage_2` |
| Columns above 0.20 per in-control week, textbook | 16 to 20, mean 18.4 | `in_control.textbook_alert_band_per_batch` |
| Columns whose edge rose above 0.20 | 20: 16 target encodings, `id_13` 1.104, `D11` 0.427, `vblockmean_V1` 0.393, `id_30_cat` 0.265 | `calibration.edges_raised` |
| Score PSI per in-control week | 0.0059 to 0.0628 | `in_control.score_psi_per_batch` |
| Score edge | 0.20 | `calibration.score_alert_edge` |
| Held-out weeks with one alertable column past its edge / two | 4 of 10 / 0 of 10 | `in_control.leave_one_out.counts` |
| `feature_count_edge` | 1 | `calibration.feature_count_edge` |
| Held-out false alert rate, batch flag | 0.0 | `in_control.leave_one_out.rate_any_alert` |

The four held-out single-column alerts were `vblockmean_V1` (val 0), `D11` (test 0), `id_13`
(test 1) and `id_30_cat` (test 4): three of the four are stage 2's drifters, moving week to week
around the edge their own worst week set.

## Consequences

- The alert edges are this model's on this file. A new champion rebuilds its reference from its
  own fit window and calibrates on the windows it was accepted on (`lifecycle_demo.py` does this
  on promotion, with four batches rather than ten; fewer batches, a looser edge).
- The held-out false alert rate is measured on ten weeks. On a new in-control week it is an
  estimate with the precision ten weeks give.
- A shift confined to one column does not raise the batch flag unless it moves the score. The
  probe in `docs/notes/stage-09.md` that shifted `C13` alone raised one column and a score PSI of
  0.016; that drift reaches the lifecycle through the decay job or not at all.
- The `_te` finding is a finding about the pipeline, not the monitor: the served distribution of
  a target encoding is not the distribution the model was trained on. Whether that costs accuracy
  is not established here; `reports/adversarial_validation.json` measures how completely it
  separates train from test, and the stage document reads it.
