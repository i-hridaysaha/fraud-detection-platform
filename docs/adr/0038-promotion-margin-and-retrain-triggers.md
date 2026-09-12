# 0038: The promotion margin and the retrain triggers come from stage 6's noise band

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 9

## Context

Stage 6 measured the noise on the primary metric and stopped there: `reports/metric_variance.json`
carries a `noise_band` block whose definition says a promotion margin inside it promotes on
noise and that stage 9 sets its margin above it. Three numbers in this stage have to come from
that band or from nothing: the margin a challenger must clear at the gate, the drop that makes a
matured batch a decay, and what "sustained" means for the drift flag. Every one of them could be
guessed to a plausible-looking value, and the rule this repo runs on is that none of them is.

## Options considered

**The margin as a round number from practice, 0.01 or 0.02.** Rejected. Both sit inside the
band: the widest paired half-width stage 6 measured is 0.0172 by row and 0.0592 by card. A gate
at 0.02 promotes on resampling noise at the card grain.

**The margin as the row-resample band, 0.0172 rounded to 0.02.** Rejected. The card resample
draws whole entities and respects the card-level labels ADR 0001 measured, its half-widths are
three to four times the row ones, and it is the one the gate uses. The narrower band belongs to
a resample the gate does not run.

**The margin as the widest paired half-width by card, rounded up at the second decimal.**
Chosen. The arithmetic, from `reports/metric_variance.json` (`make train`, `--sections
variance`; 1,000 resamples, seed 42, the ten stage 6 models on the test split):

    widest paired 95 percent half-width by row    0.0172   logistic_regression__smote vs xgboost__smote
    widest paired 95 percent half-width by card   0.0592   logistic_regression__class_weight vs random_forest__none
    noise_band.half_width                         0.0592   the larger
    rounded up at the second decimal              0.06     PROMOTION_MARGIN

A point advantage at least this large has a lower bound at or above zero for every pair stage 6
measured, which is what a margin fixed before any challenger exists can promise. The gate also
requires the pair's own paired interval, computed at gate time on the gate window, to lie above
zero, so a challenger that clears the margin on a window too small to support it is refused too.

**The decay tolerance as its own number, smaller than the margin so the trigger is sensitive.**
Rejected. A champion that has lost less than the margin cannot be replaced by a challenger that
merely recovers the loss: that challenger's advantage at the gate is the loss, under the margin,
refused. A trigger below the margin asks for work the gate discards by construction. So
`DECAY_TOLERANCE = PROMOTION_MARGIN`, and a decay is a drop of at least 0.06 whose
independent-draw interval excludes zero. The in-control check: the largest drop on any of the ten
in-control weekly batches is 0.0510 (`reports/monitoring/decay.json`, test batch 2), so the
tolerance clears every in-control week, by 0.009.

**Sustained drift as any number of cycles.** One cycle is a spike and is rejected by the brief.
Two is the smallest run that is not, and it is chosen; `SUSTAINED_CYCLES = 2`. The measured
basis: the batch flag's held-out false alert rate on the in-control weeks is 0.0 of 10
(`reports/monitoring/drift.json`, `in_control.leave_one_out.rate_any_alert`), and the longest
held-out run is 0, so no in-control run of two exists to calibrate against and two is the floor
the definition sets rather than a number the data chose.

**Drift as a trigger equal to decay.** Rejected (ADR 0036): drift is a reason to suspect the
failure, decay is the failure. Decay on the most recent matured batch triggers on its own;
drift triggers only sustained, and only when no decay evaluation says otherwise.

## Decision

`lifecycle.PROMOTION_MARGIN = 0.06`, with the arithmetic above in the comment beside it.
`lifecycle.DECAY_TOLERANCE = PROMOTION_MARGIN`. `lifecycle.SUSTAINED_CYCLES = 2`.
`lifecycle.gate` promotes when the paired difference on the gate window, resampled by entity, has
a point at or above the margin and a lower bound above zero, and says which condition failed
otherwise. `lifecycle.decay_flag` needs the tolerance and the interval. `lifecycle.should_retrain`
reads decay first and the drift run second.

## Evidence

The margin: `reports/metric_variance.json`, `noise_band`, and `tests/test_monitoring_artifacts.py`
asserts the constant is the half-width rounded up at the second decimal and nothing more.

The gate in both directions, on generated scores (`tests/test_lifecycle_units.py`): a better
challenger promoted, a worse one refused as not better, one better by less than the margin
refused with that reason, and one whose point clears a zero margin while its interval reaches
zero refused for the interval.

The gate on the demo (`reports/lifecycle_demo.json`, `make lifecycle-demo`, seed 42), four
challengers, entity resample, 1,000 draws, each with the paired difference and its interval:

| Cycle | Trigger | Gate window, days | Champion | Challenger | Difference | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | sustained drift | 125 to 139, clean | 0.6520 | 0.6082 | -0.0438 (-0.0644 to -0.0250) | refused: not better |
| 7 | sustained drift | 132 to 146, 12,841 of 36,852 rows drifted | 0.2875 | 0.3829 | +0.0954 (+0.0736 to +0.1185) | promoted |
| 8 | decay | 139 to 153, 33,104 of 38,502 drifted | 0.2557 | 0.2980 | +0.0424 (+0.0200 to +0.0644) | refused: better, but by less than the margin |
| 9 | decay | 146 to 160, all drifted | 0.2792 | 0.3413 | +0.0621 (+0.0403 to +0.0867) | promoted |

The decay trigger on the demo: batch 3 at PR-AUC 0.2012 against a reference of 0.3829, drop
0.1817 (lower bound 0.1385); batch 4 at 0.3025, drop 0.0803 (0.0322). Both cleared the
tolerance with the interval above zero. Batches 5 to 8 under the second promoted model: drops
0.0332, -0.0212, -0.0200, 0.0336, none with an interval excluding zero, no decay.

## Consequences

- The margin was exercised near its edge once: the cycle 9 promotion cleared 0.06 by 0.0021,
  inside that pair's own half-width of 0.0232. That verdict is the right one under the rule and
  it is not a firm one; a different resample seed could refuse it. The cycle 8 refusal at
  +0.0424 with an interval above zero is the margin doing what it is for: a real improvement,
  too small to act on.
- The reference the decay job compares with moves on promotion to the new champion's gate
  window, so the bar can ratchet down: the second promoted model's reference is 0.3413, and
  its batches at 0.31 to 0.36 are "no decay" against it while being far below the 0.5451 the
  shipped model was accepted at. Nothing here sets an absolute floor. That is a limit of the
  design as built, stated rather than fixed.
- The tolerance's headroom over the in-control weeks is 0.009 of PR-AUC on ten weeks. A longer
  decay batch would widen it and react later; the choice is not made here.
- Two consecutive cycles is a definition, not a measurement. On the demo the run reached six.
