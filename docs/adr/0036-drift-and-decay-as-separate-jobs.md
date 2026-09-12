# 0036: Input drift and performance decay are two jobs on two clocks, never one

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 9

## Context

A served model fails in two ways that look alike from a dashboard and are nothing alike
underneath. Its inputs can move: new cards, a browser release, an upstream change to how a column
is computed. Its accuracy can fall: the patterns it learned stop holding, whether or not the
inputs moved. The first is visible the day a batch closes, from the rows alone. The second is
visible only once the batch's labels have arrived, which on chargeback-labelled data is weeks
later (ADR 0032 could not say how many, and this stage cannot either). A monitor that reports one
number for "model health" has to have folded the two together, and whichever way it folded them
it is wrong on one clock: either it retrains on a harmless shift, or it waits for labels to
confirm a failure the inputs announced a month earlier.

The stage brief names this as the design point that matters. This record is what the two jobs
each can and cannot say, measured on this file, and why the join between them lives in the
lifecycle and nowhere in the monitoring.

## Options considered

**One health job that reads inputs and labels together and emits one flag.** Rejected. Its
schedule is the slower clock's, because a metric needs labels, so the drift signal is delayed by
the maturity window for no reason. Its flag has one meaning, so a retrain triggered by it cannot
be told from a retrain triggered by decay, and ADR 0038 needs to tell them apart to weight them
differently. And a job that reads labels on the day a batch closes reads immature ones, which
ADR 0032 measured as the worst thing a label can do.

**A drift job only, with decay inferred from the drift.** Rejected on a measurement in this
stage. A probe tripled the seven C columns the shipped model splits on hardest and cost it a
third of its test PR-AUC (0.5451 to 0.4094, scratch probe before the demo was built); the score
distribution moved by a PSI of 0.076, under the alert band. The model's own summary of its inputs
did not announce the damage. Drift cannot stand in for decay.

**A decay job only.** Rejected. It is the failure itself and the primary trigger, and it is
blind for the maturity window. In the demo (`reports/lifecycle_demo.json`) the injected shift is
visible to the drift job on the batch it starts in and to the decay job five cycles later; on
the assumed thirty-day maturity that is a month of a model scoring changed inputs with no
signal at all, and the early trigger had a replacement promoted one cycle before the first
decayed batch could be read.

**Two jobs, two schedules, two artifacts, joined only in the retrain decision.** Chosen.

## Decision

`monitoring.drift_report` runs when a batch closes. It takes the model's prepared input columns
and its scores, never a label (it raises if handed one), and compares them with a reference
stored from the training window: PSI per column, PSI of the score distribution, the bands, and a
batch flag on edges calibrated as ADR 0037 records. It writes `reports/monitoring/drift.json`
and reads `reports/monitoring/drift_reference.json`. What it can say: the inputs moved, which
columns, by how much, and whether the model's output distribution moved with them. What it
cannot say: whether the model is worse.

`monitoring.performance_report` runs when a batch's labels have matured, `MATURITY_DAYS` after
it closed, and refuses the batch before that (`assert_mature`, to the second). It takes the
labels and scores of that batch and computes precision and recall at the shipped threshold and
PR-AUC with a row bootstrap, against the reference the champion was accepted at, as an
independent-draw difference. It writes `reports/monitoring/decay.json`. What it can say: the
model is worse than it was accepted at, by how much, and whether that clears the noise. What it
cannot say: why, or anything before the labels arrive.

Neither function imports the other's result. `lifecycle.should_retrain` reads both histories and
is the only place they meet: decay on the most recent mature batch is the primary trigger, the
drift flag on `SUSTAINED_CYCLES` consecutive batches the secondary, earlier one (ADR 0038).

## Evidence

`reports/monitoring/drift.json` and `reports/monitoring/decay.json`, regenerated with `make
monitor-drift` and `make monitor-decay`, on the ten weekly batches of the val and test windows
with nothing injected.

- The drift job, no labels: the score PSI against the training window ran 0.0059 to 0.0628 across
  the ten batches; the batch flag was raised on none.
- The decay job, labels mature: PR-AUC per batch ran 0.4942 to 0.7458 against a reference of
  0.5451 (0.5274 to 0.5614), precision 0.5739 to 0.7628, recall 0.3944 to 0.6485 at the shipped
  threshold; the largest drop was 0.0510 on test batch 2, no batch was called a decay.
- The two clocks on the demo (`reports/lifecycle_demo.json`, `make lifecycle-demo`): the shift
  injected at day 141.0 raised the drift flag at the close of the batch it started in, day
  148.0 (score PSI 1.8274, 15 columns past their edges); that batch's labels matured at day
  178.0 and the decay flag was raised on them at the next cycle, day 183.0 (PR-AUC 0.2012
  against the then champion's reference of 0.3829). Five cycles apart, on the assumed thirty-day
  maturity.

The probe that ruled out drift as a proxy for decay is recorded in `docs/notes/stage-09.md` as a
scratch measurement, not reproducible from a committed command. The demo injects the same C
shift with an amount change added, which the score does announce (PSI 1.8274 on the first
drifted batch); the shipped model scores 0.2875 at the gate on the first window holding drifted
rows against 0.6520 on the clean window one cycle earlier (`cycles[6]` and `cycles[7]`,
`challenger.gate.champion`).

## Consequences

- Two schedules. The drift job is cron on the batch close; the decay job is cron on the batch
  close plus the maturity window, and it does the same work for a batch that closed a month ago.
  A deployment that runs one job runs the wrong one.
- The maturity window is an input (`lifecycle.MATURITY_DAYS`, 30, assumed). The decay job is
  honest about what it refuses and wrong about nothing only if the window is at least the true
  delay; a window shorter than the delay reads immature labels and reports a decay that is not
  there, in the direction ADR 0032 measured.
- A shift the drift job sees and the decay job never confirms is the expected case, not a bug:
  most drift on this file is population turnover (`reports/adversarial_validation.json`, stage
  document). The drift flag trains a challenger; the gate decides.
- A decay the drift job never announced is the case the primary trigger exists for. Nothing here
  makes it faster than the labels.
