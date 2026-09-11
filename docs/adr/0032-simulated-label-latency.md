# 0032: Label latency is simulated as a deterministic N day cutoff, swept over five values

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 7

## Context

A fraud label on card-present or card-not-present data arrives as a chargeback, weeks to months
after the transaction. At any training time the most recent rows carry labels that are not yet
final: a transaction that has not been charged back is not confidently legitimate, it is
undecided. A training set that treats those rows as legitimate teaches the model that the newest
patterns are safe. ADR 0014 lagged the target encoding for exactly this reason and said what it
could not say: "the lag is not a claim about how long a chargeback takes. Nothing in this repo
establishes that."

This file carries no chargeback dates. The labels in it are final, which is the one thing that
makes the experiment possible: a model fitted on every label is available as the reference, and
an immature training set can be manufactured from it. The decision here is how to manufacture it,
what to compare it with, and how far to trust what comes out.

## Options considered

**Take a chargeback delay distribution from the literature and sample maturation dates from it.**
Rejected. No distribution was fetched and read this session, and one from memory would violate
the rule this repo runs on. It would also dress a simulation as a measurement: whatever the
sampled delays were, the file has no way to check them.

**Do not simulate; state the risk and move on.** Rejected. The stage exists to turn assertions
into measurements, and the shape of the failure (does an immature window bias the model a little
or a lot, in which direction, and how fast with the window) can be measured without knowing the
true delay. Only the correction cannot.

**A deterministic latency of N days, swept.** Chosen. At training time every fraud in the final N
days of the training window reads as legitimate; nothing else changes. It is the crudest
simulation with the failure in it, it has one parameter, and sweeping that parameter is what
turns the result from a number into a shape.

**How to treat the immature window.** Two treatments, because the experiment is the comparison
between them: **immature**, fit on every train row with the immature labels treated as final,
which is what a pipeline that does not know about latency does; **excluded**, fit on the train
rows more than N days before the boundary, which is the standard correction. Each against the
reference fitted on every row with the file's labels.

**What else sees the immature labels.** The target encoding tables (ADR 0014) are fitted on
labels too. They are refitted on the same rows and labels the model fits on, at the recorded lag
and smoothing, so nothing at training time reads a label the simulation says has not arrived. A
version that left the shipped encodings in place would hand the immature model the true labels
through a side door, and the experiment would measure the door rather than the latency.

## Decision

`scripts/label_latency.py` sweeps N over **15, 30, 45, 60 and 90 days**, chosen to bracket the
weeks-to-months a chargeback takes with nothing in this repo saying where in that range the truth
sits. The recent window is `TransactionDT >= TRAIN_END_DT - N * 86,400`, so it is the final N
days of the 119.81 day training window by the clock, not by row count. The model, the
hyperparameters, the seed and the 186 column stack are the shipped model's, as in ADR 0031.

Measured per treatment and per N: test PR-AUC and ROC-AUC with a paired bootstrap interval
against the reference over 1,000 row resamples; the fraud rate the model implies on test (its
mean score) with the same interval, against the observed rate; and recall at the shipped
threshold of ADR 0029's tuned vertex, which is what a queue would see.

**The latency is simulated and the true chargeback timing is unknown here.** The artifact says
so in its first block, the figure says so in its axis label, and no number from this experiment
is a correction. What the sweep establishes is the direction of the bias, its size relative to the
noise band, and how it grows with N.

## Evidence

`reports/label_latency.json`, regenerated with `make latency`. Test PR-AUC and the fraud rate the
model implies on test, both with 95 percent intervals; the observed test rate is 0.0348 and the
reference model implies 0.0327.

| N, days | Fraud rows masked | Immature PR-AUC | Immature minus reference | Immature implied rate | Excluded PR-AUC | Excluded minus reference | Excluded implied rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 15 | 2,112 of 14,538 | 0.5106 | -0.0345, -0.0428 to -0.0268 | 0.0299 | 0.5327 | -0.0124, -0.0190 to -0.0059 | 0.0302 |
| 30 | 3,883 | 0.4593 | -0.0858, -0.0961 to -0.0754 | 0.0195 | 0.4877 | -0.0574, -0.0662 to -0.0493 | 0.0316 |
| 45 | 5,551 | 0.4163 | -0.1289, -0.1413 to -0.1170 | 0.0115 | 0.4839 | -0.0613, -0.0706 to -0.0528 | 0.0253 |
| 60 | 7,586 | 0.3435 | -0.2017, -0.2163 to -0.1881 | 0.0004 | 0.4732 | -0.0719, -0.0825 to -0.0616 | 0.0211 |
| 90 | 11,168 | 0.1629 | -0.3823, -0.3964 to -0.3664 | 0.0000 | 0.4493 | -0.0959, -0.1069 to -0.0855 | 0.0215 |

Immature is below excluded at every N, with the paired interval of the gap excluding zero at
every N (-0.0221 at 15 to -0.2864 at 90). The bias in the implied rate is downward in both
treatments and collapses under the immature one: at N=60 the immature model implies a fraud rate
of 0.0004 against an observed 0.0348 and recalls 0.0032 of the test fraud at the shipped
threshold, 0.3 alerts per day against the reference's 78.2. The excluded model at the same N
implies 0.0211 and recalls 0.3568.

## Consequences

- A training pipeline on this kind of data needs a maturation window, and on this simulation the
  cheapest window, 15 days, already costs 0.0124 of test PR-AUC against a reference that does not
  exist in production. The number to compare that with is the -0.0345 of not having one.
- The immature failure is not a gradual loss of signal. The model finds the recent rows and
  learns that they are never fraud, and every row at serving time is a recent row. The artifact
  records per fit which columns carried the gain; the stage document reads them.
- The excluded treatment is not free of bias either: its implied rate is below the observed one
  at every N and falls with N, because the fit set is older. Which of the two biases is smaller
  is what the sweep measures, and at every N it is the exclusion's.
- No value of N is recommended. The true delay is not established, and a later stage that
  obtains a chargeback date column writes a new record with a measured distribution and
  supersedes this one.
