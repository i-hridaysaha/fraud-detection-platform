# 0014: The target encoding label lag is 14 days

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

A target encoding replaces a category with a summary of the labels attached to it. That is useful
and it is also the easiest way to build a leak, because the label is exactly the thing the model
is not allowed to know. The usual precaution is out-of-fold encoding, which stops a row from
seeing its own label. On a chronological problem it is not enough.

The reason is when the label arrives. A fraud label on this kind of data comes from a chargeback,
and a chargeback matures over weeks. A transaction on day 100 does not have a label on day 100. So
an encoding built from "every training row for this card" hands the model, at scoring time on day
100, a summary that includes labels which in production would still be in the post. The model
learns to rely on information that will not exist when it runs, and the failure appears in
production as a quiet loss of accuracy that no validation split will show, because the validation
split has the same future labels available to it.

Stage 0 did not establish the host's labelling rule and neither does this stage. What can be
established is the price of each choice.

## Options considered

**No lag, out-of-fold only.** Encode a row from every other training row. Rejected: it prevents a
row seeing its own label and does nothing about the timing, which is the failure this problem has.

**A fixed conservative lag chosen by convention, say 30 or 60 days.** Safe and unmeasured.
Rejected because a lag is a cost as well as a protection: a longer lag means the encoding is built
from a smaller and staler slice of the training window, and choosing that cost without measuring
it is guessing in the safe direction rather than deciding.

**Sweep the lag and choose against validation.** Chosen, with an important qualification about
what to choose on, below.

## Decision

A row on day `d` is encoded from training rows on day `d - LAG` or earlier, and from nothing
later. `LAG = 14` days. The smoothing at that lag is 20.

The encoder holds cumulative label counts per level per day plus the cumulative prior, and both
are read at the row's own lagged boundary at transform time. That is what makes one fitted object
serve every row: a training row on day 40 and a validation row on day 130 read the same tables at
different boundaries, and neither ever sees a label from its own day.

Three states get three answers, and keeping them apart is part of the decision.

- A level the window has counted zero times before the boundary falls to the prior at that
  boundary, which is what the smoothed formula says and what the window actually knows. A level
  never seen at all is the same case.
- A null input encodes to null. There is no level to look up, and the vocabulary encoder's
  `(missing)` token is where that state lives.
- A row whose lagged boundary precedes every training row encodes to null, because at that point
  the window knows nothing at all, prior included. There is no global fallback. A fallback would
  put a number from outside the row's own information set into the column, which is the leak the
  lag exists to prevent.

**What the lag is chosen on.** Not the validation AUC. A longer lag can only remove information,
so the best validation AUC sits at or near lag 0 by construction, and choosing on it would choose
the leak. The sweep's useful column is the other one: the gap between the in-sample train AUC and
the validation AUC. An encoding that is summarising a row's own label separates the rows it was
fitted on much better than the rows it was not, and the gap is what that looks like.

The rule, stated before the sweep was read: take the smallest lag whose mean train-minus-
validation gap is indistinguishable from zero across the swept columns, and check that the
validation AUC it costs against the best lag is inside what the split can resolve. Smallest,
because a lag is a delay on information and the shortest defensible one keeps the most.

## Evidence

`reports/encoding_spec.json`, block `encoders.target`, regenerated with `.venv/bin/python
scripts/prepare.py --sections encoding`. `figures/target_encoding_lag.png` draws both panels.

Six lags by eight smoothing values over 18 identifier columns, 864 points, one fit. The tables do
not depend on either swept parameter, so every point is a different read of the same fit. No model
is trained: the encoded column is already a probability-like score and its AUC against the label
is what the encoding is worth on its own.

Averaged over the smoothing grid and the 18 columns:

| Lag, days | Train AUC | Validation AUC | Gap | 95 percent half width of the gap | Validation cost against lag 0 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.68911 | 0.60794 | +0.08117 | 0.02961 | 0.00000 |
| 1 | 0.65913 | 0.60595 | +0.05319 | 0.02429 | 0.00199 |
| 3 | 0.64902 | 0.60594 | +0.04308 | 0.02307 | 0.00200 |
| 7 | 0.63452 | 0.60309 | +0.03143 | 0.02243 | 0.00486 |
| 14 | 0.60882 | 0.60113 | +0.00768 | 0.01729 | 0.00681 |
| 28 | 0.58946 | 0.59627 | -0.00681 | 0.01362 | 0.01167 |

The gap falls monotonically from 0.08117 at lag 0 to below zero at lag 28. That is the leak
signature and it behaves exactly as the argument predicts. At lag 0 an identifier's encoding
separates the training rows it was built from by 0.081 AUC more than it separates rows it was not,
on identical columns and identical arithmetic; the only difference is which labels were in scope.

Two lags satisfy the gap test, 14 and 28, and the rule takes the smaller. Its validation cost
against the best-scoring lag is 0.00681, well inside the 0.02185 that stage 2's class balance says
this validation split can resolve for an unpaired comparison at the observed AUC. The half width
on the gap is the standard error across the 18 columns, so it measures how consistently the swept
columns agree about the direction, not the sampling error of one column's AUC, and the artifact
says so.

Smoothing at lag 14, mean validation AUC over the 18 columns:

| Smoothing | 1 | 5 | 10 | 20 | 50 | 100 | 200 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Validation AUC | 0.60125 | 0.60199 | 0.60269 | 0.60282 | 0.60251 | 0.60182 | 0.59954 | 0.59644 |

The maximum is at 20 and it is an interior maximum, which is the one useful thing to say about it.
The one thing that should not be said is that the sweep resolved it: every value on the grid sits
inside the paired interval of the best, and `resolved: false` in the artifact records that. The
point estimate picks 20 and the interval says how much that pick is worth, which is not much. What
the sweep does establish is that the ends of the grid cost something: 500 gives up 0.0064 against
20, and that is a real direction even if the interior is flat.

## Consequences

- The 14 day lag makes the encoding undefined for the first 14 days of the training window. That
  is a null, the tree path reads nulls natively (ADR 0012), and the artifact records the count per
  sweep point.
- The lag is not a claim about how long a chargeback takes. Nothing in this repo establishes that,
  and if a later stage does, this ADR is superseded rather than edited. What the sweep establishes
  is the price of each choice and the shape of the leak it removes.
- Stage 8 has to honour the same rule at serving time. A serving path that looks up "the rate for
  this card as of now" rather than "as of 14 days ago" reintroduces exactly the gap this removes,
  and the fitted object carries the lag so it cannot be applied without one.
- The sweep is on the encoded column alone and not on a model. Whether the lag changes a full
  model's validation AUC by more than the split resolves is a stage 5 question and is not answered
  here.
- The entity key is excluded from target encoding entirely and for a different reason. See ADR
  0015: no lag repairs it, because the labels it reuses are not stale, they are the same entity's.
