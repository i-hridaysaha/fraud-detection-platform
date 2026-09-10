# Design

Written 2026-09-10, before the first-pass audit. Everything here is a hypothesis. Later stages
amend it and date each revision; where a revision contradicts this text, the revision wins and
says so.

## What the system is meant to do

Score a card transaction for fraud risk at authorisation time, return a calibrated probability
and a reason for it, and keep doing both as the fraud mix moves underneath the model.

The unit of work is a single transaction arriving with its own fields plus whatever the platform
already knows about the entities behind it (the card, the billing address, the email domain, the
device). The output is a score, a decision against a threshold, and a short explanation. The
system is judged on whether the score is trustworthy in production over months, not on a
leaderboard position.

## The four constraints that shape it

**Class imbalance:** fraud is a low single-digit percentage of transactions. Accuracy is
meaningless, ROC AUC is close to meaningless because the negative class dominates the false
positive rate, and a threshold chosen without reference to review capacity is arbitrary. This
pushes the design toward precision-recall framing, toward a primary metric that is read at a
fixed operating point, and toward reporting an interval rather than a point estimate.

**Delayed labels:** a chargeback is reported days or weeks after the transaction it belongs to.
At any moment, recent transactions have unreliable labels: absence of a fraud label means either
"not fraud" or "not reported yet". This has three consequences. Training data has to respect a
maturity window. Monitoring cannot rely on labels alone, because the freshest slice is the one
being asked about. Any evaluation of a live model is a moving estimate that firms up over time.

**Temporal non-stationarity:** attack patterns, merchant mix, and the population of legitimate
transactions all drift. A model validated on a random split of the same period is measuring
interpolation, not the thing production asks of it. Splits are chronological, features are
strictly causal (a feature computed for a transaction may read only what was knowable strictly
before it), and drift is measured rather than assumed away.

**Inference latency ceiling:** the score is produced inside an authorisation flow, so the budget
is on the order of tens of milliseconds. That rules out computing entity aggregates from a
warehouse scan at request time. Aggregates are maintained in an online store and read by key,
which then creates the real risk in the design: the online path and the training path can
disagree. Train/serve parity is a first-class concern, not a deployment detail.

## Components that follow

**Data layer:** chronological split with a fixed boundary, an entity key chosen from measurement
rather than convention, and a held-out slice that is never touched during model selection.

**Feature layer:** transaction-level fields plus entity aggregates (counts, velocities, recency,
distinct-value counts) computed with a strict as-of rule. One definition of each feature, used by
both the training job and the serving path.

**Feature store:** an online key-value store holding the current aggregate state per entity, with
the same code path writing it in backfill and updating it in the stream. Parity is tested, not
asserted: the same transaction scored offline and online must produce the same feature vector.

**Model:** gradient-boosted trees as the working baseline, with calibration on a held-out slice
so the score can be read as a probability and the threshold can be set against review capacity.

**Scoring service:** FastAPI, one endpoint, feature lookup plus model call plus SHAP attribution,
under the latency ceiling. The explanation ships with the score because a reviewer acting on a
decline needs a reason, and because an unexplainable score is unauditable.

**Monitoring:** population stability on the input distribution and on the score, plus a
label-decay view that reports metrics by transaction age so a fresh-window drop is not mistaken
for a model failure when it is only unreported chargebacks.

**Retrain loop:** gated, not scheduled-and-shipped. A candidate model is promoted only if it beats
the incumbent by a margin derived from measured variance, on the same slice, at the same
operating point.

## Open questions this design does not answer yet

- What the entity key is. Stage 0 measures it.
- Where the split boundaries fall and how much entity overlap they leave. Stage 0 measures it.
- What the primary metric and the operating point are. They follow from the review-capacity
  assumption, which is not yet written down.
- What the promotion margin is. It cannot be chosen before the run-to-run variance is measured.
