# 0015: The entity key is excluded from target encoding

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 3

## Context

ADR 0001 defined the entity key as `card1 + addr1 + floor(TransactionDT / 86400 - D1)`. Stage 0
measured that 0.9664 of multi-transaction entities are label-pure at that key, and stage 2
replicated it on the train split at 0.9675: of the 68,359 entities with more than one transaction,
0.9466 are entirely non-fraud and 0.0208 are entirely fraud.

Target-encoding a key like that is not an ordinary feature decision. If knowing the entity is
almost the same as knowing the label, then a column holding the entity's historical fraud rate is
close to a lookup table from entity to label, and it will report a large validation AUC while
doing something the phrase "predictive feature" does not describe.

ADR 0014 fixed a label lag for the identifier columns. It is worth being clear that the lag does
not address this case. A lag protects against reusing labels that would not yet have arrived. Here
the problem is not that the labels are recent, it is that they belong to the same entity, and a
longer lag does not change whose labels they are.

The other half of the context is coverage. Stage 1 measured that 0.6598 of test rows sit on an
entity the training window never saw, so whatever an entity encoding is worth, it is worth it on a
minority of the rows a deployed model scores, and that minority shrinks with time since training.

## Options considered

**Ship it, because the validation AUC is good.** It is: 0.7835 at the chosen lag. Rejected, and
the reason is the whole of this ADR. A composite validation AUC over a split that is 0.469 seen
entities and 0.531 unseen ones averages a lookup and a constant, and reporting that average as
skill is how a leak survives review.

**Exclude it outright on the label purity number alone.** Defensible, and it was tempting.
Rejected because 0.9675 is a measurement about entities, not about what an encoding of them does,
and the repository rule is to decide with the number that answers the question asked.

**Measure it at several lags, decompose validation on entity overlap, and decide from that.**
Chosen.

## Decision

The entity key is excluded from target encoding. `fraud_platform.encoders.NEVER_TARGET_ENCODED`
holds `entity_id` and `card_start_day`, and `fit_target_encoding` raises if either is passed to
it. The exclusion is enforced in the encoder rather than left to whoever assembles the pipeline,
and a test asserts the raise.

The rule was written before the numbers were read. Exclude when both of these hold on validation:

- the encoding, restricted to rows whose entity the training window saw, separates the classes
  better than it does over validation as a whole, by more than the split can resolve. All of its
  power sits on the subset where the key is known;
- the encoding, restricted to rows whose entity the training window did not see, is inside the
  resolvable band of 0.5. It carries nothing at all where the key is new.

Together those describe an entity-to-label table gated by entity overlap, rather than a feature.

The probe that produced this measurement necessarily bypasses the exclusion it justifies.
`_fit_unchecked` exists for that and for nothing else.

## Evidence

`reports/encoding_spec.json`, block `entity_target_encoding`, regenerated with
`.venv/bin/python scripts/prepare.py --sections encoding`.

164,702 train entities, 46,884 validation entities, 41,536 of the 88,581 validation rows on an
entity the training window saw, a share of 0.4689. That reproduces the 0.5311 unseen share stage 1
recorded in `reports/split_summary.json` under `entity_overlap.val`.

| Lag, days | Train AUC | Validation AUC |
| --- | --- | --- |
| 0 | 0.9922 | 0.8068 |
| 1 | 0.7829 | 0.8047 |
| 3 | 0.7513 | 0.8038 |
| 7 | 0.7210 | 0.7966 |
| 14 | 0.6850 | 0.7835 |
| 28 | 0.6488 | 0.7556 |

A train AUC of 0.9922 at lag 0 is what an entity encoding without a lag does: it reads a row's own
label back out of its entity. The lag removes that, which is what ADR 0014's machinery is for, and
the validation number barely moves. That is the point where the naive reading says ship it.

The decomposition, taken at the chosen lag of 14 and split on whether the training window ever saw
the row's entity:

| Quantity | Value | Rows |
| --- | --- | --- |
| Validation AUC over the whole split | 0.7835 | 88,581 |
| Validation AUC on rows with a seen entity | 0.9877 | 41,536 |
| Validation AUC on rows with an unseen entity | 0.4915 | 47,045 |
| Validation AUC of the seen-or-unseen indicator alone | 0.4204 | 88,581 |

On the rows where an entity history exists, the encoded rate recovers the validation label at an
AUC of 0.9877. Stage 2 measured entity label purity at 0.9675. Those two numbers are the same
fact, and the second is the mechanism for the first. On the rows where the entity is new, the AUC
is 0.4915, inside the 0.0191 band this split resolves around 0.5, which is to say the column is a
constant there and carries nothing.

So 0.7835 is not a model separating fraud from non-fraud. It is a near-perfect lookup on 0.469 of
the rows and a constant on the rest, and the two groups happen to sit at different base rates.
The seen-or-unseen indicator alone scores 0.4204, below chance, because stage 1 measured that
unseen-entity rows run at a higher fraud rate, 0.0443 against 0.0231 on val. The concentration is
therefore not the indicator wearing a rate's clothing: it is the entity's own labels.

Both conditions hold, and the column is excluded.

## Consequences

- No entity target encoding reaches stage 5, and no model's validation number in this repo will be
  a measurement of entity overlap.
- Entity history is not banned. What is banned is summarising the entity's **labels**. Stage 4 is
  free to build entity aggregates over behaviour, and stage 2 already named the strongest
  candidate: the median gap between consecutive transactions is 3,177 seconds for an entity that
  ever sees fraud against 165,394 seconds for one that never does, a factor of 52, measured with
  no labels in the feature at all.
- Whatever stage 4 builds on entity history has a defined behaviour for a cold entity, and that
  behaviour is exercised on 0.6598 of test rows. Stage 2 said the same thing and this ADR is the
  first place it costs something.
- This does not establish that entity-level label history is illegitimate in general. A production
  fraud system does know a card's chargeback history, and on a stable customer identifier that is
  a real feature. What it establishes is that on **this** key, whose purity is 0.9675 and whose
  overlap with test is 0.3402, the column measured here is not that feature.
- The exclusion is enforced by a raise, not a convention, so a later stage that wants to revisit it
  has to change the constant and say why in a new ADR.
