# 0001: Entity key

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 0

## Context

Fraud in this dataset is not an independent event per row. Transactions cluster into whatever
real-world thing was compromised, and any leakage-safe split, any entity aggregate, and any
online feature lookup needs a key that names that thing. The dataset has no card identifier.
It has card1 to card6, addr1 and addr2, and a set of D columns of which D1 is described as a
day counter.

Three candidates were measured, all on the 590,540 training rows:

| Candidate | Entities | Transactions per entity (mean, median, p90, max) | Singleton entities | Label purity of multi-transaction entities |
| --- | --- | --- | --- | --- |
| card1 | 13,553 | 43.573, 4.0, 46.0, 14,932 | 0.2541 | 0.8480 |
| card1 + addr1 | 39,974 | 14.773, 2.0, 22.0, 9,928 | 0.3941 | 0.8899 |
| card1 + addr1 + card_start_day | 217,850 | 2.711, 1.0, 6.0, 1,414 | 0.5748 | 0.9664 |

card_start_day is floor(TransactionDT / 86400 - D1).

Label purity here means the share of entities with two or more transactions whose transactions
all carry the same label. It is the direct test of whether a key names the thing the label is
attached to. A key that splits one real card across several entities loses history; a key that
merges several real cards into one entity lets a fraud label bleed across cards, which is the
failure that matters, because it turns a split into a leak.

## Options considered

**card1 alone.** Simple, no nulls in the column, and it survives any missing addr1 or D1. It was
rejected because it is not a card. Two measurements say so. Its groups are far too large for a
card: mean 43.573 transactions per value, max 14,932. And within a single card1 value the implied
card start day moves constantly: only 0.1539 of card1 values with two or more non-null start days
have a single one, the median group holds 4.0 distinct start days, the p90 holds 27.0, and the
median spread between the earliest and latest implied start day is 155.0 days. card1 behaves like
a card attribute shared by many cards, not an identifier. Its label purity of 0.8480 is the cost:
0.1520 of its multi-transaction groups are label-mixed.

**card1 + addr1.** Better on every count and still tolerant of a missing D1, but it inherits the
same problem in weaker form. Purity reaches 0.8899, and 0.2373 of its groups have a single
implied start day, with a median spread of 113.0 days. It is still merging cards. It also
introduces 65,706 rows with a null key component, which have to be bucketed somewhere.

**card1 + addr1 + card_start_day.** Purity 0.9664, the highest of the three. Entity size falls to
a median of 1 transaction and a mean of 2.711, which is what a card looks like over a 182 day
window in a sampled dataset. The cost is that 0.5748 of entities are singletons holding 0.2120 of
all rows, and that 66,794 rows have a null component.

## Decision

Use **card1 + addr1 + floor(TransactionDT / 86400 - D1)** as the entity key. Null components are
kept as their own bucket rather than dropped, so every row belongs to exactly one entity and the
row counts reconcile.

This key defines the entity for the purposes of leakage reasoning and label analysis. It does not
by itself decide the grain at which features are aggregated. Coarser keys stay available as
aggregation levels, and choosing among them is a stage 4 decision with its own ADR, not something
settled here.

## Evidence

- **Artifact:** reports/audit.json, sections `entity_key_candidates`, `label_granularity_by_candidate`,
  `card_start_day_check`, `d1_profile`, `card1_profile`.
- **Command:** `make audit`
- Supporting measurement for treating D1 as a day counter: D1 has minimum 0.0 and maximum 640.0
  over an observation window of 181.999 days, no negative values, 1,269 nulls (0.0021 of rows),
  and 0 rows where the implied card start day falls after the transaction day. A counter that ran
  the other way, or that was not a day count, would violate the last of those.
- The claim that the competition host propagates a reported chargeback to later transactions
  linked by account, email, or billing address is **not verified in this repo**. The Kaggle page
  is rendered client side and returned no text when fetched this session, and the rule is not
  restated here as fact. What is measured is the consequence: labels cluster at entity level, and
  the tighter the key the purer the clusters, reaching 0.9664 at the chosen key.

## Consequences

- Under this key, 0.6666 of test-window entities were never seen in the training window, and
  those entities carry 0.6598 of test rows. Any entity-history feature is absent for most test
  rows, so the feature layer has to work when history is missing rather than treat it as an edge
  case. See ADR 0002 for the split those numbers come from.
- Rows on unseen entities in the test window carry a fraud rate of 0.0440 against 0.0170 for
  rows on entities the training window had seen. New entities are riskier, which means "no
  history" is itself signal and must not be imputed away.
- 0.2120 of rows sit in singleton entities, so entity aggregates computed at this grain are
  empty for one row in five.
- Any later change of key needs a superseding ADR. tests/test_audit_report.py asserts the key
  composition, so a quiet edit fails the suite.
