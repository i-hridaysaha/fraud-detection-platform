# 0023: Geographic deviation is keyed on the card, not on the entity

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 4

## Context

The stage 4 brief asks for a geographic deviation feature: whether `addr1` differs from the
entity's modal `addr1` so far, and the same for `addr2`. It names the signal it is after, which is
the transaction region against the account's home region, and that is a standard and useful fraud
feature.

It cannot be built at the entity grain in this repo, and the reason is a property of the key
rather than of the data. ADR 0001 defined the entity as
`card1 + addr1 + floor(TransactionDT / 86400 - D1)`. `addr1` is a **component of the key**, so an
entity holds exactly one `addr1` value by construction. A deviation from an entity's own modal
`addr1` is therefore identically zero wherever it is defined, and null everywhere else.

This is worth a record rather than a silent substitution for two reasons. A feature that is
structurally constant is not a weak feature, it is a non-feature, and the difference matters when
the repository rule says an empty section is marked rather than filled. And the same argument
partly applies to `addr2`, which is not in the key but turns out to be nearly pinned by it anyway,
so the grain question has to be answered with a measurement rather than by reading the key
definition.

## Options considered

**Build it at the entity grain and report it.** Rejected. The column would be 0 or null on every
row of every split and would occupy width in the frame, in the catalogue, in the serving store and
in every stage 6 importance table, while carrying nothing. Shipping a column known in advance to
be constant is worse than shipping nothing, because a reader of the feature list would assume it
had been measured and found weak.

**Drop the geography family and mark the section not applicable.** Defensible under the repository
rule, and rejected because the rule is about sections with nothing real to say, and there is
something real to say here. The signal the brief is after exists in this file; it just lives one
level up from the key.

**Key the family on `card1` and measure both grains so the choice is evidence.** Chosen. The rule
was written before the numbers were read: build the deviation at the coarsest grain at which the
address can actually vary, and record the measurement at both grains so the choice is checkable.

**Key it on `card1 + addr2`, or on some other composite.** Rejected as unmotivated. `card1` is the
natural home-region owner, it is the coarser component of the entity key rather than a new
construction, and stage 0 measured it null-free over all 590,540 rows so the grain is defined on
every row.

## Decision

The geography family is keyed on `card1`. Four features ship:

| Feature | Grain | Definition |
| --- | --- | --- |
| geo_addr1_differs_from_card_mode | card | addr1 differs from the card's modal addr1 over strictly earlier rows |
| geo_addr2_differs_from_card_mode | card | the same for addr2 |
| geo_n_distinct_addr1_on_card | card | distinct addr1 values over the card's strictly earlier rows |
| geo_dist1_bucket | row | the train-fitted quantile bucket dist1 falls in, no key state |

The modal value is computed from prior rows only, with ties broken on the smaller value, which is
why the level codes behind it are ordered by value rather than by first appearance.

`fraud_platform.features.NOT_BUILT` records the entity-grain version and why it is absent, so a
reader of the module finds the answer without reading this file. A test asserts that no shipped
feature name contains `differs_from_entity_mode`, and another asserts that every
`differs_from_card_mode` feature declares the card grain.

The velocity family stays at the entity grain as the brief specified, and additionally at the card
grain for the separate reason ADR 0021 gives. This ADR is only about geography.

## Evidence

`reports/feature_summary.json`, block `entity_key_pins_addr1`, regenerated with
`.venv/bin/python scripts/features.py --sections catalogue`. Measured on the train split.

| Grain | Keys | Column | Max distinct per key | Share of keys with more than one | Mean distinct per key |
| --- | --- | --- | --- | --- | --- |
| entity | 164,702 | addr1 | 1 | 0.000000 | 0.9205 |
| entity | 164,702 | addr2 | 2 | 0.000115 | 0.9206 |
| card | 12,242 | addr1 | 63 | 0.331972 | 2.6720 |
| card | 12,242 | addr2 | 5 | 0.009966 | 0.9081 |

The first row is the decision. Not one of the 164,702 train entities holds two `addr1` values, and
the maximum is 1, so the entity-grain feature is a constant.

The second row is the one that needed measuring rather than reasoning. `addr2` is not in the key,
so it was not obvious that it would be pinned too, and it very nearly is: 19 of 164,702 entities
hold two values, a share of 0.000115. At the entity grain that column would be effectively
constant, which is the category stage 3's column plan drops at a top-value share of 0.99.

At the card grain both columns vary. A third of cards have used more than one `addr1` and one card
has used 63. `addr2` is weaker at the card grain too, 0.009966 of cards, and it is kept rather
than dropped because 122 cards changing a coarse address field is a small population with a
plausible mechanism rather than a measurement artifact, and because stage 6 is where a feature is
dropped on a model measurement. Its measured thinness is recorded here so nobody reads a large
importance for it without knowing the support.

A mean distinct count below 1 is not an error: an entity or card whose `addr1` is null everywhere
contributes zero distinct values. Stage 2 measured `addr1` and `addr2` as their own two-column
missingness block at a missing rate of 0.11495 on train.

The null rates the shipped features carry follow from this. `geo_addr1_differs_from_card_mode` and
`geo_addr2_differs_from_card_mode` are both null on 0.1415 of train rows, and they are null on
exactly the same rows because the two columns share a null mask exactly: they are a block of two
in `reports/eda/missingness.json`, both at 0.11495. A null in the feature means the row's own
address is missing, or no earlier row of the card carries one.

## Consequences

- The geography family reads a third keyspace at serving time: per card, a count per address value
  observed. `reports/feature_summary.json` records `state_key: card` for three of the four
  features, so stage 8 maintains entity state, card state and content state.
- The card counter is the least bounded structure in the stage, since it grows with the distinct
  addresses a card has used. `reports/feature_summary.json`, `state_footprint.train.card`, is what
  bounds it on this data: over the 10,985 cards that carry any address, a mean of 2.978 distinct
  `addr1` values, a 99th percentile of 33.16 and a maximum of 63. A production version would cap
  it and the cap has a measured distribution to be set against.
- This does not weaken ADR 0001. The entity key was chosen on label purity, 0.8480 under card1
  against 0.9664 under the three-part key, and it still does that job. What this ADR records is
  that a key chosen for purity pins the columns it is built from, so any deviation feature over a
  key component has to move up a grain. That is a general consequence of the choice and it is
  stated here for the first time.
- **The same argument applies to `card1` and `D1` and nobody has tested it yet.** Both are key
  components, so a within-entity deviation on either is equally constant by construction. No such
  feature is proposed, and if one is, this ADR is the reason it needs a grain other than the
  entity.
- `geo_dist1_bucket` is in this family for topic and not for mechanism. It needs no key state, a
  tree reads raw `dist1` identically, and it exists for the stage 5 models that read a value
  rather than its rank and for the reason codes in stage 7. That is recorded in the catalogue next
  to the feature rather than left to be inferred.
