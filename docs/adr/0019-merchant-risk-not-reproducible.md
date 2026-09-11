# 0019: Merchant risk is not reproducible on this dataset

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 4

## Context

`docs/design.md` describes a system that scores a card transaction at authorisation. In a real
authorisation flow, one of the strongest features available is a statistic about the seller: its
chargeback rate, its dispute rate, the share of its volume that is card-not-present, how long it
has been accepting cards. A card that has never transacted before is unknown; a merchant that
declines a thousand cards an hour is not.

Stage 4 is where that feature would be built, and it cannot be built here. The reason is not that
it is hard or that the signal is weak. It is that the identifier the feature is keyed on does not
exist in the file.

This has to be recorded properly rather than passed over, for two reasons. A reader who knows
fraud detection will look for merchant risk in the feature list and needs to find the answer
rather than an absence. And anyone comparing a number from this repo against a published fraud
result needs to know that a whole family of features is missing on one side of the comparison.

## Options considered

**Build a merchant proxy and call it merchant risk.** Rejected, and it is the option worth naming
because it is the tempting one. `ProductCD` has five levels and something to do with what was
bought. `P_emaildomain` correlates with where people shop. A distance column plausibly relates a
buyer to a seller. Each of these carries measurable signal, and calling any of them a merchant
feature would put a familiar-sounding row in the feature table at the cost of the reader believing
something untrue. A proxy for a category is not an identifier, and a feature keyed on a category
cannot hold a seller's own history.

**Reconstruct a seller identifier by clustering.** Group transactions by amount pattern, product
class, distance and time-of-day and treat a cluster as a merchant. Rejected on two grounds. There
is no way to validate it: nothing in the file says whether two transactions went to the same
seller, so the clustering could not be checked against anything, and this repo does not write down
a construction it cannot measure. And the clusters would in part recover the entity, since amount
and address are already entity-linked, which would import the leakage ADR 0015 was written to
keep out.

**Say the feature cannot be built, list what stands in for it, and record that the strength is not
comparable.** Chosen. This is the case the repository rule for an empty section covers: a section
with nothing real to say is marked not applicable rather than filled.

## Decision

No merchant risk feature is built. `fraud_platform.features.NOT_BUILT` records the absence in
code, next to the other things this stage deliberately does not build, so a reader of the module
finds it without reading this file.

The features that carry adjacent information are kept and are named for what they are. None is
described as a merchant feature anywhere in the repo, and `reports/features/merchant_proxies.json`
carries an explicit statement that no number in it should be read as one.

## Evidence

`reports/features/merchant_proxies.json`, regenerated with `.venv/bin/python scripts/features.py
--sections merchant`.

**There is no merchant identifier and no merchant category code.** The screen is a regex over
every column name in both files, `merch|mcc|categor|\bsic\b|acquir|terminal|store|retail|seller|
vendor`, case-insensitively. It matches 0 of 434 columns. That on its own would only rule out a
column named for a merchant, so the artifact also lists the complete set of columns that are not
numbered members of the V, C, D, M or `id_` blocks. There are 19 of them and the whole list is:

TransactionID, isFraud, TransactionDT, TransactionAmt, ProductCD, card1, card2, card3, card4,
card5, card6, addr1, addr2, dist1, dist2, P_emaildomain, R_emaildomain, DeviceType, DeviceInfo.

Every other column is an unlabelled member of a numbered block. So there is no column that can be
asserted to identify a seller, and nothing in the numbered blocks can be claimed to either,
because stage 2 established that what the V columns are is not established and stage 4 could not
establish what the C columns are.

**The proxies, with the signal stage 2 measured for each.** Information values are read from
`reports/eda/bivariate.json` rather than recomputed, and a test asserts they match field by field.
The second column is the total with the missing bin's term removed, which is what the observed
values carry once the fact of being observed is set aside.

| Column | Proxy for | Information value | Without the missing bin |
| --- | --- | --- | --- |
| R_emaildomain | the recipient's email domain | 0.5787 | 0.4192 |
| ProductCD | the product class of the purchase, five levels | 0.5161 | 0.5161 |
| addr1 | the billing region, a property of the account | 0.4257 | 0.0996 |
| addr2 | a coarser address field | 0.4200 | 0.0938 |
| card6 | the card funding type, credit against debit | 0.2515 | 0.2515 |
| P_emaildomain | the purchaser's email domain | 0.1850 | 0.1810 |
| dist1 | a distance, unit not established, between two unnamed points | 0.1593 | 0.1183 |
| dist2 | a second distance on a different missingness block | 0.1525 | 0.1360 |
| card4 | the card network | 0.0079 | 0.0079 |

Two things in that table are worth reading rather than skipping. The address columns are mostly
missingness: addr1 carries 0.4257 in total and 0.0996 without the missing bin, and stage 2 already
said so and called it a missingness feature rather than a value feature. The email and distance
columns are not: R_emaildomain keeps 0.4192 of its 0.5787 and dist1 keeps 0.1183 of its 0.1593,
so those two carry real value signal despite being missing on 0.7507 and 0.6187 of rows
respectively.

The strongest merchant-adjacent number in the repo is not a column at all. Stage 2 measured
purchaser and recipient email domain agreement carrying more rate than either domain on its own:
of the 96,265 rows where both are present, 77,595 agree at a fraud rate of 0.09051 and 18,670
disagree at 0.02689, against 0.02011 on the 252,817 rows where only the recipient is missing.

None of this is merchant risk. ProductCD is the closest thing to a seller category and it has
five levels. The rest describe the buyer, the buyer's region, or the card.

## Consequences

- The feature table in this repo has no merchant family, and the absence is recorded in three
  places: `NOT_BUILT` in the module, this ADR, and the artifact.
- **Signal strength here is not comparable with a production fraud system's, in either
  direction.** A deployed model with merchant history has a feature this one cannot have, so a
  lower number here does not mean a worse pipeline. And nothing here can be presented as evidence
  about how much merchant risk is worth, because that quantity was not measured. The gap is a
  property of the dataset.
- Comparisons against other work on **this** dataset stay valid, because nobody working on this
  file has a merchant identifier either. The comparison that breaks is against fraud results on
  data that includes one.
- Stage 7's reason codes will not be able to say "this seller has a high dispute rate", which is
  one of the most useful things a reviewer can be told. What they can say is which entity
  behaviour and which region deviation drove the score.
- If a later stage acquires a dataset with a merchant identifier, this decision does not carry
  over and a new ADR supersedes it. Nothing in the feature code assumes the absence beyond not
  referring to a column that is not there.
