# Stage 0 notes

Working notes: dead ends, surprises, and the numbers that changed a decision. The tidy version
of stage 0 is in the ADRs; this is what it actually looked like.

## The number that changed my mind

I expected `card1` to be a card, or close enough to one to act as the entity. The label purity
gap was suggestive (0.8480 for card1 against 0.9664 for the full composite), but the number that
settled it was the spread of the implied card start day inside a single card1 value. If card1
were a card, `floor(TransactionDT / 86400 - D1)` would be one value per card1 group. It is not:
only 0.1539 of multi-transaction card1 values have a single implied start day, the median group
holds 4 distinct ones, the p90 holds 27, and the median gap between the earliest and latest is
155 days. A card that began on 4 different days is not a card. card1 is an attribute that many
cards share.

That inverted the framing I started with. I had planned to verify the brief's claim that
TransactionDT/86400 minus D1 is roughly constant per card. Measured against card1 it is plainly
false, and the useful reading is the contrapositive: the term is worth having in the key precisely
because it disagrees with card1, splitting 13,553 card1 values into 217,850 entities that are
much more label-pure.

## The cost I did not expect

Choosing the purest key is not free, and the bill arrives in the split. Under card1, 0.1225 of
test entities are new. Under the composite it is 0.6666, and those entities carry 0.6598 of test
rows. Two thirds of the evaluation set has no entity history at all.

For a while I read that as a reason to prefer a coarser key. It is not. The coarse key does not
give those rows real history, it gives them somebody else's history, and the 0.1520 mixed-label
fraction under card1 is exactly that contamination showing up as a measurement. The right
conclusion is that the feature layer has to treat "no history" as the common case rather than the
exception, and that the difference between the two fraud rates in the test window (0.0440 on
unseen entities against 0.0170 on seen ones) means absence of history is itself a signal. Stage 4
inherits that problem rather than dodging it.

## Dead ends

**Trying to verify the labelling rule from the source.** The brief describes the competition
host's rule about propagating a reported chargeback to associated transactions. Fetching the
Kaggle data page returned nothing but the page title, because the content is rendered client
side. Rather than restate the rule from memory, nothing in this repo asserts it. What is written
down is the measured consequence: labels cluster at entity level, and the tighter the key the
purer the clusters. That is weaker than the original claim and it is the part that is actually
established here.

**Fixing test failures that were really fixture bugs.** Three unit tests failed on the first run.
Two were my test helper building a DataFrame from tuples, which gives object dtype for an all-null
column, while `read_csv` gives float64. Real data never hits that path. The helper now casts, with
a comment saying why. The third failure was my own arithmetic: I asserted that a synthetic test
window would be entirely made of unseen entities when it obviously straddles the boundary and is
half seen.

That third one was worth the time. The bug it did surface was real: `entity_key_stats` divided by
the count of multi-transaction entities without checking it was non-zero, which is fine on 590,540
rows and a crash on a 2 row frame. Rates now come back as JSON null when the denominator is empty,
and the report writer runs `json.dumps(..., allow_nan=False)` so a NaN can never quietly become
invalid JSON that a downstream reader half-accepts.

**Joining the key columns with `.agg('|'.join)`.** Clean-looking and wrong twice over. It throws
on a null component, and being row-wise it is slow on 590,540 rows. Replaced with a vectorised
concatenation that maps nulls to the literal "NA", which also has the advantage of matching
`groupby(dropna=False)` elsewhere, so entity counts computed two different ways agree.

## Surprises

**Identity coverage is label-dependent.** 0.5477 of fraud rows have an identity record against
0.2332 of non-fraud rows. Whatever produces an identity row is correlated with the outcome, so
the missingness is signal and cannot be imputed away quietly. It also means any model feature
built on identity columns is unavailable for three quarters of traffic.

**Nearly half the rows have D1 equal to 0.** 0.4744 of them. Under the chosen key that reads as a
first observed transaction for that entity, which is consistent with 0.5748 of entities being
singletons. Whether the column also defaults to 0 when the true value is unknown is not
established, and it matters, because those two cases want different treatment. Left open for
stage 2.

**The split fractions came out exact.** 413,378 / 88,581 / 88,581 gives 0.7000 / 0.1500 / 0.1500
even though the boundaries are timestamp thresholds rather than row cuts. With 573,349 distinct
TransactionDT values over 590,540 rows, ties near a boundary are rare enough not to move the
fraction at four decimal places. Worth noting because the guard against a straddled timestamp is
still needed, it just happened to cost nothing here.

## Housekeeping that will matter later

`make reproduce` exists and currently chains `data` and `audit`. It is a stub in the sense that
there is little to chain yet, but the target is real and every later stage appends to it. That
was the right call to make on day one, because the alternative is discovering in stage 8 that
half the artifacts were produced by commands nobody wrote down.

Server-side branch protection could not be enabled: both the branch-protection and the ruleset
APIs refuse on a private repo under the current plan. A local pre-push hook refuses direct pushes
to main instead. It is weaker (it lives on one machine and can be bypassed with `--no-verify`)
and it is recorded as such rather than glossed.
