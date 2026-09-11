# Features

Stage 4 turns isolated transactions into behavioural patterns. One rule binds the whole stage:
**a feature for a transaction at time t reads only rows strictly before t.** Not before t within
the fold. Strictly before t, on the whole stream, because that is what an authorisation endpoint
has when it is asked for a score.

Everything below is measured on the committed artifacts under `reports/`. Regenerate the lot with
`make features`.

36 features ship, across seven families and three keyspaces. One native column, D1, is listed as
a feature without being rebuilt. Two constructions that a reader will look for are deliberately
absent and are recorded as absent: the published UID aggregation (ADR 0020) and merchant risk
(ADR 0019).

---

## 0. Why there is no train-only guard on the feature build

Stages 1 to 3 route every fitted object through `assert_train_only`, and stage 4 has exactly one
fitted object: the dist1 bucket edges. Everything else is computed over the whole 590,540 row
stream, train and later windows together, and that is correct rather than tolerated.

The reason is that nothing in `fraud_platform.features.build_features` learns a statistic. Every
value is read from the rows of the frame that fall strictly before the row being computed, which
is exactly the information the serving path has. A validation row reads earlier training rows, as
it would in production. A training row cannot read a validation row, because validation rows are
later in time. A window check would pass whatever it was handed, so the guard is not what protects
this stage.

`tests/test_causality.py` is what protects it, and it checks three different things:

| Property | What it catches |
| --- | --- |
| Brute-force agreement | every feature equals a reference recomputed over `TransactionDT < t` by filtering the whole frame, on a sample built to include cold rows, tied rows and rows with history |
| Truncation invariance | features on the whole stream equal features on the stream cut at time T, for every row before T, compared bit for bit with no tolerance |
| Tie exclusion | a transaction sharing its timestamp with another on the same key is not in its history, and `side="right"` is shown to produce a different answer |

The reference shares nothing with the module: it does not sort, does not use searchsorted, does
not factorize, and builds its own device combination and content key out of the raw columns. Four
deliberate breaks were applied, run and reverted, and the table is in `docs/notes/stage-04.md`.

**Truncation invariance is the one that changed the implementation.** It failed on the real frame
against a global `np.cumsum`, by 2e-4 on `amt_zscore_within_entity` for entities that repeat one
amount. A global cumulative sum carries a rounding error proportional to what it has accumulated,
so the last bits of a ten-minute window sum depended on every amount earlier in the file and, once
the difference was taken, on rows later than the one being computed. Immaterial to any model, and
a real dependency on the future. The arithmetic changed rather than the test's tolerance: window
sums restart at every key and the running moments use Welford, which is also the three scalars a
serving path would hold. ADR 0018.

---

## 1. The implementation is two indices, not a rolling join

A trailing window is two positions in a sorted array:

```python
lo = np.searchsorted(ts, ts - window, side="left")   # first row at or after t - window
hi = np.searchsorted(ts, ts, side="left")            # first row at t
count = hi - lo
```

There is no window argument to misconfigure and no `closed=` to set. The slice `[lo, hi)` cannot
contain the current row, because `hi` is the index of the first row whose timestamp equals t.
Including it would mean changing `side="left"` to `side="right"` on a line whose only purpose is
that choice, and a test asserts the difference the two produce rather than asserting a convention.

**Same-timestamp ties are excluded.** A transaction sharing its timestamp with another on the same
key does not see it, in either direction. The reason is the serving path: at authorisation you do
not know about a transaction being authorised at the same instant somewhere else. Stage 0 measured
573,349 distinct TransactionDT values across 590,540 rows, so ties are common enough that the
choice moves real numbers rather than being a formality.

The same rule governs the sequential features. `_blocks` walks the stream in blocks of equal
`(key, timestamp)`, every row of a block reads the state as it stood before the block, and the
whole block updates the state afterwards. Row by row, the first of two simultaneous transactions
would inform the second.

One measured consequence of the tie rule: on train, 164,711 rows have no earlier transaction on
their entity, against the 164,702 entities stage 2 counted. The 9 extra are tied first rows, where
two transactions on one entity share a timestamp and neither counts as prior to the other.

---

## 2. Three grains, and why there is more than one

| Grain | Key | Families | Keys on the whole file |
| --- | --- | --- | --- |
| entity | card1 + addr1 + floor(day - D1), ADR 0001 | velocity, recency, device, amount | 217,850 |
| card | card1 | velocity, geography | 13,553 |
| content | the six fields stage 2 called the same purchase | duplicate | 365,153 |

The entity grain is what the stage brief asked for. The other two are each there for a measured
reason.

**The card grain exists because the entity key pins its own components.** `addr1` is a component
of the ADR 0001 key, so an entity holds exactly one `addr1` value: measured at a maximum of 1
distinct value across all 164,702 train entities, and 0.000000 of entities with more than one. A
deviation from an entity's own modal `addr1` is identically zero. At the card grain 0.3320 of
cards have used more than one `addr1` and one card has used 63, so that is where the transaction
region against home region signal lives. `addr2` is not in the key and turns out to be nearly
pinned by it anyway, 19 of 164,702 entities holding two values, which is why the grain question
needed a measurement rather than a reading of the key definition. ADR 0023.

The card grain also carries velocity, for the separate reason in section 4.

**The content grain is a second keyspace, not a wider row in the first.** Stage 8 maintains three
stores. `reports/feature_summary.json`, block `running_state`, is the list.

---

## 3. What ships

| Family | Features | Grain | State at serve time |
| --- | --- | --- | --- |
| velocity | 18 | entity and card | the key's (timestamp, amount) pairs inside the widest window |
| recency | 4 | entity | first prior timestamp, last prior timestamp, prior count |
| amount | 5 | entity | prior count, running mean, running M2, plus the 24 hour amount ring |
| geography | 4 | card, and one per-row | a count per address value the card has used |
| device | 2 | entity | the set of device combinations the entity has used |
| duplicate | 3 | content | prior count, last prior timestamp, timestamps inside the hour |
| tenure | 1 listed, 0 built | row | none |

35 of the 37 catalogue entries require running state. The two that do not are `D1`, which arrives
on the request, and `geo_dist1_bucket`, whose edges are a fitted artifact read from the model
bundle.

**The windows are 600, 3,600 and 86,400 seconds and they are not round by accident.** The day is
the unit stage 0 established. The two shorter windows bracket the measurement stage 2 made on the
inter-transaction gap: for an entity that ever carries fraud the median gap is 3,177 seconds and
the 25th percentile is 460, against 165,394 and 4,243 for an entity that never does. 600 sits
under that 25th percentile and 3,600 sits just above the median.

**The recency mean gap is a telescoping sum, and that is a serving decision.** The mean of the
gaps between an entity's consecutive prior transactions is `(last_prior - first_prior) /
(n_prior - 1)`, so it needs two timestamps and a count rather than a list of gaps. That is why the
recency family's state is three scalars.

**Null and zero are different answers and the rule is stated once.** A count of history is zero
when there is no history, because zero is the true count. A comparison against history is null,
because there is nothing to compare against and zero would merge "no history" with "history, and
it matched". Stage 2 spent a section separating exactly that pair on D1 and ADR 0012 wrote the
reading down. So `vel_amount_sum_10m_entity` is 0.0 on a first transaction and
`vel_amount_mean_10m_entity` is null: the entity spent nothing in the last ten minutes, and the
average transaction was not worth nothing.

Every feature that carries a null declares what the null means, and a test asserts that a feature
carrying nulls has a stated meaning and that a feature carrying none does not declare one.

**Tenure is listed and not rebuilt.** D1 is days since the card began; it is already in the frame
and copying it would be a second source for one number. Its normalised origin was measured and is
not shipped: `D1_origin` carries PSI 1.4428 against val and 1.8304 against test, and that 1.8304
is the number ADR 0013 measured for the same column under another name. It is `card_start_day`
without the floor, differing from it by between 0.0 and 0.9999884 across 589,271 rows. Raw D1
carries PSI 0.0226 against test. The switch exists and is off.

---

## 4. Own velocity against the native C columns

This is the comparison that exists in no other public treatment of this dataset, and the reason it
is worth making is that the two blocks appear to be doing the same job with very different
guarantees.

C1 to C14 are integer valued, null-free over all 590,540 rows, and stage 2 described the block as
running counters after measuring C8 with C10 at Spearman 0.9710. Their meaning is widely repeated
as counts associated with the payment card. **This repo has not established that.** Stage 0 could
not fetch the competition's data description, which is rendered client side, and the same fetch
failed again in stage 4. So the C block is treated as fourteen unlabelled counters whose structure
is measured and whose construction is unknown, which means nothing can be said about whether they
respect a point-in-time rule.

The comparison happens at the card grain, because the native counters are card-associated and the
entity key is finer than a card by construction: 13,553 card1 values against 217,850 entities.
Comparing an entity-grain velocity with a card-associated counter would not be like for like.

**They are not the same measurement.** Spearman on the train split, all 84 pairs, computed exactly
rather than screened:

| Quantity | Value |
| --- | --- |
| Largest absolute rho | 0.2773, `vel_count_24h_entity` with C7 |
| Median absolute rho | 0.0658 |
| Pairs at or above 0.50 | 0 |
| Stage 2's redundancy threshold | 0.95 |

Nothing reaches half the line at which stage 2 would call two columns the same column, so both
blocks are kept and stage 6 measures the contribution of each.

**And the native block is stronger.** Reported as the fraud rate where the count is positive
against where it is zero, because six of these columns are zero-inflated enough that ten
equal-frequency bins collapse and the information value understates them:

| Feature | Share positive | Rate positive | Rate zero | Ratio | Information value | Bins |
| --- | --- | --- | --- | --- | --- | --- |
| C7 | 0.1183 | 0.11232 | 0.02482 | 4.525 | 0.4899 | 10 |
| C12 | 0.1521 | 0.09938 | 0.02365 | 4.201 | 0.5013 | 10 |
| C4 | 0.2522 | 0.07800 | 0.02072 | 3.764 | 0.5629 | 10 |
| C10 | 0.2536 | 0.07502 | 0.02163 | 3.469 | 0.5226 | 10 |
| C8 | 0.2637 | 0.07386 | 0.02131 | 3.466 | 0.5532 | 10 |
| vel_count_24h_entity | 0.2983 | 0.06140 | 0.02402 | 2.556 | 0.1747 | 3 |
| vel_count_10m_entity | 0.0925 | 0.07748 | 0.03086 | 2.511 | 0.0000 | 1 |
| vel_count_1h_entity | 0.1615 | 0.06821 | 0.02881 | 2.368 | 0.0738 | 2 |
| vel_count_10m_card | 0.1958 | 0.05199 | 0.03107 | 1.673 | 0.0165 | 2 |
| vel_count_24h_card | 0.7873 | 0.03815 | 0.02414 | 1.580 | 0.0386 | 8 |
| vel_count_1h_card | 0.4279 | 0.04227 | 0.02986 | 1.415 | 0.0174 | 4 |

The strongest native counter separates the label about 1.8 times as sharply as the strongest built
velocity, and its information value is more than three times as large. That is not the result this
section was expected to produce.

Two readings are available and this repo cannot choose between them, so both are recorded. The C
block may be a better feature, built over a longer history or a wider notion of what to count, in
which case a production system wants the quantity it holds and stage 8 has a problem, because it
cannot compute a column whose definition is unknown. Or the C block may not be point-in-time, in
which case it separates better for the reason a non-causal feature always does. Nothing here can
distinguish the two: the necessary evidence is the construction. The one adjacent measurement does
not settle it either, since stage 2's time-consistency screen flagged 59 columns and none of C1 to
C14 is among them.

**An information value of 0.0000 on a zero-inflated count is a binning artifact, not a verdict.**
`vel_count_10m_entity` is zero on 0.9075 of train rows, so ten equal-frequency bins collapse into
one and the information value of a single bin is zero by construction. The same column separates
the label 2.5 to 1. The artifact records the bin count beside every information value and the
zero-against-positive comparison beside that. Two features in this stage would have been read as
worthless on the naive number and one of them is the sharpest duplicate feature in the file.

`figures/velocity_vs_native_c.png` draws both panels. ADR 0021.

---

## 5. Every separation is reported on all three splits, and one of them reverses

This section is the one that changed during the stage.

Stage 2 measured a fraud lift inside content-duplicate groups: rows agreeing on card1, card2,
addr1, TransactionAmt, ProductCD and P_emaildomain ran at 0.04157 against 0.02917 outside, a ratio
of 1.425 over 200,113 of 413,378 rows. That measurement is not a feature, because a groupby marks
every member of a group, including its first row, from a statistic that includes the group's later
rows. Stage 4 built the causal version: a row is marked when at least one strictly earlier
transaction shares its content.

**The causal version reproduces stage 2's count through an unrelated implementation.** Stage 2
hashed the six fields and compared rows inside each bucket. Stage 4 factorizes them and counts
strictly earlier matches with searchsorted.

| Quantity | Value |
| --- | --- |
| Stage 2, rows in a group beyond the first | 139,803 |
| Stage 4, rows with a strictly earlier identical row | 139,792 |
| Difference | 11 |
| Rows in a content group sharing a timestamp with a companion | 102 |

The 11 are bounded and explained: rows whose only identical companion shares their timestamp,
which a groupby counts and the tie exclusion does not. That is 0.008 percent, and it is the
strongest cross-check in the stage.

**And on the train split the lift survives, at 1.333.** Which is where the first version of this
section stopped, and the wrong place to stop.

| Split | Rows marked | Rate marked | Rate unmarked | Ratio |
| --- | --- | --- | --- | --- |
| train | 139,792 | 0.04213 | 0.03161 | 1.333 |
| val | 40,921 | 0.03057 | 0.03758 | 0.814 |
| test | 44,663 | 0.03267 | 0.03698 | 0.883 |

**The sign flips, and it flips outside the intervals.** On train, marked runs 0.04109 to 0.04320
against an unmarked 0.03096 to 0.03227. On test, marked runs 0.03106 to 0.03436 against an
unmarked 0.03525 to 0.03878. Having repeated an earlier purchase is a risk factor on the training
window and a mild protective factor on the two later ones.

**Restricted to an hour it holds on all three.**

| Split | Rows marked | Rate marked | Rate unmarked | Ratio |
| --- | --- | --- | --- | --- |
| train | 31,697 | 0.08815 | 0.03077 | 2.865 |
| val | 6,302 | 0.07521 | 0.03121 | 2.410 |
| test | 8,876 | 0.06377 | 0.03158 | 2.019 |

So the signal is not that a purchase repeats. It is that it repeats **soon**, and the all-history
count dilutes that until it reverses. The mechanism is in the recency column's decile table:

| Seconds since the last identical purchase | Fraud rate |
| --- | --- |
| 1 to 186 | 0.10689 |
| 187 to 1,431 | 0.07399 |
| 1,432 to 63,534 | 0.06282 |
| 63,539 to 198,660 | 0.04292 |
| 198,670 to 450,308 | 0.02926 |
| 450,321 to 785,133 | 0.02418 |
| 785,204 to 1,291,560 | 0.02103 |
| 1,291,562 to 2,148,136 | 0.02060 |
| 2,148,153 to 3,467,542 | 0.01924 |
| 3,467,794 to 10,315,537 | 0.02039 |
| no earlier identical purchase | 0.03161 |

Six of the ten deciles sit below the base rate and they hold 0.6000 of the marked population, so a
flag that marks all of them together is averaging a strong positive over a large mild negative.
Which side wins depends on the mix of recent against old repeats in the window, and that mix is
not stationary. `dup_seconds_since_prev` carries an information value of 0.1802 against 0.0086 for
`dup_prior_count`, and it ships as a continuous column so a tree can use both directions.

`dup_prior_count` is kept as a **candidate** and is not claimed to carry signal. ADR 0009's
precedent is that a single-feature marginal does not drop a feature, and a tree can use a column
whose marginal reverses if it interacts with recency. Stage 6 settles it with a model.

**Every count feature in the stage now carries the same per-split comparison**, whether or not it
survives it, and all six velocity features do: same sign, disjoint intervals, on all three splits.
`vel_count_24h_entity` is the steadiest thing in the stage at 2.556, 2.625 and 2.482.

Of the fourteen C columns, eleven keep the sign of their separation across the three splits and
three do not, and those three are an artifact rather than a finding: C2 is zero on 404 of 413,378
train rows, so the rate on its zero side is computed from a few hundred transactions. Six of the
eleven stable columns are stably inverse, with the zero side carrying the higher rate, which a
tree reads without difficulty. The artifact reports `positive_is_riskier` and `direction_stable`
separately for that reason.

`figures/duplicate_content.png` draws both panels. ADR 0022.

---

## 6. Coverage, and what it says about ADR 0015

Stage 3 excluded entity target encoding because the encoding recovered the validation label at an
AUC of 0.9877 on rows whose entity the training window had seen and 0.4915 on rows it had not, and
stage 1 had measured 0.6598 of test rows sitting on an unseen entity. Stage 3's own summary told
stage 4 that entity-history features are missing for 0.66 of test rows by construction.

**They are not.** That number is about a train-fitted statistic and it does not apply to a
point-in-time one.

| Split | Entity seen in the training window | Has an earlier row anywhere in the stream |
| --- | --- | --- |
| val | 0.4689 | 0.6964 |
| test | 0.3402 | 0.7036 |

An entity the training window never saw can still have transacted earlier inside the split it
belongs to, and at authorisation the online store holds that history whatever window the model was
fitted on. So a train-fitted entity statistic reaches 0.3402 of test rows and a point-in-time
entity feature reaches 0.7036, which is roughly twice as many. The first column reproduces stage
1's unseen share complemented, and a test asserts that it does.

Cold coverage per grain, where cold means the stream holds no strictly earlier transaction on that
key:

| Split | Entity, all history | Card, 24 hours | Content, all history |
| --- | --- | --- | --- |
| train | 0.3985 | 0.2127 | 0.6618 |
| val | 0.3036 | 0.2277 | 0.5380 |
| test | 0.2964 | 0.2204 | 0.4958 |

Coverage improves on the later splits rather than degrading, because the stream is longer by then.

**One thing degrades and it is worth naming.** The unbounded "has this entity transacted before"
signal collapses out of sample while the windowed version does not:

| Comparison | train | val | test |
| --- | --- | --- | --- |
| rec_prior_count positive against zero | 1.558 | 1.134 | 1.084 |
| vel_count_24h_entity positive against zero | 2.556 | 2.625 | 2.482 |

That is the same finding as the duplicate reversal, from the other direction. A bound on how far
back a feature looks is not only a serving convenience; on this data it is what makes the feature
hold.

---

## 7. The serving footprint

`reports/feature_summary.json`, block `state_footprint`, measured on the whole stream. Stage 8
reads it.

| Keyspace | Keys | Transactions per key, mean | p99 | Max |
| --- | --- | --- | --- | --- |
| entity | 217,850 | 2.711 | 20.0 | 1,414 |
| card | 13,553 | 43.573 | 786.4 | 14,932 |
| content | 365,153 | 1.617 | 9.0 | 553 |

Two structures grow rather than being bounded by a window, and these are the numbers that bound
them on this data:

| Structure | Keys with any | Mean | p99 | Max |
| --- | --- | --- | --- | --- |
| distinct addr1 per card | 12,066 | 3.110 | 35.35 | 64 |
| distinct addr2 per card | 12,066 | 1.015 | 2.0 | 5 |
| distinct devices per entity | 69,186 | 1.361 | 7.0 | 25 |

The velocity ring holds a key's (timestamp, amount) pairs inside the widest window and nothing
older, so it is bounded by activity inside 24 hours rather than by lifetime count. The recency and
amount families hold scalars. A production version would cap the two counters above, and the
distributions here are what a cap should be set against.

---

## 8. Drift baseline

PSI of every shipped feature against val and test, binned on train, the same convention stage 2
used. This is stage 9's baseline.

Six features clear 0.10 against test and every one of them reads all of a key's history rather
than a window:

| Feature | PSI against test | What it reads |
| --- | --- | --- |
| dup_seconds_since_prev | 0.1914 | all history on the content key |
| dup_prior_count | 0.1597 | all history on the content key |
| rec_mean_gap | 0.1444 | all history on the entity |
| rec_gap_deviation | 0.1279 | all history on the entity |
| rec_prior_count | 0.1194 | all history on the entity |
| geo_n_distinct_addr1_on_card | 0.1127 | all history on the card |

Nothing else clears 0.10. All 18 velocity features sit below 0.02 against test, the largest being
0.0118, and `dev_n_distinct_so_far` and `dup_prior_count_1h` read 0.0000 to four places.

That pattern is the third time this stage has found the same thing. A feature bounded by a window
drifts less, separates the label more stably out of sample (section 5), and reaches more rows
(section 6) than the unbounded version of itself. None of the three was expected going in, and
none of them is a claim about why.

---

## What is not established

- What the C columns count. Their structure is measured and their construction is not, so whether
  they are point-in-time cannot be said. This is the largest open question the stage leaves and it
  lands on stage 8 if stage 6 finds they contribute.
- Whether the content-duplicate groups are card testing or ordinary repeat business. Stage 2 left
  this open and stage 4 has not closed it. The recency table is suggestive of card testing and
  suggestive is not measured.
- Why `dup_prior_count` reverses. The mix of recent against old repeats differs between the
  windows and that is arithmetic, not a mechanism. What changed about the population is not
  established.
- Whether `dup_prior_count` carries anything to a model despite its marginal. Stage 6.
- What the right window lengths are. The three are argued from stage 2's gap distribution, which
  makes them defensible rather than optimal, and no sweep was run.
- Whether the addr2 deviation is worth its width. 122 of 12,242 cards ever change the column.
