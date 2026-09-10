# Stage 1: the data layer

What the stage built: a typed loader, a chronological split with fixed boundaries, a runtime guard
against fitting on the wrong rows, and two artifacts with commands behind them. What follows is
the parts that were not obvious in advance.

## Watching the tests fail

A test that has never failed is not a guarantee, it is a hope with a green tick next to it. So the
split and the guard were each broken on purpose and the suite was run against the damage.

**Break 1: a random split.** The body of `time_based_split` was replaced with a shuffle and three
slices at 70 and 85 percent, which is the split ADR 0002 rejects. Ten tests failed. The first
message was the one worth having:

```
AssertionError: train reaches 15811131 but val starts at 156365
```

That is the contract naming the property that broke, not a metric looking slightly off. The
train-only guard failed separately and independently, reporting 124,020 of 413,378 rows at or
after the boundary, which is the shape of a shuffled 70 percent slice of a 182 day file.

**Break 2: overlapping windows.** The val mask was widened backwards by 500,000 units so train and
val share five and a half days. Five tests failed:

```
AssertionError: train reaches 10423792 but val starts at 9951526
```

Worth doing separately from break 1 because it is the subtler failure. The split still looks
chronological, still returns three ordered frames, still has the right total row count. Only the
direct comparison of the boundary values catches it, which is the argument for asserting the
contract rather than inferring it.

**Break 3: the guard neutered.** `assert_train_only` was made to return its argument unchanged.
Nine tests failed. Nothing else in the suite noticed, which is the point: with the split intact,
nothing else in the repo has any reason to notice, and that is the whole case for the guard being
a check rather than a convention.

All three were restored and the full suite runs clean. `test_a_random_split_fails_the_contract`
keeps the first of the three permanently: it builds a shuffled split and asserts the contract
rejects it, so the failing case is a test rather than an afternoon someone once had.

## The measurement that would not hold still

Peak resident set was going to be the headline number for the typing work. It is not, because it
does not repeat.

Frame bytes is deterministic. Five runs of the same strategy give the same value, and the profile
script raises if they do not. Peak resident set of the same five runs is another matter. In
reports/memory_profile.json the naive strategy's runs spread over 0.2103 of their own median,
while the typed strategy's spread over 0.0008. Identical work, same machine, no other load.

The first version of the profile ran each strategy once, and that single draw claimed the typed
load had a substantially lower peak than the naive one. Running the same command again this
session produced a draw where the two overlapped and the ordering could not be called. Both were
single samples of a quantity with a very wide spread on one side, and the disagreement is the
measurement telling you what it is worth.

What is probably going on: this kernel compresses inactive pages, so peak resident set is a floor
on what a process needed rather than a ceiling, and 31 columns of small string objects give it
more to compress than a few large contiguous float buffers. That is a guess and it is written down
as one. What is not a guess is that the number moves, so the script now runs five times per
strategy and reports the spread, and the artifact compares pairs by whether their ranges are
disjoint rather than by comparing medians. Two of the comparisons survive that treatment cleanly
and they are the two the ADR leans on: the pyarrow engine's peak sits entirely above the C
engine's, and chunked reading's sits entirely above reading whole.

The typing decision rests on frame bytes, 0.459 of naive, which does repeat.

## Dead ends

**The pyarrow engine.** Same dtype map, same frame to within 87,274 bytes, and roughly twice as
fast: median 2.401 seconds against 4.641. It lost on the only axis the exercise was about. Its
five runs bottom out at 2,987,425,792 bytes of peak resident set while the C engine tops out at
2,038,153,216, so it needs the better part of a gigabyte more headroom to save two seconds on a
read that no later stage is waiting on. Kept
in the profile script as a measured comparison rather than deleted, because "we tried it" is worth
more with a number attached.

**Chunked reading.** Reading in 100,000 row chunks and concatenating was the intuition for
lowering the parse peak. It raises it, because the chunks and the result are alive at the same
time. It also does something worse quietly: pandas builds a category set per chunk, and
concatenating chunks with different sets falls back to string, so 8 of the 31 categorical columns
arrive as plain strings and the frame comes out 42,577,454 bytes larger than reading whole. The
strategy that was supposed to save memory both used more and undid the typing, and it does not
announce either.

**Downcasting after the read.** Never got as far as being measured. If the float64 frame has to
exist before it can be narrowed, nothing is saved during the parse, which is where the memory
actually goes. The dtype map has to reach read_csv.

## The number that changed my mind

0.000375.

The plan was float32 for every numeric column, on the reasoning that these are counts, flags and
Vesta aggregates and none of them needs fifteen significant figures. The round trip measurement
mostly agreed: across the 385 numeric columns outside the integer group, the largest move any
column makes through float32 is 2.2737e-13, on V126. Every one of them is safe.

Except TransactionAmt, which moves by 0.000375 at its maximum of 31,937.391. Small, and still the
wrong kind of small, because it is the one column denominated in money and 0.000375 is a fourth
decimal place in a currency amount. Holding it at float64 costs 4 bytes per row, 2,362,160 bytes,
which is 0.2 percent of what the typing saves. A fifth of a percent to never have to explain why
an amount does not reconcile.

The general lesson from it: a uniform rule that is right for 384 columns can still be wrong for
the one column a person will check by hand.

## An unplanned cross check

scripts/split_summary.py reads six columns through the stage 1 loader and calls
`data_loader.time_based_split`. The stage 0 audit read the same file through a different function
and cut it with quantiles computed on the spot. The two share no code.

They agree exactly: 413,378 / 88,581 / 88,581 rows, 164,702 / 46,884 / 45,348 entities, 0.5736 and
0.6666 of val and test entities unseen in training, carrying 0.5311 and 0.6598 of rows. Two
implementations landing on the same numbers is a stronger statement about the boundaries than
either one alone, and it was free.

## Left undetermined

- Whether peak resident set can be measured usefully on this platform at all. Frame bytes is used
  instead and peak is reported with its spread. A Linux runner would settle it, and CI does not
  have the data files.
- Column provenance. The guard checks which rows a fit sees, not which rows the values in those
  rows were computed from. A column that already carries a whole-file statistic passes it. That
  needs a different mechanism and belongs to the feature layer in stage 4.
- Whether `card_start_day` should be attached at load time or at feature time. It is attached by
  `add_entity_key`, which the loader offers and does not call, so a caller who only wants the raw
  frame does not pay for it. Stage 4 may want it earlier.
