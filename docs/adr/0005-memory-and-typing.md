# 0005: Memory and typing strategy for the joined frame

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 1

## Context

train_transaction.csv left joined to train_identity.csv is 590,540 rows by 434 columns. Read with
pandas defaults it holds 2,080,582,559 bytes across 399 float64 columns, 4 int64 and 31 string.
Nearly all of that is float64 columns holding whole numbers, and 31 string columns holding a
handful of distinct values repeated half a million times.

This frame is loaded by the EDA, the feature build and every training run, so a factor of two here
is a factor of two on every later stage. It also decides how much room is left for the
intermediate frames those stages build on top of it, which is usually what actually runs a machine
out of memory.

Two quantities are involved and they are not the same thing. Frame bytes is
`memory_usage(deep=True).sum()`, the payload of the frame, and it is deterministic. Peak resident
set is the high-water mark of the process that built the frame, parser buffers included. Both are
measured below. Only the first is stable enough to decide on, for reasons the evidence section
gives.

## Options considered

**Read with pandas defaults, downcast afterwards.** The obvious version, and it does not work.
The point of a dtype is to never allocate the wide frame at all. Downcasting after the read means
the float64 frame exists first, so nothing is saved while the file is being parsed and the only
gain is on what the process holds once it is done. Rejected on that alone.

**Read through the pyarrow engine.** Same dtype map, a different parser. It produces almost the
same frame, 954,959,344 bytes against 955,046,618, in roughly half the time: median 2.401 seconds
against 4.641. Rejected on peak resident set, where it is the worst of the four by a wide margin:
its five runs range from 2,987,425,792 to 3,526,950,912 bytes, entirely above the C engine's
range, which tops out at 2,038,153,216. Halving the read time is not worth needing at least 0.9 GB more
headroom on a frame this size, and the read is not the slow part of any later stage.

**Read in 100,000 row chunks and concatenate.** Rejected on both counts. Peak resident set is
higher than reading whole (2,282,618,880 to 2,283,339,776 against 2,036,514,816 to 2,038,153,216,
two ranges that do not touch) because the chunks and the concatenated result exist at the same
time. And it quietly loses the typing: pandas builds a category set per chunk, and concatenating
chunks whose sets differ falls back to string, so 8 of the 31 categorical columns come out as
plain strings and the frame is 42,577,454 bytes larger. A strategy that discards part of the thing
it was asked to preserve, without saying so, is worse than the slow one.

**Explicit dtype map passed to read_csv, C engine.** Chosen. The wide frame is never materialised,
so the saving is real while the file is being parsed and not only afterwards.

## Decision

Build the dtype map from the column names and hand it to `read_csv`.
`fraud_platform.data_loader.build_dtype_map` applies four rules in order.

| Rule | Columns | Dtype | Basis |
| --- | --- | --- | --- |
| Measured null-free integers | 18 | int8, int16, int32 | Narrowest type the measured range fits |
| String columns | 31 | category | Highest cardinality is 1,786 against 590,540 rows |
| The amount | 1 | float64 | float32 moves it by up to 0.000375 |
| Everything else numeric | 384 | float32 | float32 moves nothing else by more than 2.3e-13 |

The 18 integer columns, with the measured range each dtype was chosen for: TransactionID 2,987,000
to 3,577,539 as int32; isFraud 0 to 1 as int8; TransactionDT 86,400 to 15,811,131 as int32; card1
1,000 to 18,396 as int16; C1 to C14 as int16, the widest of them C2 at 0 to 5,691. All 18 have
zero nulls. Declaring them as integers rather than floats is also a check: if a null ever appears
in one, read_csv raises instead of quietly promoting the column to float, which is how a change
in the input file goes unnoticed for a stage or two.

Every other numeric column becomes float32, and that is lossless for all of them. Of the 385
numeric columns outside the integer group, 305 hold whole numbers and 80 hold fractions. The
largest move any of them makes through a float32 round trip, excluding the amount, is 2.2737e-13
absolute on V126, a relative move of 1.83e-15.

The amount is the exception. A float32 round trip moves TransactionAmt by up to 0.000375 on a
maximum value of 31,937.391, a relative move of 5.879e-08. That is a fourth decimal place in a
number denominated in money, and holding the column at float64 costs 4 bytes per row, 2,362,160
bytes, which is 0.2 percent of what the typing saves. Keeping money exact for a fifth of a percent
is the easy side of that trade.

Categories are sorted after loading. read_csv assigns codes in the order values are first seen,
which depends on how the file happens to be ordered. Sorting makes the code for a value a function
of the category set alone, so a frame loaded from a subset of columns and a frame loaded whole
agree wherever their category sets do.

## Evidence

- **Artifact:** reports/memory_profile.json
- **Command:** `make memory-profile`
- Five runs per strategy, each in its own subprocess, because peak resident set is a high-water
  mark for a whole process and measuring two strategies in one process would report the larger
  twice. Measured on macOS-27.0-arm64.

| Strategy | Frame bytes | Categorical columns | Peak RSS, median | Peak RSS, min to max | Spread over median | Seconds, median |
| --- | --- | --- | --- | --- | --- | --- |
| naive | 2,080,582,559 | 0 | 2,580,463,616 | 2,192,457,728 to 2,735,079,424 | 0.2103 | 4.880 |
| typed | 955,046,618 | 31 | 2,036,908,032 | 2,036,514,816 to 2,038,153,216 | 0.0008 | 4.641 |
| typed, pyarrow engine | 954,959,344 | 31 | 3,508,944,896 | 2,987,425,792 to 3,526,950,912 | 0.1538 | 2.401 |
| typed, 100k chunks | 997,624,072 | 23 | 2,282,995,712 | 2,282,618,880 to 2,283,339,776 | 0.0003 | 4.833 |

The typed frame is 0.459 of the naive frame, a saving of 1,125,535,941 bytes. That number is
deterministic: the script asserts frame bytes agree across the five runs of a strategy and raises
if they do not.

Peak resident set is not deterministic, and it is unstable unevenly. The naive strategy's five
runs of identical work spread over 21 percent of their own median; the typed strategy's spread
over 0.08 percent. The kernel on this platform compresses inactive pages, so peak resident set is
a floor on what a process needed rather than a ceiling, and a strategy that allocates many small
string objects gives it more to work with than one that allocates a few large contiguous buffers.
That is the likeliest reason the naive figure moves and the typed figure does not, and it is a
guess, not a measurement.

So the artifact reports each pair of strategies as disjoint or overlapping rather than comparing
medians, in section `peak_rss_comparisons`, and only a disjoint pair is given a direction. Two
comparisons carry the rejections above and both are disjoint by a wide margin with a stable
measurement on at least one side: pyarrow's peak sits entirely above the C engine's, and chunked's
sits entirely above reading whole. The naive against typed comparison is disjoint in this run too,
with naive higher, but a spread of 21 percent is not a measurement to lean on and the decision
does not rest on it.

Round trip measurements are in section `float32_roundtrip`, integer ranges in
`integer_column_ranges`, the whole-number against fractional count in `numeric_column_shape`, and
categorical cardinality in `categorical_cardinality`, where the 31 columns run from 2 distinct
values to 1,786, the widest being DeviceInfo.

## Consequences

- Later stages get a frame under 1 GB, so an intermediate copy fits alongside it on a machine
  where two copies of the naive frame would not.
- float32 is the working precision for every model input except the amount. Anything that sums a
  float32 column over hundreds of thousands of rows should accumulate in float64. The per value
  storage error is 1e-13 or smaller, but a float32 accumulator does not have the precision to
  hold a running total over 590,540 values, and that is a property of the accumulator, not of the
  storage.
- The 18 integer columns raise on a null rather than promote. Intended: a future data refresh that
  introduces one fails at the read rather than three stages later.
- A new string column not listed in `config.CATEGORICAL_COLUMNS` gets float32 and read_csv raises.
  Also intended. tests/test_data_loader.py asserts the categorical set equals the set of string
  columns in the real files, so the failure arrives with an explanation.
- Peak resident set is not a number this repo quotes outside this ADR. Frame bytes is.
