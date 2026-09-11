"""Behavioural features over the ADR 0001 entity key, every one of them point-in-time.

One rule binds this module, and everything in it is arranged so the rule is structural rather
than careful: **a feature for a transaction at time t reads only rows strictly before t.** Not
"before t within the same fold". Strictly before t, on the whole stream, because that is what an
authorisation endpoint has when it is asked for a score.

**Why searchsorted and not a rolling join.** A trailing window is two indices into a sorted
array. `np.searchsorted(ts, ts - w, side="left")` is the first row at or after `t - w`, and
`np.searchsorted(ts, ts, side="left")` is the first row at `t`, so the slice `[lo, hi)` holds
every row in `[t - w, t)` and the current row lies outside it by construction. There is no window
argument to get wrong, no `closed=` to set, and no way to include the present row without
changing `side` on purpose. A rolling join computes the same numbers and puts the exclusion in a
parameter instead of in the arithmetic. ADR 0018.

**Same-timestamp ties are excluded, deliberately.** `side="left"` on the upper bound stops at the
first row whose timestamp equals `t`, so a simultaneous transaction on the same key is not
counted. At authorisation you do not know about a transaction being authorised at the same instant
somewhere else. Stage 0 measured 573,349 distinct TransactionDT values across 590,540 rows, so
ties exist and the choice moves real numbers.

**Three grains, and why there is more than one.**

- `entity`, the ADR 0001 key `card1 + addr1 + floor(TransactionDT / 86400 - D1)`. Velocity,
  recency, device novelty and amount behaviour are keyed here, which is what the stage asked for.
- `card`, plain `card1`. The geography family has to be keyed here and the reason is measured:
  `addr1` is a component of the entity key, so an entity holds exactly one `addr1` value and a
  deviation from its own modal value is identically zero. The velocity family is computed at this
  grain as well, because the native C columns are card-associated and a comparison between them
  and this repo's own velocity is only like for like at the same grain. ADR 0021 and ADR 0023.
- `content`, the six fields stage 2 called the same purchase. The duplicate family is keyed here,
  which is a second keyspace rather than a wider row in the first, and
  `reports/feature_summary.json` says so per feature because stage 8 has to hold both.

**What is null and what is zero.** One rule, applied everywhere: a count of history is zero when
there is no history, because zero is the true count; a comparison against history is null,
because there is nothing to compare against and zero would merge "no history" with "history, and
it matched". Stage 2 spent a section separating exactly that pair on D1, where a zero is a first
observed transaction and not a disguised null, and ADR 0012 wrote the reading down. A null here
means what `missing_policy.SENTINEL_MEANING` says: the stream holds no earlier event of this kind.

**Exactly, not approximately.** Every running quantity is accumulated inside its own key rather
than read off a global cumulative sum, and the amount moments use Welford rather than a sum of
squares. Both choices are there so that truncating the stream leaves every feature bit for bit
identical, which is what `tests/test_causality.py` asserts. A global cumulative sum would have
made the last bits of a window sum depend on every amount in the file, which is immaterial to a
model and is still a dependency on the future.

**What this module does not build.** `NOT_BUILT` names it, and the first entry is the published
UID aggregation. It is not implemented, not behind a switch and not reachable from here. ADR 0020.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraud_platform import config, encoders, transforms
from fraud_platform.data_loader import train_only

# --- choices ------------------------------------------------------------------------------

# The trailing windows, as (suffix, seconds). Ten minutes, an hour and a day. The day is the unit
# stage 0 established for this dataset. The two shorter windows bracket the measurement stage 2
# made on the inter-transaction gap: for an entity that ever carries fraud the median gap is 3,177
# seconds and the 25th percentile is 460, against 165,394 and 4,243 for an entity that never does.
# 600 seconds sits under that 25th percentile and 3,600 sits just above the median, so the pair
# straddles the distribution the signal lives in rather than being round numbers.
WINDOWS: tuple[tuple[str, int], ...] = (("10m", 600), ("1h", 3_600), ("24h", 86_400))

# The window the amount-repeat flag looks back over, and the one the duplicate family counts
# inside. A day for the first because a repeated amount is a daily-scale pattern and the widest
# velocity window already holds the state it needs. An hour for the second, on the same number as
# above: 3,600 seconds is the median inter-transaction gap of an entity that carries fraud.
AMOUNT_REPEAT_WINDOW_SECONDS = 86_400
DUPLICATE_WINDOW_SECONDS = 3_600

# The definition of the same purchase, taken from stage 2 rather than re-invented.
# quality.json `duplicates.repeated_core_fields` measured these six fields grouping 200,113 of
# 413,378 train rows at a fraud rate of 0.04157 against 0.02917 outside, while the whole-row
# definitions found 10 rows in 413,378 because the C and V blocks are running counters. ADR 0022.
CONTENT_COLUMNS: tuple[str, ...] = (
    "card1",
    "card2",
    "addr1",
    config.AMOUNT_COLUMN,
    "ProductCD",
    "P_emaildomain",
)

# The device combination, on the normalised columns stage 3 built. DeviceInfo and id_31 are the
# two free-text columns and `encoders.add_normalised_free_text` is what bounds them: stage 3
# measured DeviceInfo from 1,546 levels to 281 and id_31 from 108 to 43, with unseen test rows
# falling from 500 to 30 and from 8,068 to 149. A novelty feature on the raw strings would call a
# browser point release a new device.
DEVICE_COMBINATION_COLUMNS: tuple[str, ...] = (
    encoders.normalised_name("DeviceInfo"),
    "DeviceType",
    "id_30",
    encoders.normalised_name("id_31"),
)

# dist1 is bucketed on train quantiles. Ten buckets, because stage 2 binned every numeric feature
# into deciles for its information value and PSI work (`eda.DEFAULT_BINS`), so the edges here
# follow that convention rather than introducing a second one.
DIST_COLUMN = "dist1"
N_DIST_BUCKETS = 10

# The native column groups stage 6 measures against these features. Behind a switch rather than
# assumed useful or assumed useless.
NATIVE_GROUPS: tuple[str, ...] = ("count", "timedelta", "match", "vesta")

FAMILIES: tuple[str, ...] = (
    "velocity",
    "recency",
    "tenure",
    "geography",
    "device",
    "amount",
    "duplicate",
)

GRAINS: tuple[str, ...] = ("entity", "card", "content")

# A null component of a key is a level of its own, so two rows null in the same field share it.
# That is stage 2's rule from `eda.duplicate_groups`, and reproducing it is what makes the
# duplicate counts here comparable with the ones in quality.json.
_NULL_LEVEL = -1

# Group code and timestamp are packed into one ascending int64 key so a single searchsorted covers
# every group at once. The stride has to exceed the largest timestamp for the packing to preserve
# order; stage 0 measured TransactionDT at a maximum of 15,811,131 and the column is int32, so
# 2**32 clears it with room and keeps the product inside int64 for any key count this data makes.
_KEY_STRIDE = 1 << 32

NOT_BUILT: tuple[str, ...] = (
    "the published UID aggregation: groupby the entity over the C, M and normalised D columns, "
    "with each aggregate taken over every row of the group rather than over the rows before t. "
    "Not implemented, not behind a switch, not reachable from this module. ADR 0020 is the "
    "argument and stage 7 is where its offline gain is measured against its serving cost",
    "merchant risk. There is no merchant identifier and no merchant category code among the 434 "
    "columns, so the feature a fraud platform would most want cannot be built here at all. "
    "ADR 0019 lists the proxies used instead and states that their strength is not comparable",
    "any target-derived entity statistic. ADR 0015 excluded entity target encoding on a measured "
    "validation AUC of 0.9877 on seen entities against 0.4915 on unseen ones, and nothing here "
    "summarises an entity's labels. Behaviour only",
    "a deviation of addr1 from the entity's modal addr1. addr1 is a component of the entity key, "
    "so an entity holds one addr1 value: measured at 164,702 train entities with a maximum of 1 "
    "distinct value each. The feature is identically zero and the geography family is keyed on "
    "card1 instead. ADR 0023",
)


# --- the primitive --------------------------------------------------------------------------


def trailing_bounds(
    ts: npt.NDArray[np.int64], window: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Half-open bounds `[lo, hi)` of the window `[t - window, t)` for one key's stream.

    `ts` is that key's timestamps, ascending. This is the whole causal mechanism, written once so
    that it can be read:

    - `side="left"` on the lower bound puts a row exactly at `t - window` inside the window;
    - `side="left"` on the upper bound stops before the first row at `t`, which excludes the
      current row and every other row sharing its timestamp.

    The second is the one to be deliberate about. A simultaneous transaction on the same key is
    not knowable at authorisation, so it is not counted, and `side="right"` would count it.

    `grouped_trailing_bounds` is what the pipeline calls. This exists because it is the
    definition, and because `tests/test_causality.py` checks the grouped version against a loop
    over this one.
    """
    lo = np.searchsorted(ts, ts - window, side="left")
    hi = np.searchsorted(ts, ts, side="left")
    return lo.astype("int64"), hi.astype("int64")


def _packed_keys(
    codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """`(group, time)` packed into one int64 key, plus the key each group starts at."""
    if codes.size and int(codes.min()) < 0:
        raise ValueError("group codes must be non-negative")
    if ts.size and (int(ts.min()) < 0 or int(ts.max()) >= _KEY_STRIDE):
        raise ValueError(
            f"timestamps must lie in [0, {_KEY_STRIDE}) for the packing to preserve order, got "
            f"[{int(ts.min())}, {int(ts.max())}]"
        )
    starts = codes.astype("int64") * _KEY_STRIDE
    return starts + ts.astype("int64"), starts


def grouped_trailing_bounds(
    codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64], window: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """`trailing_bounds` for every group at once, over a stream sorted by group then time.

    Packing the group code and the timestamp into one ascending key turns one searchsorted per key
    into one searchsorted for the whole frame. The lower search target is clamped to the group's
    own start, so a window wider than the group's first timestamp cannot reach back into the
    previous group's rows.
    """
    keys, starts = _packed_keys(codes, ts)
    lo = np.searchsorted(keys, np.maximum(keys - int(window), starts), side="left")
    hi = np.searchsorted(keys, keys, side="left")
    return lo.astype("int64"), hi.astype("int64")


def grouped_prior_bounds(
    codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Bounds of every row of the group strictly before `t`: the expanding version of the above.

    Same upper bound, so ties are excluded here too. The lower bound is the group's first row.
    """
    keys, starts = _packed_keys(codes, ts)
    lo = np.searchsorted(keys, starts, side="left")
    hi = np.searchsorted(keys, keys, side="left")
    return lo.astype("int64"), hi.astype("int64")


def group_starts(codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """The index each row's group begins at, over a stream sorted by group then time."""
    keys, starts = _packed_keys(codes, ts)
    return np.searchsorted(keys, starts, side="left").astype("int64")


def grouped_prefix_sum(
    codes: npt.NDArray[np.int64], values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Cumulative sum restarted at every group, as one array of length n + 1.

    Restarted, and not a global cumulative sum with the group's base subtracted, and the
    difference is the whole reason this function exists. A global `np.cumsum` carries a rounding
    error proportional to the total it has accumulated so far, which means the last bits of a
    ten-minute window sum would depend on every amount earlier in the file, including amounts
    from rows after the one being computed once the difference is taken. That is immaterial to
    any model and it is still a dependency on the future, and `tests/test_causality.py` asserts
    that truncating the stream changes nothing at all rather than changes little. Accumulating
    inside each group makes the claim exact: a group's sums are a function of that group's own
    rows in time order, so removing later rows removes later rows and nothing else.

    Index it with `window_total`, which handles the group boundary.
    """
    out = np.zeros(values.size + 1, dtype="float64")
    if values.size == 0:
        return out
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    edges = [0, *boundaries.tolist(), values.size]
    for start, stop in itertools.pairwise(edges):
        np.cumsum(values[start:stop], out=out[start + 1 : stop + 1])
    return out


def window_total(
    prefix: npt.NDArray[np.float64],
    lo: npt.NDArray[np.int64],
    hi: npt.NDArray[np.int64],
    starts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Total of `[lo, hi)` from a `grouped_prefix_sum`.

    A restarted cumulative sum has no slot for "zero at this group's first row": the boundary
    index belongs to the previous group's last entry. So an index that sits on a group start
    reads as zero here rather than reading the previous group's total.
    """
    lower = np.where(lo == starts, 0.0, prefix[lo])
    upper = np.where(hi == starts, 0.0, prefix[hi])
    return upper - lower


# --- the stream ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    """One grain's rows in (group, time) order, with the map back to the caller's row order."""

    codes: npt.NDArray[np.int64]
    ts: npt.NDArray[np.int64]
    order: npt.NDArray[np.int64]

    @property
    def n_rows(self) -> int:
        return int(self.codes.size)

    def sorted_like(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """A caller-order array put into this stream's order."""
        return values[self.order]


def make_stream(frame: pd.DataFrame, codes: npt.NDArray[np.int64]) -> Stream:
    """Sort by group then time.

    `np.lexsort` is stable, so rows sharing a `(group, time)` pair keep the order they arrived in.
    No feature here can see that order, because a tie is excluded from its own window either way.
    """
    ts = frame[config.TIME_COLUMN].to_numpy(dtype="int64")
    order = np.lexsort((ts, codes))
    return Stream(codes=codes[order], ts=ts[order], order=order)


def _blocks(codes: npt.NDArray[np.int64], ts: npt.NDArray[np.int64]) -> Iterator[tuple[int, int]]:
    """Walk a sorted stream in blocks of equal `(group, timestamp)`.

    Every sequential feature in this module is computed against the state as it stood before the
    block and then updates the state with the whole block, which is how the same-timestamp
    exclusion survives a forward scan. Row by row, the first of two simultaneous transactions
    would inform the second.
    """
    n = codes.size
    start = 0
    while start < n:
        code = int(codes[start])
        moment = int(ts[start])
        stop = start + 1
        while stop < n and int(codes[stop]) == code and int(ts[stop]) == moment:
            stop += 1
        yield start, stop
        start = stop


def _scatter_float(stream: Stream, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    out = np.empty(values.size, dtype="float64")
    out[stream.order] = values
    return out


def _scatter_int(stream: Stream, values: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    out = np.empty(values.size, dtype="int32")
    out[stream.order] = values.astype("int32")
    return out


def _sorted_level_codes(series: pd.Series) -> npt.NDArray[np.int64]:
    """Dense codes ordered by value, with -1 for a null.

    Ordered rather than first-seen because a modal value is read off these codes and a tie has to
    break on the smaller value, not on whichever row the file happened to list first.
    """
    codes, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
    return np.asarray(codes, dtype="int64")


def _row_codes(parts: Sequence[npt.NDArray[np.int64]]) -> npt.NDArray[np.int64]:
    """Dense code per distinct tuple of already-coded columns. Exact, no hashing."""
    stacked = np.stack(list(parts), axis=1)
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    return np.asarray(inverse, dtype="int64").reshape(-1)


def entity_codes(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Dense code per `entity_id`. `data_loader.add_entity_key` never leaves it null."""
    if config.ENTITY_ID_COLUMN not in frame.columns:
        raise KeyError(
            f"features need {config.ENTITY_ID_COLUMN}; call data_loader.add_entity_key first"
        )
    codes, _ = pd.factorize(frame[config.ENTITY_ID_COLUMN], sort=True, use_na_sentinel=False)
    return np.asarray(codes, dtype="int64")


def card_codes(frame: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Dense code per `card1`. Stage 0 measured it null-free over all 590,540 rows."""
    if "card1" not in frame.columns:
        raise KeyError("the card grain needs card1")
    codes, _ = pd.factorize(frame["card1"], sort=True, use_na_sentinel=False)
    return np.asarray(codes, dtype="int64")


def content_codes(
    frame: pd.DataFrame, columns: Sequence[str] = CONTENT_COLUMNS
) -> npt.NDArray[np.int64]:
    """Dense code per distinct value of the six core fields, nulls as a level of their own."""
    present = [column for column in columns if column in frame.columns]
    if not present:
        raise KeyError(f"the content grain needs at least one of {list(columns)}")
    return _row_codes([_sorted_level_codes(frame[column]) for column in present])


def device_codes(
    frame: pd.DataFrame, columns: Sequence[str] = DEVICE_COMBINATION_COLUMNS
) -> npt.NDArray[np.int64]:
    """Dense code per device combination, and -1 where the row carries no identity record.

    A null component of a combination that has at least one present component is a level, which is
    what stage 3's vocabulary encoder does with a null (`encoders.MISSING_TOKEN`). A row with all
    four components null is a row where the identity join failed, and stage 3's policy for the
    identity group leaves that as a null with `has_identity_record` carrying the join state. So
    the combination is undefined there and the device features are null, not novel.
    """
    present = [column for column in columns if column in frame.columns]
    if not present:
        return np.full(len(frame), _NULL_LEVEL, dtype="int64")
    known = np.zeros(len(frame), dtype=bool)
    parts: list[npt.NDArray[np.int64]] = []
    for column in present:
        series = frame[column]
        known |= series.notna().to_numpy(dtype=bool)
        parts.append(_sorted_level_codes(series))
    codes = _row_codes(parts)
    codes[~known] = _NULL_LEVEL
    return codes


GRAIN_CODERS = {
    "entity": entity_codes,
    "card": card_codes,
    "content": content_codes,
}


# --- dist1 buckets, the one fitted object in the stage ---------------------------------------


@train_only
def fit_dist_buckets(
    frame: pd.DataFrame, column: str = DIST_COLUMN, n_buckets: int = N_DIST_BUCKETS
) -> dict[str, Any]:
    """Bucket edges for `dist1`, at quantiles of its present values on the train split.

    Fitted, so it goes through the guard. Duplicate edges collapse, which is why the artifact
    records the number of buckets the edges actually produce next to the number asked for: a
    column concentrated on a few values cannot be cut into ten equal parts, and saying so is
    cheaper than pretending it can.
    """
    values = frame[column].dropna().astype("float64")
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)[1:-1]
    edges = np.unique(np.quantile(values.to_numpy(), quantiles)) if values.size else np.array([])
    return {
        "column": column,
        "n_buckets_requested": int(n_buckets),
        "n_buckets": int(edges.size) + 1,
        "quantiles": [float(q) for q in quantiles],
        "edges": [float(edge) for edge in edges],
        "n_train_rows": len(frame),
        "n_present_train": int(values.size),
        "missing_rate_train": float(frame[column].isna().mean()),
    }


def apply_dist_buckets(frame: pd.DataFrame, fitted: Mapping[str, Any]) -> npt.NDArray[np.float64]:
    """Bucket index per row, null where the column is null.

    Missingness is carried by the distance block indicator stage 3 built, not by folding a null
    into bucket zero: `bivariate.json` measured 255,775 missing dist1 rows at a fraud rate of
    0.0444 against a present-row trough of 0.0128, so the two states are not the same state.
    """
    column = str(fitted["column"])
    edges = np.asarray(fitted["edges"], dtype="float64")
    values = frame[column].to_numpy(dtype="float64")
    buckets = np.searchsorted(edges, values, side="right").astype("float64")
    buckets[np.isnan(values)] = np.nan
    return buckets


# --- the families --------------------------------------------------------------------------


def _velocity(
    stream: Stream,
    amount: npt.NDArray[np.float64],
    grain: str,
    windows: Sequence[tuple[str, int]],
) -> dict[str, npt.NDArray[Any]]:
    """Count, amount sum and amount mean over each trailing window, current row excluded.

    The mean is null on an empty window and the sum is zero, and those are different statements.
    No transaction in the last ten minutes means the key spent nothing in the last ten minutes,
    which is a sum of zero. It does not mean the average transaction was worth nothing.
    """
    prefix = grouped_prefix_sum(stream.codes, amount)
    starts = group_starts(stream.codes, stream.ts)
    out: dict[str, npt.NDArray[Any]] = {}
    for suffix, window in windows:
        lo, hi = grouped_trailing_bounds(stream.codes, stream.ts, window)
        count = hi - lo
        total = window_total(prefix, lo, hi, starts)
        mean = np.divide(
            total, count, out=np.full(count.shape, np.nan, dtype="float64"), where=count > 0
        )
        out[velocity_name("count", suffix, grain)] = _scatter_int(stream, count)
        out[velocity_name("amount_sum", suffix, grain)] = _scatter_float(stream, total)
        out[velocity_name("amount_mean", suffix, grain)] = _scatter_float(stream, mean)
    return out


def velocity_name(statistic: str, window_suffix: str, grain: str) -> str:
    """`vel_count_1h_card`. The grain is in the name because there is more than one."""
    return f"vel_{statistic}_{window_suffix}_{grain}"


def _recency(stream: Stream) -> dict[str, npt.NDArray[Any]]:
    """Seconds since the entity's previous transaction, its running mean gap, and the deviation.

    The mean gap is `(last_prior - first_prior) / (n_prior - 1)`, which is the mean of the gaps
    between consecutive prior transactions written as a telescoping sum. It needs two timestamps
    and a count rather than the list of gaps, and that is why the serving state for this family is
    three scalars per entity instead of a history.
    """
    lo, hi = grouped_prior_bounds(stream.codes, stream.ts)
    n_prior = hi - lo
    ts = stream.ts.astype("float64")
    previous = ts[np.maximum(hi - 1, 0)]
    first = ts[lo]

    since = np.where(n_prior > 0, ts - previous, np.nan)
    span = np.where(n_prior >= 2, previous - first, np.nan)
    mean_gap = np.divide(
        span,
        np.maximum(n_prior - 1, 1).astype("float64"),
        out=np.full(span.shape, np.nan, dtype="float64"),
        where=n_prior >= 2,
    )
    return {
        "rec_seconds_since_prev": _scatter_float(stream, since),
        "rec_prior_count": _scatter_int(stream, n_prior),
        "rec_mean_gap": _scatter_float(stream, mean_gap),
        "rec_gap_deviation": _scatter_float(stream, since - mean_gap),
    }


def _duplicate(stream: Stream, window: int) -> dict[str, npt.NDArray[Any]]:
    """Earlier transactions with identical content: how many, how long ago, how many inside `window`.

    The grain is the content group, not the entity, which is why these three carry a different
    `state_key` in the catalogue. Stage 2 measured the lift that makes them worth building and ADR
    0022 argues the decision to ship them.

    The stage asked for "whether any fell inside a short window". The count inside the window is
    that fact plus the multiplicity, and `dup_prior_count_1h > 0` recovers the boolean exactly, so
    the count is what is stored.
    """
    lo, hi = grouped_prior_bounds(stream.codes, stream.ts)
    n_prior = hi - lo
    ts = stream.ts.astype("float64")
    previous = ts[np.maximum(hi - 1, 0)]
    since = np.where(n_prior > 0, ts - previous, np.nan)
    window_lo, window_hi = grouped_trailing_bounds(stream.codes, stream.ts, window)
    return {
        "dup_prior_count": _scatter_int(stream, n_prior),
        "dup_seconds_since_prev": _scatter_float(stream, since),
        "dup_prior_count_1h": _scatter_int(stream, window_hi - window_lo),
    }


def _mode_of(counts: dict[int, int]) -> int:
    """Most frequent level so far, ties broken on the smaller level. Codes are value-ordered."""
    best_level = _NULL_LEVEL
    best_count = -1
    for level, count in counts.items():
        if count > best_count or (count == best_count and level < best_level):
            best_level = level
            best_count = count
    return best_level


def _geography(
    stream: Stream, addr1: npt.NDArray[np.int64], addr2: npt.NDArray[np.int64]
) -> dict[str, npt.NDArray[Any]]:
    """Whether the address differs from the card's modal address so far, and how many it has used.

    Keyed on the card, not the entity, and the reason is measured rather than argued: at the ADR
    0001 key every one of the 164,702 train entities holds exactly one addr1 value, so the same
    feature at the entity grain is identically zero. At the card grain 0.3320 of cards have used
    more than one addr1 and one card has used 63. ADR 0023.
    """
    n = stream.n_rows
    addr1_differs = np.full(n, np.nan, dtype="float64")
    addr2_differs = np.full(n, np.nan, dtype="float64")
    n_distinct_addr1 = np.zeros(n, dtype="int64")

    counts1: dict[int, int] = {}
    counts2: dict[int, int] = {}
    current = -1
    for start, stop in _blocks(stream.codes, stream.ts):
        code = int(stream.codes[start])
        if code != current:
            counts1, counts2 = {}, {}
            current = code
        mode1 = _mode_of(counts1)
        mode2 = _mode_of(counts2)
        n_seen1 = len(counts1)
        for i in range(start, stop):
            if mode1 != _NULL_LEVEL and addr1[i] != _NULL_LEVEL:
                addr1_differs[i] = 0.0 if addr1[i] == mode1 else 1.0
            if mode2 != _NULL_LEVEL and addr2[i] != _NULL_LEVEL:
                addr2_differs[i] = 0.0 if addr2[i] == mode2 else 1.0
            n_distinct_addr1[i] = n_seen1
        for i in range(start, stop):
            if addr1[i] != _NULL_LEVEL:
                counts1[int(addr1[i])] = counts1.get(int(addr1[i]), 0) + 1
            if addr2[i] != _NULL_LEVEL:
                counts2[int(addr2[i])] = counts2.get(int(addr2[i]), 0) + 1

    return {
        "geo_addr1_differs_from_card_mode": _scatter_float(stream, addr1_differs),
        "geo_addr2_differs_from_card_mode": _scatter_float(stream, addr2_differs),
        "geo_n_distinct_addr1_on_card": _scatter_int(stream, n_distinct_addr1),
    }


def _entity_scan(
    stream: Stream,
    device: npt.NDArray[np.int64],
    amount: npt.NDArray[np.float64],
    amount_level: npt.NDArray[np.int64],
    repeat_window: int,
) -> dict[str, npt.NDArray[Any]]:
    """Device novelty, the repeated-amount flag, and the running amount moments.

    All five come out of one forward pass because they need the same thing: state that a sum over
    a slice cannot give. A set of device combinations, a set of recent amounts, and a running
    mean and spread.

    **The moments are Welford and not a sum of squares.** `mean` and `M2` updated in place is
    what an online store holds, three scalars per key, so this is the serving path's own
    arithmetic rather than an offline shortcut. It is also the numerically sound one: the
    difference form, `(sum(x^2) - n * mean^2) / (n - 1)`, cancels badly when a key's amounts are
    large and close together, and on this file it produces a running standard deviation of the
    wrong order on entities that repeat the same amount. That is the case a fraud feature most
    wants to be right about.

    Amounts are compared as level codes for the repeat flag, which is exact equality on the
    values the file holds and faster than hashing floats.
    """
    n = stream.n_rows
    is_new = np.full(n, np.nan, dtype="float64")
    n_distinct = np.zeros(n, dtype="int64")
    repeat = np.full(n, np.nan, dtype="float64")
    prior_mean = np.full(n, np.nan, dtype="float64")
    prior_std = np.full(n, np.nan, dtype="float64")
    deviation = np.full(n, np.nan, dtype="float64")
    zscore = np.full(n, np.nan, dtype="float64")

    devices_seen: set[int] = set()
    recent: list[tuple[int, int]] = []
    count = 0
    mean = 0.0
    m2 = 0.0
    current = -1
    for start, stop in _blocks(stream.codes, stream.ts):
        code = int(stream.codes[start])
        if code != current:
            devices_seen, recent = set(), []
            count, mean, m2 = 0, 0.0, 0.0
            current = code
        moment = int(stream.ts[start])
        cutoff = moment - repeat_window
        recent = [entry for entry in recent if entry[0] >= cutoff]
        recent_amounts = {entry[1] for entry in recent}
        n_seen = len(devices_seen)
        seen_mean = mean if count > 0 else np.nan
        seen_std = np.sqrt(max(m2, 0.0) / (count - 1)) if count >= 2 else np.nan

        for i in range(start, stop):
            n_distinct[i] = n_seen
            if device[i] != _NULL_LEVEL and devices_seen:
                is_new[i] = 0.0 if int(device[i]) in devices_seen else 1.0
            if recent_amounts:
                repeat[i] = 1.0 if int(amount_level[i]) in recent_amounts else 0.0
            prior_mean[i] = seen_mean
            prior_std[i] = seen_std
            deviation[i] = amount[i] - seen_mean
            if seen_std > 0.0:
                zscore[i] = (amount[i] - seen_mean) / seen_std

        for i in range(start, stop):
            if device[i] != _NULL_LEVEL:
                devices_seen.add(int(device[i]))
            recent.append((moment, int(amount_level[i])))
            count += 1
            delta = amount[i] - mean
            mean += delta / count
            m2 += delta * (amount[i] - mean)

    return {
        "dev_combination_is_new": _scatter_float(stream, is_new),
        "dev_n_distinct_so_far": _scatter_int(stream, n_distinct),
        "amt_entity_mean": _scatter_float(stream, prior_mean),
        "amt_entity_std": _scatter_float(stream, prior_std),
        "amt_deviation_from_entity_mean": _scatter_float(stream, deviation),
        "amt_zscore_within_entity": _scatter_float(stream, zscore),
        "amt_repeat_within_24h": _scatter_float(stream, repeat),
    }


# --- configuration -------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """What to build and what to pass through. Every field is a switch stage 6 can move.

    `native_groups` is what item 5 of the stage brief asks for: the C, D, M and V blocks stay
    available so stage 6 measures their contribution against these features rather than assuming
    it either way. `include_d_origin_tenure` is off because ADR 0013 measured the D-column
    normalisation raising PSI on all 15 columns and lowering validation AUC on 12 of them, and
    D1's origin is `card_start_day` without the floor, the loudest PSI in the frame at 1.841.
    """

    families: tuple[str, ...] = FAMILIES
    velocity_grains: tuple[str, ...] = ("entity", "card")
    native_groups: tuple[str, ...] = NATIVE_GROUPS
    windows: tuple[tuple[str, int], ...] = WINDOWS
    amount_repeat_window_seconds: int = AMOUNT_REPEAT_WINDOW_SECONDS
    duplicate_window_seconds: int = DUPLICATE_WINDOW_SECONDS
    include_d_origin_tenure: bool = False

    def with_families(self, *families: str) -> FeatureConfig:
        return replace(self, families=tuple(families))

    def to_dict(self) -> dict[str, Any]:
        return {
            "families": list(self.families),
            "velocity_grains": list(self.velocity_grains),
            "native_groups": list(self.native_groups),
            "windows": [{"suffix": suffix, "seconds": window} for suffix, window in self.windows],
            "amount_repeat_window_seconds": self.amount_repeat_window_seconds,
            "duplicate_window_seconds": self.duplicate_window_seconds,
            "include_d_origin_tenure": self.include_d_origin_tenure,
        }


DEFAULT_CONFIG = FeatureConfig()

NATIVE_GROUP_COLUMNS: dict[str, tuple[str, ...]] = {
    "count": config.COUNT_COLUMNS,
    "timedelta": config.TIMEDELTA_COLUMNS,
    "match": config.MATCH_COLUMNS,
    "vesta": config.VESTA_COLUMNS,
}

# The V block does not survive stage 3 as V columns. ADR 0016 shipped the block-mean strategy, so
# the prepared frame carries `vblockmean_*` where the V columns were, and the switch has to cover
# the derived columns or it would switch nothing off.
_DERIVED_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {"vesta": ("vblockmean_", "vpca_")}


def native_group_columns(frame: pd.DataFrame, group: str) -> list[str]:
    """Columns of one native group present in the frame, raw and stage-3-derived alike."""
    if group not in NATIVE_GROUP_COLUMNS:
        raise KeyError(
            f"unknown native group {group!r}, expected one of {list(NATIVE_GROUP_COLUMNS)}"
        )
    named = set(NATIVE_GROUP_COLUMNS[group])
    prefixes = _DERIVED_GROUP_PREFIXES.get(group, ())
    return [
        str(column)
        for column in frame.columns
        if str(column) in named or (bool(prefixes) and str(column).startswith(prefixes))
    ]


def select_native_groups(frame: pd.DataFrame, cfg: FeatureConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Drop the native groups the config switches off. Nothing else is touched.

    A group that is on passes through untouched, which is the point: stage 6 compares a model with
    the native blocks against one without, and that comparison is like for like only if the single
    difference is which columns were present.
    """
    wanted = set(cfg.native_groups)
    drop: list[str] = []
    for group in NATIVE_GROUP_COLUMNS:
        if group not in wanted:
            drop.extend(native_group_columns(frame, group))
    return frame.drop(columns=sorted(set(drop)))


# --- the catalogue -------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """One feature's declared shape. `reports/feature_summary.json` is this plus measured nulls."""

    name: str
    family: str
    grain: str
    derived: bool
    window_seconds: int | None
    state_key: str
    state_fields: tuple[str, ...]
    definition: str
    null_meaning: str | None

    @property
    def requires_running_state(self) -> bool:
        return self.state_key != "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.name,
            "family": self.family,
            "grain": self.grain,
            "derived_in_stage_4": self.derived,
            "window_seconds": self.window_seconds,
            "requires_running_state": self.requires_running_state,
            "state_key": self.state_key,
            "state_fields": list(self.state_fields),
            "definition": self.definition,
            "null_meaning": self.null_meaning,
        }


_RECENCY_STATE: tuple[str, ...] = (
    "first prior timestamp",
    "last prior timestamp",
    "prior transaction count",
)
_AMOUNT_STATE: tuple[str, ...] = (
    "prior transaction count",
    "prior amount sum",
    "prior amount sum of squares",
)
_NO_HISTORY = "the stream holds no earlier transaction on this key to compare against"


def _velocity_specs(cfg: FeatureConfig) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for grain in cfg.velocity_grains:
        state = (
            f"the {grain}'s (timestamp, amount) pairs inside the widest window, as a bounded ring",
        )
        for suffix, window in cfg.windows:
            common = {
                "family": "velocity",
                "grain": grain,
                "derived": True,
                "window_seconds": window,
                "state_key": grain,
                "state_fields": state,
            }
            specs.append(
                FeatureSpec(
                    name=velocity_name("count", suffix, grain),
                    definition=(
                        f"transactions on the {grain} key in [t - {window}s, t), current row and "
                        "same-timestamp ties excluded"
                    ),
                    null_meaning=None,
                    **common,  # type: ignore[arg-type]
                )
            )
            specs.append(
                FeatureSpec(
                    name=velocity_name("amount_sum", suffix, grain),
                    definition="sum of TransactionAmt over the same window",
                    null_meaning=None,
                    **common,  # type: ignore[arg-type]
                )
            )
            specs.append(
                FeatureSpec(
                    name=velocity_name("amount_mean", suffix, grain),
                    definition="mean of TransactionAmt over the same window",
                    null_meaning="the window holds no transaction, so there is no mean",
                    **common,  # type: ignore[arg-type]
                )
            )
    return specs


def catalogue(cfg: FeatureConfig = DEFAULT_CONFIG) -> tuple[FeatureSpec, ...]:
    """Every feature the config builds, declared.

    The artifact reads this rather than a list of names typed out a second time, so a feature
    cannot be shipped without a family, a grain and a serving footprint.
    """
    specs: list[FeatureSpec] = []

    if "velocity" in cfg.families:
        specs.extend(_velocity_specs(cfg))

    if "recency" in cfg.families:
        specs.extend(
            [
                FeatureSpec(
                    name="rec_seconds_since_prev",
                    family="recency",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_RECENCY_STATE,
                    definition="t minus the timestamp of the entity's latest strictly earlier row",
                    null_meaning=_NO_HISTORY,
                ),
                FeatureSpec(
                    name="rec_prior_count",
                    family="recency",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_RECENCY_STATE,
                    definition="transactions on the entity strictly before t, over all history",
                    null_meaning=None,
                ),
                FeatureSpec(
                    name="rec_mean_gap",
                    family="recency",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_RECENCY_STATE,
                    definition=(
                        "mean gap between the entity's consecutive prior transactions, as "
                        "(last_prior - first_prior) / (n_prior - 1)"
                    ),
                    null_meaning="fewer than two earlier transactions, so no gap to average",
                ),
                FeatureSpec(
                    name="rec_gap_deviation",
                    family="recency",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_RECENCY_STATE,
                    definition="rec_seconds_since_prev minus rec_mean_gap",
                    null_meaning="either input is null",
                ),
            ]
        )

    if "tenure" in cfg.families:
        specs.append(
            FeatureSpec(
                name="D1",
                family="tenure",
                grain="row",
                derived=False,
                window_seconds=None,
                state_key="none",
                state_fields=(),
                definition=(
                    "days since the card began, native column, carried through stage 3 unchanged. "
                    "Listed because it is the tenure feature and stage 6 needs to know that; not "
                    "recomputed, because it already exists and there is one source per number"
                ),
                null_meaning="null on 0.0021 of rows (stage 0 audit, d1_profile)",
            )
        )
        if cfg.include_d_origin_tenure:
            specs.append(
                FeatureSpec(
                    name=transforms.d_origin_name("D1"),
                    family="tenure",
                    grain="row",
                    derived=True,
                    window_seconds=None,
                    state_key="none",
                    state_fields=(),
                    definition=(
                        "TransactionDT / 86400 - D1, the day the tenure delta counts from. Off by "
                        "default: ADR 0013 measured the normalisation raising PSI on all 15 D "
                        "columns, and this is card_start_day without the floor"
                    ),
                    null_meaning="D1 is null, so there is no origin to compute",
                )
            )

    if "geography" in cfg.families:
        specs.extend(
            [
                FeatureSpec(
                    name="geo_addr1_differs_from_card_mode",
                    family="geography",
                    grain="card",
                    derived=True,
                    window_seconds=None,
                    state_key="card",
                    state_fields=("a count per addr1 value the card has used",),
                    definition=(
                        "whether addr1 differs from the card's modal addr1 over strictly earlier "
                        "rows, ties broken on the smaller value"
                    ),
                    null_meaning="addr1 is null, or no earlier row of the card carries one",
                ),
                FeatureSpec(
                    name="geo_addr2_differs_from_card_mode",
                    family="geography",
                    grain="card",
                    derived=True,
                    window_seconds=None,
                    state_key="card",
                    state_fields=("a count per addr2 value the card has used",),
                    definition="the same for addr2",
                    null_meaning="addr2 is null, or no earlier row of the card carries one",
                ),
                FeatureSpec(
                    name="geo_n_distinct_addr1_on_card",
                    family="geography",
                    grain="card",
                    derived=True,
                    window_seconds=None,
                    state_key="card",
                    state_fields=("a count per addr1 value the card has used",),
                    definition="distinct addr1 values over the card's strictly earlier rows",
                    null_meaning=None,
                ),
                FeatureSpec(
                    name="geo_dist1_bucket",
                    family="geography",
                    grain="row",
                    derived=True,
                    window_seconds=None,
                    state_key="none",
                    state_fields=(),
                    definition=(
                        "index of the train-fitted quantile bucket dist1 falls in. No key state: "
                        "the edges are a fitted artifact read from the model bundle"
                    ),
                    null_meaning="dist1 is null; the distance block indicator carries that state",
                ),
            ]
        )

    if "device" in cfg.families:
        device_state = ("the set of device combinations the entity has used",)
        specs.extend(
            [
                FeatureSpec(
                    name="dev_combination_is_new",
                    family="device",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=device_state,
                    definition=(
                        "whether the normalised (DeviceInfo, DeviceType, id_30, id_31) combination "
                        "appears in no strictly earlier row of the entity"
                    ),
                    null_meaning=(
                        "the row carries no identity record, or the entity has no earlier row that "
                        "does. Stage 3's identity policy leaves that null and lets "
                        "has_identity_record carry the join state"
                    ),
                ),
                FeatureSpec(
                    name="dev_n_distinct_so_far",
                    family="device",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=device_state,
                    definition="distinct device combinations over the entity's strictly earlier rows",
                    null_meaning=None,
                ),
            ]
        )

    if "amount" in cfg.families:
        specs.extend(
            [
                FeatureSpec(
                    name="amt_entity_mean",
                    family="amount",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_AMOUNT_STATE,
                    definition="mean TransactionAmt over the entity's strictly earlier rows",
                    null_meaning=_NO_HISTORY,
                ),
                FeatureSpec(
                    name="amt_entity_std",
                    family="amount",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_AMOUNT_STATE,
                    definition="sample standard deviation of the same, clamped at zero",
                    null_meaning="fewer than two earlier transactions",
                ),
                FeatureSpec(
                    name="amt_deviation_from_entity_mean",
                    family="amount",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_AMOUNT_STATE,
                    definition="TransactionAmt minus amt_entity_mean",
                    null_meaning=_NO_HISTORY,
                ),
                FeatureSpec(
                    name="amt_zscore_within_entity",
                    family="amount",
                    grain="entity",
                    derived=True,
                    window_seconds=None,
                    state_key="entity",
                    state_fields=_AMOUNT_STATE,
                    definition="the same deviation divided by amt_entity_std",
                    null_meaning="fewer than two earlier transactions, or a zero running spread",
                ),
                FeatureSpec(
                    name="amt_repeat_within_24h",
                    family="amount",
                    grain="entity",
                    derived=True,
                    window_seconds=AMOUNT_REPEAT_WINDOW_SECONDS,
                    state_key="entity",
                    state_fields=(
                        f"the entity's amounts inside the last {AMOUNT_REPEAT_WINDOW_SECONDS} "
                        "seconds",
                    ),
                    definition=(
                        "whether TransactionAmt equals the amount of a strictly earlier "
                        f"transaction of the entity inside {AMOUNT_REPEAT_WINDOW_SECONDS} seconds"
                    ),
                    null_meaning="the window holds no earlier transaction of the entity",
                ),
            ]
        )

    if "duplicate" in cfg.families:
        content_state = (
            "per content key: the prior count, the last prior timestamp, and the timestamps inside "
            f"the last {cfg.duplicate_window_seconds} seconds",
        )
        specs.extend(
            [
                FeatureSpec(
                    name="dup_prior_count",
                    family="duplicate",
                    grain="content",
                    derived=True,
                    window_seconds=None,
                    state_key="content",
                    state_fields=content_state,
                    definition=(
                        f"strictly earlier transactions agreeing on all of {list(CONTENT_COLUMNS)}, "
                        "nulls compared as a level"
                    ),
                    null_meaning=None,
                ),
                FeatureSpec(
                    name="dup_seconds_since_prev",
                    family="duplicate",
                    grain="content",
                    derived=True,
                    window_seconds=None,
                    state_key="content",
                    state_fields=content_state,
                    definition="t minus the timestamp of the latest strictly earlier identical row",
                    null_meaning="no earlier transaction shares this content",
                ),
                FeatureSpec(
                    name="dup_prior_count_1h",
                    family="duplicate",
                    grain="content",
                    derived=True,
                    window_seconds=cfg.duplicate_window_seconds,
                    state_key="content",
                    state_fields=content_state,
                    definition=(
                        "the same count restricted to "
                        f"[t - {cfg.duplicate_window_seconds}s, t). The stage's 'whether any fell "
                        "inside a short window' is this column being positive"
                    ),
                    null_meaning=None,
                ),
            ]
        )

    return tuple(specs)


def feature_names(cfg: FeatureConfig = DEFAULT_CONFIG) -> list[str]:
    """Names of the columns `build_features` returns, in catalogue order."""
    return [spec.name for spec in catalogue(cfg) if spec.derived]


# --- the assembly ---------------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = (
    config.TIME_COLUMN,
    config.AMOUNT_COLUMN,
    config.ENTITY_ID_COLUMN,
)


def _levels_or_null(frame: pd.DataFrame, column: str) -> npt.NDArray[np.int64]:
    if column not in frame.columns:
        return np.full(len(frame), _NULL_LEVEL, dtype="int64")
    return _sorted_level_codes(frame[column])


def build_features(
    frame: pd.DataFrame,
    dist_buckets: Mapping[str, Any] | None = None,
    cfg: FeatureConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Every feature the config asks for, as a frame indexed like `frame`.

    There is no fit here and no train-only guard, and both are deliberate. Nothing in this
    function learns a statistic from the frame it is given: every value is read from the rows of
    that frame which fall strictly before the row being computed, which is exactly the information
    the serving path has. Passing the whole stream is therefore correct rather than merely
    permitted. A validation row reads earlier training rows, as it would in production, and a
    training row cannot read a validation row because validation rows are later in time.

    The one fitted object in the stage is the dist1 bucket edges, and it arrives as an argument so
    that it has to have been fitted somewhere the guard could see it.

    `tests/test_causality.py` is what makes the claim checkable. It recomputes each feature by
    brute force over all rows with TransactionDT < t, and it asserts that truncating the stream at
    a time T leaves every feature on the rows before T unchanged, which is the property any global
    statistic would break.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"build_features needs {missing}")
    unknown = [family for family in cfg.families if family not in FAMILIES]
    if unknown:
        raise ValueError(
            f"unknown feature families {unknown}, expected a subset of {list(FAMILIES)}"
        )
    unknown_grains = [grain for grain in cfg.velocity_grains if grain not in GRAIN_CODERS]
    if unknown_grains:
        raise ValueError(f"unknown velocity grains {unknown_grains}, expected from {list(GRAINS)}")

    amount = frame[config.AMOUNT_COLUMN].to_numpy(dtype="float64")
    columns: dict[str, npt.NDArray[Any]] = {}

    if "velocity" in cfg.families:
        for grain in cfg.velocity_grains:
            stream = make_stream(frame, GRAIN_CODERS[grain](frame))
            columns.update(_velocity(stream, stream.sorted_like(amount), grain, cfg.windows))

    needs_entity_stream = bool({"recency", "amount", "device"} & set(cfg.families))
    if needs_entity_stream:
        entity = make_stream(frame, entity_codes(frame))
        if "recency" in cfg.families:
            columns.update(_recency(entity))
        if {"device", "amount"} & set(cfg.families):
            scanned = _entity_scan(
                entity,
                entity.sorted_like(device_codes(frame)),
                entity.sorted_like(amount),
                entity.sorted_like(_levels_or_null(frame, config.AMOUNT_COLUMN)),
                repeat_window=cfg.amount_repeat_window_seconds,
            )
            for spec in catalogue(cfg):
                if spec.name in scanned and spec.family in cfg.families:
                    columns[spec.name] = scanned[spec.name]

    if "geography" in cfg.families:
        card = make_stream(frame, card_codes(frame))
        columns.update(
            _geography(
                card,
                card.sorted_like(_levels_or_null(frame, "addr1")),
                card.sorted_like(_levels_or_null(frame, "addr2")),
            )
        )
        if dist_buckets is None:
            raise ValueError(
                "the geography family needs the dist1 bucket edges from fit_dist_buckets; pass "
                "them rather than letting this frame fit its own"
            )
        columns["geo_dist1_bucket"] = apply_dist_buckets(frame, dist_buckets)

    if "tenure" in cfg.families and cfg.include_d_origin_tenure:
        origin = transforms.d_origin_name("D1")
        columns[origin] = transforms.to_d_origin(frame, ["D1"])[origin].to_numpy(dtype="float64")

    if "duplicate" in cfg.families:
        content = make_stream(frame, content_codes(frame))
        columns.update(_duplicate(content, cfg.duplicate_window_seconds))

    ordered = [name for name in feature_names(cfg) if name in columns]
    return pd.DataFrame({name: columns[name] for name in ordered}, index=frame.index)


def add_features(
    frame: pd.DataFrame,
    dist_buckets: Mapping[str, Any] | None = None,
    cfg: FeatureConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """`build_features` concatenated onto the frame, with the native groups the config keeps."""
    built = build_features(frame, dist_buckets, cfg)
    return pd.concat([select_native_groups(frame, cfg), built], axis=1)


# --- reporting -------------------------------------------------------------------------------


def null_rates(frame: pd.DataFrame, names: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Null count and rate per named column.

    A column the frame does not carry is reported absent rather than as zero, because a zero would
    read as "measured, none found".
    """
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        if name not in frame.columns:
            out[name] = {
                "present": False,
                "n_rows": len(frame),
                "n_null": None,
                "null_rate": None,
            }
            continue
        n_null = int(frame[name].isna().sum())
        out[name] = {
            "present": True,
            "n_rows": len(frame),
            "n_null": n_null,
            "null_rate": n_null / len(frame) if len(frame) else None,
        }
    return out


def _distinct_per_group(
    codes: npt.NDArray[np.int64], values: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Distinct non-null values per group code, as one count per group that has any."""
    mask = values != _NULL_LEVEL
    if not bool(mask.any()):
        return np.zeros(0, dtype="int64")
    pairs = np.unique(np.stack([codes[mask], values[mask]], axis=1), axis=0)
    _, counts = np.unique(pairs[:, 0], return_counts=True)
    return counts.astype("int64")


def _spread(counts: npt.NDArray[np.int64]) -> dict[str, Any]:
    if counts.size == 0:
        return {"n_groups": 0, "mean": None, "p95": None, "p99": None, "max": None}
    return {
        "n_groups": int(counts.size),
        "mean": float(counts.mean()),
        "p95": float(np.quantile(counts, 0.95)),
        "p99": float(np.quantile(counts, 0.99)),
        "max": int(counts.max()),
    }


def state_footprint(frame: pd.DataFrame, cfg: FeatureConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """How much per-key state a serving path has to hold, measured rather than estimated.

    Stage 8 reads this. Two of the structures look unbounded when written down, the counter over a
    card's addr values and the set of an entity's device combinations, and these are the numbers
    that bound them on this data. The velocity ring is bounded by the widest window instead, not by
    a key's lifetime count, and the content keyspace is a second store rather than a wider row in
    the first.
    """
    out: dict[str, Any] = {"n_rows": len(frame)}
    for grain in ("entity", "card"):
        codes = GRAIN_CODERS[grain](frame)
        _, per_key = np.unique(codes, return_counts=True)
        out[grain] = {
            "n_keys": int(per_key.size),
            "transactions_per_key": _spread(per_key.astype("int64")),
        }
    card = card_codes(frame)
    entity = entity_codes(frame)
    for label in ("addr1", "addr2"):
        if label not in frame.columns:
            continue
        out["card"][f"distinct_{label}_per_key"] = _spread(
            _distinct_per_group(card, _sorted_level_codes(frame[label]))
        )
    out["entity"]["distinct_devices_per_key"] = _spread(
        _distinct_per_group(entity, device_codes(frame))
    )
    content = content_codes(frame)
    _, per_content = np.unique(content, return_counts=True)
    out["content"] = {
        "n_keys": int(per_content.size),
        "window_seconds": cfg.duplicate_window_seconds,
        "transactions_per_key": _spread(per_content.astype("int64")),
    }
    out["note"] = (
        "the velocity ring holds a key's (timestamp, amount) pairs inside the widest window and "
        "nothing older, so it is bounded by activity inside that window rather than by lifetime "
        "count. The addr counter and the device set are bounded by the distinct-value counts "
        "above. The duplicate family reads a second keyspace, keyed by content."
    )
    return out
