"""Row-level transformations: changes to a raw column that need no history to compute.

The line between this module and stage 4 is whether the value of one row depends on any other
row. `TransactionAmt` becomes `log1p(TransactionAmt)` by looking at that row and nothing else,
so it belongs here. "Transactions on this card in the last hour" needs the rest of the frame,
so it does not. Hour-of-day is on this side of the line for the same reason: it is arithmetic on
`TransactionDT`, not an aggregate over anything.

Four things live here.

**D-column normalisation.** The D columns are time deltas measured backwards from the
transaction, so a card observed on day 30 and again on day 110 carries a D that has grown by 80
in between. Subtracting the delta from the day index gives the origin the delta is measured
from, `D_n = TransactionDT / 86400 - D`, which does not move with the calendar. That is the
argument. Whether it works is a per-column measurement, `d_origin_psi` computes it, and ADR 0013
records that on some D columns the answer is no.

**Amount.** log1p and the cents fraction, both on the numbers stage 2 measured.

**Clipping.** A train-fitted quantile bound for the models that cannot see a tail, and nothing
at all for the trees, which can.

**Clock columns.** Day index, week index and position within the day, all relative, because
stage 0 did not establish what TransactionDT is measured from.

Everything with a fitted parameter goes through `data_loader.train_only`, so a bound fitted on
a frame that reaches past the training boundary raises instead of quietly working.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config
from fraud_platform.data_loader import train_only

# --- choices ------------------------------------------------------------------------------

# The suffix a normalised D column takes. `D4_origin` is the day the delta in D4 counts from.
D_ORIGIN_SUFFIX = "_origin"

# Clipping quantile for the models that need bounded inputs. 0.999 rather than 0.99, and the
# reason is measured: stage 2 found TransactionAmt U-shaped against the label, with the top
# decile at a fraud rate of 0.0509 against a trough of 0.0192, so a bound at the 99th percentile
# would flatten a bin the analysis found informative. `clip_cost` reports how many rows and how
# many fraud rows a given quantile actually touches, so the choice is checkable rather than
# asserted.
CLIP_QUANTILE = 0.999

# Names of the derived amount columns.
AMOUNT_LOG_COLUMN = "TransactionAmt_log1p"
AMOUNT_CENTS_COLUMN = "TransactionAmt_cents"
AMOUNT_IS_ROUND_COLUMN = "TransactionAmt_is_round"

# Names of the derived clock columns. All four are positions in a relative cycle, not wall-clock
# values: stage 0 did not establish what TransactionDT counts from.
DAY_INDEX_COLUMN = "day_index"
WEEK_INDEX_COLUMN = "week_index"
HOUR_POSITION_COLUMN = "hour_position"
DOW_POSITION_COLUMN = "dow_position"

CLOCK_COLUMNS: tuple[str, ...] = (
    DAY_INDEX_COLUMN,
    WEEK_INDEX_COLUMN,
    HOUR_POSITION_COLUMN,
    DOW_POSITION_COLUMN,
)

# Stage 2 measured the fraud rate by hour position running 0.0217 to 0.1001, a ratio of 4.62, and
# by day-of-week position 0.0322 to 0.0379, a ratio of 1.18. The second is derived because the
# lag machinery and the reports want it, and excluded from the feature set on that number.
CLOCK_FEATURE_COLUMNS: tuple[str, ...] = (HOUR_POSITION_COLUMN,)


def day_index(frame: pd.DataFrame) -> pd.Series:
    """Day number since the TransactionDT origin, as an integer. Relative, not a date."""
    days = np.floor(frame[config.TIME_COLUMN].to_numpy(dtype="float64") / config.SECONDS_PER_DAY)
    return pd.Series(days.astype("int32"), index=frame.index)


def add_clock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach day index, week index, hour position and day-of-week position.

    Idempotent: the four columns are a function of TransactionDT alone, so applying this twice
    writes the same values twice. A test asserts that rather than trusting it.
    """
    out = frame.copy()
    day = day_index(frame)
    out[DAY_INDEX_COLUMN] = day
    out[WEEK_INDEX_COLUMN] = (day // 7).astype("int32")
    out[HOUR_POSITION_COLUMN] = (
        (frame[config.TIME_COLUMN] % config.SECONDS_PER_DAY) // 3600
    ).astype("int8")
    out[DOW_POSITION_COLUMN] = (day % 7).astype("int8")
    return out


# --- D columns ------------------------------------------------------------------------------


def d_origin_name(column: str) -> str:
    return f"{column}{D_ORIGIN_SUFFIX}"


def to_d_origin(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Add `<D>_origin = TransactionDT / 86400 - <D>` for each named column.

    Nulls pass through as nulls. A D column that is missing has no origin to compute, and
    inventing one would put a real number where the frame says there is no event to measure
    from. The missing-value policy in `fraud_platform.missing_policy` is what handles that.

    The transformation is invertible given TransactionDT, and a test checks the round trip
    rather than taking the arithmetic on trust.
    """
    out = frame.copy()
    days = frame[config.TIME_COLUMN] / config.SECONDS_PER_DAY
    for column in columns:
        if column not in frame.columns:
            continue
        out[d_origin_name(column)] = (days - frame[column]).astype("float64")
    return out


def from_d_origin(frame: pd.DataFrame, column: str) -> pd.Series:
    """Recover the original delta from its origin column. The inverse of `to_d_origin`."""
    days = frame[config.TIME_COLUMN] / config.SECONDS_PER_DAY
    return (days - frame[d_origin_name(column)]).astype("float64")


def d_origin_psi(
    train: pd.DataFrame,
    later: pd.DataFrame,
    columns: Sequence[str],
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """PSI of each D column against a later split, before and after normalisation.

    Both numbers are binned on the train distribution of the column being measured, which is the
    same rule stage 2 used, so the before number here reproduces the one in temporal.json and a
    test checks that it does.

    The delta is the point. A D column whose PSI falls has been converted from something that
    moves with the calendar into something that does not. A D column whose PSI rises has had the
    calendar moved into it, which is what happens when the origin itself marches forward, and
    stage 2 already measured the extreme case: card_start_day, the origin of D1, has the highest
    PSI in the frame at 1.841.
    """
    from fraud_platform import eda

    train_with = to_d_origin(train, columns)
    later_with = to_d_origin(later, columns)
    rows: list[dict[str, Any]] = []
    for column in columns:
        if column not in train.columns:
            continue
        origin = d_origin_name(column)
        before = eda.population_stability_index(train[column], later[column], n_bins)
        after = eda.population_stability_index(train_with[origin], later_with[origin], n_bins)
        rows.append(
            {
                "column": column,
                "origin_column": origin,
                "psi_before": before["psi"],
                "psi_after": after["psi"],
                "psi_delta": after["psi"] - before["psi"],
                "improves": after["psi"] < before["psi"],
                "n_bins_before": before["n_levels"],
                "n_bins_after": after["n_levels"],
                "missing_rate_train": float(train[column].isna().mean()),
            }
        )
    return rows


# --- amount ---------------------------------------------------------------------------------


def add_amount_features(frame: pd.DataFrame) -> pd.DataFrame:
    """log1p of the amount, the cents fraction, and whether the amount is a whole unit.

    Stage 2 measured the raw skew at 17.639 and the log1p skew at 0.499, and the raw excess
    kurtosis at 1,603.6 against 0.765. The raw column stays: a monetary threshold is expressed
    in currency, not in log currency, and stage 7 will need it in the units a reviewer reads.

    The cents definition is stage 2's, `round(amount * 100) mod 100`, so the distribution this
    produces is the one univariate.json already reports: 0.5255 of rows at 0, 0.2712 at 95.
    """
    out = frame.copy()
    amount = frame[config.AMOUNT_COLUMN].astype("float64")
    out[AMOUNT_LOG_COLUMN] = np.log1p(amount)
    cents = (amount * 100).round().astype("int64") % 100
    out[AMOUNT_CENTS_COLUMN] = cents.astype("int16")
    out[AMOUNT_IS_ROUND_COLUMN] = (cents == 0).astype("int8")
    return out


def invert_amount_log(values: pd.Series) -> pd.Series:
    """expm1 of the log column, which returns the amount. Checked by a test, not assumed."""
    return pd.Series(
        np.expm1(values.to_numpy(dtype="float64")), index=values.index, dtype="float64"
    )


# --- clipping --------------------------------------------------------------------------------


@train_only
def fit_clip_bounds(
    frame: pd.DataFrame, columns: Sequence[str], quantile: float = CLIP_QUANTILE
) -> dict[str, Any]:
    """Lower and upper bounds per column, at symmetric quantiles of the train distribution.

    Fitted on train and applied unchanged to every later window, which is the only arrangement
    that means anything: a bound refitted on the window being scored moves with that window and
    clips nothing that matters.

    Trees do not get this. A gradient boosted tree cuts on rank, so the largest value in a column
    is the largest value whether it is 31,937 or a clipped 5,000, and clipping costs information
    without buying stability. It exists for the models in stage 5 that cannot see a tail: a
    linear model, a distance, anything that reads the value rather than its position.
    """
    lower_q = 1.0 - quantile
    bounds: dict[str, Any] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].dropna().astype("float64")
        if values.empty:
            continue
        bounds[column] = {
            "lower": float(values.quantile(lower_q)),
            "upper": float(values.quantile(quantile)),
        }
    return {
        "quantile": quantile,
        "lower_quantile": lower_q,
        "n_columns": len(bounds),
        "n_train_rows": len(frame),
        "bounds": bounds,
    }


def apply_clip(frame: pd.DataFrame, fitted: dict[str, Any]) -> pd.DataFrame:
    """Clip every column the bounds cover. Idempotent: clipping a clipped frame changes nothing."""
    out = frame.copy()
    for column, bound in fitted["bounds"].items():
        if column not in out.columns:
            continue
        out[column] = out[column].clip(lower=bound["lower"], upper=bound["upper"])
    return out


@train_only
def clip_cost(
    frame: pd.DataFrame, columns: Sequence[str], quantiles: Sequence[float], target: str
) -> list[dict[str, Any]]:
    """How many rows and how many fraud rows a clip at each quantile would touch.

    This is what makes CLIP_QUANTILE a decision rather than a habit. A quantile that moves 1
    percent of rows is cheap if those rows are ordinary and expensive if the analysis found a
    fraud rate in them, and the only way to know which is to count.
    """
    labels = frame[target].to_numpy(dtype="int64")
    n_fraud = int(labels.sum())
    rows: list[dict[str, Any]] = []
    for quantile in quantiles:
        n_touched = 0
        n_fraud_touched = 0
        per_column: list[dict[str, Any]] = []
        for column in columns:
            if column not in frame.columns:
                continue
            values = frame[column].astype("float64")
            clean = values.dropna()
            if clean.empty:
                continue
            low = float(clean.quantile(1.0 - quantile))
            high = float(clean.quantile(quantile))
            touched = ((values < low) | (values > high)).fillna(False).to_numpy()
            n_touched += int(touched.sum())
            n_fraud_touched += int(labels[touched].sum())
            per_column.append(
                {
                    "column": column,
                    "lower": low,
                    "upper": high,
                    "n_touched": int(touched.sum()),
                    "n_fraud_touched": int(labels[touched].sum()),
                    "fraud_rate_touched": (
                        float(labels[touched].mean()) if touched.sum() else None
                    ),
                }
            )
        rows.append(
            {
                "quantile": quantile,
                "n_values_touched": n_touched,
                "n_fraud_values_touched": n_fraud_touched,
                "share_of_fraud_rows_touched": n_fraud_touched / n_fraud if n_fraud else None,
                "per_column": per_column,
            }
        )
    return rows
