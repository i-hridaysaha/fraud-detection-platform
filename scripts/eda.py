"""Stage 2 exploratory analysis. Writes one JSON artifact per section under reports/eda/.

Usage:
    make eda
    .venv/bin/python scripts/eda.py --out-dir reports/eda
    .venv/bin/python scripts/eda.py --sections correlation temporal

Two rules govern the whole stage and both are enforced here rather than left to care.

Everything is computed on the train split. The frame is cut with
`fraud_platform.data_loader.time_based_split` and the train part goes through
`assert_train_only` before any section runs, so a section that quietly reached for the whole
frame would have to defeat the guard to do it. The three sections that are about the difference
between splits say so in their own metadata: the PSI block in temporal.json and the split
comparison in target.json. See ADR 0007.

Every number a later document quotes lands in one of these artifacts. docs/eda.md reads them and
retypes nothing, and every artifact carries the command that regenerates it.

Sections and their outputs:

    target        reports/eda/target.json       rate overall, by period, by category, balance
    univariate    reports/eda/univariate.json   per column distribution, the amount in detail
    missingness   reports/eda/missingness.json  rates, column blocks, missingness against label
    correlation   reports/eda/correlation.json  Spearman, V-block reduction, duplicate pairs
    bivariate     reports/eda/bivariate.json    information value, fraud rate by decile
    temporal      reports/eda/temporal.json     PSI, the time consistency screen, cold entities
    entities      reports/eda/entities.json     size, purity and gap distributions per entity
    quality       reports/eda/quality.json      duplicates, impossible values, emails, devices
    train_only    reports/eda/train_only.json   what an EDA over the whole file would have hidden
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, data_loader, eda

SECTIONS = (
    "target",
    "univariate",
    "missingness",
    "correlation",
    "bivariate",
    "temporal",
    "entities",
    "quality",
    "train_only",
)

# Resampling budget for the rate intervals. Large enough that the interval endpoints are stable
# to the fourth decimal across seeds, small enough to run in a second. A choice, not a
# measurement.
N_BOOTSTRAP = 20_000

# Assumed AUC values for the minimum detectable effect table. Nothing in this repo has measured
# an AUC yet; these are the range a fraud model of this kind is being aimed at, and the table
# says what each of them would allow. Labelled assumed in docs/eda.md.
ASSUMED_AUC = (0.90, 0.93, 0.95)
ASSUMED_RECALL = (0.50, 0.70)

# The screen threshold for the correlation pass. Every pair above it is recomputed exactly.
# Set at 0.80 against a measured worst case: on a random sample of pairs the screen and
# pandas differ by at most the number reported in correlation.json, well inside the 0.15
# margin this leaves below the 0.95 redundancy threshold.
SCREEN_RHO = 0.80

# The fields that make two rows the same purchase for the duplicate check that ignores the
# running counters. A choice about what "the same transaction" means, not a measurement.
CORE_DUPLICATE_COLUMNS: tuple[str, ...] = (
    "card1",
    "card2",
    "addr1",
    "TransactionAmt",
    "ProductCD",
    "P_emaildomain",
)

# A column binned into more than this many levels gets a flag: at that width information
# value rewards granularity as much as separation.
HIGH_CARDINALITY_BINS = 50

# Columns given a decile table because a reader can reason about them, whatever their rank.
REFERENCE_DECILE_COLUMNS: tuple[str, ...] = (
    "TransactionAmt",
    "C1",
    "C13",
    "C14",
    "D2",
    "D15",
    "dist1",
    "card_start_day",
)

# How many features get a decile table and a figure. The rest carry their information value only.
TOP_FEATURES_FOR_DECILES = 12

# The time consistency screen splits the train span into its first and last 30 day blocks.
CONSISTENCY_WINDOW_DAYS = 30
# A feature is flagged when the early window separates the classes by at least this much above
# 0.5 and the later window falls below 0.5. Both halves of the rule are choices.
CONSISTENCY_EARLY_MIN = 0.51
CONSISTENCY_LATE_MAX = 0.50


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def envelope(section: str, command: str, n_train_rows: int) -> dict[str, Any]:
    """The header every artifact carries: what made it, from what, and how to make it again."""
    return {
        "section": section,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": f"make eda  # or: .venv/bin/python scripts/eda.py --sections {section}",
        "command": command,
        "computed_on": "train split only, dt < config.TRAIN_END_DT, guarded by assert_train_only",
        "n_train_rows": n_train_rows,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
    }


def add_clock_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Day index, week index, hour position and day-of-week position, all relative.

    TransactionDT is an offset from an origin this repo has not established (stage 0), so none of
    these is a wall-clock value. `hour_position` is the position inside an 86,400 unit cycle and
    `dow_position` the position inside a seven day cycle. Stage 0 measured that the 86,400 cycle
    carries the diurnal shape of card activity, which is why the hour column is worth cutting on;
    nothing has established which real weekday position 0 is.
    """
    out = frame.copy()
    day = np.floor(out[config.TIME_COLUMN] / config.SECONDS_PER_DAY).astype("int32")
    out["day_index"] = day
    out["week_index"] = ((day - int(day.min())) // 7).astype("int32")
    out["hour_position"] = ((out[config.TIME_COLUMN] % config.SECONDS_PER_DAY) // 3600).astype(
        "int16"
    )
    out["dow_position"] = (day % 7).astype("int8")
    return out


def period_table(frame: pd.DataFrame, period: str) -> list[dict[str, Any]]:
    """Rows, fraud count and fraud rate for each value of a period column, in period order."""
    grouped = frame.groupby(period, observed=True)[config.TARGET].agg(["size", "sum"]).sort_index()
    return [
        {
            "period": int(value),
            "n": int(row["size"]),
            "n_fraud": int(row["sum"]),
            "fraud_rate": float(row["sum"]) / float(row["size"]),
        }
        for value, row in grouped.iterrows()
    ]


# --- section builders --------------------------------------------------------------------


def build_target(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, split_summary: dict[str, Any] | None
) -> dict[str, Any]:
    """Fraud rate: overall, per split, per period, per category, and what the balance allows.

    Per-split rates are the one place this section reads rows outside the training window, and
    they exist here because the intervals do not exist anywhere else. The point estimates are
    checked against reports/split_summary.json rather than being a second opinion on it.
    """
    report = envelope(
        "target",
        "train, val, test = data_loader.time_based_split(data_loader.load_raw()); "
        "rates and intervals on each",
        len(train),
    )
    keyed = add_clock_columns(train)
    labels = train[config.TARGET].to_numpy(dtype="int64")

    per_split: dict[str, Any] = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        y = part[config.TARGET].to_numpy(dtype="int64")
        per_split[name] = {
            "n_rows": len(part),
            "n_fraud": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "wilson_95": eda.wilson_interval(int(y.sum()), len(part)),
            "bootstrap_95": eda.bootstrap_rate_interval(y, N_BOOTSTRAP, config.SEED),
        }

    if split_summary is not None:
        per_split["reproduces_split_summary"] = {
            name: bool(
                abs(per_split[name]["fraud_rate"] - split_summary["splits"][name]["fraud_rate"])
                < 1e-12
            )
            for name in config.SPLIT_NAMES
        }
        per_split["split_summary_note"] = (
            "point estimates come from the same rows as reports/split_summary.json and are "
            "checked against it; the intervals are new here"
        )

    by_day = period_table(keyed, "day_index")
    by_week = period_table(keyed, "week_index")
    # The train window is 120 days, so the last week holds one day. A partial period is a
    # smaller sample at the end of the series, exactly where a trend test is most sensitive, so
    # the trend is reported twice: over every week and over the complete weeks only.
    days_per_week = keyed.groupby("week_index", observed=True)["day_index"].nunique()
    complete_weeks = {int(w) for w, n in days_per_week.items() if int(n) == 7}
    by_week_complete = [row for row in by_week if row["period"] in complete_weeks]

    report.update(
        {
            "overall_train": {
                "n_rows": len(train),
                "n_fraud": int(labels.sum()),
                "fraud_rate": float(labels.mean()),
                "wilson_95": eda.wilson_interval(int(labels.sum()), len(train)),
                "bootstrap_95": eda.bootstrap_rate_interval(labels, N_BOOTSTRAP, config.SEED),
            },
            "per_split": per_split,
            "by_day": by_day,
            "by_week": by_week,
            "days_per_week": {str(int(w)): int(n) for w, n in days_per_week.items()},
            "stationarity": {
                "note": (
                    "three tests on the train span. Kendall tau is Mann-Kendall and asks for "
                    "monotone trend; least squares gives the size of a linear one; chi-square "
                    "homogeneity asks whether the periods share a rate at all."
                ),
                "weekly": eda.trend_tests(
                    [row["period"] for row in by_week],
                    [row["fraud_rate"] for row in by_week],
                    [row["n"] for row in by_week],
                ),
                "weekly_complete_weeks_only": {
                    "n_weeks_dropped": len(by_week) - len(by_week_complete),
                    "weeks_kept": sorted(complete_weeks),
                    **eda.trend_tests(
                        [row["period"] for row in by_week_complete],
                        [row["fraud_rate"] for row in by_week_complete],
                        [row["n"] for row in by_week_complete],
                    ),
                },
                "daily": eda.trend_tests(
                    [row["period"] for row in by_day],
                    [row["fraud_rate"] for row in by_day],
                    [row["n"] for row in by_day],
                ),
            },
            "by_category": {
                column: eda.rate_by_group(keyed, column, config.TARGET)
                for column in ("ProductCD", "card4", "card6", "DeviceType")
                if column in keyed.columns
            },
            "by_clock": {
                column: eda.rate_by_group(keyed, column, config.TARGET)
                for column in ("hour_position", "dow_position")
            },
            "class_balance": {
                "note": (
                    "what the measured positive counts allow. The AUC and recall values are "
                    "assumed inputs, not measurements: nothing in this repo has fitted a model "
                    "yet. See ADR 0010."
                ),
                "positives_per_split": {
                    name: per_split[name]["n_fraud"] for name in config.SPLIT_NAMES
                },
                "negatives_per_split": {
                    name: per_split[name]["n_rows"] - per_split[name]["n_fraud"]
                    for name in config.SPLIT_NAMES
                },
                "min_detectable_auc_difference": {
                    name: [
                        eda.detectable_auc_difference(
                            auc,
                            per_split[name]["n_fraud"],
                            per_split[name]["n_rows"] - per_split[name]["n_fraud"],
                        )
                        for auc in ASSUMED_AUC
                    ]
                    for name in ("val", "test")
                },
                "recall_precision": {
                    name: [
                        eda.recall_precision_of_positives(per_split[name]["n_fraud"], recall)
                        for recall in ASSUMED_RECALL
                    ]
                    for name in ("val", "test")
                },
            },
        }
    )
    return report


def skew_summary(numeric: list[dict[str, Any]]) -> dict[str, Any]:
    """How skewed the numeric feature columns are, in one block. Evidence for ADR 0008."""
    values = sorted(
        abs(float(p["skew"]))
        for p in numeric
        if p["role"] == "feature" and p.get("skew") is not None
    )
    if not values:
        return {"n_columns": 0}
    array = np.array(values)
    worst = max(
        (p for p in numeric if p["role"] == "feature" and p.get("skew") is not None),
        key=lambda p: abs(float(p["skew"])),
    )
    return {
        "n_columns": len(values),
        "median_abs_skew": float(np.median(array)),
        "p90_abs_skew": float(np.quantile(array, 0.90)),
        "max_abs_skew": float(array.max()),
        "n_abs_skew_above_0_5": int((array > 0.5).sum()),
        "n_abs_skew_below_0_5": int((array <= 0.5).sum()),
        "n_abs_skew_above_1": int((array > 1).sum()),
        "n_abs_skew_above_5": int((array > 5).sum()),
        "n_abs_skew_above_10": int((array > 10).sum()),
        "n_abs_skew_above_100": int((array > 100).sum()),
        "most_skewed_column": worst["column"],
        "most_skewed_skew": float(worst["skew"]),
        "most_skewed_kurtosis": float(worst["kurtosis"]),
    }


def build_univariate(train: pd.DataFrame) -> dict[str, Any]:
    """Per column distribution for every column, and TransactionAmt in detail."""
    report = envelope(
        "univariate",
        "eda.numeric_profile / eda.categorical_profile over every column of the train split",
        len(train),
    )
    numeric: list[dict[str, Any]] = []
    categorical: list[dict[str, Any]] = []
    for column in train.columns:
        if column == config.ENTITY_ID_COLUMN:
            continue
        series = train[column]
        role = "not a feature" if column in eda.NON_FEATURE_COLUMNS else "feature"
        if isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object:
            profile = eda.categorical_profile(series, column)
            profile["role"] = role
            categorical.append(profile)
        else:
            profile = eda.numeric_profile(series, column)
            profile["role"] = role
            numeric.append(profile)

    amount = train[config.AMOUNT_COLUMN]
    cents = (amount * 100).round().astype("int64") % 100
    is_round = cents == 0
    labels = train[config.TARGET].to_numpy(dtype="int64")
    cent_counts = cents.value_counts().sort_values(ascending=False)

    report.update(
        {
            "n_columns_profiled": len(numeric) + len(categorical),
            "n_numeric": len(numeric),
            "n_categorical": len(categorical),
            "thresholds": {
                "effectively_constant_share": eda.CONSTANT_SHARE,
                "single_value_dominated_share": eda.DOMINATED_SHARE,
                "zero_inflated_share": eda.ZERO_INFLATED_SHARE,
                "rare_tail_share": eda.RARE_TAIL_SHARE,
            },
            "flags": {
                "effectively_constant": sorted(
                    p["column"] for p in numeric + categorical if p.get("effectively_constant")
                ),
                "single_value_dominated": sorted(
                    p["column"]
                    for p in numeric + categorical
                    if p.get("single_value_dominated") and not p.get("effectively_constant")
                ),
                "zero_inflated": sorted(p["column"] for p in numeric if p.get("zero_inflated")),
                "all_null": sorted(p["column"] for p in numeric + categorical if p.get("all_null")),
            },
            # ADR 0008 rests on these counts, so they are computed here rather than being read
            # off the per-column table by whoever quotes them.
            "skew_summary": skew_summary(numeric),
            "numeric": numeric,
            "categorical": categorical,
            "amount": {
                "column": config.AMOUNT_COLUMN,
                "skew_raw": float(amount.skew()),
                "skew_log1p": float(np.log1p(amount).skew()),
                "kurtosis_raw": float(amount.kurtosis()),
                "kurtosis_log1p": float(np.log1p(amount).kurtosis()),
                "histogram_raw": eda.histogram(amount, 100),
                "histogram_log1p": eda.histogram(pd.Series(np.log1p(amount)), 100),
                "cents": {
                    "definition": "round(amount * 100) mod 100",
                    "n_distinct": int(cents.nunique()),
                    "top_10": [
                        {"cents": int(v), "n": int(c), "share": float(c) / len(train)}
                        for v, c in cent_counts.head(10).items()
                    ],
                    "share_zero_cents": float(is_round.mean()),
                },
                "round_against_not_round": {
                    "definition": "round means the amount is a whole number of currency units",
                    "round": {
                        "n": int(is_round.sum()),
                        "share": float(is_round.mean()),
                        "n_fraud": int(labels[is_round.to_numpy()].sum()),
                        "fraud_rate": float(labels[is_round.to_numpy()].mean()),
                        "mean_amount": float(amount[is_round].mean()),
                        "median_amount": float(amount[is_round].median()),
                        "interval": eda.wilson_interval(
                            int(labels[is_round.to_numpy()].sum()), int(is_round.sum())
                        ),
                    },
                    "not_round": {
                        "n": int((~is_round).sum()),
                        "share": float((~is_round).mean()),
                        "n_fraud": int(labels[(~is_round).to_numpy()].sum()),
                        "fraud_rate": float(labels[(~is_round).to_numpy()].mean()),
                        "mean_amount": float(amount[~is_round].mean()),
                        "median_amount": float(amount[~is_round].median()),
                        "interval": eda.wilson_interval(
                            int(labels[(~is_round).to_numpy()].sum()), int((~is_round).sum())
                        ),
                    },
                },
            },
        }
    )
    return report


def build_missingness(train: pd.DataFrame, identity_ids: set[int]) -> dict[str, Any]:
    """Missing rate per column, the blocks that share a mask, and missingness against the label."""
    report = envelope(
        "missingness",
        "mask = train.isna(); eda.missingness_blocks(mask); eda.missingness_against_target(mask, y)",
        len(train),
    )
    columns = [c for c in train.columns if c != config.ENTITY_ID_COLUMN]
    mask = train[columns].isna()
    target = train[config.TARGET]

    rates = [
        {
            "column": column,
            "missing_rate": float(mask[column].mean()),
            "n_missing": int(mask[column].sum()),
        }
        for column in columns
    ]
    rates.sort(key=lambda r: -float(r["missing_rate"]))

    blocks = eda.missingness_blocks(mask)
    multi = [b for b in blocks if b["n_columns"] > 1]
    # The heatmap is about columns that go missing together, so the blocks with no nulls at all
    # are left out of it. They are still in `all_blocks`.
    multi_with_nulls = [b for b in multi if b["missing_rate"] > 0]

    against_target = eda.missingness_against_target(mask, target)
    linked = [r for r in against_target if r["label_linked"]]

    section: dict[str, Any] = {
        "n_columns": len(columns),
        "n_columns_with_nulls": sum(1 for r in rates if r["n_missing"] > 0),
        "n_columns_without_nulls": sum(1 for r in rates if r["n_missing"] == 0),
        "missing_rate_by_column": rates,
        "blocks": {
            "definition": "columns whose null mask is identical row for row",
            "n_blocks": len(blocks),
            "n_blocks_with_more_than_one_column": len(multi),
            "n_columns_in_multi_column_blocks": sum(b["n_columns"] for b in multi),
            "largest_block_columns": multi[0]["n_columns"] if multi else 0,
            "all_blocks": blocks,
        },
        "block_overlap": eda.block_overlap_matrix(mask, multi_with_nulls[:20]),
        "against_target": {
            "thresholds": {"abs_gap": eda.MNAR_ABS_GAP, "p_value": eda.MNAR_P_VALUE},
            "note": (
                "a column flagged label_linked is one whose missingness carries rate. Imputing "
                "it without a was-missing indicator erases that. Stage 3 reads this list."
            ),
            "n_columns_compared": len(against_target),
            "n_label_linked": len(linked),
            "label_linked_columns": [r["column"] for r in linked],
            "per_column": against_target,
        },
    }

    y = target.to_numpy(dtype="int64")
    present = train[config.ID_COLUMN].isin(identity_ids).to_numpy(dtype=bool)
    identity_columns = [c for c in config.IDENTITY_COLUMNS if c in mask.columns]
    id_blocks = [b for b in blocks if any(c in identity_columns for c in b["columns"])]
    section["identity_join"] = {
        "note": (
            "a row joins the identity table or it does not, so the identity columns are "
            "missingness at block level. Whether they are missing as ONE block is measured "
            "rather than assumed: n_blocks_covering_identity_columns says how many distinct "
            "masks those columns actually take, and columns_whose_mask_is_exactly_the_join "
            "names the ones that go missing precisely when the join fails."
        ),
        "membership_source": "TransactionID present in train_identity.csv",
        "n_rows_with_identity": int(present.sum()),
        "share_rows_with_identity": float(present.mean()),
        "fraud_rate_with_identity": float(y[present].mean()),
        "fraud_rate_without_identity": float(y[~present].mean()),
        "interval_with_identity": eda.wilson_interval(int(y[present].sum()), int(present.sum())),
        "interval_without_identity": eda.wilson_interval(
            int(y[~present].sum()), int((~present).sum())
        ),
        "n_identity_columns": len(identity_columns),
        "n_blocks_covering_identity_columns": len(id_blocks),
        "identity_column_missing_rates": {c: float(mask[c].mean()) for c in identity_columns},
        "n_identity_columns_null_on_every_joined_row": sum(
            1 for c in identity_columns if bool(mask.loc[present, c].all())
        ),
        "columns_whose_mask_is_exactly_the_join": sorted(
            c for c in identity_columns if np.array_equal(mask[c].to_numpy(dtype=bool), ~present)
        ),
    }
    report.update(section)
    return report


def build_correlation(train: pd.DataFrame) -> dict[str, Any]:
    """Spearman among numeric columns, the V-block reduction, and near-duplicate pairs.

    Two stages. A screen matrix over all 401 numeric columns, ranked once per column and
    correlated over each pair's complete rows, finds every pair that could plausibly be
    redundant. Every pair the screen returns is then recomputed with pandas, which re-ranks
    inside the pair. The screen is exact when two columns share a missingness mask and is only
    approximate when they do not, and both cases are measured here rather than asserted.
    """
    report = envelope(
        "correlation",
        "ranks = eda.rank_matrix(numeric); rho, n = eda.spearman_matrix(ranks); "
        "eda.exact_spearman(train, eda.screened_pairs(names, rho, SCREEN_RHO))",
        len(train),
    )
    numeric_columns = [
        c
        for c in train.columns
        if c not in eda.NON_FEATURE_COLUMNS
        and c != config.ENTITY_ID_COLUMN
        and not isinstance(train[c].dtype, pd.CategoricalDtype)
        and train[c].dtype != object
    ]
    numeric = train[numeric_columns]
    index_of = {name: i for i, name in enumerate(numeric_columns)}

    started = time.perf_counter()
    ranks = eda.rank_matrix(numeric)
    rho, pair_counts = eda.spearman_matrix(ranks)
    del ranks
    gc.collect()
    screen_seconds = time.perf_counter() - started

    blocks = eda.missingness_blocks(numeric.isna())
    multi_blocks = [b for b in blocks if b["n_columns"] > 1]

    rng = np.random.default_rng(config.SEED)
    within_block_sample: list[tuple[str, str]] = []
    for block in multi_blocks:
        members = block["columns"]
        for _ in range(min(5, len(members))):
            a, b = rng.choice(members, size=2, replace=False)
            within_block_sample.append((str(a), str(b)))
    random_sample = [
        (numeric_columns[int(i)], numeric_columns[int(j)])
        for i, j in rng.choice(len(numeric_columns), size=(200, 2))
        if i != j
    ]

    started = time.perf_counter()
    pairs = eda.screened_pairs(numeric_columns, rho, SCREEN_RHO)
    exact = eda.exact_spearman(numeric, pairs)
    exact_seconds = time.perf_counter() - started

    v_blocks: list[dict[str, Any]] = []
    total_removed = 0
    total_v_columns = 0
    for block in blocks:
        members = [c for c in block["columns"] if c.startswith("V")]
        if len(members) < 2:
            continue
        total_v_columns += len(members)
        positions = [index_of[c] for c in members]
        sub = rho[np.ix_(positions, positions)]
        # Priority is the dataset's own column order, best first. Inside a block every member
        # has the same missing rate, so there is nothing else to break the tie on.
        priority = [-float(i) for i in range(len(members))]
        groups = eda.correlated_groups(members, sub, eda.REDUNDANT_RHO, priority)
        removed = len(members) - len(groups)
        total_removed += removed
        v_blocks.append(
            {
                "block_missing_rate": block["missing_rate"],
                "n_columns": len(members),
                "n_groups": len(groups),
                "n_removed_by_reduction": removed,
                "groups": groups,
            }
        )
    v_blocks.sort(key=lambda b: -int(b["n_columns"]))

    report.update(
        {
            "method": (
                "Spearman, not Pearson. The skew measured in univariate.json is why: rank "
                "correlation is invariant to the monotone transform that would be needed to "
                "make Pearson mean anything on these columns. See ADR 0008."
            ),
            "n_numeric_columns": len(numeric_columns),
            "columns": numeric_columns,
            "screen": {
                "method": (
                    "each column ranked once over the train split, then correlated over the "
                    "rows where both columns of a pair are present, accumulated over 50,000 row "
                    "chunks. Exact for a pair that shares a missingness mask, approximate "
                    "otherwise."
                ),
                "threshold": SCREEN_RHO,
                "n_pairs_total": len(numeric_columns) * (len(numeric_columns) - 1) // 2,
                "n_pairs_screened_in": len(pairs),
                "n_pairs_recomputed_exactly": len(exact),
                "seconds_screen": screen_seconds,
                "seconds_exact": exact_seconds,
                "agreement_within_missingness_blocks": eda.compare_against_pandas(
                    numeric, within_block_sample, rho, index_of
                ),
                "agreement_on_random_pairs": eda.compare_against_pandas(
                    numeric, random_sample, rho, index_of
                ),
            },
            "v_block_reduction": {
                "threshold": eda.REDUNDANT_RHO,
                "note": (
                    "grouping runs inside each missingness block, so two V columns that are "
                    "never observed together cannot be called redundant on a handful of rows, "
                    "and the screen matrix is exact within a block."
                ),
                "n_v_columns_in_multi_column_blocks": total_v_columns,
                "n_columns_a_reduction_would_remove": total_removed,
                "n_blocks": len(v_blocks),
                "blocks": v_blocks,
            },
            "near_duplicate_pairs": eda.near_duplicate_pairs(
                exact, pair_counts, index_of, eda.NEAR_DUPLICATE_RHO, 300
            ),
            "redundant_pairs": eda.near_duplicate_pairs(
                exact, pair_counts, index_of, eda.REDUNDANT_RHO, 50
            ),
            "strongest_pairs_outside_v": eda.near_duplicate_pairs(
                exact, pair_counts, index_of, eda.REDUNDANT_RHO, 100, exclude_prefix="V"
            ),
            "missing_rate_by_column": {c: float(numeric[c].isna().mean()) for c in numeric_columns},
        }
    )
    del rho, pair_counts
    gc.collect()
    return report


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Every column a model could see, which is everything but the identifier, label and clock."""
    return [
        c
        for c in frame.columns
        if c not in eda.NON_FEATURE_COLUMNS
        and c not in (config.ENTITY_ID_COLUMN, "day_index", "week_index")
    ]


def build_bivariate(train: pd.DataFrame) -> dict[str, Any]:
    """Information value per feature, and the shape of the strongest relationships."""
    report = envelope(
        "bivariate",
        "eda.information_value(train[column], train[isFraud]) for every feature column",
        len(train),
    )
    target = train[config.TARGET]
    rows: list[dict[str, Any]] = []
    for column in feature_columns(train):
        values = train[column]
        result = eda.information_value(values, target)
        present = values.notna()
        present_only = (
            eda.information_value(values[present], target[present])["iv"]
            if int(present.sum()) > 0 and target[present].nunique() > 1
            else None
        )
        rows.append(
            {
                "column": column,
                "iv": result["iv"],
                # IV sums over bins, so the missing bin's term comes out exactly. A column whose
                # IV is nearly all missing bin is telling you it joined, not what it holds.
                "iv_missing_bin": result["iv_missing_bin"],
                "iv_present_bins": result["iv_present_bins"],
                # Recomputed on the present rows alone, where the class balance is different.
                # Not a decomposition of the line above, a second view of the same column.
                "iv_present_rows_only": present_only,
                "n_bins": result["n_bins"],
                "n_present": int(present.sum()),
                "missing_rate": float(values.isna().mean()),
                "is_categorical": bool(isinstance(values.dtype, pd.CategoricalDtype)),
                "high_cardinality": bool(result["n_bins"] > HIGH_CARDINALITY_BINS),
                "entity_identifying": column in eda.ENTITY_IDENTIFYING_COLUMNS,
            }
        )
    rows.sort(key=lambda r: -float(r["iv"]))
    behavioural = [r for r in rows if not r["entity_identifying"]]
    by_present_bins = sorted(rows, key=lambda r: -float(r["iv_present_bins"]))

    decile_columns = [r["column"] for r in behavioural if not r["is_categorical"]][
        :TOP_FEATURES_FOR_DECILES
    ]

    report.update(
        {
            "method": (
                "information value with missing as a level and a 0.5 event smoothing. Bins are "
                "deciles of the train distribution for a numeric column and the levels "
                "themselves for a categorical one."
            ),
            "entity_identifying_note": (
                "information value on a column that identifies the entity is inflated by the "
                "label clustering measured in stage 0. Those columns are flagged and ranked "
                "separately rather than dropped from the table."
            ),
            "entity_identifying_columns": list(eda.ENTITY_IDENTIFYING_COLUMNS),
            "n_features": len(rows),
            "iv_bands": {
                "above_0_5": sum(1 for r in rows if r["iv"] > 0.5),
                "0_1_to_0_5": sum(1 for r in rows if 0.1 < r["iv"] <= 0.5),
                "0_02_to_0_1": sum(1 for r in rows if 0.02 < r["iv"] <= 0.1),
                "below_0_02": sum(1 for r in rows if r["iv"] <= 0.02),
            },
            "per_feature": rows,
            "top_20_overall": rows[:20],
            "top_20_behavioural": behavioural[:20],
            "top_20_by_iv_present_bins": by_present_bins[:20],
            "high_cardinality_note": (
                f"a column binned into more than {HIGH_CARDINALITY_BINS} levels earns "
                "information value from having many small bins as much as from separating the "
                "classes. Those columns carry high_cardinality and are read with that in mind."
            ),
            "n_high_cardinality": sum(1 for r in rows if r["high_cardinality"]),
            "deciles": {
                column: eda.decile_table(train[column], target) for column in decile_columns
            },
            "deciles_note": (
                "the top of the information value table is a block of V columns that take three "
                "bins, so their tables show a step and not a curve. The reference set below is "
                "chosen by name, not by rank, so the shape of the columns a person can reason "
                "about is on the record too."
            ),
            "deciles_reference": {
                column: eda.decile_table(train[column], target)
                for column in REFERENCE_DECILE_COLUMNS
                if column in train.columns
            },
        }
    )
    return report


def build_temporal(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """PSI against the later splits, the time consistency screen, and cold entities in train."""
    report = envelope(
        "temporal",
        "eda.population_stability_index(train[c], val[c]) and (train[c], test[c]); "
        "eda.time_consistency(first 30 days, last 30 days) per column",
        len(train),
    )
    columns = feature_columns(train)

    psi_rows: list[dict[str, Any]] = []
    for column in columns:
        against_val = eda.population_stability_index(train[column], val[column])
        against_test = eda.population_stability_index(train[column], test[column])
        psi_rows.append(
            {
                "column": column,
                "psi_val": against_val["psi"],
                "psi_test": against_test["psi"],
                "n_levels": against_val["n_levels"],
                "n_new_levels_val": against_val["n_new_levels"],
                "n_new_levels_test": against_test["n_new_levels"],
            }
        )
    psi_rows.sort(key=lambda r: -float(r["psi_test"]))

    day = np.floor(train[config.TIME_COLUMN] / config.SECONDS_PER_DAY).astype("int64")
    first_day, last_day = int(day.min()), int(day.max())
    early_mask = (day <= first_day + CONSISTENCY_WINDOW_DAYS - 1).to_numpy(dtype=bool)
    late_mask = (day >= last_day - CONSISTENCY_WINDOW_DAYS + 1).to_numpy(dtype=bool)
    early, late = train[early_mask], train[late_mask]
    early_y, late_y = early[config.TARGET], late[config.TARGET]

    started = time.perf_counter()
    consistency: list[dict[str, Any]] = []
    for column in columns:
        result = eda.time_consistency(
            early[column], early_y, late[column], late_y, seed=config.SEED
        )
        result["column"] = column
        consistency.append(result)
    consistency_seconds = time.perf_counter() - started

    flagged = [
        r
        for r in consistency
        if r["early_auc"] is not None
        and r["late_auc"] is not None
        and r["early_auc"] >= CONSISTENCY_EARLY_MIN
        and r["late_auc"] < CONSISTENCY_LATE_MAX
    ]
    flagged.sort(key=lambda r: float(r["early_auc"]) - float(r["late_auc"]), reverse=True)
    consistency.sort(key=lambda r: -(r["late_auc"] if r["late_auc"] is not None else -1.0))

    report.update(
        {
            "psi": {
                "method": (
                    "bins are deciles of the train distribution with missing as its own level; "
                    "the later split is scored against those bins and never rebinned. Empty "
                    "bins are floored so one of them cannot contribute an infinity."
                ),
                "floor": eda.PSI_FLOOR,
                "n_features": len(psi_rows),
                "per_feature": psi_rows,
                "summary": {
                    "n_psi_test_above_0_10": sum(1 for r in psi_rows if r["psi_test"] > 0.10),
                    "n_psi_test_above_0_25": sum(1 for r in psi_rows if r["psi_test"] > 0.25),
                    "n_psi_val_above_0_10": sum(1 for r in psi_rows if r["psi_val"] > 0.10),
                    "n_psi_val_above_0_25": sum(1 for r in psi_rows if r["psi_val"] > 0.25),
                    "note": (
                        "0.10 and 0.25 are the conventional PSI bands. They are a convention "
                        "this repo has adopted, not a measurement, and stage 9 sets the "
                        "monitoring thresholds against this baseline rather than against them."
                    ),
                },
            },
            "time_consistency": {
                "method": (
                    "one LightGBM classifier per column fitted on the first 30 day block of "
                    "train and scored on the last 30 day block of the same split. The early AUC "
                    "is in-sample and is not a performance estimate; it is there so a feature "
                    "with no early signal can be told from one whose signal inverted."
                ),
                "model": {
                    "estimator": "lightgbm.LGBMClassifier",
                    "n_estimators": 50,
                    "num_leaves": 15,
                    "min_child_samples": 200,
                    "learning_rate": 0.1,
                    "random_state": config.SEED,
                },
                "windows": {
                    "window_days": CONSISTENCY_WINDOW_DAYS,
                    "early_day_index_range": [first_day, first_day + CONSISTENCY_WINDOW_DAYS - 1],
                    "late_day_index_range": [last_day - CONSISTENCY_WINDOW_DAYS + 1, last_day],
                    "n_early_rows": int(early_mask.sum()),
                    "n_late_rows": int(late_mask.sum()),
                    "n_early_fraud": int(early_y.sum()),
                    "n_late_fraud": int(late_y.sum()),
                },
                "flag_rule": {
                    "early_auc_at_least": CONSISTENCY_EARLY_MIN,
                    "late_auc_below": CONSISTENCY_LATE_MAX,
                    "note": "both numbers are choices; the artifact lists every column's pair",
                },
                "seconds": consistency_seconds,
                "n_flagged": len(flagged),
                "flagged_columns": [r["column"] for r in flagged],
                "flagged": flagged,
                "per_feature": consistency,
            },
            "cold_entities": eda.cold_warm_within(
                train, config.ENTITY_ID_COLUMN, config.TARGET, config.TIME_COLUMN
            ),
        }
    )
    return report


def build_train_only(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    """What an EDA over the whole file would have said differently. Evidence for ADR 0007.

    Deliberately the one section that computes a statistic over all 590,540 rows, and it does so
    in order to measure the size of the mistake rather than to make it. Two comparisons: how far
    a summary statistic moves when the later windows are folded in, and how many categorical
    levels exist only outside the training window.
    """
    report = envelope(
        "train_only",
        "the same statistic computed on train and on train + val + test, compared",
        len(train),
    )
    whole = pd.concat([train, val, test], axis=0)
    columns = feature_columns(train)

    numeric_shift: list[dict[str, Any]] = []
    unseen_levels: list[dict[str, Any]] = []
    for column in columns:
        series_train, series_all = train[column], whole[column]
        if isinstance(series_train.dtype, pd.CategoricalDtype):
            in_train = set(series_train.dropna().unique())
            later = set(pd.concat([val[column], test[column]]).dropna().unique())
            only_later = later - in_train
            rows_on_unseen = int(
                pd.concat([val[column], test[column]]).isin(list(only_later)).sum()
            )
            unseen_levels.append(
                {
                    "column": column,
                    "n_levels_in_train": len(in_train),
                    "n_levels_only_after_train": len(only_later),
                    "n_later_rows_on_a_level_train_never_saw": rows_on_unseen,
                    "share_of_later_rows": rows_on_unseen / (len(val) + len(test)),
                }
            )
            continue
        clean_train = series_train.dropna().astype("float64")
        clean_all = series_all.dropna().astype("float64")
        if clean_train.empty or clean_all.empty:
            continue
        for statistic, name in ((0.99, "p99"), (0.50, "median")):
            on_train = float(clean_train.quantile(statistic))
            on_all = float(clean_all.quantile(statistic))
            if on_train != 0.0:
                numeric_shift.append(
                    {
                        "column": column,
                        "statistic": name,
                        "on_train": on_train,
                        "on_all_rows": on_all,
                        "relative_shift": abs(on_all - on_train) / abs(on_train),
                    }
                )
    numeric_shift.sort(key=lambda r: -float(r["relative_shift"]))
    unseen_levels.sort(key=lambda r: -int(r["n_levels_only_after_train"]))

    return report | {
        "note": (
            "the numbers here are the cost of getting the rule wrong, measured once so the rule "
            "does not have to be argued for again. Every other section in this directory is "
            "computed on train alone."
        ),
        "n_rows_train": len(train),
        "n_rows_all": len(whole),
        "quantile_shift": {
            "definition": (
                "the same quantile computed on the train split and on all 590,540 rows, as a "
                "relative move"
            ),
            "n_columns_compared": len({row["column"] for row in numeric_shift}),
            "n_statistics_moving_more_than_10_percent": sum(
                1 for row in numeric_shift if row["relative_shift"] > 0.10
            ),
            "largest_50": numeric_shift[:50],
        },
        "categorical_levels_only_after_train": {
            "definition": (
                "levels a categorical column takes in val or test and never takes in train. An "
                "EDA over the whole file lists them as ordinary categories; a model trained on "
                "train has never seen one."
            ),
            "n_columns_with_unseen_levels": sum(
                1 for row in unseen_levels if row["n_levels_only_after_train"] > 0
            ),
            "per_column": unseen_levels,
        },
    }


def build_entities(train: pd.DataFrame) -> dict[str, Any]:
    """Size, label purity and inter-transaction gap distributions at the ADR 0001 entity key."""
    report = envelope(
        "entities",
        "eda.entity_structure(train, entity_id, isFraud, TransactionDT)",
        len(train),
    )
    report.update(
        {
            "entity_key": list(config.ENTITY_KEY_COLUMNS),
            "entity_key_source": "ADR 0001",
            "structure": eda.entity_structure(
                train, config.ENTITY_ID_COLUMN, config.TARGET, config.TIME_COLUMN
            ),
        }
    )
    return report


def build_quality(train: pd.DataFrame) -> dict[str, Any]:
    """Duplicates, impossible values, the email domains, and the free text device string."""
    report = envelope(
        "quality",
        "eda.duplicate_groups(train, subset); range checks; email and device tables",
        len(train),
    )
    all_columns = [c for c in train.columns if c != config.ENTITY_ID_COLUMN]
    content_columns = [c for c in all_columns if c not in (config.ID_COLUMN, config.TIME_COLUMN)]
    keep_time_columns = [c for c in all_columns if c != config.ID_COLUMN]
    labels = train[config.TARGET].to_numpy(dtype="int64")

    duplicates = {
        "exact_rows": {
            "definition": "identical on every column including TransactionID",
            **eda.duplicate_groups(train, all_columns, config.TARGET),
        },
        "same_content_same_time": {
            "definition": "identical on every column except TransactionID",
            **eda.duplicate_groups(train, keep_time_columns, config.TARGET),
        },
        "same_content": {
            "definition": "identical on every column except TransactionID and TransactionDT",
            **eda.duplicate_groups(train, content_columns, config.TARGET),
        },
        "repeated_core_fields": {
            "definition": (
                "a different question from the three above: rows agreeing on the fields a person "
                "would call the same purchase, ignoring the counters and the V block that move "
                "every time. Card testing repeats the purchase, not the running totals, so a "
                "whole-row comparison cannot see it."
            ),
            "columns": list(CORE_DUPLICATE_COLUMNS),
            **eda.duplicate_groups(train, CORE_DUPLICATE_COLUMNS, config.TARGET),
        },
        "note": (
            "repeated identical transactions are a known card testing shape. The number is "
            "recorded here and the decision on what to do about it is stage 3's."
        ),
        "transaction_id_is_unique": bool(train[config.ID_COLUMN].is_unique),
    }

    amount = train[config.AMOUNT_COLUMN]
    numeric_columns = [
        c
        for c in all_columns
        if not isinstance(train[c].dtype, pd.CategoricalDtype) and train[c].dtype != object
    ]
    negative = {
        c: int((train[c] < 0).sum()) for c in numeric_columns if int((train[c] < 0).sum()) > 0
    }
    impossible = {
        "n_negative_amounts": int((amount < 0).sum()),
        "n_zero_amounts": int((amount == 0).sum()),
        "n_non_finite_amounts": int((~np.isfinite(amount)).sum()),
        "min_amount": float(amount.min()),
        "max_amount": float(amount.max()),
        "columns_with_negative_values": negative,
        "n_columns_with_negative_values": len(negative),
        "categorical_levels": {
            c: sorted(str(v) for v in train[c].dropna().unique())
            for c in ("ProductCD", "card4", "card6", "DeviceType")
            if c in train.columns and train[c].nunique() <= 20
        },
        "n_rows_with_null_target": int(train[config.TARGET].isna().sum()),
        "n_rows_with_null_time": int(train[config.TIME_COLUMN].isna().sum()),
    }

    emails: dict[str, Any] = {}
    for column in config.EMAIL_COLUMNS:
        if column not in train.columns:
            continue
        emails[column] = {
            "profile": eda.categorical_profile(train[column], column),
            "by_domain": [
                row for row in eda.rate_by_group(train, column, config.TARGET) if row["n"] >= 100
            ],
            "min_rows_listed": 100,
        }
    if all(c in train.columns for c in config.EMAIL_COLUMNS):
        p = train["P_emaildomain"].astype("object")
        r = train["R_emaildomain"].astype("object")
        both = p.notna() & r.notna()
        agree = both & (p == r)
        disagree = both & (p != r)
        r_only_missing = p.notna() & r.isna()
        emails["agreement"] = {
            "definition": "compared as raw strings, on rows where both are present",
            "n_both_present": int(both.sum()),
            "share_both_present": float(both.mean()),
            "agree": {
                "n": int(agree.sum()),
                "n_fraud": int(labels[agree.to_numpy()].sum()),
                "fraud_rate": float(labels[agree.to_numpy()].mean()) if agree.any() else None,
                "interval": eda.wilson_interval(
                    int(labels[agree.to_numpy()].sum()), int(agree.sum())
                ),
            },
            "disagree": {
                "n": int(disagree.sum()),
                "n_fraud": int(labels[disagree.to_numpy()].sum()),
                "fraud_rate": (
                    float(labels[disagree.to_numpy()].mean()) if disagree.any() else None
                ),
                "interval": eda.wilson_interval(
                    int(labels[disagree.to_numpy()].sum()), int(disagree.sum())
                ),
            },
            "recipient_missing_only": {
                "n": int(r_only_missing.sum()),
                "n_fraud": int(labels[r_only_missing.to_numpy()].sum()),
                "fraud_rate": (
                    float(labels[r_only_missing.to_numpy()].mean())
                    if r_only_missing.any()
                    else None
                ),
                "interval": eda.wilson_interval(
                    int(labels[r_only_missing.to_numpy()].sum()), int(r_only_missing.sum())
                ),
            },
        }

    device: dict[str, Any] = {}
    if "DeviceInfo" in train.columns:
        values = train["DeviceInfo"].astype("object").dropna()
        counts = values.value_counts()
        device = {
            "profile": eda.categorical_profile(train["DeviceInfo"], "DeviceInfo"),
            "note": (
                "free text. The tail is what makes it free text: the count of distinct strings "
                "appearing once, against the share of rows they cover, is the size of the "
                "normalisation job stage 3 inherits."
            ),
            "n_distinct": int(counts.size),
            "n_appearing_once": int((counts == 1).sum()),
            "share_of_rows_in_strings_appearing_once": float(
                counts[counts == 1].sum() / counts.sum()
            ),
            "share_of_rows_in_top_10": float(counts.head(10).sum() / counts.sum()),
            "share_of_rows_in_top_50": float(counts.head(50).sum() / counts.sum()),
            "top_20": [
                {"value": str(v), "n": int(c), "share_of_present": float(c) / int(counts.sum())}
                for v, c in counts.head(20).items()
            ],
        }

    report.update(
        {
            "duplicates": duplicates,
            "impossible_values": impossible,
            "email_domains": emails,
            "device_info": device,
        }
    )
    return report


# --- driver ------------------------------------------------------------------------------


def write(report: dict[str, Any], path: Path) -> None:
    """Write one artifact. allow_nan=False so a NaN fails here and not in a reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=False) + "\n")
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 exploratory analysis.")
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    parser.add_argument("--out-dir", default=str(config.REPORTS_DIR / "eda"))
    parser.add_argument("--sections", nargs="*", default=list(SECTIONS), choices=list(SECTIONS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    started = time.perf_counter()

    print("loading the joined frame")
    frame = data_loader.add_entity_key(
        data_loader.load_raw(transactions_path=args.transactions, identity_path=args.identity)
    )
    print(f"  {len(frame):,} rows by {frame.shape[1]} columns")
    train, val, test = data_loader.time_based_split(frame)
    del frame
    gc.collect()

    # The whole stage rests on this line. Every section below reads `train` and nothing else,
    # except the two that are explicitly about the difference between splits.
    data_loader.assert_train_only(train, what="stage 2 exploratory analysis")
    print(f"  train {len(train):,} rows, val {len(val):,}, test {len(test):,}")

    identity_ids = set(
        pd.read_csv(args.identity, usecols=[config.ID_COLUMN])[config.ID_COLUMN].tolist()
    )

    split_summary: dict[str, Any] | None = None
    if config.SPLIT_SUMMARY_PATH.exists():
        split_summary = json.loads(config.SPLIT_SUMMARY_PATH.read_text())

    for section in args.sections:
        print(f"section {section}")
        section_started = time.perf_counter()
        if section == "target":
            report = build_target(train, val, test, split_summary)
        elif section == "univariate":
            report = build_univariate(train)
        elif section == "missingness":
            report = build_missingness(train, identity_ids)
        elif section == "correlation":
            report = build_correlation(train)
        elif section == "bivariate":
            report = build_bivariate(train)
        elif section == "temporal":
            report = build_temporal(train, val, test)
        elif section == "entities":
            report = build_entities(train)
        elif section == "train_only":
            report = build_train_only(train, val, test)
        else:
            report = build_quality(train)
        report["seconds"] = time.perf_counter() - section_started
        write(report, out_dir / f"{section}.json")
        del report
        gc.collect()

    print(f"\ndone in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
