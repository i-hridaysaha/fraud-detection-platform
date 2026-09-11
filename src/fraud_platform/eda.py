"""Analysis primitives for stage 2.

Every function here describes data. None of them changes it, and none of them fits anything a
later stage will reuse: cleaning, encoding and feature construction are stage 3 and stage 4. The
split between this module and scripts/eda.py is the same one stage 0 used. The functions return
plain dicts carrying their numbers, and the script assembles them into the artifacts under
reports/eda/ with the command that regenerates each one.

**Everything here is meant to run on the train split.** The caller enforces that with
`data_loader.assert_train_only`; this module does not check, because two of its functions are
about the difference between splits and would have to argue with the guard. The two exceptions
are named in their docstrings: `population_stability_index` compares a train distribution against
a later one, and the caller is expected to bin on train. See ADR 0007.

Thresholds that could have been picked differently are module constants with a comment saying so.
They are choices, not measurements, and docs/eda.md labels them that way.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

# --- choices -----------------------------------------------------------------------------------
# None of these is measured. Each is a line drawn somewhere, and the artifacts record the line so
# a reader can move it and recompute.

# A column is called effectively constant when one value covers this share of its non-null rows.
# Stage 2 brief sets 0.99.
CONSTANT_SHARE = 0.99
# Below the constant threshold but still dominated by one value.
DOMINATED_SHARE = 0.95
# A numeric column is called zero-inflated when this share of its non-null rows is exactly zero.
ZERO_INFLATED_SHARE = 0.50
# Two columns are called near-duplicates at this Spearman correlation.
NEAR_DUPLICATE_RHO = 0.99
# Redundancy grouping threshold for the V-block reduction, from the stage 2 brief.
REDUNDANT_RHO = 0.95
# The label a null gets when a column is binned. Missing is a level, not an exclusion.
MISSING_LEVEL = "(missing)"

# The rare tail of a categorical column: the categories that, taken from the least frequent
# upwards, cover this share of rows.
RARE_TAIL_SHARE = 0.01
# Missingness is called label-linked when the fraud rate differs by at least this much in
# absolute terms AND the two-proportion test clears the p threshold. With 413,378 train rows
# almost any difference clears a p test on its own, so the absolute gate is what does the work.
MNAR_ABS_GAP = 0.010
MNAR_P_VALUE = 1e-4
# Default bin count for information value, deciles for PSI, and the smoothing that keeps a bin
# with zero events out of a logarithm.
DEFAULT_BINS = 10
IV_SMOOTHING = 0.5
# PSI floor: a bin holding no rows in one distribution would otherwise give an infinite term.
PSI_FLOOR = 1e-6

# Stand-in for a null when rows are compared value by value. A plain tuple comparison calls two
# nulls different, and for the duplicate check they are the same content.
_NULL_SENTINEL = "\x00null"

# Columns that carry entity identity rather than transaction behaviour. Information value on
# these is inflated by the entity linkage measured in stage 0 (0.9664 of multi-transaction
# entities are label-pure), so they are reported separately rather than ranked alongside the
# rest. The list is the ADR 0001 entity key, its sources, and the row identifier.
ENTITY_IDENTIFYING_COLUMNS: tuple[str, ...] = (
    "TransactionID",
    "card1",
    "addr1",
    "D1",
    "card_start_day",
    "entity_id",
)

# Never profiled as a feature: the row identifier, the label, and the clock.
NON_FEATURE_COLUMNS: tuple[str, ...] = ("TransactionID", "isFraud", "TransactionDT")


def _float(value: Any) -> float:
    return float(value)


def _int(value: Any) -> int:
    return int(value)


def _finite(value: float) -> float | None:
    """None for a NaN or an infinity, so the artifact stays valid JSON under allow_nan=False."""
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


# --- univariate --------------------------------------------------------------------------------


PERCENTILES: tuple[float, ...] = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def numeric_profile(series: pd.Series, name: str) -> dict[str, Any]:
    """Count, missing rate, moments, percentiles and the domination flags for one numeric column.

    Skew and kurtosis are pandas' own, which are the bias-corrected sample statistics; kurtosis is
    excess kurtosis, so a normal column reads 0 and not 3. Both are None when fewer than four
    non-null values make them undefined rather than silently NaN.
    """
    n_total = len(series)
    values = series.dropna()
    n = len(values)
    profile: dict[str, Any] = {
        "column": name,
        "n_total": n_total,
        "count": n,
        "missing_rate": (n_total - n) / n_total if n_total else None,
        "n_unique": _int(values.nunique()) if n else 0,
    }
    if n == 0:
        profile["all_null"] = True
        return profile

    as_float = values.astype("float64")
    counts = values.value_counts()
    top_value = counts.index[0]
    top_share = _float(counts.iloc[0]) / n
    zero_share = _float((as_float == 0).mean())

    profile.update(
        {
            "all_null": False,
            "min": _finite(_float(as_float.min())),
            "max": _finite(_float(as_float.max())),
            "mean": _finite(_float(as_float.mean())),
            "median": _finite(_float(as_float.median())),
            "std": _finite(_float(as_float.std())) if n > 1 else None,
            "skew": _finite(_float(as_float.skew())) if n > 2 else None,
            "kurtosis": _finite(_float(as_float.kurtosis())) if n > 3 else None,
            "percentiles": {
                f"p{int(q * 100):02d}": _finite(_float(as_float.quantile(q))) for q in PERCENTILES
            },
            "top_value": _float(top_value),
            "top_value_share": top_share,
            "zero_share": zero_share,
            "zero_inflated": bool(zero_share >= ZERO_INFLATED_SHARE),
            "single_value_dominated": bool(top_share >= DOMINATED_SHARE),
            "effectively_constant": bool(top_share >= CONSTANT_SHARE),
        }
    )
    return profile


def categorical_profile(series: pd.Series, name: str) -> dict[str, Any]:
    """Cardinality, top value share and the size of the rare tail for one categorical column.

    The rare tail is counted from the least frequent category upwards: how many categories it
    takes to cover RARE_TAIL_SHARE of the non-null rows. A column where that number is large is a
    column where a frequency floor throws away many distinct values and few rows, which is the
    shape stage 3 needs to know about.
    """
    n_total = len(series)
    values = series.dropna()
    n = len(values)
    profile: dict[str, Any] = {
        "column": name,
        "n_total": n_total,
        "count": n,
        "missing_rate": (n_total - n) / n_total if n_total else None,
    }
    if n == 0:
        profile.update({"all_null": True, "cardinality": 0})
        return profile

    # value_counts on a categorical column returns every declared category, including the ones
    # no row in this split uses. The categories were fixed when the whole file was read, so on
    # the train split several of them are empty and counting them would overstate cardinality.
    counts = values.value_counts()
    counts = counts[counts > 0].sort_values(ascending=False)
    ascending = counts.sort_values(ascending=True)
    cumulative = ascending.cumsum() / n
    n_rare = _int((cumulative <= RARE_TAIL_SHARE).sum())
    top_share = _float(counts.iloc[0]) / n

    profile.update(
        {
            "all_null": False,
            "cardinality": len(counts),
            "top_value": str(counts.index[0]),
            "top_value_share": top_share,
            "top_5": [
                {"value": str(v), "count": _int(c), "share": _float(c) / n}
                for v, c in counts.head(5).items()
            ],
            "rare_tail_share_target": RARE_TAIL_SHARE,
            "n_categories_in_rare_tail": n_rare,
            "rare_tail_category_share": n_rare / len(counts),
            "rare_tail_row_share": _float(cumulative.iloc[n_rare - 1]) if n_rare else 0.0,
            "n_singleton_categories": _int((counts == 1).sum()),
            "single_value_dominated": bool(top_share >= DOMINATED_SHARE),
            "effectively_constant": bool(top_share >= CONSTANT_SHARE),
        }
    )
    return profile


def histogram(values: pd.Series, bins: int) -> dict[str, Any]:
    """Bin edges and counts, so a figure can be drawn from the artifact and not from the data."""
    clean = values.dropna().astype("float64").to_numpy()
    counts, edges = np.histogram(clean, bins=bins)
    return {
        "n": int(clean.size),
        "bins": int(bins),
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


# --- rates and intervals -----------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    """Wilson score interval for a proportion. z defaults to the two-sided 95 percent value.

    Wilson rather than the normal approximation because the fraud rate is near 0.035 and some of
    the cells cut here hold a few hundred rows, where the normal interval runs below zero.
    """
    if n == 0:
        return {"point": None, "low": None, "high": None, "n": 0, "successes": 0}
    p = successes / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = (z / denominator) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return {
        "point": p,
        "low": max(0.0, centre - half),
        "high": min(1.0, centre + half),
        "n": n,
        "successes": successes,
        "method": "wilson",
        "z": z,
    }


def bootstrap_rate_interval(
    labels: npt.NDArray[np.int_], n_boot: int, seed: int, alpha: float = 0.05
) -> dict[str, Any]:
    """Percentile bootstrap interval for a rate, resampling rows with replacement.

    Drawn as binomial counts rather than by materialising n_boot resampled label vectors, which
    is the same distribution and does not allocate 413,378 by n_boot integers.
    """
    n = int(labels.size)
    if n == 0:
        return {"point": None, "low": None, "high": None, "n": 0}
    successes = int(labels.sum())
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n, successes / n, size=n_boot) / n
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "point": successes / n,
        "low": float(low),
        "high": float(high),
        "n": n,
        "successes": successes,
        "n_boot": int(n_boot),
        "alpha": alpha,
        "seed": int(seed),
        "method": "percentile bootstrap on the row resample",
    }


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict[str, Any]:
    """Pooled two-proportion z test. Returns None for the statistic when either cell is empty."""
    if n1 == 0 or n2 == 0:
        return {"z": None, "p_value": None}
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": None, "p_value": None}
    z = (p1 - p2) / se
    return {"z": z, "p_value": float(2 * stats.norm.sf(abs(z)))}


def trend_tests(
    index: Sequence[float], rate: Sequence[float], count: Sequence[int]
) -> dict[str, Any]:
    """Three tests on a rate series against its period index, reported together.

    Kendall tau is the Mann-Kendall trend test: it asks whether the series is monotone and cares
    about order alone, so an outlying week cannot manufacture a slope. Least squares gives the
    size of a linear trend in rate per period. The chi-square test of homogeneity asks a different
    question from either: whether the periods share one fraud rate at all, trend or no trend. A
    series can be flat under Kendall and still fail homogeneity by bouncing.
    """
    x = np.asarray(index, dtype="float64")
    y = np.asarray(rate, dtype="float64")
    n = np.asarray(count, dtype="float64")
    tau = stats.kendalltau(x, y)
    fit = stats.linregress(x, y)

    successes = np.round(y * n)
    table = np.vstack([successes, n - successes])
    chi2, chi_p, dof, _ = stats.chi2_contingency(table)

    return {
        "n_periods": int(x.size),
        "kendall_tau": _finite(float(tau.statistic)),
        "kendall_p_value": _finite(float(tau.pvalue)),
        "ols_slope_per_period": _finite(float(fit.slope)),
        "ols_p_value": _finite(float(fit.pvalue)),
        "ols_r_squared": _finite(float(fit.rvalue**2)),
        "chi2_homogeneity": _finite(float(chi2)),
        "chi2_p_value": _finite(float(chi_p)),
        "chi2_dof": int(dof),
        "rate_min": float(y.min()),
        "rate_max": float(y.max()),
        "rate_range": float(y.max() - y.min()),
        "rate_max_over_min": _finite(float(y.max() / y.min())) if y.min() > 0 else None,
    }


def rate_by_group(frame: pd.DataFrame, by: str, target: str) -> list[dict[str, Any]]:
    """Fraud rate per level of one column, with counts and a Wilson interval per level.

    Nulls become their own level rather than being dropped, because for several
    of these columns the missing level is the largest one and dropping it would leave a table that
    does not add up to the split.
    """
    key = frame[by]
    if isinstance(key.dtype, pd.CategoricalDtype):
        key = key.astype("object")
    key = key.where(key.notna(), MISSING_LEVEL)
    grouped = frame.groupby(key, observed=True)[target].agg(["size", "sum"])
    grouped = grouped.sort_values("size", ascending=False)
    rows: list[dict[str, Any]] = []
    for value, row in grouped.iterrows():
        n, k = _int(row["size"]), _int(row["sum"])
        rows.append(
            {
                "value": str(value),
                "n": n,
                "n_fraud": k,
                "fraud_rate": k / n,
                "share_of_rows": n / len(frame),
                "interval": wilson_interval(k, n),
            }
        )
    return rows


# --- missingness -------------------------------------------------------------------------------


def missingness_blocks(mask: pd.DataFrame) -> list[dict[str, Any]]:
    """Group columns whose null masks are identical, row for row.

    Columns are bucketed by a hash of the packed mask and then compared exactly inside each
    bucket, so a hash collision cannot merge two blocks that differ. A column with no nulls joins
    the block whose mask is all False, which is reported like any other.
    """
    buckets: dict[bytes, list[str]] = {}
    arrays: dict[str, npt.NDArray[np.bool_]] = {}
    for column in mask.columns:
        values = mask[column].to_numpy(dtype=bool)
        arrays[column] = values
        digest = np.packbits(values).tobytes()
        buckets.setdefault(digest, []).append(column)

    blocks: list[dict[str, Any]] = []
    for columns in buckets.values():
        exact: list[list[str]] = []
        for column in columns:
            for group in exact:
                if np.array_equal(arrays[column], arrays[group[0]]):
                    group.append(column)
                    break
            else:
                exact.append([column])
        for group in exact:
            n_missing = _int(arrays[group[0]].sum())
            blocks.append(
                {
                    "n_columns": len(group),
                    "columns": sorted(group),
                    "n_rows_missing": n_missing,
                    "missing_rate": n_missing / len(mask),
                }
            )
    blocks.sort(key=lambda b: (-b["n_columns"], b["missing_rate"]))
    return blocks


def block_overlap_matrix(mask: pd.DataFrame, blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """For each pair of missingness blocks, the share of rows missing in both.

    This is the pattern heatmap in numeric form. The diagonal is each block's own missing rate,
    and a pair whose off-diagonal equals both diagonals is two blocks that are missing together
    and could be one wider block if the exact-match rule were relaxed.
    """
    representatives = [b["columns"][0] for b in blocks]
    stacked = np.column_stack([mask[c].to_numpy(dtype=bool) for c in representatives])
    counts = stacked.T.astype("int64") @ stacked.astype("int64")
    n_rows = len(mask)
    return {
        "blocks": [
            {
                "index": i,
                "representative": r,
                "n_columns": blocks[i]["n_columns"],
                "missing_rate": blocks[i]["missing_rate"],
            }
            for i, r in enumerate(representatives)
        ],
        "n_rows": n_rows,
        "share_missing_in_both": (counts / n_rows).tolist(),
    }


def missingness_against_target(mask: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    """Fraud rate when a column is present against when it is missing, per column.

    Columns with no nulls are skipped, having nothing to compare. The MNAR flag is the pair of
    module thresholds: an absolute gap in fraud rate and a p value. Both have to clear, because
    at this row count the test alone flags gaps too small to act on.
    """
    labels = target.to_numpy(dtype="int64")
    total_fraud = int(labels.sum())
    n_rows = int(labels.size)
    rows: list[dict[str, Any]] = []
    for column in mask.columns:
        missing = mask[column].to_numpy(dtype=bool)
        n_missing = int(missing.sum())
        if n_missing == 0 or n_missing == n_rows:
            continue
        k_missing = int(labels[missing].sum())
        n_present = n_rows - n_missing
        k_present = total_fraud - k_missing
        rate_missing = k_missing / n_missing
        rate_present = k_present / n_present
        test = two_proportion_test(k_missing, n_missing, k_present, n_present)
        gap = rate_missing - rate_present
        p_value = test["p_value"]
        rows.append(
            {
                "column": column,
                "n_present": n_present,
                "n_fraud_present": k_present,
                "fraud_rate_present": rate_present,
                "n_missing": n_missing,
                "n_fraud_missing": k_missing,
                "fraud_rate_missing": rate_missing,
                "gap": gap,
                "risk_ratio": (rate_missing / rate_present) if rate_present > 0 else None,
                "z": test["z"],
                "p_value": p_value,
                "label_linked": bool(
                    abs(gap) >= MNAR_ABS_GAP and p_value is not None and p_value < MNAR_P_VALUE
                ),
            }
        )
    rows.sort(key=lambda r: -abs(float(r["gap"])))
    return rows


# --- correlation and redundancy ------------------------------------------------------------


def rank_matrix(frame: pd.DataFrame) -> npt.NDArray[np.float32]:
    """Per-column average ranks scaled to (0, 1], nulls left as NaN.

    Held at float32 because the full train frame of ranks at float64 is 1.3 GB on this dataset.
    The accumulation in `spearman_matrix` upcasts each chunk to float64, so the width here costs
    storage precision only, and ranks scaled into the unit interval have far more float32
    precision than the seven significant figures a correlation is read to.
    """
    out = np.empty((len(frame), frame.shape[1]), dtype="float32")
    for position, column in enumerate(frame.columns):
        out[:, position] = frame[column].rank(method="average", pct=True).to_numpy(dtype="float32")
    return out


def spearman_matrix(
    ranks: npt.NDArray[np.float32], chunk_rows: int = 50_000
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Spearman correlation from a rank matrix, accumulated over row chunks.

    pandas' own `DataFrame.corr(method="spearman")` re-ranks each pair on that pair's complete
    cases, which is a quadratic number of ranking passes over 413,378 rows and does not finish in
    a useful time on 400 columns. This ranks each column once over the whole split and then
    correlates each pair over the rows where both are present, from four cross-product matrices:
    the pairwise count, the pairwise sums, the pairwise sums of squares and the cross term. Each
    is a column-by-column matrix, so memory is a function of the column count and not the row
    count, and the row chunk keeps the working copy bounded.

    **This is not identical to pandas.** Where two columns are missing on different rows, ranking
    once over the column and ranking again inside the pair give slightly different numbers.
    Columns that share a missingness mask, which is most of the V block, agree exactly.
    scripts/eda.py measures the gap on a random sample of pairs and records it in
    reports/eda/correlation.json rather than asserting it is small.

    Returns the correlation matrix and the pairwise complete-case counts, so a reader can see
    which entries rest on few rows.
    """
    n_columns = ranks.shape[1]
    count = np.zeros((n_columns, n_columns), dtype="float64")
    sum_x = np.zeros((n_columns, n_columns), dtype="float64")
    sum_xx = np.zeros((n_columns, n_columns), dtype="float64")
    sum_xy = np.zeros((n_columns, n_columns), dtype="float64")

    for start in range(0, ranks.shape[0], chunk_rows):
        block = ranks[start : start + chunk_rows].astype("float64")
        present = np.isfinite(block)
        filled = np.where(present, block, 0.0)
        indicator = present.astype("float64")
        count += indicator.T @ indicator
        sum_x += filled.T @ indicator
        sum_xx += (filled * filled).T @ indicator
        sum_xy += filled.T @ filled

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = sum_x / count
        # Variance of x over the rows where both x and y are present. Not symmetric: var_x[i, j]
        # is column i's variance on the complete cases of the pair, which is what Spearman on
        # pairwise-complete data asks for.
        var_x = sum_xx / count - mean_x * mean_x
        covariance = sum_xy / count - mean_x * mean_x.T
        rho = covariance / np.sqrt(var_x * var_x.T)

    rho = np.clip(np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    np.fill_diagonal(rho, 1.0)
    return rho, count.astype("int64")


def screened_pairs(
    names: Sequence[str], rho: npt.NDArray[np.float64], threshold: float
) -> list[tuple[str, str]]:
    """Column pairs the screen matrix puts at or above |rho| = threshold, strongest first."""
    upper = np.triu_indices(len(names), k=1)
    magnitude = np.abs(rho[upper])
    selected = np.flatnonzero(magnitude >= threshold)
    order = selected[np.argsort(-magnitude[selected])]
    return [(names[int(upper[0][k])], names[int(upper[1][k])]) for k in order]


def exact_spearman(frame: pd.DataFrame, pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """pandas' own pairwise-complete Spearman, one pair at a time, keyed "a|b".

    Slow per pair and correct per pair. `spearman_matrix` is the screen that decides which pairs
    are worth paying for.
    """
    out: dict[str, float] = {}
    for a, b in pairs:
        value = frame[[a, b]].corr(method="spearman").to_numpy()[0, 1]
        if math.isfinite(float(value)):
            out[f"{a}|{b}"] = float(value)
    return out


def compare_against_pandas(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    rho: npt.NDArray[np.float64],
    index_of: dict[str, int],
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """How far the screen matrix sits from pandas on a given set of pairs.

    Measured rather than assumed. The two agree to float32 rank precision on a pair whose two
    columns share a missingness mask, and can differ materially on a pair whose masks differ,
    which is why the script measures those two cases separately.
    """
    differences: list[dict[str, Any]] = []
    for a, b in pairs:
        reference = frame[[a, b]].corr(method="spearman").to_numpy()[0, 1]
        if not math.isfinite(float(reference)):
            continue
        ours = float(rho[index_of[a], index_of[b]])
        differences.append(
            {
                "a": a,
                "b": b,
                "screen": ours,
                "pandas": float(reference),
                "abs_difference": abs(ours - float(reference)),
            }
        )
    if not differences:
        return {"n_pairs_compared": 0}
    gaps = np.array([d["abs_difference"] for d in differences])
    return {
        "n_pairs_compared": len(differences),
        "tolerance": tolerance,
        "max_abs_difference": float(gaps.max()),
        "median_abs_difference": float(np.median(gaps)),
        "n_pairs_within_tolerance": int((gaps < tolerance).sum()),
        "worst_pair": max(differences, key=lambda d: float(d["abs_difference"])),
    }


def correlated_groups(
    names: Sequence[str],
    rho: npt.NDArray[np.float64],
    threshold: float,
    priority: Sequence[float],
) -> list[dict[str, Any]]:
    """Greedy single-representative grouping at |rho| >= threshold.

    Columns are visited in `priority` order, best first. The first unassigned column opens a
    group and becomes its representative; every remaining unassigned column correlated with it
    above the threshold joins. This is not a clustering with any optimality claim, and it is not
    transitive: a column can correlate above the threshold with a member of a group it did not
    join. It is the reduction a person would apply by hand, made reproducible, and the artifact
    reports the group memberships so the arbitrariness is visible.
    """
    order = sorted(range(len(names)), key=lambda i: -priority[i])
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []
    for i in order:
        if i in assigned:
            continue
        members = [i]
        assigned.add(i)
        for j in order:
            if j in assigned:
                continue
            if abs(rho[i, j]) >= threshold:
                members.append(j)
                assigned.add(j)
        groups.append(
            {
                "representative": names[i],
                "n_members": len(members),
                "members": [names[m] for m in members],
                "min_abs_rho_to_representative": (
                    float(min(abs(rho[i, m]) for m in members[1:])) if len(members) > 1 else None
                ),
            }
        )
    groups.sort(key=lambda g: -int(g["n_members"]))
    return groups


def near_duplicate_pairs(
    exact: dict[str, float],
    counts: npt.NDArray[np.int64],
    index_of: dict[str, int],
    threshold: float,
    limit: int,
    exclude_prefix: str | None = None,
) -> dict[str, Any]:
    """Column pairs at or above |rho| = threshold, strongest first, from exact correlations.

    `exact` holds only the pairs the screen sent for exact computation, so a pair the screen
    missed cannot appear here. The screen threshold and the measured screen-to-exact gap both go
    into the artifact, so the size of that blind spot is visible rather than implied.
    """
    rows: list[dict[str, Any]] = []
    for key, value in exact.items():
        a, b = key.split("|", 1)
        if exclude_prefix is not None and (
            a.startswith(exclude_prefix) or b.startswith(exclude_prefix)
        ):
            continue
        if abs(value) >= threshold:
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "rho": value,
                    "n_complete_pairs": int(counts[index_of[a], index_of[b]]),
                }
            )
    rows.sort(key=lambda r: -abs(float(r["rho"])))
    return {
        "threshold": threshold,
        "n_pairs": len(rows),
        "listed": min(len(rows), limit),
        "pairs": rows[:limit],
    }


# --- bivariate signal ------------------------------------------------------------------------


def _bin_edges(values: pd.Series, n_bins: int) -> npt.NDArray[np.float64]:
    """Quantile cut points on the non-null values, duplicates dropped.

    A column where one value covers more than a bin's worth of rows produces fewer bins than
    asked for. That is the honest outcome and the artifact records the bin count it got.
    """
    clean = values.dropna().astype("float64")
    if clean.empty:
        return np.array([], dtype="float64")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(clean.to_numpy(), quantiles)).astype("float64")
    if edges.size < 2:
        return np.array([], dtype="float64")
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def bin_series(values: pd.Series, edges: npt.NDArray[np.float64]) -> pd.Series:
    """Assign each value to a bin label, with nulls in a bin of their own."""
    if edges.size < 2:
        return pd.Series(
            np.where(values.isna(), MISSING_LEVEL, "(single bin)"),
            index=values.index,
            dtype="object",
        )
    codes = pd.cut(
        values.to_numpy(dtype="float64"), bins=edges.tolist(), labels=False, include_lowest=True
    )
    labelled = pd.Series(codes, index=values.index).map(
        lambda c: None if pd.isna(c) else f"b{int(c):02d}"
    )
    return labelled.where(labelled.notna(), MISSING_LEVEL).astype("object")


def _as_bins(values: pd.Series, n_bins: int, edges: npt.NDArray[np.float64] | None) -> pd.Series:
    """Bin a column whichever way its dtype calls for: quantiles if numeric, levels if not."""
    if isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object:
        out = values.astype("object")
        return out.where(out.notna(), MISSING_LEVEL)
    if edges is None:
        edges = _bin_edges(values, n_bins)
    return bin_series(values, edges)


def information_value(
    values: pd.Series, target: pd.Series, n_bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """Information value of one column against the label, missing treated as a level.

    IV is the sum over bins of (share of non-fraud minus share of fraud) times the log of their
    ratio. A bin holding none of one class is smoothed by IV_SMOOTHING events rather than
    dropped, because dropping it would quietly reward a column for having an empty bin.

    Missing is a level and not an exclusion. Several columns here are missing on most rows and
    the missingness itself carries rate (see `missingness_against_target`), so an IV computed on
    present rows only would be measuring a different column from the one a model would see.
    """
    bins = _as_bins(values, n_bins, None)
    table = pd.crosstab(bins, target)
    for column in (0, 1):
        if column not in table.columns:
            table[column] = 0
    good = table[0].to_numpy(dtype="float64") + IV_SMOOTHING
    bad = table[1].to_numpy(dtype="float64") + IV_SMOOTHING
    share_good = good / good.sum()
    share_bad = bad / bad.sum()
    woe = np.log(share_good / share_bad)
    contributions = (share_good - share_bad) * woe
    levels = [str(level) for level in table.index]
    missing_contribution = float(
        contributions[levels.index(MISSING_LEVEL)] if MISSING_LEVEL in levels else 0.0
    )
    return {
        "iv": float(contributions.sum()),
        # IV is a sum over bins, so the missing bin's term subtracts out exactly. The remainder
        # is what the observed values carry once the fact of being observed is set aside.
        "iv_missing_bin": missing_contribution,
        "iv_present_bins": float(contributions.sum()) - missing_contribution,
        "n_bins": int(table.shape[0]),
        "n_bins_requested": int(n_bins),
        "smoothing": IV_SMOOTHING,
    }


def decile_table(
    values: pd.Series, target: pd.Series, n_bins: int = DEFAULT_BINS
) -> list[dict[str, Any]]:
    """Rows per bin, fraud rate per bin and the bin's value range, in bin order.

    The point is shape. A column can carry a high IV from a monotone relationship or from one
    extreme bin, and the two imply different things for a linear model, so the table is reported
    alongside the scalar rather than instead of it.
    """
    edges = _bin_edges(values, n_bins)
    bins = _as_bins(values, n_bins, edges)
    numeric = values.astype("float64")
    grouped = pd.DataFrame({"bin": bins, "y": target.to_numpy(), "x": numeric.to_numpy()})
    rows: list[dict[str, Any]] = []
    for label, part in grouped.groupby("bin", observed=True):
        n = len(part)
        k = _int(part["y"].sum())
        finite = part["x"].dropna()
        rows.append(
            {
                "bin": str(label),
                "n": n,
                "n_fraud": k,
                "fraud_rate": k / n,
                "x_min": _finite(_float(finite.min())) if len(finite) else None,
                "x_max": _finite(_float(finite.max())) if len(finite) else None,
                "interval": wilson_interval(k, n),
            }
        )
    rows.sort(key=lambda r: str(r["bin"]))
    return rows


# --- drift ------------------------------------------------------------------------------------


def population_stability_index(
    train_values: pd.Series, other_values: pd.Series, n_bins: int = DEFAULT_BINS
) -> dict[str, Any]:
    """PSI of a later split against train, binned on train.

    This is one of the two functions in the module that deliberately sees rows outside the
    training window. Binning on train and scoring the later split against those bins is the whole
    point: bins recomputed on the later split would move with the distribution and report no
    drift at all.

    A bin empty in one distribution is floored at PSI_FLOOR rather than dropped, which bounds a
    single empty bin's contribution instead of letting it be infinite or invisible.
    """
    is_categorical = (
        isinstance(train_values.dtype, pd.CategoricalDtype) or train_values.dtype == object
    )
    edges = None if is_categorical else _bin_edges(train_values, n_bins)
    train_bins = _as_bins(train_values, n_bins, edges)
    other_bins = _as_bins(other_values, n_bins, edges)

    expected = train_bins.value_counts(normalize=True)
    actual = other_bins.value_counts(normalize=True)
    levels = sorted(set(expected.index) | set(actual.index), key=str)
    e = np.array([float(expected.get(level, 0.0)) for level in levels])
    a = np.array([float(actual.get(level, 0.0)) for level in levels])
    e = np.clip(e, PSI_FLOOR, None)
    a = np.clip(a, PSI_FLOOR, None)
    terms = (a - e) * np.log(a / e)
    return {
        "psi": float(terms.sum()),
        "n_levels": len(levels),
        "n_new_levels": int(sum(1 for level in levels if level not in set(expected.index))),
        "floor": PSI_FLOOR,
    }


def time_consistency(
    train_values: pd.Series,
    train_target: pd.Series,
    later_values: pd.Series,
    later_target: pd.Series,
    seed: int,
    n_estimators: int = 50,
    num_leaves: int = 15,
    min_child_samples: int = 200,
) -> dict[str, Any]:
    """Fit a one-feature model on an early window and score it on a later one.

    The screen this implements: a feature whose single-feature model separates the classes in the
    early window and separates them the wrong way round in the later window has found a pattern
    that inverted. Correlation cannot see that, because a correlation measured over the whole
    training span averages the two halves and reports whatever is left.

    The model is a small gradient boosting fit on one column, which is close to a supervised
    binning of that column: with one feature the trees can only cut it, so the AUC reported is
    what an ordering of that column's values learned on the early window is worth on the later
    one. Both AUCs are reported. The early one is in-sample and is not a performance estimate;
    it is there so a feature with no early signal at all can be told apart from one whose signal
    inverted.

    Returns nulls rather than raising when a window holds one class or the column is all null,
    because over 400 columns several of those cases exist and the artifact should say so.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    result: dict[str, Any] = {
        "n_early": len(train_values),
        "n_late": len(later_values),
        "early_auc": None,
        "late_auc": None,
        "status": "ok",
    }
    if train_values.notna().sum() == 0 or later_values.notna().sum() == 0:
        result["status"] = "all null in one window"
        return result
    if train_target.nunique() < 2 or later_target.nunique() < 2:
        result["status"] = "one class in one window"
        return result

    name = str(train_values.name)
    x_early = train_values.to_frame(name)
    x_late = later_values.to_frame(name)
    if isinstance(train_values.dtype, pd.CategoricalDtype):
        # The later window is recoded onto the earlier window's category set. A level the early
        # window never saw becomes missing, which is the honest answer: the model has no
        # information about it, and letting it inherit some other level's code would score it as
        # whatever category happens to sit in that position.
        categories = train_values.cat.categories
        as_object = later_values.astype("object")
        x_late[name] = pd.Categorical(
            as_object.where(as_object.isin(categories)), categories=categories
        )

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        learning_rate=0.1,
        random_state=seed,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(x_early, train_target.to_numpy())
    early_score = np.asarray(model.predict_proba(x_early))[:, 1]
    late_score = np.asarray(model.predict_proba(x_late))[:, 1]
    result["early_auc"] = float(roc_auc_score(train_target.to_numpy(), early_score))
    result["late_auc"] = float(roc_auc_score(later_target.to_numpy(), late_score))
    return result


# --- class balance and what it allows -----------------------------------------------------


def hanley_mcneil_se(auc: float, n_positive: int, n_negative: int) -> float:
    """Standard error of an AUC estimate at a given class count, Hanley and McNeil's formula.

    Q1 and Q2 are the exponential approximations that formula uses for the probabilities that two
    positives or two negatives both rank below a case of the other class. They are an assumption
    about the score distribution, not a measurement of one, and the artifact says so.
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    numerator = (
        auc * (1 - auc) + (n_positive - 1) * (q1 - auc * auc) + (n_negative - 1) * (q2 - auc * auc)
    )
    return math.sqrt(numerator / (n_positive * n_negative))


def detectable_auc_difference(
    auc: float,
    n_positive: int,
    n_negative: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, Any]:
    """The AUC gap two independent evaluations of this size could resolve.

    Upper bound, and deliberately so. Two models compared on the same rows produce correlated AUC
    estimates and the paired standard error is smaller by a factor that depends on a correlation
    this repo has not measured, so the number here is what an unpaired comparison would need.
    The paired figure belongs to stage 6, after run-to-run variance is measured.
    """
    z_alpha = float(stats.norm.isf(alpha / 2))
    z_beta = float(stats.norm.isf(1 - power))
    se = hanley_mcneil_se(auc, n_positive, n_negative)
    return {
        "assumed_auc": auc,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "se_auc": se,
        "alpha": alpha,
        "power": power,
        "min_detectable_difference_unpaired": (z_alpha + z_beta) * se * math.sqrt(2.0),
        "note": "unpaired upper bound; a paired comparison on the same rows resolves less",
    }


def recall_precision_of_positives(n_positive: int, recall: float) -> dict[str, Any]:
    """Binomial standard error and 95 percent width of a recall read off this many positives."""
    se = math.sqrt(recall * (1 - recall) / n_positive) if n_positive else float("nan")
    return {
        "n_positive": n_positive,
        "assumed_recall": recall,
        "se": _finite(se),
        "half_width_95": _finite(1.959963984540054 * se),
    }


# --- entity structure -------------------------------------------------------------------


def entity_structure(frame: pd.DataFrame, entity: str, target: str, time: str) -> dict[str, Any]:
    """Transactions per entity, within-entity label purity, and inter-transaction gaps.

    Label purity is revisited from stage 0 at the level of the distribution rather than the
    three-way split into pure fraud, pure non-fraud and mixed: the mixed entities are the
    interesting ones and stage 0 only counted them.
    """
    grouped = frame.groupby(entity, observed=True)[target].agg(["size", "sum"])
    size = grouped["size"].to_numpy(dtype="float64")
    rate = (grouped["sum"] / grouped["size"]).to_numpy(dtype="float64")
    multi = size > 1
    mixed = multi & (rate > 0) & (rate < 1)

    ordered = frame[[entity, time]].sort_values([entity, time], kind="stable")
    gaps = ordered.groupby(entity, observed=True)[time].diff().dropna()
    entity_has_fraud = grouped["sum"] > 0
    gap_entities = ordered.loc[gaps.index, entity]
    fraud_gaps = gaps[gap_entities.map(entity_has_fraud).to_numpy(dtype=bool)]
    clean_gaps = gaps[~gap_entities.map(entity_has_fraud).to_numpy(dtype=bool)]

    def describe(values: npt.NDArray[np.float64]) -> dict[str, Any]:
        if values.size == 0:
            return {"n": 0}
        return {
            "n": int(values.size),
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
            "percentiles": {
                f"p{int(q * 100):02d}": float(np.quantile(values, q)) for q in PERCENTILES
            },
        }

    return {
        "n_entities": len(grouped),
        "n_rows": len(frame),
        "transactions_per_entity": describe(size),
        "singleton_entity_share": float((size == 1).mean()),
        "singleton_row_share": float(size[size == 1].sum() / size.sum()),
        "label_purity": {
            "n_entities_multi": int(multi.sum()),
            "share_all_fraud": float((rate[multi] == 1).mean()) if multi.any() else None,
            "share_all_clean": float((rate[multi] == 0).mean()) if multi.any() else None,
            "n_entities_mixed": int(mixed.sum()),
            "share_mixed": float(mixed.sum() / multi.sum()) if multi.any() else None,
            "mixed_rate_percentiles": (
                {f"p{int(q * 100):02d}": float(np.quantile(rate[mixed], q)) for q in PERCENTILES}
                if mixed.any()
                else None
            ),
            "n_entities_any_fraud": int((grouped["sum"] > 0).sum()),
        },
        "seconds_between_transactions": {
            "all": describe(gaps.to_numpy(dtype="float64")),
            "entities_with_fraud": describe(fraud_gaps.to_numpy(dtype="float64")),
            "entities_without_fraud": describe(clean_gaps.to_numpy(dtype="float64")),
        },
    }


def cold_warm_within(frame: pd.DataFrame, entity: str, target: str, time: str) -> dict[str, Any]:
    """Fraud rate on an entity's first transaction against its later ones, inside one split.

    Cold here means first appearance within this frame. The cross-split version, where cold means
    an entity the training window never saw, is measured in reports/split_summary.json and is not
    recomputed here so that each number has one source.
    """
    ordered = frame[[entity, time, target]].sort_values([entity, time], kind="stable")
    first = ~ordered.duplicated(subset=[entity], keep="first").to_numpy(dtype=bool)
    labels = ordered[target].to_numpy(dtype="int64")
    n_cold, n_warm = int(first.sum()), int((~first).sum())
    k_cold, k_warm = int(labels[first].sum()), int(labels[~first].sum())
    return {
        "n_rows": len(ordered),
        "cold": {
            "n_rows": n_cold,
            "share_of_rows": n_cold / len(ordered),
            "n_fraud": k_cold,
            "fraud_rate": k_cold / n_cold if n_cold else None,
            "interval": wilson_interval(k_cold, n_cold),
        },
        "warm": {
            "n_rows": n_warm,
            "share_of_rows": n_warm / len(ordered),
            "n_fraud": k_warm,
            "fraud_rate": k_warm / n_warm if n_warm else None,
            "interval": wilson_interval(k_warm, n_warm),
        },
        "definition": "cold is an entity's first row inside this frame, warm is any later row",
        "cross_split_source": "reports/split_summary.json entity_overlap",
    }


# --- duplicates and data quality ---------------------------------------------------------


def duplicate_groups(frame: pd.DataFrame, subset: Sequence[str], target: str) -> dict[str, Any]:
    """Rows identical across `subset`, and whether they carry a different fraud rate.

    Rows are bucketed by a 64 bit hash of the subset and then compared value by value inside each
    bucket, so a hash collision cannot merge two rows that differ. The exact comparison puts
    nulls into a sentinel of their own first, because two rows null in the same hundred V columns
    and equal everywhere else are the same content, and a raw tuple comparison would call them
    different.
    """
    columns = [c for c in subset if c in frame.columns]
    hashes = pd.util.hash_pandas_object(frame[columns], index=False)
    duplicated_hash = hashes.duplicated(keep=False).to_numpy(dtype=bool)

    positions = np.flatnonzero(duplicated_hash)
    n_groups = 0
    member_positions: list[int] = []
    if positions.size:
        block = frame.iloc[positions][columns].astype("object")
        # copy=True because to_numpy can hand back a read-only view of the block's own buffer,
        # and the sentinel is written into this array in place.
        values = block.to_numpy(dtype=object, copy=True)
        values[pd.isna(block).to_numpy(dtype=bool)] = _NULL_SENTINEL
        seen: dict[tuple[Any, ...], list[int]] = {}
        for row, position in zip(values, positions, strict=True):
            seen.setdefault(tuple(row), []).append(int(position))
        for members in seen.values():
            if len(members) > 1:
                n_groups += 1
                member_positions.extend(members)

    labels = frame[target].to_numpy(dtype="int64")
    in_group = np.zeros(len(frame), dtype=bool)
    in_group[member_positions] = True
    n_in, n_out = int(in_group.sum()), int((~in_group).sum())
    k_in, k_out = int(labels[in_group].sum()), int(labels[~in_group].sum())
    return {
        "n_columns_compared": len(columns),
        "n_rows": len(frame),
        "n_groups": n_groups,
        "n_rows_in_groups": n_in,
        "share_rows_in_groups": n_in / len(frame),
        "n_extra_rows": n_in - n_groups,
        "fraud_rate_in_groups": k_in / n_in if n_in else None,
        "fraud_rate_outside_groups": k_out / n_out if n_out else None,
        "interval_in_groups": wilson_interval(k_in, n_in),
        "interval_outside_groups": wilson_interval(k_out, n_out),
    }


# --- a count feature against the label, per split ------------------------------------------------


def zero_against_positive(
    values: pd.Series, labels: npt.NDArray[np.int64], mask: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    """Fraud rate on the rows where a count is positive against the rows where it is zero.

    This exists because decile information value is the wrong instrument for a zero-inflated
    count, and it reports a number that reads as "carries nothing" when the column carries
    plenty. Stage 4 measured `dup_prior_count_1h` at zero on 0.9233 of train rows, so ten
    equal-frequency bins collapse into one and the information value comes back at 0.0000 while
    the positive rows run at nearly three times the base rate. Both numbers go into the artifacts,
    with this one beside the binning diagnostics that explain the other.

    Written in stage 4's driver and moved here in stage 5 because the graph features are counts
    too and are judged by the same comparison.
    """
    array = values.to_numpy(dtype="float64")
    positive = mask & (array > 0)
    zero = mask & (array == 0)
    n_positive, n_zero = int(positive.sum()), int(zero.sum())
    rate_positive = float(labels[positive].mean()) if n_positive else None
    rate_zero = float(labels[zero].mean()) if n_zero else None
    return {
        "n_positive": n_positive,
        "n_zero": n_zero,
        "share_positive": n_positive / int(mask.sum()) if mask.sum() else None,
        "n_distinct_train": int(np.unique(array[mask]).size),
        "fraud_rate_positive": rate_positive,
        "fraud_rate_zero": rate_zero,
        "interval_positive": wilson_interval(int(labels[positive].sum()), n_positive),
        "interval_zero": wilson_interval(int(labels[zero].sum()), n_zero),
        "lift": (
            rate_positive / rate_zero
            if rate_positive is not None and rate_zero not in (None, 0.0)
            else None
        ),
    }


def zero_against_positive_by_split(
    values: pd.Series,
    labels: npt.NDArray[np.int64],
    masks: Mapping[str, pd.Series],
) -> dict[str, Any]:
    """`zero_against_positive` on each split, so a train-only separation cannot be reported alone.

    This block exists because the first version of stage 4 reported the duplicate-content lift
    from the train split and nothing else, and the train split is not where the question is
    settled. ADR 0009 set the precedent: a relationship learned from one window is a candidate
    until a later window has seen it. Every separation is therefore reported per split, whether
    or not it holds, and the three flags at the end are the ship rule ADR 0022 wrote down.
    """
    out: dict[str, Any] = {}
    for name, mask in masks.items():
        block = zero_against_positive(values, labels, mask.to_numpy(dtype=bool))
        out[name] = block
    rates = [
        (name, block["fraud_rate_positive"], block["fraud_rate_zero"])
        for name, block in out.items()
        if block["fraud_rate_positive"] is not None and block["fraud_rate_zero"] is not None
    ]
    out["definitions"] = {
        "positive_is_riskier": (
            "the positive side of the count carries the higher fraud rate on every split. False "
            "for a column whose zero side is the risky one, which is an ordinary inverse "
            "relationship and not a fault"
        ),
        "direction_stable": (
            "the sign of (rate positive minus rate zero) is the same on all three splits. This is "
            "the instability test. A column can be stably inverse and pass it; a column whose "
            "marginal relationship flips between the training window and the later ones fails it, "
            "and that is the case ADR 0009 said to treat as a candidate rather than a result"
        ),
        "disjoint_on_every_split": (
            "the two Wilson intervals do not overlap on any split, so the separation is larger "
            "than the split resolves wherever it is measured"
        ),
    }
    out["positive_is_riskier"] = bool(rates) and all(positive > zero for _, positive, zero in rates)
    signs = {1 if positive > zero else -1 for _, positive, zero in rates}
    out["direction_stable"] = bool(rates) and len(signs) == 1
    out["disjoint_on_every_split"] = bool(rates) and all(
        out[name]["interval_positive"]["low"] > out[name]["interval_zero"]["high"]
        or out[name]["interval_zero"]["low"] > out[name]["interval_positive"]["high"]
        for name, _, _ in rates
    )
    return out
