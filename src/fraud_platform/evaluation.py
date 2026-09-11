"""Metrics, thresholds, bands and intervals for stage 6.

The primary metric is PR-AUC, computed as average precision. ADR 0025 has the measurement
behind the choice; this module only computes it. Every function here takes label and score
vectors and returns plain dictionaries, so the driver can write what it computed and a test can
recompute it from the same inputs.

Three things are fitted here and none of them on the training split: a threshold, a calibrator
and a band edge. All three are read off the validation split, which is the split model selection
is allowed to see, and the test split is scored once with them fixed. The driver enforces which
split each function receives; this module records which one it was given.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from fraud_platform import config, eda

# The sweep item 7 of the stage brief asks for: one hundredth steps from 0.01 to 0.99.
SWEEP_THRESHOLDS: tuple[float, ...] = tuple(round(0.01 * i, 2) for i in range(1, 100))

# Bins for the reliability table. Chosen.
RELIABILITY_BINS = 10

BAND_NAMES: tuple[str, ...] = ("approve", "review", "block")


def pr_auc(y: npt.NDArray[np.int_], score: npt.NDArray[np.float64]) -> float:
    """Average precision: the area under the precision-recall curve as sklearn computes it,
    the precision at each recall step weighted by the recall gained, with no interpolation."""
    return float(average_precision_score(y, score))


def roc_auc(y: npt.NDArray[np.int_], score: npt.NDArray[np.float64]) -> float:
    return float(roc_auc_score(y, score))


def span_days(ts: npt.NDArray[np.int64]) -> float:
    """Days a split covers, from its first timestamp to its last. Never below one day."""
    if ts.size == 0:
        return 1.0
    return max(1.0, float(ts.max() - ts.min()) / config.SECONDS_PER_DAY)


def at_threshold(
    y: npt.NDArray[np.int_],
    score: npt.NDArray[np.float64],
    threshold: float,
    n_days: float = 1.0,
) -> dict[str, Any]:
    """Confusion matrix and the rates read from it, alerting on `score >= threshold`."""
    y = np.asarray(y, dtype="int64")
    alert = np.asarray(score) >= threshold
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    fn = int(np.sum(~alert & (y == 1)))
    tn = int(np.sum(~alert & (y == 0)))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alerts": tp + fp,
        "alert_rate": (tp + fp) / y.size if y.size else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "alerts_per_day": (tp + fp) / n_days,
        "false_alarms_per_catch": fp / tp if tp else None,
    }


def threshold_sweep(
    y: npt.NDArray[np.int_],
    score: npt.NDArray[np.float64],
    n_days: float,
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
) -> list[dict[str, Any]]:
    return [at_threshold(y, score, t, n_days) for t in thresholds]


def best_f1_vertex(
    y: npt.NDArray[np.int_], score: npt.NDArray[np.float64], n_days: float
) -> dict[str, Any]:
    """The vertex of the precision-recall curve with the largest F1, at full precision.

    The curve has one vertex per distinct score, so the threshold returned is an actual score
    value rather than a grid point, and the confusion matrix at it is recomputed from the rows
    rather than read off the curve so both agree by construction.
    """
    precision, recall, thresholds = precision_recall_curve(y, score)
    # The last point of the curve has no threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    best = int(np.argmax(f1))
    return {
        "rule": "largest F1 on the validation split",
        **at_threshold(y, score, float(thresholds[best]), n_days),
    }


def alert_budget_vertex(
    y: npt.NDArray[np.int_],
    score: npt.NDArray[np.float64],
    n_days: float,
    alerts_per_day: float,
) -> dict[str, Any]:
    """The threshold that produces a given alert volume per day, and what it catches."""
    n_alerts = round(alerts_per_day * n_days)
    n_alerts = min(max(n_alerts, 1), int(np.asarray(score).size))
    ordered = np.sort(np.asarray(score))[::-1]
    return {
        "rule": f"{alerts_per_day:g} alerts per day on the validation split",
        **at_threshold(y, score, float(ordered[n_alerts - 1]), n_days),
    }


# --- card level ------------------------------------------------------------------------------


def card_level(
    y: npt.NDArray[np.int_], score: npt.NDArray[np.float64], keys: npt.NDArray[Any]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Collapse rows to one per key: the label is any fraud on the key, the score is the maximum.

    Labels in this file are card-level (ADR 0001 measured the entity's label purity), so a
    transaction metric counts a card once per transaction and a card with many rows dominates.
    The maximum is the score a queue sees when it works the card rather than the row.
    """
    frame = pd.DataFrame({"key": keys, "y": np.asarray(y), "s": np.asarray(score)})
    grouped = frame.groupby("key", sort=False).agg(y=("y", "max"), s=("s", "max"))
    return grouped["y"].to_numpy(dtype="int64"), grouped["s"].to_numpy(dtype="float64")


# --- paired bootstrap --------------------------------------------------------------------------


def _interval(draws: npt.NDArray[np.float64], point: float, alpha: float) -> dict[str, Any]:
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "point": float(point),
        "low": float(low),
        "high": float(high),
        "half_width": float((high - low) / 2),
        "bootstrap_mean": float(draws.mean()),
    }


def paired_bootstrap(
    y: npt.NDArray[np.int_],
    scores: Mapping[str, npt.NDArray[np.float64]],
    n_boot: int,
    seed: int,
    groups: npt.NDArray[Any] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """PR-AUC intervals for several fixed models on the same resamples of one split.

    Every model is scored on the identical resample, so the differences between them are paired
    and their intervals are narrower than two independent intervals would suggest. With `groups`
    the resample is of groups rather than rows: whole cards are drawn with replacement, which is
    the resample that respects card-level labels. `sign_agreement` is the share of resamples on
    which a difference has the same sign as its point estimate.
    """
    y = np.asarray(y, dtype="int64")
    names = list(scores)
    stacked = np.stack([np.asarray(scores[name], dtype="float64") for name in names])
    rng = np.random.default_rng(seed)
    n = y.size

    if groups is not None:
        codes, uniques = pd.factorize(np.asarray(groups), sort=False)
        order = np.argsort(codes, kind="stable")
        starts = np.searchsorted(codes[order], np.arange(uniques.size))
        ends = np.append(starts[1:], n)

    draws = np.empty((n_boot, len(names)), dtype="float64")
    n_positive = np.empty(n_boot, dtype="int64")
    for b in range(n_boot):
        if groups is None:
            index = rng.integers(0, n, n)
        else:
            picked = rng.integers(0, uniques.size, uniques.size)
            index = np.concatenate([order[starts[g] : ends[g]] for g in picked])
        y_b = y[index]
        n_positive[b] = int(y_b.sum())
        if n_positive[b] == 0 or n_positive[b] == y_b.size:
            draws[b] = np.nan
            continue
        for j in range(len(names)):
            draws[b, j] = average_precision_score(y_b, stacked[j, index])

    valid = ~np.isnan(draws).any(axis=1)
    draws = draws[valid]
    points = {name: pr_auc(y, stacked[j]) for j, name in enumerate(names)}
    per_model = {name: _interval(draws[:, j], points[name], alpha) for j, name in enumerate(names)}
    pairs: list[dict[str, Any]] = []
    for (i, first), (k, second) in itertools.combinations(enumerate(names), 2):
        diff = draws[:, i] - draws[:, k]
        point = points[first] - points[second]
        sign = np.sign(point)
        agreement = float(np.mean(np.sign(diff) == sign)) if sign != 0 else None
        pairs.append(
            {
                "first": first,
                "second": second,
                "difference": _interval(diff, point, alpha),
                "sign_agreement": agreement,
                "excludes_zero": bool(
                    np.quantile(diff, alpha / 2) > 0 or np.quantile(diff, 1 - alpha / 2) < 0
                ),
            }
        )
    widest = max(pairs, key=lambda p: p["difference"]["half_width"]) if pairs else None
    return {
        "metric": "pr_auc",
        "n_boot": int(n_boot),
        "n_valid": int(valid.sum()),
        "seed": int(seed),
        "alpha": alpha,
        "resample": "rows" if groups is None else "groups",
        "n_rows": int(n),
        "n_groups": int(uniques.size) if groups is not None else None,
        "positives_per_resample": {
            "min": int(n_positive[valid].min()),
            "median": float(np.median(n_positive[valid])),
            "max": int(n_positive[valid].max()),
        },
        "per_model": per_model,
        "pairs": pairs,
        "widest_paired_half_width": (
            {
                "pair": [widest["first"], widest["second"]],
                "half_width": widest["difference"]["half_width"],
            }
            if widest
            else None
        ),
    }


# --- calibration --------------------------------------------------------------------------------


def fit_calibrator(score: npt.NDArray[np.float64], y: npt.NDArray[np.int_]) -> IsotonicRegression:
    """Isotonic regression from score to probability. Monotone, so it moves no ranking."""
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(np.asarray(score, dtype="float64"), np.asarray(y, dtype="float64"))
    return calibrator


def calibrate(
    calibrator: IsotonicRegression, score: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return np.asarray(calibrator.predict(np.asarray(score, dtype="float64")), dtype="float64")


def reliability(
    y: npt.NDArray[np.int_], p: npt.NDArray[np.float64], n_bins: int = RELIABILITY_BINS
) -> dict[str, Any]:
    """Observed rate against mean predicted probability, in equal-count bins, plus the Brier score."""
    y = np.asarray(y, dtype="int64")
    p = np.asarray(p, dtype="float64")
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, n_bins)
    rows = []
    for i, index in enumerate(bins):
        if index.size == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(index.size),
                "mean_predicted": float(p[index].mean()),
                "observed_rate": float(y[index].mean()),
                "low_score": float(p[index].min()),
                "high_score": float(p[index].max()),
            }
        )
    gap = float(sum(r["n"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in rows) / y.size)
    return {
        "n_bins": n_bins,
        "bins": rows,
        "brier": float(brier_score_loss(y, p)),
        "expected_calibration_error": gap,
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


# --- cost bands ----------------------------------------------------------------------------------


def cost_bands(
    cost_missed_fraud: float = config.COST_MISSED_FRAUD,
    cost_false_decline: float = config.COST_FALSE_DECLINE,
    cost_review: float = config.COST_REVIEW,
) -> dict[str, Any]:
    """Two band edges from three costs, for a calibrated probability p of fraud.

    Three actions and their expected cost at p, under the assumption that an analyst review
    resolves the transaction correctly: approve costs p times the missed-fraud cost, review
    costs the review cost whatever the outcome, block costs (1 - p) times the false-decline
    cost. Review beats approve once p exceeds review / missed, and block beats review once
    (1 - p) times decline falls below review, so the edges are

        low  = cost_review / cost_missed_fraud
        high = 1 - cost_review / cost_false_decline

    The middle band exists only when low < high, which needs the review to be cheap against
    both errors; the dictionary says whether it is. The costs are inputs and the artifact
    labels them so. The arithmetic is not.
    """
    for name, value in (
        ("cost_missed_fraud", cost_missed_fraud),
        ("cost_false_decline", cost_false_decline),
        ("cost_review", cost_review),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    low = cost_review / cost_missed_fraud
    high = 1.0 - cost_review / cost_false_decline
    single = cost_false_decline / (cost_false_decline + cost_missed_fraud)
    return {
        "costs": {
            "missed_fraud": cost_missed_fraud,
            "false_decline": cost_false_decline,
            "review": cost_review,
            "status": "assumption: chosen inputs, not measured anywhere in this repo",
            "unit": "one false decline",
        },
        "assumption_about_review": "an analyst review resolves the transaction correctly",
        "expected_cost_at_p": {
            "approve": "p * missed_fraud",
            "review": "review",
            "block": "(1 - p) * false_decline",
        },
        "low": low,
        "high": high,
        "middle_band_exists": low < high,
        "single_threshold_without_review": single,
        "derivation": (
            "review beats approve when review < p * missed_fraud, so low = review / missed_fraud; "
            "block beats review when (1 - p) * false_decline < review, so "
            "high = 1 - review / false_decline; with no review band, block beats approve when "
            "p > false_decline / (false_decline + missed_fraud)"
        ),
    }


def apply_bands(p: npt.NDArray[np.float64], low: float, high: float) -> npt.NDArray[np.int64]:
    """0 approve, 1 review, 2 block. Edges are inclusive on the riskier side."""
    p = np.asarray(p, dtype="float64")
    return np.where(p >= high, 2, np.where(p >= low, 1, 0)).astype("int64")


def band_report(
    y: npt.NDArray[np.int_],
    p: npt.NDArray[np.float64],
    bands: Mapping[str, Any],
    n_days: float,
) -> dict[str, Any]:
    """What each band holds on one split, and the realised cost of the policy against the
    two single-threshold policies it replaces."""
    y = np.asarray(y, dtype="int64")
    decision = apply_bands(p, bands["low"], bands["high"])
    costs = bands["costs"]
    rows: dict[str, dict[str, Any]] = {}
    counts: dict[str, tuple[int, int]] = {}
    for code, name in enumerate(BAND_NAMES):
        mask = decision == code
        n = int(mask.sum())
        fraud = int(y[mask].sum())
        counts[name] = (n, fraud)
        rows[name] = {
            "n": n,
            "share": n / y.size if y.size else None,
            "per_day": n / n_days,
            "fraud": fraud,
            "fraud_rate": fraud / n if n else None,
            "share_of_all_fraud": fraud / y.sum() if y.sum() else None,
        }
    missed = counts["approve"][1]
    false_declines = counts["block"][0] - counts["block"][1]
    realised = (
        missed * costs["missed_fraud"]
        + counts["review"][0] * costs["review"]
        + false_declines * costs["false_decline"]
    )

    def single(threshold: float) -> dict[str, Any]:
        point = at_threshold(y, p, threshold)
        cost = point["fn"] * costs["missed_fraud"] + point["fp"] * costs["false_decline"]
        return {"threshold": threshold, "cost": cost, "cost_per_row": cost / y.size, **point}

    approve_all = float(y.sum() * costs["missed_fraud"])
    return {
        "n_rows": int(y.size),
        "n_days": n_days,
        "bands": rows,
        "realised_cost": realised,
        "realised_cost_per_row": realised / y.size if y.size else None,
        "approve_everything_cost": approve_all,
        "approve_everything_cost_per_row": approve_all / y.size if y.size else None,
        "single_threshold_at_cost_optimum": single(bands["single_threshold_without_review"]),
    }


# --- segments -----------------------------------------------------------------------------------


def segment_report(
    segments: Mapping[str, dict[str, npt.NDArray[Any]]],
    n_days: Mapping[str, float],
    bands: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Per-segment base rates, the tuned vertex and the band edges, each split reported.

    `segments[name]` holds `y_val`, `s_val`, `y_test`, `s_test` for one level of the segment
    column. The threshold and the calibrator are fitted on the segment's validation rows and the
    test rows are scored with them fixed; a segment too small to calibrate is reported with the
    global edges and says so.
    """
    out = []
    for name, parts in segments.items():
        y_val, s_val = parts["y_val"], parts["s_val"]
        y_test, s_test = parts["y_test"], parts["s_test"]
        row: dict[str, Any] = {
            "segment": name,
            "base_rate": {
                split: eda.wilson_interval(int(y.sum()), int(y.size))
                for split, y in (("val", y_val), ("test", y_test))
            },
        }
        if y_val.sum() == 0 or y_val.sum() == y_val.size:
            row["note"] = "one class only on validation; nothing can be tuned here"
            out.append(row)
            continue
        vertex = best_f1_vertex(y_val, s_val, n_days["val"])
        row["tuned_vertex"] = {
            "val": vertex,
            "test": at_threshold(y_test, s_test, vertex["threshold"], n_days["test"]),
        }
        calibrator = fit_calibrator(s_val, y_val)
        p_val = calibrate(calibrator, s_val)
        p_test = calibrate(calibrator, s_test)
        row["calibration"] = {"fitted_on": "val", "n_val_positive": int(y_val.sum())}
        row["bands"] = {
            "val": band_report(y_val, p_val, bands, n_days["val"]),
            "test": band_report(y_test, p_test, bands, n_days["test"]),
        }
        row["pr_auc"] = {"val": pr_auc(y_val, s_val), "test": pr_auc(y_test, s_test)}
        out.append(row)
    return out
