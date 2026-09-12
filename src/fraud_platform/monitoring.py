"""Two monitoring jobs on two clocks: input drift without labels, performance decay with them.

**Drift** is measured the day a batch closes. It reads the model's input columns and its scores
against a reference stored from the training window, as a population stability index per column
and one for the score distribution, and it never sees a label. What it can say is that the inputs
moved; what it cannot say is whether the model got worse, because a shift can be harmless (new
cards, a browser version) or harmful (a unit change on a column the model splits on) and the
index does not distinguish them.

**Decay** is measured once the labels for a batch have matured, weeks after it closed. It reads
precision, recall and PR-AUC on that batch against the reference the model was accepted at.
What it can say is that the model got worse; what it cannot say is why, or say it early. ADR
0032 measured what a label read before it has matured does to a fit; the same label read into a
metric reports a decay that is not there, so the job refuses an immature batch outright.

The two are separate functions writing separate artifacts on separate schedules, and nothing in
this module joins them. The join is the lifecycle's (`lifecycle.should_retrain`), where decay is
the primary trigger and drift a secondary early one. ADR 0036.

**Bands.** PSI below 0.10 is stable, 0.10 to 0.20 watch, above 0.20 alert, per column. The bands
are reported as they are, and the alert decision is not taken on them alone: stage 2 measured 30
raw columns above 0.10 and 22 above 0.25 between train and test on this file with nothing wrong,
so a column's alert edge is the larger of 0.20 and the worst value it took on an in-control batch
of the accepted windows, and the batch flag needs more columns past their edges than any held-out
in-control batch showed. The target encodings are reported and never alert: a training row is
encoded at its own lagged day (ADR 0014), so their training-window histogram is the encoder's
clock rather than a population, and on in-control weeks they sit at PSI 0.7 to 11 against it
(`reports/monitoring/drift.json`, `calibration.edges_raised`). ADR 0037.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from fraud_platform import config, eda, encoders, evaluation

# The conventional PSI bands, reported per column. The alert decision is calibrated above them.
PSI_WATCH = 0.10
PSI_ALERT = 0.20
BANDS: tuple[str, str, str] = ("stable", "watch", "alert")

# Deciles of the training distribution, the stage 2 convention (`eda.DEFAULT_BINS`), so a PSI
# here on the whole test window reproduces the stage 2 number for the same column.
N_BINS = eda.DEFAULT_BINS

# One monitoring cycle is one batch. Seven days, chosen: a batch has to hold enough rows for a
# decile histogram of a sparse column, and a week is the grain the fraud rate was measured to
# move at in stage 2 (weekly range 0.0207 to 0.0507).
BATCH_DAYS = 7

SCORE_NAME = "score"

# Columns whose histogram against the training window is not a population statement and which
# therefore never raise an alert, however far they sit from their bins. See the module docstring.
NOT_ALERTABLE_SUFFIXES: tuple[str, ...] = (encoders.TARGET_SUFFIX,)
NOT_ALERTABLE_REASON = (
    "target encodings are computed at each training row's own lagged day, so the training "
    "histogram is the encoder's clock, not the population; reported, never alerting"
)


def alertable(column: str) -> bool:
    return not column.endswith(NOT_ALERTABLE_SUFFIXES)


# The score histogram is binned on the training-window scores. In-sample scores sit closer to
# 0 and 1 than out-of-sample ones do, so the score PSI against train is positive on every batch
# by construction, which is why it carries a calibrated edge like every column does.
SCORE_REFERENCE = "shipped model scores on the training split, in sample"


class MonitoringError(RuntimeError):
    """A job was handed something it must not read: a label into drift, an immature batch."""


# --- batches -------------------------------------------------------------------------------------


def batch_bounds(ts: npt.NDArray[np.int64], batch_days: int = BATCH_DAYS) -> list[dict[str, int]]:
    """Consecutive `[start, end)` windows of `batch_days` from the first timestamp's day.

    Assignment is `start <= dt < end`, the stage 0 rule, so a timestamp shared by several rows
    never straddles two batches. The last window is cut at the data's end and its `end` is
    where it would have closed, so a partial batch is visible as one.
    """
    if ts.size == 0:
        raise ValueError("no timestamps to batch")
    if batch_days < 1:
        raise ValueError("batch_days must be at least 1")
    step = batch_days * config.SECONDS_PER_DAY
    first = int(ts.min()) // config.SECONDS_PER_DAY * config.SECONDS_PER_DAY
    last = int(ts.max())
    out: list[dict[str, int]] = []
    start = first
    index = 0
    while start <= last:
        end = start + step
        mask = (ts >= start) & (ts < end)
        out.append(
            {
                "batch": index,
                "start_dt": start,
                "end_dt": end,
                "n_rows": int(mask.sum()),
                "partial": int(end > last + 1),
            }
        )
        start = end
        index += 1
    return out


def batch_mask(ts: npt.NDArray[np.int64], bounds: Mapping[str, int]) -> npt.NDArray[np.bool_]:
    return (ts >= int(bounds["start_dt"])) & (ts < int(bounds["end_dt"]))


def day_index(dt: int) -> float:
    """A TransactionDT as a day count, the unit the notes and artifacts talk in."""
    return dt / config.SECONDS_PER_DAY


# --- one column's histogram ----------------------------------------------------------------------------


@dataclass
class BinSpec:
    """A column's bins, fixed on the training window, and the share of training rows in each.

    Numeric columns carry the decile edges (interior only; the outer edges are infinite) and
    categorical columns carry their levels. Either way the reference is a label-to-share table
    with `(missing)` as its own level, exactly as `eda.population_stability_index` bins.
    """

    kind: str
    reference: dict[str, float]
    edges: list[float] = field(default_factory=list)
    n_reference: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "edges": list(self.edges),
            "reference": dict(self.reference),
            "n_reference": self.n_reference,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> BinSpec:
        return cls(
            kind=str(raw["kind"]),
            reference={str(k): float(v) for k, v in raw["reference"].items()},
            edges=[float(e) for e in raw.get("edges", [])],
            n_reference=int(raw.get("n_reference", 0)),
        )


def _is_categorical(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object


def _full_edges(interior: Sequence[float]) -> npt.NDArray[np.float64]:
    if len(interior) == 0:
        return np.array([], dtype="float64")
    return np.array([-np.inf, *interior, np.inf], dtype="float64")


def bin_labels(spec: BinSpec, values: pd.Series) -> pd.Series:
    """Every value to its bin label under `spec`, through the stage 2 binning functions."""
    if spec.kind == "categorical":
        out = values.astype("object")
        return out.where(out.notna(), eda.MISSING_LEVEL)
    return eda.bin_series(values, _full_edges(spec.edges))


def fit_bin_spec(values: pd.Series, n_bins: int = N_BINS) -> BinSpec:
    """Bins and reference shares from the training values of one column."""
    if _is_categorical(values):
        labels = bin_labels(BinSpec(kind="categorical", reference={}), values)
        shares = labels.value_counts(normalize=True)
        return BinSpec(
            kind="categorical",
            reference={str(k): float(v) for k, v in shares.items()},
            n_reference=int(values.size),
        )
    # `eda._bin_edges` is the stage 2 cut rule: unique quantiles, outer edges infinite.
    edges = eda._bin_edges(values, n_bins)
    interior = [float(e) for e in edges[1:-1]] if edges.size >= 2 else []
    spec = BinSpec(kind="numeric", reference={}, edges=interior, n_reference=int(values.size))
    labels = bin_labels(spec, values)
    shares = labels.value_counts(normalize=True)
    spec.reference = {str(k): float(v) for k, v in shares.items()}
    return spec


def psi(spec: BinSpec, values: pd.Series) -> dict[str, Any]:
    """PSI of `values` against the reference shares, the stage 2 arithmetic exactly.

    Every level in either table contributes; a level empty on one side is floored at
    `eda.PSI_FLOOR` rather than dropped. On the whole test split this reproduces
    `reports/eda/temporal.json` for a column both stages bin the same way.
    """
    actual = bin_labels(spec, values).value_counts(normalize=True)
    expected = spec.reference
    levels = sorted(set(expected) | {str(k) for k in actual.index}, key=str)
    e = np.array([float(expected.get(level, 0.0)) for level in levels])
    a = np.array([float(actual.get(level, 0.0)) for level in levels])
    e = np.clip(e, eda.PSI_FLOOR, None)
    a = np.clip(a, eda.PSI_FLOOR, None)
    terms = (a - e) * np.log(a / e)
    return {
        "psi": float(terms.sum()),
        "n_levels": len(levels),
        "n_new_levels": int(sum(1 for level in levels if level not in expected)),
    }


def band(value: float) -> str:
    if value < PSI_WATCH:
        return BANDS[0]
    if value < PSI_ALERT:
        return BANDS[1]
    return BANDS[2]


# --- the stored reference --------------------------------------------------------------------------


@dataclass
class DriftReference:
    """What the drift job compares a batch with: the training window, and the in-control range.

    `specs` holds one histogram per model column and `score_spec` one for the score. The
    calibration block holds, per column, the PSI of the whole accepted windows (the stage 2
    grain) and the worst PSI over the in-control batches of those windows (the monitoring
    grain). `alert_edges` is the larger of `PSI_ALERT` and that worst in-control batch, and
    `feature_count_edge` the most alertable columns any held-out in-control batch put past their
    edges, which is what the batch flag has to exceed.
    """

    columns: list[str]
    specs: dict[str, BinSpec]
    score_spec: BinSpec
    batch_days: int
    calibration: dict[str, Any]
    alert_edges: dict[str, float]
    score_alert_edge: float
    feature_count_edge: int
    built_from: dict[str, Any]

    @property
    def alertable_columns(self) -> list[str]:
        return [column for column in self.columns if alertable(column)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "batch_days": self.batch_days,
            "bands": {"watch": PSI_WATCH, "alert": PSI_ALERT, "names": list(BANDS)},
            "n_bins": N_BINS,
            "floor": eda.PSI_FLOOR,
            "score_reference": SCORE_REFERENCE,
            "built_from": dict(self.built_from),
            "calibration": self.calibration,
            "alertable_columns": self.alertable_columns,
            "not_alertable": {
                "columns": [c for c in self.columns if not alertable(c)],
                "reason": NOT_ALERTABLE_REASON,
            },
            "alert_edges": dict(self.alert_edges),
            "score_alert_edge": self.score_alert_edge,
            "feature_count_edge": self.feature_count_edge,
            "score_spec": self.score_spec.to_dict(),
            "specs": {column: spec.to_dict() for column, spec in self.specs.items()},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DriftReference:
        return cls(
            columns=[str(c) for c in raw["columns"]],
            specs={str(c): BinSpec.from_dict(s) for c, s in raw["specs"].items()},
            score_spec=BinSpec.from_dict(raw["score_spec"]),
            batch_days=int(raw["batch_days"]),
            calibration=dict(raw["calibration"]),
            alert_edges={str(c): float(v) for c, v in raw["alert_edges"].items()},
            score_alert_edge=float(raw["score_alert_edge"]),
            feature_count_edge=int(raw["feature_count_edge"]),
            built_from=dict(raw["built_from"]),
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n")

    @classmethod
    def read(cls, path: Path) -> DriftReference:
        return cls.from_dict(json.loads(path.read_text()))


def _psi_table(
    specs: Mapping[str, BinSpec], frame: pd.DataFrame, columns: Sequence[str]
) -> dict[str, float]:
    return {column: psi(specs[column], frame[column])["psi"] for column in columns}


def build_reference(
    train: pd.DataFrame,
    train_scores: npt.NDArray[np.float64],
    columns: Sequence[str],
    windows: Mapping[str, tuple[pd.DataFrame, npt.NDArray[np.float64]]],
    batch_days: int = BATCH_DAYS,
    n_bins: int = N_BINS,
    built_from: Mapping[str, Any] | None = None,
) -> DriftReference:
    """Fit the histograms on the training window and calibrate the edges on the accepted ones.

    `windows` maps a name (val, test) to a prepared frame and its scores. Each is cut into
    batches of `batch_days` and every batch is scored against the training histograms; the
    worst value per column over those batches is the in-control range at the monitoring grain.
    The whole-window PSI is recorded too, which is the stage 2 grain and the number to check
    against `reports/eda/temporal.json`.
    """
    if config.TARGET in columns:
        raise MonitoringError("the drift reference does not carry the label column")
    specs = {column: fit_bin_spec(train[column], n_bins) for column in columns}
    score_spec = fit_bin_spec(pd.Series(train_scores, name=SCORE_NAME), n_bins)

    whole: dict[str, dict[str, float]] = {}
    whole_score: dict[str, float] = {}
    batches: list[dict[str, Any]] = []
    worst = dict.fromkeys(columns, 0.0)
    worst_score = 0.0
    for name, (frame, scores) in windows.items():
        whole[name] = _psi_table(specs, frame, columns)
        whole_score[name] = psi(score_spec, pd.Series(scores, name=SCORE_NAME))["psi"]
        ts = frame[config.TIME_COLUMN].to_numpy(dtype="int64")
        for bounds in batch_bounds(ts, batch_days):
            mask = batch_mask(ts, bounds)
            part = frame.loc[mask]
            table = _psi_table(specs, part, columns)
            score_psi = psi(score_spec, pd.Series(scores[mask], name=SCORE_NAME))["psi"]
            for column, value in table.items():
                worst[column] = max(worst[column], value)
            worst_score = max(worst_score, score_psi)
            batches.append(
                {
                    "window": name,
                    **bounds,
                    "psi": table,
                    "score_psi": score_psi,
                }
            )

    alert_edges = {column: max(PSI_ALERT, worst[column]) for column in columns}
    held_out = leave_one_out_counts(batches, [c for c in columns if alertable(c)])
    calibration = {
        "rule": (
            "alert edge per column is max(PSI_ALERT, worst PSI over the in-control batches); a "
            "column alerts strictly above its edge; the batch flag is the score past its edge "
            "or more alertable columns past theirs than any held-out in-control batch showed"
        ),
        "windows": list(windows),
        "batch_days": batch_days,
        "n_batches": len(batches),
        "whole_window_psi": whole,
        "whole_window_score_psi": whole_score,
        "worst_batch_psi": worst,
        "worst_batch_score_psi": worst_score,
        "leave_one_out_counts": held_out,
        "batches": batches,
    }
    return DriftReference(
        columns=list(columns),
        specs=specs,
        score_spec=score_spec,
        batch_days=batch_days,
        calibration=calibration,
        alert_edges=alert_edges,
        score_alert_edge=max(PSI_ALERT, worst_score),
        feature_count_edge=max(held_out) if held_out else 0,
        built_from=dict(built_from or {}),
    )


def leave_one_out_counts(batches: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[int]:
    """Per in-control batch, how many of `columns` exceed an edge calibrated on the others.

    A column's edge is the larger of `PSI_ALERT` and its worst value over every other batch, so
    the count is what the rule would have said about a week it had not seen. The largest of
    these counts is the batch flag's edge: a week is called drifted when it puts more columns
    past their edges than any in-control week did when held out.
    """
    out: list[int] = []
    for i, held in enumerate(batches):
        others = [b for j, b in enumerate(batches) if j != i]
        count = 0
        for column in columns:
            edge = max(PSI_ALERT, max(float(b["psi"][column]) for b in others))
            if float(held["psi"][column]) > edge:
                count += 1
        out.append(count)
    return out


# --- the drift job ----------------------------------------------------------------------------------


def drift_report(
    reference: DriftReference,
    frame: pd.DataFrame,
    scores: npt.NDArray[np.float64],
    bounds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """PSI per column and for the score on one batch, against the stored reference. No labels.

    Raises if the frame carries the label column: the job is defined by not reading it, and a
    frame that has it is a frame assembled the wrong way round.
    """
    if config.TARGET in frame.columns:
        raise MonitoringError(
            f"the drift job reads no labels; drop {config.TARGET} before calling it"
        )
    if len(frame) != scores.size:
        raise ValueError(f"{len(frame)} rows but {scores.size} scores")
    if len(frame) == 0:
        raise MonitoringError("an empty batch has no distribution to compare")

    per_feature: list[dict[str, Any]] = []
    for column in reference.columns:
        stat = psi(reference.specs[column], frame[column])
        edge = reference.alert_edges[column]
        per_feature.append(
            {
                "column": column,
                "psi": stat["psi"],
                "band": band(stat["psi"]),
                "n_new_levels": stat["n_new_levels"],
                "alert_edge": edge,
                "alertable": alertable(column),
                "alert": bool(alertable(column) and stat["psi"] > edge),
            }
        )
    values = np.array([f["psi"] for f in per_feature])
    bands = [f["band"] for f in per_feature]
    alerts = [f for f in per_feature if f["alert"]]
    score_stat = psi(reference.score_spec, pd.Series(scores, name=SCORE_NAME))
    score_alert = bool(score_stat["psi"] > reference.score_alert_edge)
    aggregate = {
        "n_features": len(per_feature),
        "median_psi": float(np.median(values)),
        "mean_psi": float(values.mean()),
        "max_psi": float(values.max()),
        "n_stable": bands.count(BANDS[0]),
        "n_watch": bands.count(BANDS[1]),
        "n_alert_band": bands.count(BANDS[2]),
        "n_alertable": len(reference.alertable_columns),
        "n_alert": len(alerts),
        "feature_count_edge": reference.feature_count_edge,
        "feature_alert": bool(len(alerts) > reference.feature_count_edge),
        "alert_columns": sorted((f["column"] for f in alerts), key=str),
    }
    return {
        "job": "drift",
        "reads_labels": False,
        "batch": dict(bounds) if bounds is not None else None,
        "n_rows": len(frame),
        "score": {
            "psi": score_stat["psi"],
            "band": band(score_stat["psi"]),
            "alert_edge": reference.score_alert_edge,
            "alert": score_alert,
        },
        "aggregate": aggregate,
        "drift_alert": aggregate_alert(score_alert, aggregate),
        "per_feature": per_feature,
    }


def aggregate_alert(score_alert: bool, aggregate: Mapping[str, Any]) -> bool:
    """The batch-level drift flag the lifecycle reads: the score moved, or the inputs did.

    Two paths, because each misses what the other sees. The score is the model's own summary
    of its inputs and moves when a column it splits on hard changes scale; it stayed under 0.08
    on a probe that tripled the seven C columns and cost the model a third of its PR-AUC, because
    the histogram of scores barely moved while the rows under each score changed. The column
    count catches that and misses a shift the model absorbs. Either raises the flag: it is the
    cheap side of the lifecycle, a challenger the gate still has to pass.
    """
    return bool(score_alert or bool(aggregate["feature_alert"]))


def leave_one_out_false_alerts(reference: DriftReference) -> dict[str, Any]:
    """How often the calibrated rule fires on an in-control batch it was not calibrated on.

    Each in-control batch is held out, the edges are recomputed from the others, and the held
    out batch is scored against those edges. This is the honest false alert rate of the rule;
    the rate on the calibration batches themselves is zero by construction.
    """
    batches = list(reference.calibration["batches"])
    columns = reference.alertable_columns
    counts = leave_one_out_counts(batches, columns)
    out: list[dict[str, Any]] = []
    for i, held in enumerate(batches):
        others = [b for j, b in enumerate(batches) if j != i]
        feature_alerts = []
        for column in columns:
            edge = max(PSI_ALERT, max(float(b["psi"][column]) for b in others))
            if float(held["psi"][column]) > edge:
                feature_alerts.append(column)
        # The count edge a held-out week faces is the largest count among the other weeks.
        count_edge = max(c for j, c in enumerate(counts) if j != i)
        score_edge = max(PSI_ALERT, max(float(b["score_psi"]) for b in others))
        score_alert = float(held["score_psi"]) > score_edge
        textbook = sum(1 for column in reference.columns if float(held["psi"][column]) > PSI_ALERT)
        out.append(
            {
                "window": held["window"],
                "batch": held["batch"],
                "n_feature_alerts": len(feature_alerts),
                "feature_alerts": feature_alerts,
                "feature_count_edge": count_edge,
                "feature_alert": bool(len(feature_alerts) > count_edge),
                "score_alert": bool(score_alert),
                "any_alert": bool(score_alert or len(feature_alerts) > count_edge),
                "n_textbook_alert_band": textbook,
            }
        )
    n = len(out)
    return {
        "n_batches": n,
        "batches": out,
        "counts": counts,
        "rate_any_column_past_its_edge": sum(1 for b in out if b["n_feature_alerts"]) / n,
        "rate_feature_alert": sum(1 for b in out if b["feature_alert"]) / n,
        "rate_score_alert": sum(1 for b in out if b["score_alert"]) / n,
        "rate_any_alert": sum(1 for b in out if b["any_alert"]) / n,
        "mean_textbook_alert_band": float(np.mean([b["n_textbook_alert_band"] for b in out])),
        "min_textbook_alert_band": min(b["n_textbook_alert_band"] for b in out),
    }


# --- the decay job ----------------------------------------------------------------------------------


def matures_at(end_dt: int, maturity_days: int) -> int:
    """The clock value at which every label in a batch closing at `end_dt` is treated as final."""
    return int(end_dt) + int(maturity_days) * config.SECONDS_PER_DAY


def is_mature(end_dt: int, as_of_dt: int, maturity_days: int) -> bool:
    return int(as_of_dt) >= matures_at(end_dt, maturity_days)


def assert_mature(end_dt: int, as_of_dt: int, maturity_days: int) -> None:
    """Refuse a batch whose labels have not had `maturity_days` to arrive.

    A fraud that has not been charged back yet reads as legitimate. Scored into precision it is
    a false positive that is not one; into recall it is a miss that is not one; into PR-AUC it
    is a decay that is not there. ADR 0032 measured the fit-time version of this failure and
    found no small version of it; the metric-time version is refused rather than corrected.
    """
    if not is_mature(end_dt, as_of_dt, maturity_days):
        raise MonitoringError(
            f"batch closing at {end_dt} (day {day_index(end_dt):.1f}) is immature at "
            f"{as_of_dt} (day {day_index(as_of_dt):.1f}): labels are read only "
            f"{maturity_days} days after the batch closes, at day "
            f"{day_index(matures_at(end_dt, maturity_days)):.1f}"
        )


def reference_draws(
    y: npt.NDArray[np.int_], scores: npt.NDArray[np.float64], n_boot: int, seed: int
) -> dict[str, Any]:
    """Bootstrap draws of the reference window, the shape `independent_difference` wants."""
    return evaluation.bootstrap_draws(y, {"reference": scores}, n_boot, seed)


def performance_report(
    y: npt.NDArray[np.int_],
    scores: npt.NDArray[np.float64],
    threshold: float,
    reference: Mapping[str, Any],
    bounds: Mapping[str, Any],
    as_of_dt: int,
    maturity_days: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Precision, recall and PR-AUC on one mature batch, against the reference window.

    `reference` is a `reference_draws` result on the window the champion was accepted at. The
    batch is bootstrapped by row on its own resamples and the drop is the independent difference
    of the two, which is the interval there is for one model scored on two windows.
    """
    assert_mature(int(bounds["end_dt"]), as_of_dt, maturity_days)
    y = np.asarray(y, dtype="int64")
    if y.size == 0:
        raise MonitoringError("an empty batch has no metric")
    n_days = (int(bounds["end_dt"]) - int(bounds["start_dt"])) / config.SECONDS_PER_DAY
    at = evaluation.at_threshold(y, scores, threshold, n_days)
    drawn = evaluation.bootstrap_draws(y, {"batch": scores}, n_boot, seed)
    interval = evaluation.paired_summary(drawn)["per_model"]["batch"]
    drop = evaluation.independent_difference(reference, "reference", drawn, "batch")
    return {
        "job": "decay",
        "reads_labels": True,
        "batch": dict(bounds),
        "as_of_dt": int(as_of_dt),
        "maturity_days": int(maturity_days),
        "matured_at_dt": matures_at(int(bounds["end_dt"]), maturity_days),
        "n_rows": int(y.size),
        "n_fraud": int(y.sum()),
        "fraud_rate": float(y.mean()),
        "threshold": float(threshold),
        "precision": at["precision"],
        "recall": at["recall"],
        "alerts_per_day": at["alerts_per_day"],
        "false_alarms_per_catch": at["false_alarms_per_catch"],
        "pr_auc": interval,
        "roc_auc": float(evaluation.roc_auc(y, scores)) if 0 < y.sum() < y.size else None,
        "reference_pr_auc": float(reference["points"]["reference"]),
        "drop": {
            "point": drop["difference"]["point"],
            "low": drop["difference"]["low"],
            "high": drop["difference"]["high"],
            "half_width": drop["difference"]["half_width"],
            "excludes_zero": drop["excludes_zero"],
            "resample": drop["resample"],
        },
    }
