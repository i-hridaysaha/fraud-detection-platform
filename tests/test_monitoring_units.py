"""Unit tests for the two monitoring jobs, on generated data.

The drift job's arithmetic is checked against the stage 2 function it must reproduce, its
reference is checked to round-trip through JSON, and its refusal to read a label is checked.
The decay job's refusal of an immature batch is the point of the clock and is checked at the
boundary, one second either side.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, eda, evaluation, monitoring

DAY = config.SECONDS_PER_DAY


def _frame(rng: np.random.Generator, n: int, start_day: int, shift: float = 0.0) -> pd.DataFrame:
    ts = (start_day * DAY + rng.integers(0, 28 * DAY, n)).astype("int64")
    frame = pd.DataFrame(
        {
            config.TIME_COLUMN: np.sort(ts),
            "x": rng.normal(shift, 1.0, n),
            "count": rng.integers(0, 5, n).astype("int16"),
            "sparse": np.where(rng.random(n) < 0.3, np.nan, rng.normal(0, 1, n)),
            "level": pd.Categorical(rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1])),
            "level_te": rng.random(n),
        }
    )
    return frame


@pytest.fixture(scope="module")
def train() -> pd.DataFrame:
    return _frame(np.random.default_rng(config.SEED), 6000, 1)


@pytest.fixture(scope="module")
def later() -> pd.DataFrame:
    return _frame(np.random.default_rng(config.SEED + 1), 3000, 30)


COLUMNS = ["x", "count", "sparse", "level", "level_te"]


# --- batches --------------------------------------------------------------------------------------


def test_batches_are_consecutive_half_open_windows_from_the_first_day() -> None:
    ts = np.array([3 * DAY + 5, 3 * DAY + 6, 10 * DAY, 17 * DAY - 1, 17 * DAY], dtype="int64")
    bounds = monitoring.batch_bounds(ts, 7)
    assert [(b["start_dt"], b["end_dt"]) for b in bounds] == [
        (3 * DAY, 10 * DAY),
        (10 * DAY, 17 * DAY),
        (17 * DAY, 24 * DAY),
    ]
    assert [b["n_rows"] for b in bounds] == [2, 2, 1]
    assert [b["partial"] for b in bounds] == [0, 0, 1]
    assert sum(b["n_rows"] for b in bounds) == ts.size


def test_batches_refuse_nothing_to_batch() -> None:
    with pytest.raises(ValueError):
        monitoring.batch_bounds(np.array([], dtype="int64"))
    with pytest.raises(ValueError):
        monitoring.batch_bounds(np.array([DAY], dtype="int64"), 0)


# --- psi ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("column", COLUMNS)
def test_psi_reproduces_the_stage_2_function(
    train: pd.DataFrame, later: pd.DataFrame, column: str
) -> None:
    spec = monitoring.fit_bin_spec(train[column])
    here = monitoring.psi(spec, later[column])
    stage_2 = eda.population_stability_index(train[column], later[column])
    assert here["psi"] == pytest.approx(stage_2["psi"], abs=1e-12)
    assert here["n_levels"] == stage_2["n_levels"]
    assert here["n_new_levels"] == stage_2["n_new_levels"]


def test_psi_is_zero_against_itself_and_grows_with_a_shift(train: pd.DataFrame) -> None:
    spec = monitoring.fit_bin_spec(train["x"])
    assert monitoring.psi(spec, train["x"])["psi"] == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    small = monitoring.psi(spec, pd.Series(rng.normal(0.2, 1, 4000)))["psi"]
    large = monitoring.psi(spec, pd.Series(rng.normal(1.5, 1, 4000)))["psi"]
    assert 0 < small < large


def test_bands() -> None:
    assert monitoring.band(0.0) == "stable"
    assert monitoring.band(0.0999) == "stable"
    assert monitoring.band(0.1) == "watch"
    assert monitoring.band(0.1999) == "watch"
    assert monitoring.band(0.2) == "alert"
    assert monitoring.band(5.0) == "alert"


def test_a_new_categorical_level_is_counted(train: pd.DataFrame) -> None:
    spec = monitoring.fit_bin_spec(train["level"])
    later = pd.Series(pd.Categorical(["a"] * 50 + ["z"] * 50))
    out = monitoring.psi(spec, later)
    assert out["n_new_levels"] == 1
    assert out["psi"] > monitoring.PSI_ALERT


# --- the reference --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def reference(train: pd.DataFrame, later: pd.DataFrame) -> monitoring.DriftReference:
    rng = np.random.default_rng(3)
    scores_train = rng.beta(0.5, 8, len(train))
    scores_later = rng.beta(0.5, 8, len(later))
    return monitoring.build_reference(
        train,
        scores_train,
        COLUMNS,
        {"later": (later, scores_later)},
        batch_days=7,
        built_from={"test": True},
    )


def test_reference_round_trips_through_json(
    reference: monitoring.DriftReference, tmp_path: Path
) -> None:
    path = tmp_path / "reference.json"
    reference.write(path)
    back = monitoring.DriftReference.read(path)
    assert back.columns == reference.columns
    assert back.alert_edges == reference.alert_edges
    assert back.score_alert_edge == reference.score_alert_edge
    assert back.feature_count_edge == reference.feature_count_edge
    for column in COLUMNS:
        assert back.specs[column].to_dict() == reference.specs[column].to_dict()
    raw = json.loads(path.read_text())
    assert raw["not_alertable"]["columns"] == ["level_te"]
    assert set(raw["alertable_columns"]) == set(COLUMNS) - {"level_te"}


def test_edges_are_the_larger_of_the_band_and_the_worst_in_control_batch(
    reference: monitoring.DriftReference,
) -> None:
    worst = reference.calibration["worst_batch_psi"]
    for column in COLUMNS:
        assert reference.alert_edges[column] == max(monitoring.PSI_ALERT, worst[column])
        assert reference.alert_edges[column] >= monitoring.PSI_ALERT
    assert reference.score_alert_edge == max(
        monitoring.PSI_ALERT, reference.calibration["worst_batch_score_psi"]
    )
    assert reference.calibration["n_batches"] == len(reference.calibration["batches"])
    assert reference.feature_count_edge == max(reference.calibration["leave_one_out_counts"])


def test_calibration_batches_never_alert_but_a_held_out_week_can(
    reference: monitoring.DriftReference, later: pd.DataFrame
) -> None:
    ts = later[config.TIME_COLUMN].to_numpy(dtype="int64")
    rng = np.random.default_rng(3)
    scores = rng.beta(0.5, 8, len(later))
    for bounds in monitoring.batch_bounds(ts, 7):
        mask = monitoring.batch_mask(ts, bounds)
        report = monitoring.drift_report(reference, later.loc[mask], scores[mask], bounds)
        assert report["aggregate"]["n_alert"] == 0
        assert report["aggregate"]["feature_alert"] is False
    loo = monitoring.leave_one_out_false_alerts(reference)
    assert loo["n_batches"] == reference.calibration["n_batches"]
    assert loo["counts"] == reference.calibration["leave_one_out_counts"]
    assert 0.0 <= loo["rate_any_alert"] <= 1.0


# --- the drift job --------------------------------------------------------------------------------


def test_drift_job_refuses_a_frame_with_the_label(
    reference: monitoring.DriftReference, later: pd.DataFrame
) -> None:
    with_label = later.assign(**{config.TARGET: 0})
    with pytest.raises(monitoring.MonitoringError, match="reads no labels"):
        monitoring.drift_report(reference, with_label, np.zeros(len(later)))


def test_drift_job_refuses_an_empty_batch_and_a_length_mismatch(
    reference: monitoring.DriftReference, later: pd.DataFrame
) -> None:
    with pytest.raises(ValueError):
        monitoring.drift_report(reference, later, np.zeros(3))
    with pytest.raises(monitoring.MonitoringError):
        monitoring.drift_report(reference, later.iloc[:0], np.zeros(0))


def test_drift_job_fires_on_a_shift_and_never_on_a_target_encoding(
    reference: monitoring.DriftReference, later: pd.DataFrame
) -> None:
    rng = np.random.default_rng(5)
    shifted = later.copy()
    shifted["x"] = shifted["x"] + 3.0
    shifted["count"] = (shifted["count"] * 3 + 7).astype("int16")
    shifted["level_te"] = shifted["level_te"] + 100.0
    report = monitoring.drift_report(reference, shifted, rng.beta(0.5, 8, len(shifted)))
    by_column = {f["column"]: f for f in report["per_feature"]}
    assert by_column["x"]["alert"] and by_column["count"]["alert"]
    assert by_column["level_te"]["band"] == "alert"
    assert by_column["level_te"]["alertable"] is False
    assert by_column["level_te"]["alert"] is False
    assert report["aggregate"]["n_alert"] == 2
    assert report["aggregate"]["feature_alert"] is True
    assert report["drift_alert"] is True
    assert report["reads_labels"] is False


def test_score_shift_alone_raises_the_flag(
    reference: monitoring.DriftReference, later: pd.DataFrame
) -> None:
    rng = np.random.default_rng(6)
    report = monitoring.drift_report(reference, later, rng.beta(4, 2, len(later)))
    assert report["score"]["alert"] is True
    assert report["aggregate"]["n_alert"] == 0
    assert report["drift_alert"] is True


def test_the_batch_flag_is_either_path() -> None:
    assert monitoring.aggregate_alert(True, {"feature_alert": False}) is True
    assert monitoring.aggregate_alert(False, {"feature_alert": True}) is True
    assert monitoring.aggregate_alert(False, {"feature_alert": False}) is False


# --- the clock ------------------------------------------------------------------------------------


def test_maturity_is_the_batch_close_plus_the_window_to_the_second() -> None:
    end = 100 * DAY
    assert monitoring.matures_at(end, 30) == 130 * DAY
    assert monitoring.is_mature(end, 130 * DAY, 30)
    assert not monitoring.is_mature(end, 130 * DAY - 1, 30)
    with pytest.raises(monitoring.MonitoringError, match="immature"):
        monitoring.assert_mature(end, 130 * DAY - 1, 30)
    monitoring.assert_mature(end, 130 * DAY, 30)


# --- the decay job --------------------------------------------------------------------------------


def _scored(rng: np.random.Generator, n: int, separation: float) -> tuple[Any, Any]:
    y = (rng.random(n) < 0.05).astype("int64")
    score = np.clip(rng.normal(0.1 + separation * y, 0.15), 0, 1)
    return y, score


def test_decay_job_refuses_an_immature_batch() -> None:
    rng = np.random.default_rng(7)
    y, s = _scored(rng, 2000, 0.5)
    reference = monitoring.reference_draws(y, s, 50, 1)
    bounds = {"batch": 0, "start_dt": 100 * DAY, "end_dt": 107 * DAY}
    with pytest.raises(monitoring.MonitoringError, match="immature"):
        monitoring.performance_report(y, s, 0.3, reference, bounds, 137 * DAY - 1, 30, 50, 1)


def test_decay_job_reports_the_threshold_metrics_and_the_drop() -> None:
    rng = np.random.default_rng(8)
    y_ref, s_ref = _scored(rng, 4000, 0.6)
    reference = monitoring.reference_draws(y_ref, s_ref, 200, 1)
    y, s = _scored(rng, 2000, 0.2)
    bounds = {"batch": 0, "start_dt": 100 * DAY, "end_dt": 107 * DAY}
    report = monitoring.performance_report(y, s, 0.3, reference, bounds, 137 * DAY, 30, 200, 1)
    at = evaluation.at_threshold(y, s, 0.3, 7.0)
    assert report["precision"] == at["precision"]
    assert report["recall"] == at["recall"]
    assert report["alerts_per_day"] == at["alerts_per_day"]
    assert report["pr_auc"]["point"] == pytest.approx(evaluation.pr_auc(y, s))
    assert report["reference_pr_auc"] == pytest.approx(evaluation.pr_auc(y_ref, s_ref))
    assert report["drop"]["point"] == pytest.approx(
        report["reference_pr_auc"] - report["pr_auc"]["point"]
    )
    assert report["drop"]["resample"] == "independent"
    assert report["drop"]["point"] > 0 and report["drop"]["excludes_zero"]
    assert report["matured_at_dt"] == 137 * DAY
    assert report["reads_labels"] is True
