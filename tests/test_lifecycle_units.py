"""Unit tests for the lifecycle: the triggers, the gate in both directions, the registry moves.

The gate is the test that matters. It is given a challenger that is genuinely better and must
promote it, one that is worse and must refuse it, one that is better by less than the margin
and must refuse that too, and the refusals have to come with the reason. The registry moves run
on a temporary SQLite registry with two real registrations, and the rollback is checked against
what the aliases say afterwards, not against what the code meant to do.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fraud_platform import config, data_loader, evaluation, lifecycle, serving

DAY = config.SECONDS_PER_DAY


# --- the constants ------------------------------------------------------------------------------


def test_the_margin_is_the_tolerance_and_sustained_is_more_than_one() -> None:
    assert lifecycle.DECAY_TOLERANCE == lifecycle.PROMOTION_MARGIN
    assert lifecycle.SUSTAINED_CYCLES >= 2
    assert lifecycle.MATURITY_DAYS > 0


# --- the retrain decision ---------------------------------------------------------------------------


def _cycles(*pairs: tuple[bool | None, bool | None]) -> list[dict[str, Any]]:
    return [
        {"cycle": k, "drift_alert": drift, "decay": decay} for k, (drift, decay) in enumerate(pairs)
    ]


def test_nothing_fires_on_a_quiet_history() -> None:
    out = lifecycle.should_retrain(_cycles((False, None), (False, None), (False, False)))
    assert out == {
        "retrain": False,
        "reason": None,
        "latest_decay_cycle": 2,
        "latest_decay": False,
        "consecutive_drift_alerts": 0,
        "sustained_cycles": lifecycle.SUSTAINED_CYCLES,
    }


def test_one_spike_of_drift_is_not_sustained() -> None:
    out = lifecycle.should_retrain(_cycles((False, None), (True, None), (False, None)))
    assert out["retrain"] is False
    assert out["consecutive_drift_alerts"] == 0
    out = lifecycle.should_retrain(_cycles((False, None), (True, None)))
    assert out["retrain"] is False
    assert out["consecutive_drift_alerts"] == 1


def test_two_consecutive_drift_alerts_are_the_early_trigger() -> None:
    out = lifecycle.should_retrain(_cycles((False, None), (True, None), (True, None)))
    assert out["retrain"] is True
    assert out["reason"] == "sustained_drift"
    assert out["consecutive_drift_alerts"] == 2


def test_a_cycle_with_no_batch_breaks_the_run() -> None:
    out = lifecycle.should_retrain(_cycles((True, None), (None, None), (True, None)))
    assert out["retrain"] is False
    assert out["consecutive_drift_alerts"] == 1


def test_decay_is_the_primary_trigger_and_wins() -> None:
    out = lifecycle.should_retrain(_cycles((True, None), (True, True)))
    assert out["reason"] == "decay"
    out = lifecycle.should_retrain(_cycles((False, None), (False, True), (False, None)))
    assert out["retrain"] is True and out["reason"] == "decay"
    assert out["latest_decay_cycle"] == 1


def test_the_most_recent_evaluated_batch_decides_decay() -> None:
    out = lifecycle.should_retrain(_cycles((False, True), (False, None), (False, False)))
    assert out["retrain"] is False
    assert out["latest_decay_cycle"] == 2


def test_sustained_must_be_at_least_two() -> None:
    with pytest.raises(ValueError):
        lifecycle.should_retrain(_cycles((True, None)), sustained_cycles=1)


def test_decay_flag_needs_the_tolerance_and_the_interval() -> None:
    def report(point: float, excludes_zero: bool) -> dict[str, Any]:
        return {"drop": {"point": point, "excludes_zero": excludes_zero}}

    assert lifecycle.decay_flag(report(0.10, True)) is True
    assert lifecycle.decay_flag(report(lifecycle.DECAY_TOLERANCE, True)) is True
    assert lifecycle.decay_flag(report(0.10, False)) is False
    assert lifecycle.decay_flag(report(lifecycle.DECAY_TOLERANCE - 1e-9, True)) is False
    assert lifecycle.decay_flag(report(-0.10, True)) is False


# --- the windows ------------------------------------------------------------------------------------


def test_challenger_windows_are_contiguous_and_newest_first() -> None:
    windows = lifecycle.ChallengerWindows(fit_days=60, calibration_days=14, gate_days=14)
    bounds = lifecycle.challenger_bounds(200 * DAY, windows)
    assert bounds["gate_end_dt"] == 200 * DAY
    assert bounds["gate_start_dt"] == bounds["calibration_end_dt"] == 186 * DAY
    assert bounds["calibration_start_dt"] == bounds["fit_end_dt"] == 172 * DAY
    assert bounds["fit_start_dt"] == 112 * DAY
    with pytest.raises(ValueError):
        lifecycle.ChallengerWindows(fit_days=0, calibration_days=14, gate_days=14)


# --- the moved boundary ---------------------------------------------------------------------------


def test_training_boundary_moves_the_guard_and_puts_it_back() -> None:
    import pandas as pd

    later = pd.DataFrame({config.TIME_COLUMN: [config.TRAIN_END_DT + 5 * DAY]})
    with pytest.raises(data_loader.TrainOnlyError):
        data_loader.assert_train_only(later, what="a later fit")
    before = config.TRAIN_END_DT
    with data_loader.training_boundary(config.TRAIN_END_DT + 10 * DAY) as end:
        assert end == config.TRAIN_END_DT
        data_loader.assert_train_only(later, what="a later fit")
        with pytest.raises(data_loader.TrainOnlyError):
            data_loader.assert_train_only(
                pd.DataFrame({config.TIME_COLUMN: [end]}), what="at the moved boundary"
            )
    assert before == config.TRAIN_END_DT
    with pytest.raises(data_loader.TrainOnlyError):
        data_loader.assert_train_only(later, what="a later fit")


def test_training_boundary_is_restored_after_an_error() -> None:
    before = config.TRAIN_END_DT
    with pytest.raises(RuntimeError), data_loader.training_boundary(before + DAY):
        raise RuntimeError("inside")
    assert before == config.TRAIN_END_DT
    with pytest.raises(ValueError), data_loader.training_boundary(0):
        pass


# --- the gate, both directions ----------------------------------------------------------------------


def _gate_case(
    seed: int, champion_separation: float, challenger_separation: float, n: int = 6000
) -> tuple[Any, Any, Any, Any]:
    """Labels, two score vectors and card groups. A larger separation is a better model, and
    the two share their noise so the paired difference is what it would be for two models on
    the same rows."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype("int64")
    noise = rng.normal(0, 0.2, n)
    champion = np.clip(0.1 + champion_separation * y + noise, 0, 1)
    challenger = np.clip(0.1 + challenger_separation * y + noise + rng.normal(0, 0.02, n), 0, 1)
    groups = rng.integers(0, n // 3, n)
    return y, champion, challenger, groups


def test_the_gate_promotes_a_genuinely_better_challenger() -> None:
    y, champion, challenger, groups = _gate_case(1, 0.3, 0.9)
    out = lifecycle.gate(y, champion, challenger, groups, 200, config.SEED)
    assert out["promote"] is True
    assert out["clears_margin"] and out["above_noise"]
    assert out["difference"]["point"] >= lifecycle.PROMOTION_MARGIN
    assert out["difference"]["low"] > 0
    assert out["resample"] == "groups"
    assert out["reason"].startswith("promoted")
    assert out["challenger"]["point"] == pytest.approx(evaluation.pr_auc(y, challenger))
    assert out["champion"]["point"] == pytest.approx(evaluation.pr_auc(y, champion))


def test_the_gate_refuses_a_worse_challenger() -> None:
    y, champion, challenger, groups = _gate_case(2, 0.9, 0.3)
    out = lifecycle.gate(y, champion, challenger, groups, 200, config.SEED)
    assert out["promote"] is False
    assert out["difference"]["point"] < 0
    assert out["reason"] == "refused: not better"


def test_the_gate_refuses_a_challenger_inside_the_margin() -> None:
    y, champion, challenger, groups = _gate_case(3, 0.60, 0.63)
    out = lifecycle.gate(y, champion, challenger, groups, 200, config.SEED)
    assert 0 < out["difference"]["point"] < lifecycle.PROMOTION_MARGIN
    assert out["promote"] is False
    assert out["clears_margin"] is False
    assert out["reason"] == "refused: better, but by less than the margin"


def test_the_gate_refuses_a_margin_the_interval_cannot_support() -> None:
    # Seed 5 at these separations: the point clears a zero margin, the interval reaches zero.
    y, champion, challenger, groups = _gate_case(5, 0.30, 0.32, n=300)
    out = lifecycle.gate(y, champion, challenger, groups, 200, config.SEED, margin=0.0)
    assert out["difference"]["point"] > 0
    assert out["difference"]["low"] < 0
    assert out["clears_margin"] is True
    assert out["above_noise"] is False
    assert out["promote"] is False
    assert out["reason"] == "refused: clears the margin on the point but the interval reaches zero"


def test_the_gate_by_row_when_no_groups() -> None:
    y, champion, challenger, _ = _gate_case(6, 0.3, 0.9)
    out = lifecycle.gate(y, champion, challenger, None, 100, config.SEED)
    assert out["resample"] == "rows" and out["promote"] is True


# --- the registry ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry(tmp_path_factory: pytest.TempPathFactory) -> Iterator[lifecycle.Registry]:
    root = tmp_path_factory.mktemp("registry")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    previous = os.environ.get(serving.ENV_TRACKING_URI)
    os.environ[serving.ENV_TRACKING_URI] = uri
    yield lifecycle.Registry(uri, root / "artifacts", name="lifecycle-test")
    if previous is None:
        os.environ.pop(serving.ENV_TRACKING_URI, None)
    else:
        os.environ[serving.ENV_TRACKING_URI] = previous


def _dummy_bundle(directory: Path, tag: str) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "booster": directory / "model.ubj",
        "preparation": directory / "preparation.pkl",
        "serving": directory / "serving.json",
    }
    for path in paths.values():
        path.write_text(tag)
    return paths


def test_promote_keeps_the_previous_and_rollback_restores_it(
    registry: lifecycle.Registry, tmp_path: Path
) -> None:
    assert registry.version_for(lifecycle.PRODUCTION_ALIAS) is None
    first = registry.register(
        _dummy_bundle(tmp_path / "one", "one"), {"role": "champion"}, lifecycle.PRODUCTION_ALIAS
    )
    second = registry.register(
        _dummy_bundle(tmp_path / "two", "two"), {"role": "challenger"}, lifecycle.CHALLENGER_ALIAS
    )
    v1, v2 = int(first["version"]), int(second["version"])
    assert registry.aliases() == {
        lifecycle.PRODUCTION_ALIAS: v1,
        lifecycle.PREVIOUS_ALIAS: None,
        lifecycle.CHALLENGER_ALIAS: v2,
        lifecycle.ROLLED_BACK_ALIAS: None,
    }
    with pytest.raises(lifecycle.LifecycleError, match="nothing to roll back"):
        registry.rollback()

    moved = registry.promote(v2)
    assert moved == {"action": "promote", "production": v2, "previous": v1}
    assert registry.aliases() == {
        lifecycle.PRODUCTION_ALIAS: v2,
        lifecycle.PREVIOUS_ALIAS: v1,
        lifecycle.CHALLENGER_ALIAS: None,
        lifecycle.ROLLED_BACK_ALIAS: None,
    }
    with pytest.raises(lifecycle.LifecycleError, match="already"):
        registry.promote(v2)

    back = registry.rollback()
    assert back == {"action": "rollback", "production": v1, "rolled_back": v2}
    assert registry.aliases() == {
        lifecycle.PRODUCTION_ALIAS: v1,
        lifecycle.PREVIOUS_ALIAS: None,
        lifecycle.CHALLENGER_ALIAS: None,
        lifecycle.ROLLED_BACK_ALIAS: v2,
    }
    with pytest.raises(lifecycle.LifecycleError, match="nothing to roll back"):
        registry.rollback()
