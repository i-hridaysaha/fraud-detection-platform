"""The stage 9 artifacts hold together, on a clean checkout without the data.

Every assertion here recomputes something from the artifact or checks it against another
committed artifact: the drift edges against the calibration block, the decay reference against
stage 6's bootstrap, the promotion margin against stage 6's noise band, the adversarial
ranking against its own gain shares, and the demo's gate verdicts against its own numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import config, lifecycle, monitoring

REPORTS = config.REPORTS_DIR
MONITORING = REPORTS / "monitoring"


def load(path: Path) -> dict[str, Any]:
    assert path.exists(), f"{path.relative_to(config.ROOT)} is missing"
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    return load(MONITORING / "drift_reference.json")


@pytest.fixture(scope="module")
def drift() -> dict[str, Any]:
    return load(MONITORING / "drift.json")


@pytest.fixture(scope="module")
def decay() -> dict[str, Any]:
    return load(MONITORING / "decay.json")


@pytest.fixture(scope="module")
def adversarial() -> dict[str, Any]:
    return load(REPORTS / "adversarial_validation.json")


@pytest.fixture(scope="module")
def demo() -> dict[str, Any]:
    return load(REPORTS / "lifecycle_demo.json")


@pytest.fixture(scope="module")
def variance() -> dict[str, Any]:
    return load(REPORTS / "metric_variance.json")


@pytest.fixture(scope="module")
def shipped() -> dict[str, Any]:
    return load(config.ROOT / "models" / "shipped_model.json")


# --- the drift reference ------------------------------------------------------------------------


def test_reference_covers_the_shipped_stack(
    reference: dict[str, Any], shipped: dict[str, Any]
) -> None:
    assert reference["columns"] == shipped["columns"]
    assert set(reference["specs"]) == set(shipped["columns"])
    assert reference["batch_days"] == monitoring.BATCH_DAYS
    assert reference["bands"] == {"watch": 0.1, "alert": 0.2, "names": ["stable", "watch", "alert"]}
    assert reference["n_bins"] == 10
    for column, spec in reference["specs"].items():
        shares = spec["reference"]
        assert math.isclose(sum(shares.values()), 1.0, abs_tol=1e-9), column
        assert spec["kind"] in ("numeric", "categorical")
        if spec["kind"] == "numeric":
            assert spec["edges"] == sorted(spec["edges"])


def test_edges_are_calibrated_on_the_accepted_windows(reference: dict[str, Any]) -> None:
    calibration = reference["calibration"]
    assert calibration["windows"] == ["val", "test"]
    assert calibration["n_batches"] == len(calibration["batches"]) == 10
    worst = calibration["worst_batch_psi"]
    for column in reference["columns"]:
        batch_max = max(float(b["psi"][column]) for b in calibration["batches"])
        assert math.isclose(worst[column], batch_max, abs_tol=1e-12)
        assert reference["alert_edges"][column] == max(monitoring.PSI_ALERT, worst[column])
    assert reference["score_alert_edge"] == max(
        monitoring.PSI_ALERT, calibration["worst_batch_score_psi"]
    )
    assert reference["feature_count_edge"] == max(calibration["leave_one_out_counts"])
    not_alertable = reference["not_alertable"]["columns"]
    assert not_alertable and all(c.endswith("_te") for c in not_alertable)
    assert set(reference["alertable_columns"]) | set(not_alertable) == set(reference["columns"])


def test_the_reference_is_readable_by_the_module(reference: dict[str, Any]) -> None:
    loaded = monitoring.DriftReference.from_dict(reference)
    assert loaded.columns == reference["columns"]
    assert loaded.feature_count_edge == reference["feature_count_edge"]


# --- the drift job over the in-control batches ----------------------------------------------------


def test_drift_job_reads_no_labels_and_reproduces_stage_2(drift: dict[str, Any]) -> None:
    assert drift["reads_labels"] is False
    against = drift["against_stage_2"]
    assert against["n_shared_columns"] > 0
    assert against["n_exact_to_1e_9"] == against["n_shared_columns"]
    assert against["max_abs_difference_test"] == 0.0
    cache = drift["scores_against_stage_6_cache"]
    assert cache["max_abs_difference_val"] == 0.0
    assert cache["max_abs_difference_test"] == 0.0


def test_the_textbook_band_fires_on_every_in_control_week_and_the_calibrated_rule_on_none(
    drift: dict[str, Any],
) -> None:
    in_control = drift["in_control"]
    assert min(in_control["textbook_alert_band_per_batch"]) > 0
    assert in_control["calibrated_alerts_per_batch"] == [0] * len(drift["batches"])
    assert not any(in_control["drift_alert_per_batch"])
    assert in_control["longest_run_of_drift_alerts"] == 0
    loo = in_control["leave_one_out"]
    assert loo["n_batches"] == len(drift["batches"])
    assert loo["rate_any_alert"] == 0.0
    assert loo["rate_score_alert"] == 0.0
    assert max(loo["counts"]) == drift["calibration"]["feature_count_edge"]
    assert loo["min_textbook_alert_band"] > 0


def test_per_batch_records_are_consistent(drift: dict[str, Any], reference: dict[str, Any]) -> None:
    for batch in drift["batches"]:
        aggregate = batch["aggregate"]
        assert aggregate["n_stable"] + aggregate["n_watch"] + aggregate["n_alert_band"] == len(
            reference["columns"]
        )
        assert aggregate["n_alertable"] == len(reference["alertable_columns"])
        assert batch["score"]["alert"] == (batch["score"]["psi"] > reference["score_alert_edge"])
        assert batch["drift_alert"] == (batch["score"]["alert"] or aggregate["feature_alert"])
        assert batch["n_rows"] > 0


# --- the decay job over the in-control batches ---------------------------------------------------


def test_decay_reference_is_stage_6s_bootstrap(
    decay: dict[str, Any], variance: dict[str, Any]
) -> None:
    assert decay["reads_labels"] is True
    reference = decay["reference"]
    recorded = variance["by_row"]["per_model"]["shipped"]
    for key in ("point", "low", "high"):
        assert math.isclose(reference["pr_auc"][key], recorded[key], abs_tol=1e-12)
    assert reference["max_abs_difference_vs_stage_6"] == 0.0
    assert reference["n_boot"] == variance["n_boot"]
    assert reference["seed"] == variance["seed"]


def test_no_in_control_batch_is_a_decay_and_the_headroom_is_recorded(
    decay: dict[str, Any],
) -> None:
    in_control = decay["in_control"]
    evaluated = [b for b in decay["batches"] if b["status"] == "evaluated"]
    assert in_control["n_evaluated"] == len(evaluated) == len(decay["batches"])
    assert in_control["n_refused"] == 0
    assert in_control["n_decay"] == 0
    assert in_control["decay_tolerance"] == lifecycle.DECAY_TOLERANCE
    assert in_control["largest_drop"] == max(b["drop"]["point"] for b in evaluated)
    assert in_control["largest_drop"] < lifecycle.DECAY_TOLERANCE
    assert math.isclose(
        in_control["headroom"], lifecycle.DECAY_TOLERANCE - in_control["largest_drop"]
    )
    for batch in evaluated:
        assert batch["decay"] == lifecycle.decay_flag(batch)
        assert math.isclose(
            batch["drop"]["point"], decay["reference"]["pr_auc"]["point"] - batch["pr_auc"]["point"]
        )
        assert batch["matures_at_day"] <= decay["as_of_day"]
        assert 0 < batch["precision"] < 1 and 0 < batch["recall"] < 1


# --- the constants against stage 6 ---------------------------------------------------------------


def test_the_promotion_margin_sits_just_above_the_stage_6_noise_band(
    variance: dict[str, Any],
) -> None:
    band = variance["noise_band"]
    half_width = float(band["half_width"])
    assert half_width < lifecycle.PROMOTION_MARGIN
    # Rounded up at the second decimal, and nothing more than that.
    assert math.ceil(half_width * 100) / 100 == lifecycle.PROMOTION_MARGIN
    by_card = max(p["difference"]["half_width"] for p in variance["by_card"]["pairs"])
    by_row = max(p["difference"]["half_width"] for p in variance["by_row"]["pairs"])
    assert math.isclose(half_width, max(by_card, by_row))
    assert band["from"] == "by_card"


# --- adversarial validation ------------------------------------------------------------------------


def test_adversarial_fits_are_internally_consistent(
    adversarial: dict[str, Any], shipped: dict[str, Any]
) -> None:
    assert adversarial["n_columns"] == len(shipped["columns"])
    fits = adversarial["fits"]
    assert list(fits) == ["full", "without_target_encodings", "without_entity_and_clock"]
    for fit in fits.values():
        assert 0.5 <= fit["auc"]["out_of_fold"] <= 1.0
        assert len(fit["auc"]["per_fold"]) == adversarial["model"]["n_folds"]
        shares = [t["gain_share"] for t in fit["top"]]
        assert shares == sorted(shares, reverse=True)
        assert [t["rank"] for t in fit["top"]] == list(range(1, len(fit["top"]) + 1))
        total = sum(g["gain_share"] for g in fit["by_group"].values())
        assert math.isclose(total, 1.0, abs_tol=1e-9)
        assert sum(g["n_columns"] for g in fit["by_group"].values()) == fit["n_columns"]
    assert fits["full"]["n_columns"] > fits["without_target_encodings"]["n_columns"]
    assert (
        fits["without_target_encodings"]["n_columns"]
        > fits["without_entity_and_clock"]["n_columns"]
    )


def test_the_target_encodings_separate_the_windows_by_construction(
    adversarial: dict[str, Any],
) -> None:
    full = adversarial["fits"]["full"]
    assert full["auc"]["out_of_fold"] > 0.99
    assert full["by_group"]["target_encoding"]["gain_share"] > 0.9
    assert full["top"][0]["column"].endswith("_te")
    without = adversarial["fits"]["without_target_encodings"]
    assert without["by_group"]["target_encoding"]["n_columns"] == 0
    assert without["auc"]["out_of_fold"] < full["auc"]["out_of_fold"]
    reduced = adversarial["fits"]["without_entity_and_clock"]
    assert reduced["entity_and_clock_share_of_top_10_gain"] == 0.0
    assert reduced["auc"]["out_of_fold"] < without["auc"]["out_of_fold"]


def test_the_reading_against_stage_2_names_its_columns(adversarial: dict[str, Any]) -> None:
    against = adversarial["against_stage_2"]
    assert against["read_on_fit"] == "without_target_encodings"
    assert against["n_shared_raw_columns"] > 0
    assert -1.0 <= against["spearman_gain_vs_stage_2_psi"]["rho"] <= 1.0
    top_30 = [t["column"] for t in adversarial["fits"]["without_target_encodings"]["top"]]
    for column in against["stage_2_top_10_in_adversarial_top_30"]:
        assert column in top_30 and column in against["stage_2_top_10_by_psi"]
    for column in against["time_inconsistent_columns_in_adversarial_top_30"]:
        assert column in top_30 and column in against["time_inconsistent_columns_in_stack"]
    for row in against["time_inconsistent_sources_in_adversarial_top_30"]:
        assert row["column"] in top_30 and row["flagged_sources"]


# --- the lifecycle demo ------------------------------------------------------------------------------


def test_demo_inputs_are_the_modules_constants(demo: dict[str, Any]) -> None:
    inputs = demo["inputs"]
    assert inputs["batch_days"] == monitoring.BATCH_DAYS
    assert inputs["maturity_days"] == lifecycle.MATURITY_DAYS
    assert inputs["promotion_margin"] == lifecycle.PROMOTION_MARGIN
    assert inputs["decay_tolerance"] == lifecycle.DECAY_TOLERANCE
    assert inputs["sustained_cycles"] == lifecycle.SUSTAINED_CYCLES
    assert demo["seed"] == config.SEED
    assert demo["stream"]["n_batches"] == len([c for c in demo["cycles"] if c["drift"]])


def test_drift_fires_from_the_injection_and_not_before(demo: dict[str, Any]) -> None:
    from_batch = demo["inputs"]["drift_from_batch"]
    for cycle in demo["cycles"]:
        drift = cycle["drift"]
        if drift is None:
            continue
        if drift["batch"] < from_batch:
            assert drift["drifted_rows"] == 0
            assert drift["drift_alert"] is False
        else:
            assert drift["drifted_rows"] == drift["n_rows"]
            assert drift["drift_alert"] is True
        assert drift["drift_alert"] == (drift["score_alert"] or drift["feature_alert"])


def test_labels_are_read_only_once_mature_and_each_batch_once(demo: dict[str, Any]) -> None:
    seen: list[int] = []
    maturity = demo["inputs"]["maturity_days"]
    for cycle in demo["cycles"]:
        for d in cycle["decay"]:
            assert d["matured_at_day"] <= cycle["as_of_day"]
            assert math.isclose(d["matured_at_day"], d["end_day"] + maturity)
            assert d["decay"] == lifecycle.decay_flag(
                {"drop": {"point": d["drop"], "excludes_zero": d["drop_excludes_zero"]}}
            )
            seen.append(d["batch"])
        for batch in cycle["immature"]:
            assert batch not in seen
    assert sorted(seen) == list(range(demo["stream"]["n_batches"]))


def test_the_retrain_decision_is_the_modules(demo: dict[str, Any]) -> None:
    history: list[dict[str, Any]] = []
    for cycle in demo["cycles"]:
        drift_alert = cycle["drift"]["drift_alert"] if cycle["drift"] else None
        decay = any(d["decay"] for d in cycle["decay"]) if cycle["decay"] else None
        history.append({"cycle": cycle["cycle"], "drift_alert": drift_alert, "decay": decay})
        decision = lifecycle.should_retrain(history)
        assert decision["retrain"] == cycle["decision"]["retrain"]
        assert decision["reason"] == cycle["decision"]["reason"]
        challenger = cycle.get("challenger")
        if challenger and challenger.get("promotion"):
            history = []


def test_every_gate_verdict_follows_from_its_numbers(demo: dict[str, Any]) -> None:
    trained = [c["challenger"] for c in demo["cycles"] if c.get("challenger", {}).get("trained")]
    assert trained, "the demo trained no challenger"
    assert len(trained) == demo["n_challengers"]
    verdicts = [t["gate"]["promote"] for t in trained]
    assert True in verdicts and False in verdicts, "the gate was exercised in one direction only"
    for challenger in trained:
        gate = challenger["gate"]
        difference = gate["difference"]
        assert gate["margin"] == lifecycle.PROMOTION_MARGIN
        assert gate["clears_margin"] == (difference["point"] >= gate["margin"])
        assert gate["above_noise"] == (difference["low"] > 0)
        assert gate["promote"] == (gate["clears_margin"] and gate["above_noise"])
        assert math.isclose(
            difference["point"], gate["challenger"]["point"] - gate["champion"]["point"]
        )
        assert gate["resample"] == "groups"
        assert challenger["fit"]["n_fit_rows"] > 0
        windows = challenger["windows"]
        assert windows["fit_end_dt"] == windows["calibration_start_dt"]
        assert windows["calibration_end_dt"] == windows["gate_start_dt"]


def test_promotion_moved_the_alias_and_rollback_moved_it_back(demo: dict[str, Any]) -> None:
    promotion = demo["promotion"]
    assert promotion is not None
    assert promotion["gate"]["promote"] is True
    rollback = demo["rollback"]
    assert rollback["performed"] is True
    before, after = rollback["aliases_before"], rollback["aliases_after"]
    assert before["production"] == promotion["version"]
    assert before["previous"] == promotion["previous_version"]
    assert after["production"] == before["previous"]
    assert after["previous"] is None
    assert after["rolled-back"] == before["production"]
    assert rollback["production_is_the_previous_version"] is True
    assert rollback["loaded"]["version"] == after["production"]
    assert rollback["smoke"]["max_abs_score_difference_serving_vs_batch"] == 0.0
