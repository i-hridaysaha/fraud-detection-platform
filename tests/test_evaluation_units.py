"""The pieces of `fraud_platform.evaluation`, on hand-built vectors so every number is checkable.

The confusion arithmetic, the tuned vertex against a brute-force search, the alert budget, the
card-level collapse, the paired bootstrap's pairing, the calibrator and the reliability table,
the cost-band derivation and the band report's cost reconciliation, and the segment report.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from fraud_platform import config, evaluation


def ranked_case() -> tuple[np.ndarray, np.ndarray]:
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    return y, score


def test_perfect_ranking_scores_one() -> None:
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    assert evaluation.pr_auc(y, score) == 1.0
    assert evaluation.roc_auc(y, score) == 1.0


def test_at_threshold_arithmetic() -> None:
    y, score = ranked_case()
    point = evaluation.at_threshold(y, score, 0.65, n_days=2.0)
    assert (point["tp"], point["fp"], point["fn"], point["tn"]) == (2, 1, 1, 6)
    assert point["precision"] == pytest.approx(2 / 3)
    assert point["recall"] == pytest.approx(2 / 3)
    assert point["f1"] == pytest.approx(2 / 3)
    assert point["alerts"] == 3 and point["alerts_per_day"] == 1.5
    assert point["alert_rate"] == 0.3
    assert point["false_alarms_per_catch"] == 0.5
    assert point["false_positive_rate"] == pytest.approx(1 / 7)
    nothing = evaluation.at_threshold(y, score, 0.95)
    assert nothing["precision"] is None and nothing["false_alarms_per_catch"] is None
    assert nothing["recall"] == 0.0


def test_sweep_covers_the_grid() -> None:
    y, score = ranked_case()
    sweep = evaluation.threshold_sweep(y, score, 1.0)
    assert len(sweep) == 99
    assert sweep[0]["threshold"] == 0.01 and sweep[-1]["threshold"] == 0.99
    alerts = [row["alerts"] for row in sweep]
    assert alerts == sorted(alerts, reverse=True)


def test_best_f1_vertex_matches_a_brute_force_search() -> None:
    rng = np.random.default_rng(config.SEED)
    score = rng.random(500)
    y = (rng.random(500) < score).astype("int64")
    vertex = evaluation.best_f1_vertex(y, score, n_days=3.0)
    brute = max(evaluation.at_threshold(y, score, t)["f1"] for t in np.unique(score))
    assert vertex["f1"] == pytest.approx(brute)
    assert vertex["threshold"] in set(score)
    assert vertex["rule"].startswith("largest F1")


def test_alert_budget_vertex_hits_the_volume() -> None:
    y, score = ranked_case()
    vertex = evaluation.alert_budget_vertex(y, score, n_days=2.0, alerts_per_day=2.0)
    assert vertex["alerts"] == 4
    assert vertex["threshold"] == 0.6
    assert evaluation.alert_budget_vertex(y, score, 1.0, 100.0)["alerts"] == 10


def test_span_days() -> None:
    assert evaluation.span_days(np.array([0, 3 * config.SECONDS_PER_DAY])) == 3.0
    assert evaluation.span_days(np.array([5, 10])) == 1.0
    assert evaluation.span_days(np.zeros(0, dtype="int64")) == 1.0


def test_card_level_takes_the_maximum_score_and_any_fraud() -> None:
    y = np.array([0, 1, 0, 0, 0])
    score = np.array([0.2, 0.9, 0.5, 0.1, 0.3])
    keys = np.array(["a", "a", "b", "b", "c"])
    y_card, s_card = evaluation.card_level(y, score, keys)
    assert y_card.tolist() == [1, 0, 0]
    assert s_card.tolist() == [0.9, 0.5, 0.3]


# --- paired bootstrap ---------------------------------------------------------------------------


def test_paired_bootstrap_pairs_the_models() -> None:
    rng = np.random.default_rng(config.SEED)
    n = 400
    score = rng.random(n)
    y = (rng.random(n) < score).astype("int64")
    better = np.clip(score + 0.3 * (y - 0.5), 0, 1)
    report = evaluation.paired_bootstrap(
        y, {"a": score, "a_again": score.copy(), "b": better}, n_boot=100, seed=1
    )
    assert report["n_valid"] == 100 and report["resample"] == "rows"
    for name in ("a", "a_again", "b"):
        interval = report["per_model"][name]
        assert interval["low"] <= interval["point"] <= interval["high"]
    same = next(p for p in report["pairs"] if {p["first"], p["second"]} == {"a", "a_again"})
    assert same["difference"]["point"] == 0.0
    assert same["difference"]["half_width"] == 0.0
    assert same["sign_agreement"] is None and not same["excludes_zero"]
    gap = next(p for p in report["pairs"] if {p["first"], p["second"]} == {"a", "b"})
    assert gap["excludes_zero"] and gap["sign_agreement"] == 1.0
    assert report["widest_paired_half_width"]["half_width"] == max(
        p["difference"]["half_width"] for p in report["pairs"]
    )
    again = evaluation.paired_bootstrap(y, {"a": score, "b": better}, n_boot=100, seed=1)
    assert again["per_model"]["a"]["low"] == report["per_model"]["a"]["low"]


def test_group_bootstrap_draws_whole_groups() -> None:
    rng = np.random.default_rng(config.SEED)
    n = 300
    score = rng.random(n)
    y = (rng.random(n) < score).astype("int64")
    groups = np.repeat(np.arange(30), 10)
    report = evaluation.paired_bootstrap(y, {"a": score}, n_boot=50, seed=2, groups=groups)
    assert report["resample"] == "groups" and report["n_groups"] == 30
    assert report["pairs"] == [] and report["widest_paired_half_width"] is None
    assert report["positives_per_resample"]["min"] > 0


def test_bootstrap_metric_switch_and_the_summary_is_the_report() -> None:
    rng = np.random.default_rng(config.SEED)
    n = 400
    score = rng.random(n)
    y = (rng.random(n) < score).astype("int64")
    for metric, compute in (
        ("pr_auc", evaluation.pr_auc),
        ("roc_auc", evaluation.roc_auc),
        ("mean_score", evaluation.mean_score),
    ):
        drawn = evaluation.bootstrap_draws(y, {"a": score}, n_boot=30, seed=4, metric=metric)
        assert drawn["metric"] == metric and drawn["draws"].shape == (30, 1)
        assert drawn["points"]["a"] == pytest.approx(compute(y, score))
        summary = evaluation.paired_summary(drawn)
        report = evaluation.paired_bootstrap(y, {"a": score}, n_boot=30, seed=4, metric=metric)
        assert summary == report
        assert report["metric"] == metric
    assert evaluation.mean_score(y, score) == pytest.approx(float(score.mean()))
    with pytest.raises(ValueError, match="unknown metric"):
        evaluation.bootstrap_draws(y, {"a": score}, n_boot=5, seed=4, metric="f1")


def test_independent_difference_is_wider_than_the_paired_one() -> None:
    rng = np.random.default_rng(config.SEED)
    n = 400
    score = rng.random(n)
    y = (rng.random(n) < score).astype("int64")
    better = np.clip(score + 0.3 * (y - 0.5), 0, 1)
    paired = evaluation.paired_bootstrap(y, {"a": score, "b": better}, n_boot=200, seed=5)
    first = evaluation.bootstrap_draws(y, {"b": better}, n_boot=200, seed=6)
    second = evaluation.bootstrap_draws(y, {"a": score}, n_boot=200, seed=7)
    independent = evaluation.independent_difference(first, "b", second, "a")
    pair = paired["pairs"][0]
    assert independent["resample"] == "independent" and independent["metric"] == "pr_auc"
    assert independent["n_draws"] == 200
    assert independent["difference"]["point"] == pytest.approx(-pair["difference"]["point"])
    assert independent["difference"]["half_width"] > pair["difference"]["half_width"]
    assert independent["difference"]["low"] <= independent["difference"]["point"]
    assert independent["difference"]["point"] <= independent["difference"]["high"]


# --- calibration ----------------------------------------------------------------------------------


def test_calibrator_is_monotone_and_reliability_partitions() -> None:
    rng = np.random.default_rng(config.SEED)
    score = rng.random(2_000)
    y = (rng.random(2_000) < score**2).astype("int64")
    calibrator = evaluation.fit_calibrator(score, y)
    p = evaluation.calibrate(calibrator, score)
    order = np.argsort(score)
    assert np.all(np.diff(p[order]) >= 0)
    assert p.min() >= 0.0 and p.max() <= 1.0
    table = evaluation.reliability(y, p, n_bins=5)
    assert sum(row["n"] for row in table["bins"]) == 2_000
    assert table["brier"] == pytest.approx(brier_score_loss(y, p))
    assert (
        table["expected_calibration_error"]
        < evaluation.reliability(y, score, 5)["expected_calibration_error"]
    )


# --- cost bands -----------------------------------------------------------------------------------


def test_cost_bands_derive_from_the_config_costs() -> None:
    bands = evaluation.cost_bands()
    assert bands["low"] == pytest.approx(config.COST_REVIEW / config.COST_MISSED_FRAUD)
    assert bands["high"] == pytest.approx(1 - config.COST_REVIEW / config.COST_FALSE_DECLINE)
    assert bands["single_threshold_without_review"] == pytest.approx(
        config.COST_FALSE_DECLINE / (config.COST_FALSE_DECLINE + config.COST_MISSED_FRAUD)
    )
    assert bands["middle_band_exists"]
    assert bands["costs"]["status"].startswith("assumption")
    expensive_review = evaluation.cost_bands(10.0, 1.0, 5.0)
    assert not expensive_review["middle_band_exists"]
    with pytest.raises(ValueError):
        evaluation.cost_bands(0.0, 1.0, 0.5)


def test_bands_are_the_cheapest_action_at_every_probability() -> None:
    bands = evaluation.cost_bands(10.0, 1.0, 0.5)
    p = np.linspace(0.0, 1.0, 201)
    approve = p * 10.0
    review = np.full_like(p, 0.5)
    block = (1 - p) * 1.0
    cheapest = np.argmin(np.stack([approve, review, block]), axis=0)
    decided = evaluation.apply_bands(p, bands["low"], bands["high"])
    # Ties at the edges go to the riskier action by the inclusive rule.
    assert np.all(
        (decided == cheapest) | (np.isclose(p, bands["low"]) | np.isclose(p, bands["high"]))
    )


def test_band_report_reconciles_its_cost() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    p = np.array([0.01, 0.03, 0.2, 0.9, 0.6, 0.04, 0.95, 0.5])
    bands = evaluation.cost_bands(10.0, 1.0, 0.5)
    report = evaluation.band_report(y, p, bands, n_days=2.0)
    assert report["bands"]["approve"]["n"] == 3 and report["bands"]["approve"]["fraud"] == 1
    # 0.5 sits on the high edge and goes to block by the inclusive rule.
    assert report["bands"]["review"]["n"] == 1 and report["bands"]["block"]["n"] == 4
    assert report["bands"]["block"]["fraud"] == 1
    expected = 1 * 10.0 + 1 * 0.5 + 3 * 1.0
    assert report["realised_cost"] == pytest.approx(expected)
    assert report["approve_everything_cost"] == 30.0
    assert report["bands"]["review"]["per_day"] == 0.5
    single = report["single_threshold_at_cost_optimum"]
    assert single["cost"] == pytest.approx(single["fn"] * 10.0 + single["fp"] * 1.0)


def test_segment_report_tunes_each_segment_on_its_own_validation_rows() -> None:
    rng = np.random.default_rng(config.SEED)
    segments = {}
    for name in ("W", "C"):
        s_val = rng.random(400)
        s_test = rng.random(300)
        segments[name] = {
            "y_val": (rng.random(400) < s_val).astype("int64"),
            "s_val": s_val,
            "y_test": (rng.random(300) < s_test).astype("int64"),
            "s_test": s_test,
        }
    segments["empty"] = {
        "y_val": np.zeros(20, dtype="int64"),
        "s_val": rng.random(20),
        "y_test": np.zeros(10, dtype="int64"),
        "s_test": rng.random(10),
    }
    report = evaluation.segment_report(segments, {"val": 2.0, "test": 1.0}, evaluation.cost_bands())
    assert [row["segment"] for row in report] == ["W", "C", "empty"]
    assert "note" in report[2] and "tuned_vertex" not in report[2]
    for row in report[:2]:
        assert row["base_rate"]["val"]["method"] == "wilson"
        assert row["tuned_vertex"]["val"]["threshold"] == row["tuned_vertex"]["test"]["threshold"]
        assert row["calibration"]["fitted_on"] == "val"
        assert set(row["bands"]) == {"val", "test"}


def test_bootstrap_skips_a_resample_with_one_class() -> None:
    # Six rows with one positive: some resamples draw no positive and are dropped, not scored.
    y = np.array([1, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5])
    report = evaluation.paired_bootstrap(y, {"a": score}, n_boot=200, seed=3)
    assert 0 < report["n_valid"] < 200
    assert report["positives_per_resample"]["min"] >= 1


def test_reliability_with_more_bins_than_rows_drops_empty_bins() -> None:
    table = evaluation.reliability(np.array([0, 1, 1]), np.array([0.2, 0.6, 0.9]), n_bins=5)
    assert len(table["bins"]) == 3
    assert sum(row["n"] for row in table["bins"]) == 3
