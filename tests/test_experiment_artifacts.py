"""Internal consistency of the stage 7 artifacts, checked without the data.

The artifacts are committed, so these run on a clean checkout and on every push. Three kinds of
check, as in stages 2 to 6:

- **The script is the source.** The variants are the script's list, the aggregate columns are
  the script's, the latencies swept are the script's, and the model is the shipped bundle's
  family, stack, columns and hyperparameters at the module defaults.
- **Numbers quoted from earlier stages are the earlier stages' numbers.** Both baselines
  reproduce the shipped model's test PR-AUC from `reports/operating_points.json`; the
  chronological test window has stage 1's row count.
- **Every decision is the rule applied to the numbers beside it.** A delta is the difference of
  its two points; an interval contains its point; `inflates` is the rule on the row resample;
  the fit-row and positive counts under each latency reconcile with the recent-window counts.

One more, because it is the stage's constraint: the leaky machinery lives in scripts/ and the
train-only guard it switches off is restored when it is done.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, data_loader, modelling, prepare

REPORTS = config.REPORTS_DIR
SRC = config.ROOT / "src" / "fraud_platform"
sys.path.insert(0, str(config.ROOT / "scripts"))

import label_latency as latency_script  # noqa: E402
import leakage_delta as leakage_script  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def leakage() -> dict[str, Any]:
    return load(REPORTS / "leakage_delta.json")


@pytest.fixture(scope="module")
def latency() -> dict[str, Any]:
    return load(REPORTS / "label_latency.json")


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    return load(config.ROOT / "models" / "shipped_model.json")


@pytest.fixture(scope="module")
def points() -> dict[str, Any]:
    return load(REPORTS / "operating_points.json")


@pytest.fixture(scope="module")
def split_summary() -> dict[str, Any]:
    return load(REPORTS / "split_summary.json")


def contains(interval: dict[str, Any]) -> bool:
    return bool(interval["low"] - 1e-12 <= interval["point"] <= interval["high"] + 1e-12)


# --- the model held fixed -----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["leakage", "latency"])
def test_the_model_is_the_shipped_one_at_the_module_defaults(
    request: pytest.FixtureRequest, name: str, bundle: dict[str, Any]
) -> None:
    report = request.getfixturevalue(name)
    assert report["stage"] == 7 and report["seed"] == config.SEED
    model = report["model"]
    assert model["family"] == bundle["family"] == "xgboost"
    assert model["imbalance"] == bundle["imbalance"] == "none"
    assert model["stack"] == bundle["stack"]
    assert model["n_columns"] == len(bundle["columns"])
    assert model["hyperparameters"] == bundle["hyperparameters"]
    assert model["hyperparameters"] == modelling.DEFAULT_HYPERPARAMETERS["xgboost"]
    assert model["hyperparameters_are_module_defaults"] is True
    assert report["split"]["train_end_dt"] == config.TRAIN_END_DT
    assert report["split"]["val_end_dt"] == config.VAL_END_DT


@pytest.mark.parametrize("name", ["leakage", "latency"])
def test_the_baseline_reproduces_the_shipped_model(
    request: pytest.FixtureRequest, name: str, points: dict[str, Any]
) -> None:
    report = request.getfixturevalue(name)
    shipped = points["primary_metric"]["test"]["pr_auc"]
    assert report["shipped_model_test_pr_auc"] == shipped
    if name == "leakage":
        baseline = next(v for v in report["variants"] if v["id"] == "baseline")
    else:
        baseline = next(r for r in report["runs"] if r["id"] == "reference")
    assert baseline["test"]["pr_auc"] == pytest.approx(shipped, abs=1e-3)


# --- experiment 1: the leakage delta ---------------------------------------------------------------


def test_the_variants_are_the_scripts_list(leakage: dict[str, Any]) -> None:
    expected = [dict(v) for v in leakage_script.VARIANTS]
    got = leakage["variants"]
    assert [v["id"] for v in got] == [v["id"] for v in expected]
    for variant, spec in zip(got, expected, strict=True):
        for key in (
            "letters",
            "description",
            "encoder_fit_rows",
            "entity_aggregates",
            "postprocessing",
            "split",
        ):
            assert variant[key] == spec[key]
    letters = {v["id"]: set(v["letters"]) for v in expected}
    assert letters["abc_chronological"] == {"a", "b", "c"}
    assert letters["abcd_random"] == {"a", "b", "c", "d"}
    assert leakage["deltas"]["per_variant"].keys() == {v["id"] for v in expected} - {"baseline"}


def test_the_aggregation_is_the_published_construction(
    leakage: dict[str, Any], bundle: dict[str, Any]
) -> None:
    aggregation = leakage["aggregation"]
    assert aggregation["key"] == config.ENTITY_ID_COLUMN
    assert aggregation["groups"] == {
        k: list(v) for k, v in leakage_script.AGGREGATE_COLUMNS.items()
    }
    assert aggregation["columns"] == leakage_script.aggregate_column_names()
    assert aggregation["n_columns"] == 2 * (
        len(config.COUNT_COLUMNS) + len(config.MATCH_COLUMNS) + len(config.TIMEDELTA_COLUMNS)
    )
    n_shipped = len(bundle["columns"])
    for variant in leakage["variants"]:
        expected = n_shipped + (aggregation["n_columns"] if variant["entity_aggregates"] else 0)
        assert variant["n_columns"] == expected, variant["id"]


def test_the_postprocessing_variant_is_the_baseline_fit(leakage: dict[str, Any]) -> None:
    by_id = {v["id"]: v for v in leakage["variants"]}
    for key in ("n_columns", "n_fit_rows", "n_fit_positive", "fit_seconds"):
        assert by_id["c_entity_mean_postprocessing"][key] == by_id["baseline"][key]


def test_the_windows_reconcile(leakage: dict[str, Any], split_summary: dict[str, Any]) -> None:
    chrono = leakage["population"]["chronological"]
    random = leakage["population"]["random"]
    total = sum(chrono[s]["n_rows"] for s in config.SPLIT_NAMES)
    assert total == sum(random[s]["n_rows"] for s in config.SPLIT_NAMES)
    for split in config.SPLIT_NAMES:
        assert chrono[split]["n_rows"] == split_summary["splits"][split]["n_rows"]
        assert chrono[split]["n_positive"] == split_summary["splits"][split]["n_fraud"]
    assert chrono["train"]["max_dt"] < config.TRAIN_END_DT <= chrono["val"]["min_dt"]
    assert chrono["val"]["max_dt"] < config.VAL_END_DT <= chrono["test"]["min_dt"]
    # The random split interleaves the windows; the chronological one cannot.
    assert random["test"]["min_dt"] < config.TRAIN_END_DT < random["train"]["max_dt"]
    assert random["test"]["seen_entity_row_share"] > chrono["test"]["seen_entity_row_share"]
    fractions = leakage["random_split"]["fractions"]
    assert fractions == list(leakage_script.RANDOM_SPLIT_FRACTIONS)
    assert random["train"]["n_rows"] == int(fractions[0] * total)


def test_every_delta_is_the_difference_of_its_points(leakage: dict[str, Any]) -> None:
    by_id = {v["id"]: v for v in leakage["variants"]}
    baseline = by_id["baseline"]
    for variant_id, entry in leakage["deltas"]["per_variant"].items():
        assert entry["split"] == by_id[variant_id]["split"]
        for label, metric in (
            ("pr_auc_by_row", "pr_auc"),
            ("roc_auc_by_row", "roc_auc"),
            ("pr_auc_by_card", "pr_auc"),
        ):
            block = entry[label]
            assert contains(block["variant"]) and contains(block["baseline"])
            assert contains(block["difference"])
            variant_point = by_id[variant_id]["test"][metric]
            baseline_point = baseline["test"][metric]
            assert block["variant"]["point"] == pytest.approx(variant_point)
            assert block["baseline"]["point"] == pytest.approx(baseline_point)
            assert block["difference"]["point"] == pytest.approx(variant_point - baseline_point)
            paired = entry["split"] == "chronological"
            assert (block["resample"] == "paired") == paired
            assert block["excludes_zero"] == (
                block["difference"]["low"] > 0 or block["difference"]["high"] < 0
            )
        primary = entry["pr_auc_by_row"]
        assert entry["inflates"] == (
            primary["difference"]["point"] > 0 and primary["excludes_zero"]
        )


def test_the_bootstrap_summaries_agree_with_the_deltas(leakage: dict[str, Any]) -> None:
    for label in ("pr_auc_by_row", "roc_auc_by_row", "pr_auc_by_card"):
        for kind in ("chronological", "random"):
            summary = leakage["bootstrap"][label][kind]
            assert summary["n_boot"] == leakage["deltas"]["n_boot"]
            assert summary["resample"] == ("groups" if label.endswith("card") else "rows")
            for interval in summary["per_model"].values():
                assert contains(interval)
            for pair in summary["pairs"]:
                assert contains(pair["difference"])


def test_the_guard_was_on_for_the_baseline_and_is_switched_off_by_name(
    leakage: dict[str, Any],
) -> None:
    assert leakage["guard"]["baseline"].startswith("on")
    assert "guard_switched_off" in leakage["guard"]["leaky_variants"]


# --- experiment 2: simulated label latency ---------------------------------------------------------


def test_the_latencies_are_the_scripts_and_the_simulation_says_so(latency: dict[str, Any]) -> None:
    simulation = latency["simulation"]
    assert simulation["latency_days"] == list(latency_script.LATENCY_DAYS)
    assert "simulated" in simulation["statement"]
    assert "not known" in simulation["statement"]
    assert set(simulation["treatments"]) == {*latency_script.TREATMENTS, "reference"}
    ids = [r["id"] for r in latency["runs"]]
    expected = ["reference"] + [
        f"{t}_{n:02d}" for n in latency_script.LATENCY_DAYS for t in latency_script.TREATMENTS
    ]
    assert ids == expected
    assert [row["latency_days"] for row in latency["sweep"]] == list(latency_script.LATENCY_DAYS)


def test_the_fit_rows_reconcile_with_the_recent_window(latency: dict[str, Any]) -> None:
    train = latency["train_window"]
    by_id = {r["id"]: r for r in latency["runs"]}
    reference = by_id["reference"]
    assert reference["n_fit_rows"] == train["n_rows"]
    assert reference["n_fit_positive"] == train["n_positive"]
    previous_recent = 0
    for n in latency_script.LATENCY_DAYS:
        immature = by_id[f"immature_{n:02d}"]
        excluded = by_id[f"excluded_{n:02d}"]
        assert immature["n_recent_rows"] == excluded["n_recent_rows"] > previous_recent
        assert immature["n_recent_positive"] == excluded["n_recent_positive"]
        previous_recent = immature["n_recent_rows"]
        assert immature["n_fit_rows"] == train["n_rows"]
        assert immature["n_fit_positive"] == train["n_positive"] - immature["n_recent_positive"]
        assert excluded["n_fit_rows"] == train["n_rows"] - excluded["n_recent_rows"]
        assert excluded["n_fit_positive"] == immature["n_fit_positive"]
        assert immature["recent_positive_share_of_train"] == pytest.approx(
            immature["n_recent_positive"] / train["n_positive"]
        )
        assert f"{config.TRAIN_END_DT} - {n} * {config.SECONDS_PER_DAY}" in immature["recent_rule"]


def test_every_sweep_entry_is_the_difference_of_its_points(latency: dict[str, Any]) -> None:
    by_id = {r["id"]: r for r in latency["runs"]}
    reference = by_id["reference"]
    for row in latency["sweep"]:
        for treatment in latency_script.TREATMENTS:
            entry = row[treatment]
            run = by_id[entry["id"]]
            assert entry["n_fit_rows"] == run["n_fit_rows"]
            for label, metric in (
                ("test_pr_auc", "pr_auc"),
                ("test_roc_auc", "roc_auc"),
                ("test_implied_fraud_rate", "implied_fraud_rate"),
            ):
                block = entry[label]
                assert contains(block["run"]) and contains(block["reference"])
                assert contains(block["difference"])
                assert block["run"]["point"] == pytest.approx(run["test"][metric])
                assert block["reference"]["point"] == pytest.approx(reference["test"][metric])
                assert block["difference"]["point"] == pytest.approx(
                    run["test"][metric] - reference["test"][metric]
                )
            assert entry["test_implied_minus_observed"] == pytest.approx(
                run["test"]["implied_fraud_rate"] - run["test"]["observed_fraud_rate"]
            )
            assert run["test"]["implied_over_observed"] == pytest.approx(
                run["test"]["implied_fraud_rate"] / run["test"]["observed_fraud_rate"]
            )
        gap = row["immature_minus_excluded_test_pr_auc"]
        immature = by_id[row["immature"]["id"]]["test"]["pr_auc"]
        excluded = by_id[row["excluded"]["id"]]["test"]["pr_auc"]
        assert gap["point"] == pytest.approx(immature - excluded)
        assert gap["low"] <= gap["point"] <= gap["high"]


def test_the_refit_reproduced_the_cache(latency: dict[str, Any]) -> None:
    assert latency["model"]["refit_reproduces_cache_within"] <= 1e-9
    assert latency["model"]["lag_days"] == prepare.read_decisions()["lag_days"]
    assert latency["model"]["smoothing"] == prepare.read_decisions()["smoothing"]
    assert latency["shipped_threshold"]["value"] > 0


# --- the constraint: leaks live in scripts/ ---------------------------------------------------------


def test_nothing_in_src_reassigns_the_guard() -> None:
    for path in SRC.glob("*.py"):
        text = path.read_text()
        assert "assert_train_only =" not in text, path.name
        assert "guard_switched_off" not in text, path.name
        assert "uid_" not in text, path.name


def test_the_guard_is_restored_after_a_leaky_fit() -> None:
    original = data_loader.assert_train_only
    late = pd.DataFrame({config.TIME_COLUMN: [config.TRAIN_END_DT + 1]})
    with pytest.raises(data_loader.TrainOnlyError):
        data_loader.assert_train_only(late, what="a late frame")
    with leakage_script.guard_switched_off():
        assert data_loader.assert_train_only(late, what="a late frame") is late
        assert prepare.assert_train_only(late, what="a late frame") is late
    assert data_loader.assert_train_only is original
    assert prepare.assert_train_only is original
    with pytest.raises(data_loader.TrainOnlyError):
        prepare.assert_train_only(late, what="a late frame")


def test_the_postprocessing_and_the_aggregation_reach_later_rows() -> None:
    keys = np.array(["x", "y", "x", "x", "z"])
    scores = np.array([0.1, 0.5, 0.4, 0.7, 0.9])
    out = leakage_script.entity_mean_postprocessing(scores, keys)
    assert out[0] == out[2] == out[3] == pytest.approx(0.4)
    assert out[1] == 0.5 and out[4] == 0.9

    frame = pd.DataFrame(
        {
            config.TIME_COLUMN: [86_400, 172_800, 259_200, 345_600],
            config.ENTITY_ID_COLUMN: ["a", "a", "a", "b"],
            **{c: [1.0, 2.0, 6.0, 4.0] for c in config.COUNT_COLUMNS},
            **{c: pd.Categorical(["F", "T", "T", None]) for c in config.MATCH_COLUMNS},
            **{c: [0.0, 1.0, np.nan, 3.0] for c in config.TIMEDELTA_COLUMNS},
        }
    )
    aggregates = leakage_script.full_data_entity_aggregates(frame)
    assert list(aggregates.columns) == leakage_script.aggregate_column_names()
    # The first row of entity a carries the mean of all three of its rows, the later two included.
    assert aggregates.loc[0, "uid_C1_mean"] == pytest.approx(3.0)
    assert aggregates.loc[0, "uid_M1_mean"] == pytest.approx(2 / 3)
    assert np.isnan(aggregates.loc[3, "uid_C1_std"])
    assert aggregates.loc[0, "uid_D1_origin_mean"] == pytest.approx(np.mean([1.0, 1.0]))
