"""The stage 8 artifacts, checked against the modules and the earlier stages.

Four artifacts, none of which needs the data or a server to be read: the registration record
(`reports/serving_bundle.json`), the latency benchmark (`reports/latency.json`), the parity
measurement (`reports/parity.json`) and the load test (`loadtest/results.json`, with the
markdown rendered from it). Each is checked for the things a reader would otherwise have to
take on trust: that the bundle reproduces stage 6 to the last digit, that every summary number
in the latency artifact is recomputed from the per-process runs it claims to summarise, that
the pinning comment in `serving.py` rests on a measured difference in the direction it states,
that the parity run was bit-identical, and that the load test's saturation point is the rule
applied to its own sweep and its markdown is its JSON rendered.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fraud_platform import (
    config,
    evaluation,
    features,
    graph_features,
    modelling,
    online_store,
    serving,
)

ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict[str, Any]:
    return dict(json.loads((ROOT / path).read_text()))


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    return load("reports/serving_bundle.json")


@pytest.fixture(scope="module")
def latency() -> dict[str, Any]:
    return load("reports/latency.json")


@pytest.fixture(scope="module")
def parity() -> dict[str, Any]:
    return load("reports/parity.json")


@pytest.fixture(scope="module")
def loadtest() -> dict[str, Any]:
    return load("loadtest/results.json")


# --- the registration record ---------------------------------------------------------------------------


def test_the_bundle_was_registered_under_the_alias_the_service_loads(
    bundle: dict[str, Any],
) -> None:
    registry = bundle["registry"]
    assert registry["uri"] == serving.registry_uri() == "models:/fraud-detector@production"
    assert registry["model_name"] == serving.MODEL_NAME
    assert registry["alias"] == serving.MODEL_ALIAS
    assert registry["loaded_back"]["version"] == registry["version"] >= 1
    assert registry["python_model"] == serving.MODEL_CODE.name
    assert sorted(registry["artifacts"]) == ["booster", "preparation", "serving"]


def test_the_bundle_carries_the_shipped_model_and_the_stage_3_contract(
    bundle: dict[str, Any],
) -> None:
    shipped = load("models/shipped_model.json")
    schema_report = load("reports/prep/schema.json")
    assert bundle["bundle"]["columns"] == len(shipped["columns"])
    assert bundle["bundle"]["stack"] == shipped["stack"]
    assert bundle["bundle"]["schema_columns_declared"] == len(schema_report["schema"]["columns"])
    # The serving contract is the stage 3 contract without the label, and nothing else.
    assert (
        bundle["bundle"]["schema_columns_serving"]
        == bundle["bundle"]["schema_columns_declared"] - 1
    )
    assert bundle["bundle"]["raw_columns"] == load("reports/column_plan.json")["n_columns"]


def test_the_band_edges_reproduce_stage_6(bundle: dict[str, Any]) -> None:
    bands = evaluation.cost_bands()
    points = load("reports/operating_points.json")
    edges = bundle["band_edges"]
    assert edges["on_probability"] == {"low": bands["low"], "high": bands["high"]}
    assert edges["on_probability"] == {k: points["band_edges"][k] for k in ("low", "high")}
    recorded = points["bands"]["edges_as_raw_score"]
    for name in ("low", "high"):
        assert edges["as_raw_score_stage_6"][name] == recorded[name]
        assert abs(edges["as_raw_score_this_fit"][name] - recorded[name]) <= 1e-12
    assert edges["max_abs_difference"] <= 1e-12


def test_the_serving_path_reproduces_the_cached_scores_and_the_shap_example(
    bundle: dict[str, Any],
) -> None:
    smoke = bundle["smoke"]
    assert smoke["n_rows"] >= 100
    assert smoke["max_abs_score_difference_vs_this_scripts_batch"] <= 1e-6
    assert smoke["max_abs_score_difference_batch_vs_stage_6_cache"] <= 1e-6
    example = load("reports/shap/shap_example.json")
    shap = bundle["shap"]
    assert shap["found"] and shap["transaction_id"] == example["transaction"][config.ID_COLUMN]
    assert shap["score_stage_6"] == example["transaction"]["score"]
    assert abs(shap["score_serving"] - shap["score_stage_6"]) <= 1e-6
    assert abs(shap["base_value_serving"] - example["expected_value"]) <= 1e-6
    assert (
        shap["n_factors_compared"] == shap["n_factors_in_artifact"] == len(example["contributions"])
    )
    assert shap["max_abs_contribution_difference"] <= 1e-6
    assert (
        shap["top_factor_serving"]
        == shap["top_factor_stage_6"]
        == example["contributions"][0]["column"]
    )


def test_the_calibrator_breakpoints_reproduce_sklearn(bundle: dict[str, Any]) -> None:
    calibrator = bundle["calibrator"]
    assert calibrator["n_breakpoints"] > 2
    assert calibrator["fitted_on"] == "val"
    assert calibrator["max_abs_difference_interp_vs_sklearn_on_val"] <= 1e-12


# --- the latency benchmark -------------------------------------------------------------------------------


def test_every_repeat_timed_every_family_on_the_same_rows(latency: dict[str, Any]) -> None:
    runs = latency["runs"]
    assert len(runs) == latency["summary"]["n_repeats"] >= 3
    assert len({run["pid"] for run in runs}) == len(runs), "every repeat is its own process"
    n = latency["method"]["rows"]["n"]
    for run in runs:
        for family in ("xgboost", "random_forest", "logistic_regression"):
            assert run["pinned"][family]["n"] == n
        for family in ("xgboost", "random_forest"):
            assert run["unpinned"][family]["n"] == n
    assert latency["method"]["rows"]["split"] == "test"
    assert len(latency["method"]["rows"]["positions"]) == n


def test_the_families_are_at_the_shipped_configuration(latency: dict[str, Any]) -> None:
    shipped = load("models/shipped_model.json")
    models = latency["models"]
    assert models["stack"] == shipped["stack"]
    assert models["n_columns"] == len(shipped["columns"])
    assert models["hyperparameters"]["xgboost"] == shipped["hyperparameters"]
    for family in ("random_forest", "logistic_regression"):
        assert models["hyperparameters"][family] == modelling.DEFAULT_HYPERPARAMETERS[family]
    assert models["convergence"]["converged"] is True
    assert latency["pinning"]["inference_threads"] == serving.INFERENCE_THREADS == 1


def _spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "max_over_min": float(array.max() / array.min()),
    }


def test_the_summary_is_the_runs_recomputed(latency: dict[str, Any]) -> None:
    runs = latency["runs"]
    summary = latency["summary"]
    for family, spread in summary["pinned_p50_by_family"].items():
        assert spread == _spread([run["pinned"][family]["p50"] for run in runs])
    for family, spread in summary["ratio_p50_to_xgboost"].items():
        assert spread == _spread(
            [run["pinned"][family]["p50"] / run["pinned"]["xgboost"]["p50"] for run in runs]
        )
    for family, block in latency["pinning"]["comparison"].items():
        assert block["unpinned_over_pinned_p50"] == _spread(
            [run["unpinned"][family]["p50"] / run["pinned"][family]["p50"] for run in runs]
        )
    for name, spread in summary["serving_path_p50_ms"].items():
        assert spread == _spread([run["serving_path"]["steps"][name]["p50"] for run in runs])


def test_the_pinning_comment_rests_on_a_measured_difference_in_its_direction(
    latency: dict[str, Any],
) -> None:
    """`serving.PIN_REASON` points here: the pinned single-row call is the faster one."""
    assert serving.PIN_REASON.startswith("reports/latency.json")
    comparison = latency["pinning"]["comparison"]["xgboost"]
    assert comparison["unpinned_over_pinned_p50"]["min"] > 1.0
    # And the counterpoint the artifact keeps beside it: on a batch, the threads help.
    batch = latency["pinning"]["batch"]
    assert batch["rows"] >= 1_000
    assert batch["pinned_over_unpinned_p50"]["min"] > 1.0


def test_the_model_call_is_a_small_part_of_the_request(latency: dict[str, Any]) -> None:
    steps = latency["summary"]["serving_path_p50_ms"]
    assert steps["predict"]["median"] < steps["apply_preparation"]["median"]
    whole = latency["summary"]["serving_path_whole_no_store_p50_ms"]["median"]
    assert whole > sum(steps[name]["median"] for name in ("predict", "calibrate_and_band"))
    assert latency["summary"]["store_p50_ms"]["memory"]["get_features"]["median"] > 0


# --- the parity measurement --------------------------------------------------------------------------------


def test_the_parity_run_was_bit_identical_on_every_feature(parity: dict[str, Any]) -> None:
    expected = features.feature_names() + list(
        graph_features.graph_feature_names(online_store.GRAPH_ON)
    )
    assert parity["batch"]["stage_4"] + parity["batch"]["stage_5"] == expected
    for name, result in parity["backends"].items():
        assert result["bit_identical"] is True, name
        assert result["max_abs_difference_over_all_features"] == 0.0
        assert result["rows_outside_tolerance_over_all_features"] == 0
        assert list(result["per_feature"]) == expected
        for feature in result["per_feature"].values():
            assert feature["null_mask_mismatches"] == 0 and feature["rows_outside_tolerance"] == 0
    assert "memory" in parity["backends"]


def test_the_parity_stream_is_a_prefix_with_ties_in_it(parity: dict[str, Any]) -> None:
    stream = parity["stream"]
    assert stream["n_rows"] > 10_000
    assert stream["first_dt"] >= config.SECONDS_PER_DAY
    assert stream["last_dt"] < (1 + stream["n_days"]) * config.SECONDS_PER_DAY
    assert stream["ties"]["card"] > 0 and stream["ties"]["device_node"] > 0
    assert parity["tolerance"]["relative"] == 1e-9


# --- the load test -----------------------------------------------------------------------------------------


def _render(report: dict[str, Any]) -> str:
    spec = importlib.util.spec_from_file_location(
        "run_loadtest", ROOT / "loadtest" / "run_loadtest.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_loadtest"] = module
    spec.loader.exec_module(module)
    return str(module.render(report))


def test_the_sweeps_ran_every_level_on_the_same_rows_without_errors(
    loadtest: dict[str, Any],
) -> None:
    workers = [sweep["workers"] for sweep in loadtest["sweeps"]]
    assert workers[0] == 1, "the one-worker sweep is the one the stage asks for"
    for sweep in loadtest["sweeps"]:
        levels = sweep["levels"]
        concurrency = [level["concurrency"] for level in levels]
        assert concurrency == sorted(concurrency) and 1 in concurrency and len(concurrency) >= 4
        for level in levels:
            assert level["n_requests"] == loadtest["rows"]["n"]
            assert level["n_errors"] == 0 and level["n_ok"] == level["n_requests"]
            assert level["status_counts"] == {"200": level["n_requests"]}
            assert level["client_latency_ms"]["p50"] <= level["client_latency_ms"]["p99"]
            assert abs(level["throughput_rps"] - level["n_requests"] / level["wall_seconds"]) < 1e-9
            assert 0 < level["effective_concurrency"] <= level["concurrency"] + 1e-9
        assert sweep["server"]["workers"] == sweep["workers"]
        assert sweep["server"]["store"].startswith("redis")
        assert sweep["server"]["inference_threads"] == serving.INFERENCE_THREADS


def test_the_saturation_point_is_the_rule_applied_to_each_sweep(loadtest: dict[str, Any]) -> None:
    for sweep in loadtest["sweeps"]:
        levels = sorted(sweep["levels"], key=lambda level: level["concurrency"])
        sat = sweep["saturation"]
        point = None
        for previous, current in itertools.pairwise(levels):
            gain = current["throughput_rps"] / previous["throughput_rps"] - 1.0
            if point is None and gain < 0.10:
                point = current["concurrency"]
        assert sat["saturated_at_concurrency"] == point
        assert point is not None, "the sweep has to include the point where the worker saturates"
        best = max(levels, key=lambda level: level["throughput_rps"])
        assert sat["peak_at_concurrency"] == best["concurrency"]
        assert sat["peak_throughput_rps"] == best["throughput_rps"]
        # Past the saturation point the client latency grows with the number of clients.
        after = [level for level in levels if level["concurrency"] >= point]
        p50 = [level["client_latency_ms"]["p50"] for level in after]
        assert p50 == sorted(p50)


def test_one_worker_saturates_at_the_first_client(loadtest: dict[str, Any]) -> None:
    """The finding the stage records: the Python path sets the ceiling, and threads add nothing."""
    one = loadtest["sweeps"][0]
    assert one["workers"] == 1
    assert one["saturation"]["peak_at_concurrency"] == 1
    assert one["saturation"]["saturated_at_concurrency"] == 2


def test_the_markdown_is_the_json_rendered(loadtest: dict[str, Any]) -> None:
    assert (ROOT / "loadtest" / "results.md").read_text() == _render(loadtest)
