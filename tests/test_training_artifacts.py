"""Internal consistency of the stage 6 artifacts, checked without the data.

The artifacts are committed, so these run on a clean checkout and on every push. Three kinds of
check, as in stages 2 to 5:

- **The module is the source.** The comparison lists exactly the families and strategies
  `fraud_platform.modelling` offers, on the paths it assigns, with the hyperparameters it holds;
  the bands are the config costs put through `evaluation.cost_bands`; the sweep is the module's
  grid.
- **Numbers quoted from earlier stages are the earlier stages' numbers.** Row and positive counts
  per split are stage 1's from `reports/split_summary.json`; the split boundaries are config's.
- **Every decision is the rule applied to the numbers beside it.** The shipped family is the
  largest validation PR-AUC; each ablation verdict is the per-seed flags; the search picks the
  largest mean fold score; a confusion matrix sums to its split; an interval contains its point;
  the noise band is the widest paired half-width.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from fraud_platform import config, evaluation, features, modelling

REPORTS = config.REPORTS_DIR


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="module")
def comparison() -> dict[str, Any]:
    return load(REPORTS / "model_comparison.json")


@pytest.fixture(scope="module")
def ablations() -> dict[str, Any]:
    return load(REPORTS / "ablations.json")


@pytest.fixture(scope="module")
def search() -> dict[str, Any]:
    return load(REPORTS / "hyperparameter_search.json")


@pytest.fixture(scope="module")
def variance() -> dict[str, Any]:
    return load(REPORTS / "metric_variance.json")


@pytest.fixture(scope="module")
def curve() -> dict[str, Any]:
    return load(REPORTS / "threshold_curve.json")


@pytest.fixture(scope="module")
def points() -> dict[str, Any]:
    return load(REPORTS / "operating_points.json")


@pytest.fixture(scope="module")
def shap_global() -> dict[str, Any]:
    return load(REPORTS / "shap" / "shap_global.json")


@pytest.fixture(scope="module")
def shap_example() -> dict[str, Any]:
    return load(REPORTS / "shap" / "shap_example.json")


@pytest.fixture(scope="module")
def split_summary() -> dict[str, Any]:
    return load(REPORTS / "split_summary.json")


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    return load(config.ROOT / "models" / "shipped_model.json")


def split_counts(split_summary: dict[str, Any], split: str) -> tuple[int, int]:
    row = split_summary["splits"][split]
    return int(row["n_rows"]), int(row["n_fraud"])


def confusion_sums_to(point: dict[str, Any], n_rows: int, n_positive: int) -> None:
    assert point["tp"] + point["fp"] + point["fn"] + point["tn"] == n_rows
    assert point["tp"] + point["fn"] == n_positive
    assert point["alerts"] == point["tp"] + point["fp"]
    if point["tp"] + point["fp"]:
        assert point["precision"] == pytest.approx(point["tp"] / (point["tp"] + point["fp"]))
    assert point["recall"] == pytest.approx(point["tp"] / n_positive)


# --- the envelope -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "comparison",
        "ablations",
        "search",
        "variance",
        "curve",
        "points",
        "shap_global",
        "shap_example",
    ],
)
def test_envelope(name: str, request: pytest.FixtureRequest) -> None:
    report = request.getfixturevalue(name)
    assert report["stage"] == 6
    assert report["seed"] == config.SEED
    assert report["split"]["train_end_dt"] == config.TRAIN_END_DT
    assert report["split"]["val_end_dt"] == config.VAL_END_DT
    assert report["command"].startswith(".venv/bin/python scripts/train.py --sections")
    assert len(report["git_commit"]) == 40


# --- the comparison -----------------------------------------------------------------------------


def test_comparison_covers_the_module_grid(comparison: dict[str, Any]) -> None:
    names = {m["name"] for m in comparison["models"]}
    expected = {
        f"{family}__{strategy}"
        for family in modelling.FAMILIES
        for strategy in modelling.IMBALANCE_STRATEGIES
    }
    assert names == expected
    for model in comparison["models"]:
        assert model["path"] == modelling.PATH_BY_FAMILY[model["family"]]
        assert model["seed"] == config.SEED
        if model["imbalance"] == "smote":
            assert model["n_synthetic_rows"] > 0
            assert model["n_fit_rows"] == comparison["n_train_rows"] + model["n_synthetic_rows"]
        else:
            assert model["n_synthetic_rows"] == 0
            assert model["n_fit_rows"] == comparison["n_train_rows"]
    for family in modelling.FAMILIES:
        assert comparison["hyperparameters"][family] == modelling.DEFAULT_HYPERPARAMETERS[family]
    assert comparison["stack"] == modelling.DEFAULT_STACK.to_dict()


def test_comparison_counts_are_stage_1_counts(
    comparison: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    assert comparison["n_train_rows"] == split_counts(split_summary, "train")[0]
    for model in comparison["models"]:
        for split in ("val", "test"):
            n_rows, n_positive = split_counts(split_summary, split)
            assert model[split]["n_rows"] == n_rows
            assert model[split]["n_positive"] == n_positive
            assert model[split]["card_level"]["n_positive_cards"] <= n_positive
            assert model[split]["card_level"]["n_cards"] <= n_rows
            assert 0.0 <= model[split]["pr_auc"] <= 1.0
            assert 0.0 <= model[split]["roc_auc"] <= 1.0
        tuned = model["tuned_threshold"]
        assert tuned["fitted_on"] == "val"
        assert tuned["val"]["threshold"] == tuned["test"]["threshold"] == tuned["threshold"]
        confusion_sums_to(tuned["val"], *split_counts(split_summary, "val"))
        confusion_sums_to(tuned["test"], *split_counts(split_summary, "test"))


def test_logistic_regression_convergence_is_reported(comparison: dict[str, Any]) -> None:
    rows = comparison["logistic_regression_convergence"]
    assert {r["name"] for r in rows} == {
        f"logistic_regression__{s}" for s in modelling.IMBALANCE_STRATEGIES
    }
    for row in rows:
        assert (
            row["max_iter"] == modelling.DEFAULT_HYPERPARAMETERS["logistic_regression"]["max_iter"]
        )
        assert row["converged"] is (row["n_iter"] < row["max_iter"])
    for model in comparison["models"]:
        if model["family"] == "logistic_regression":
            assert model["convergence"]["converged"], model["name"]
        else:
            assert model["convergence"] is None


def test_shipped_family_is_the_largest_validation_pr_auc(comparison: dict[str, Any]) -> None:
    best = max(comparison["models"], key=lambda m: m["val"]["pr_auc"])
    selection = comparison["selection"]
    assert selection["shipped_name"] == best["name"]
    assert selection["shipped_family"] == best["family"]
    assert selection["shipped_imbalance"] == best["imbalance"]
    ranking = [r["val_pr_auc"] for r in selection["ranking_by_val_pr_auc"]]
    assert ranking == sorted(ranking, reverse=True)
    for row in selection["by_family"].values():
        scores = row["val_pr_auc_by_imbalance"]
        assert row["best_imbalance"] == max(scores, key=lambda k: scores[k])


def test_metric_agreement_is_read_off_the_models(comparison: dict[str, Any]) -> None:
    agreement = comparison["metric_agreement"]
    by_name = {m["name"]: m for m in comparison["models"]}
    assert agreement["rank_by_pr_auc"] == [
        m["name"] for m in sorted(comparison["models"], key=lambda m: -m["val"]["pr_auc"])
    ]
    assert agreement["rank_by_roc_auc"] == [
        m["name"] for m in sorted(comparison["models"], key=lambda m: -m["val"]["roc_auc"])
    ]
    discordant = 0
    for first, second in itertools.combinations(comparison["models"], 2):
        roc = first["val"]["roc_auc"] - second["val"]["roc_auc"]
        pr = first["val"]["pr_auc"] - second["val"]["pr_auc"]
        discordant += (roc > 0) != (pr > 0)
    assert agreement["n_discordant_pairs"] == discordant == len(agreement["discordant_pairs"])
    assert agreement["n_pairs"] == len(list(itertools.combinations(by_name, 2)))
    for name, fpr in agreement["false_positive_rate_at_tuned_threshold_test"].items():
        point = by_name[name]["tuned_threshold"]["test"]
        assert fpr == pytest.approx(point["fp"] / (point["fp"] + point["tn"]))


# --- the ablations ------------------------------------------------------------------------------


def test_ablation_stacks_are_named_and_referenced(ablations: dict[str, Any]) -> None:
    stacks = ablations["stacks"]
    ids = [s["id"] for s in stacks]
    assert len(ids) == len(set(ids)) == 12
    for entry in stacks:
        stack = modelling.FeatureStack(
            causal=entry["stack"]["causal"],
            graph=entry["stack"]["graph"],
            native_groups=tuple(entry["stack"]["native_groups"]),
        )
        assert entry["stack"]["name"] == stack.name
        assert [row["seed"] for row in entry["per_seed"]] == ablations["seeds"]
        summary = entry["summary"]["val_pr_auc"]
        values = [row["val_pr_auc"] for row in entry["per_seed"]]
        assert summary["mean"] == pytest.approx(sum(values) / len(values))
        assert summary["min"] == min(values) and summary["max"] == max(values)
        if entry["reference"] is None:
            assert "against_reference" not in entry
            continue
        assert entry["reference"] in ids
        reference = next(s for s in stacks if s["id"] == entry["reference"])
        against = entry["against_reference"]
        for diff, row, ref_row in zip(
            against["per_seed"], entry["per_seed"], reference["per_seed"], strict=True
        ):
            assert diff["seed"] == row["seed"] == ref_row["seed"]
            assert diff["val_pr_auc_difference"]["point"] == pytest.approx(
                row["val_pr_auc"] - ref_row["val_pr_auc"]
            )
            interval = diff["val_pr_auc_difference"]
            assert interval["low"] <= interval["point"] <= interval["high"]
            assert diff["excludes_zero"] is (interval["low"] > 0 or interval["high"] < 0)
        assert against["val_gain_positive_on_every_seed"] is all(
            d["val_pr_auc_difference"]["point"] > 0 for d in against["per_seed"]
        )
        assert against["val_interval_excludes_zero_on_every_seed"] is all(
            d["excludes_zero"] for d in against["per_seed"]
        )
    assert ablations["seeds"][0] == config.SEED


def test_ablation_decision_is_the_rule_applied(ablations: dict[str, Any]) -> None:
    by_id = {s["id"]: s for s in ablations["stacks"]}
    decision = ablations["decision"]

    def ships(stack_id: str) -> bool:
        against = by_id[stack_id]["against_reference"]
        return bool(
            against["val_gain_positive_on_every_seed"]
            and against["val_interval_excludes_zero_on_every_seed"]
        )

    def removal_shows_a_loss(stack_id: str) -> bool:
        against = by_id[stack_id]["against_reference"]
        return any(
            d["val_pr_auc_difference"]["point"] < 0 and d["excludes_zero"]
            for d in against["per_seed"]
        )

    def removal_hurts_on_every_seed(stack_id: str) -> bool:
        against = by_id[stack_id]["against_reference"]
        return bool(
            all(d["val_pr_auc_difference"]["point"] < 0 for d in against["per_seed"])
            and against["val_interval_excludes_zero_on_every_seed"]
        )

    assert decision["causal_block_ships"] is ships("prepared+causal")
    assert decision["graph_block_ships"] is ships("prepared+causal+graph")
    for group in features.NATIVE_GROUPS:
        assert decision["native_group_ships"][group] is removal_shows_a_loss(
            f"prepared+causal+graph|without={group}"
        )
        assert decision["superseded_first_draft"]["native_group_ships"][
            group
        ] is removal_hurts_on_every_seed(f"prepared+causal+graph|without={group}")
    shipped = modelling.FeatureStack(
        causal=decision["causal_block_ships"],
        graph=decision["graph_block_ships"],
        native_groups=tuple(g for g in features.NATIVE_GROUPS if decision["native_group_ships"][g]),
    )
    assert decision["shipped_stack"] == shipped.to_dict()
    two_by_two = ablations["velocity_against_native_c"]["two_by_two_val_pr_auc"]
    assert set(two_by_two) == {
        "prepared",
        "prepared|without=count",
        "prepared+causal",
        "prepared+causal|without=count",
    }


def test_ablations_use_the_shipped_family(
    ablations: dict[str, Any], comparison: dict[str, Any]
) -> None:
    assert ablations["family"] == comparison["selection"]["shipped_family"]
    assert ablations["imbalance"] == comparison["selection"]["shipped_imbalance"]
    assert ablations["hyperparameters"] == modelling.DEFAULT_HYPERPARAMETERS[ablations["family"]]


# --- the search ---------------------------------------------------------------------------------


def test_search_folds_never_fit_on_later_rows(search: dict[str, Any]) -> None:
    folds = search["cross_validation"]["folds"]
    assert len(folds) == search["cross_validation"]["n_folds"]
    previous_fit = 0
    for fold in folds:
        assert fold["fit_end_ts"] < fold["validate_start_ts"] <= fold["validate_end_ts"]
        assert fold["validate_end_ts"] < config.TRAIN_END_DT
        assert fold["n_fit"] > previous_fit
        assert fold["n_validate_positive"] > 0
        previous_fit = fold["n_fit"]


def test_search_selection_is_the_largest_mean_fold_score(
    search: dict[str, Any], ablations: dict[str, Any], comparison: dict[str, Any]
) -> None:
    trials = search["trials"]
    assert trials[0]["is_default"]
    default = modelling.DEFAULT_HYPERPARAMETERS[search["family"]]
    assert trials[0]["hyperparameters"] == {k: default.get(k) for k in search["space"]}
    for trial in trials:
        assert trial["mean_pr_auc"] == pytest.approx(
            sum(trial["fold_pr_auc"]) / len(trial["fold_pr_auc"])
        )
        for key, value in trial["hyperparameters"].items():
            assert value in search["space"][key]
    best = max(trials, key=lambda t: t["mean_pr_auc"])
    assert search["selection"]["best_index"] == best["index"]
    assert search["selection"]["best_hyperparameters"] == best["hyperparameters"]
    assert search["family"] == comparison["selection"]["shipped_family"]
    assert search["stack"] == ablations["decision"]["shipped_stack"]
    final = search["final"]
    shipped = search["shipped"]
    expected = (
        "tuned"
        if final["tuned"]["val"]["pr_auc"] >= final["default"]["val"]["pr_auc"]
        else "default"
    )
    assert shipped["label"] == expected
    assert shipped["hyperparameters"] == final[expected]["hyperparameters"]
    assert shipped["val_pr_auc"] == final[expected]["val"]["pr_auc"]


def test_shipped_bundle_matches_the_search(search: dict[str, Any], bundle: dict[str, Any]) -> None:
    assert bundle["family"] == search["family"]
    assert bundle["imbalance"] == search["imbalance"]
    assert bundle["stack"] == search["stack"]
    assert bundle["hyperparameters"] == search["shipped"]["hyperparameters"]
    assert len(bundle["columns"]) == search["n_columns"]
    # The binary is rebuilt by `make train` and not committed; only its name is asserted here.
    assert bundle["model_file"] == "shipped_model.ubj"


# --- the variance -------------------------------------------------------------------------------


def test_variance_intervals_are_paired_on_the_same_resamples(
    variance: dict[str, Any], comparison: dict[str, Any]
) -> None:
    names = variance["models"]
    assert names == [m["name"] for m in comparison["models"]] + ["shipped"]
    assert variance["n_boot"] == 1_000
    for key in ("by_row", "by_card", "card_level_by_card"):
        block = variance[key]
        assert block["n_boot"] == 1_000
        assert block["n_valid"] == 1_000
        assert set(block["per_model"]) == set(names)
        for interval in block["per_model"].values():
            assert interval["low"] <= interval["point"] <= interval["high"]
            assert interval["half_width"] == pytest.approx((interval["high"] - interval["low"]) / 2)
        assert len(block["pairs"]) == len(list(itertools.combinations(names, 2)))
        widest = max(block["pairs"], key=lambda p: p["difference"]["half_width"])
        assert block["widest_paired_half_width"]["half_width"] == widest["difference"]["half_width"]
        for pair in block["pairs"]:
            diff = pair["difference"]
            assert diff["point"] == pytest.approx(
                block["per_model"][pair["first"]]["point"]
                - block["per_model"][pair["second"]]["point"]
            )
            assert pair["excludes_zero"] is (diff["low"] > 0 or diff["high"] < 0)
            if pair["sign_agreement"] is not None:
                assert 0.0 <= pair["sign_agreement"] <= 1.0
    noise = variance["noise_band"]
    assert noise["half_width"] == max(
        variance[k]["widest_paired_half_width"]["half_width"] for k in ("by_row", "by_card")
    )
    assert variance["by_row"]["resample"] == "rows" and variance["by_card"]["resample"] == "groups"


def test_variance_points_are_the_comparison_points(
    variance: dict[str, Any], comparison: dict[str, Any], search: dict[str, Any]
) -> None:
    for model in comparison["models"]:
        assert variance["by_row"]["per_model"][model["name"]]["point"] == pytest.approx(
            model["test"]["pr_auc"]
        )
        assert variance["card_level_by_card"]["per_model"][model["name"]]["point"] == pytest.approx(
            model["test"]["card_level"]["pr_auc"]
        )
    assert variance["by_row"]["per_model"]["shipped"]["point"] == pytest.approx(
        search["shipped"]["test_pr_auc"]
    )


# --- thresholds, bands, segments ----------------------------------------------------------------


def test_threshold_curve_is_the_module_sweep(
    curve: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    assert curve["thresholds"] == list(evaluation.SWEEP_THRESHOLDS)
    for kind in ("calibrated", "raw_score"):
        for split in ("val", "test"):
            rows = curve[kind][split]
            assert [r["threshold"] for r in rows] == curve["thresholds"]
            alerts = [r["alerts"] for r in rows]
            assert alerts == sorted(alerts, reverse=True)
            n_rows, n_positive = split_counts(split_summary, split)
            for row in rows:
                confusion_sums_to(row, n_rows, n_positive)
                assert row["alerts_per_day"] == pytest.approx(
                    row["alerts"] / curve["n_days"][split]
                )
    edges = curve["band_edges"]
    bands = evaluation.cost_bands()
    assert edges["low"] == bands["low"] and edges["high"] == bands["high"]


def test_operating_points_carry_the_tuned_vertex_and_the_bands(
    points: dict[str, Any], split_summary: dict[str, Any], search: dict[str, Any]
) -> None:
    vertex = points["tuned_vertex"]
    assert vertex["fitted_on"] == "val"
    assert vertex["val"]["threshold"] == vertex["test"]["threshold"]
    confusion_sums_to(vertex["val"], *split_counts(split_summary, "val"))
    confusion_sums_to(vertex["test"], *split_counts(split_summary, "test"))
    assert points["primary_metric"]["test"]["pr_auc"] == pytest.approx(
        search["shipped"]["test_pr_auc"]
    )
    assert points["model"]["hyperparameters"] == search["shipped"]["hyperparameters"]

    bands = points["bands"]
    derived = evaluation.cost_bands()
    assert bands["low"] == derived["low"] and bands["high"] == derived["high"]
    assert bands["costs"]["missed_fraud"] == config.COST_MISSED_FRAUD
    assert bands["costs"]["false_decline"] == config.COST_FALSE_DECLINE
    assert bands["costs"]["review"] == config.COST_REVIEW
    assert bands["costs"]["status"].startswith("assumption")
    assert bands["middle_band_exists"]
    for split in ("val", "test"):
        n_rows, n_positive = split_counts(split_summary, split)
        report = bands[split]
        assert sum(b["n"] for b in report["bands"].values()) == n_rows
        assert sum(b["fraud"] for b in report["bands"].values()) == n_positive
        expected = (
            report["bands"]["approve"]["fraud"] * config.COST_MISSED_FRAUD
            + report["bands"]["review"]["n"] * config.COST_REVIEW
            + (report["bands"]["block"]["n"] - report["bands"]["block"]["fraud"])
            * config.COST_FALSE_DECLINE
        )
        assert report["realised_cost"] == pytest.approx(expected)
        assert report["approve_everything_cost"] == pytest.approx(
            n_positive * config.COST_MISSED_FRAUD
        )
    assert points["calibration"]["fitted_on"] == "val"
    for kind in ("raw_score", "calibrated"):
        for split in ("val", "test"):
            table = points["calibration"][kind][split]
            assert sum(row["n"] for row in table["bins"]) == split_counts(split_summary, split)[0]


def test_segments_partition_the_splits(
    points: dict[str, Any], split_summary: dict[str, Any]
) -> None:
    segments = points["segments"]
    assert segments["key"] == config.SEGMENT_COLUMN
    rows = segments["per_segment"]
    assert {r["segment"] for r in rows} == {"C", "H", "R", "S", "W"}
    for split in ("val", "test"):
        n_rows, n_positive = split_counts(split_summary, split)
        assert sum(r["base_rate"][split]["n"] for r in rows) == n_rows
        assert sum(r["base_rate"][split]["successes"] for r in rows) == n_positive
        assert (
            sum(
                b["n_rows"]
                for b in (v[split] for v in segments["global_bands_applied_per_segment"].values())
            )
            == n_rows
        )
    for row in rows:
        assert (
            row["base_rate"]["val"]["low"]
            <= row["base_rate"]["val"]["point"]
            <= row["base_rate"]["val"]["high"]
        )
        if "tuned_vertex" in row:
            assert (
                row["tuned_vertex"]["val"]["threshold"] == row["tuned_vertex"]["test"]["threshold"]
            )
            assert row["calibration"]["fitted_on"] == "val"


# --- shap ---------------------------------------------------------------------------------------


def test_shap_global_is_ranked_and_blocked(
    shap_global: dict[str, Any], bundle: dict[str, Any]
) -> None:
    rows = shap_global["mean_abs_shap"]
    assert [r["column"] for r in rows] and len(rows) == len(bundle["columns"])
    assert {r["column"] for r in rows} == set(bundle["columns"])
    values = [r["mean_abs_shap"] for r in rows]
    assert values == sorted(values, reverse=True)
    assert [r["rank"] for r in rows] == list(range(1, len(rows) + 1))
    total = sum(values)
    for name, block in shap_global["by_block"].items():
        assert 0.0 <= block["mean_abs_shap_share"] <= 1.0 + 1e-9, name
        assert block["mean_abs_shap_sum"] == pytest.approx(block["mean_abs_shap_share"] * total)
    assert shap_global["method"]["n_rows"] <= 5_000
    assert "card" in shap_global["caveat"]["labels"]


def test_shap_example_adds_up(shap_example: dict[str, Any], shap_global: dict[str, Any]) -> None:
    assert shap_example["expected_value"] == shap_global["expected_value"]
    assert shap_example["transaction"]["label"] == 1
    assert 0.0 <= shap_example["transaction"]["score"] <= 1.0
    assert len(shap_example["contributions"]) == 20
    magnitudes = [abs(c["shap"]) for c in shap_example["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


# --- the stage 3 decisions the cache is built from ------------------------------------------------


def test_read_decisions_reads_the_stage_3_artifacts() -> None:
    from fraud_platform import prepare

    decisions = prepare.read_decisions()
    encoding = load(REPORTS / "encoding_spec.json")["encoders"]["target"]
    assert decisions["lag_days"] == int(encoding["chosen_lag_days"])
    assert decisions["smoothing"] == float(encoding["chosen_smoothing"])
    assert (
        decisions["v_strategy"] == load(REPORTS / "prep" / "v_reduction.json")["decision"]["chosen"]
    )
    assert decisions["d_origin_columns"] == list(
        load(REPORTS / "prep" / "transforms.json")["d_normalisation"]["applied_to"]
    )
    plan = prepare.read_column_plan()
    assert plan["n_columns"] == 434


def test_readers_refuse_a_directory_without_the_artifacts(tmp_path: Path) -> None:
    from fraud_platform import prepare

    with pytest.raises(FileNotFoundError):
        prepare.read_decisions(tmp_path)
    with pytest.raises(FileNotFoundError):
        prepare.read_column_plan(tmp_path)
