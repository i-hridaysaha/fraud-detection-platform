"""Print every number the README quotes, read from the committed artifacts and formatted as the
README shows them.

Usage:
    make readme-numbers
    .venv/bin/python scripts/readme_numbers.py [--reports-dir reports] [--loadtest-dir loadtest]

The three tables in the README's Results section are this script's output pasted verbatim, and
every inline number the README carries is one of the named values printed after them.
tests/test_readme.py asserts both, so the README cannot quote a number the artifacts do not
hold. Nothing here is computed from the data: the script reads JSON and formats it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# The variants the leakage table shows, in the artifact's order, with the label the README uses.
LEAKAGE_LABELS = {
    "baseline": "Causal baseline (the shipped configuration)",
    "a_full_fit_encoders": "(a) Encoders fitted on every row",
    "b_full_data_entity_aggregates": "(b) Entity aggregates over every row",
    "c_entity_mean_postprocessing": "(c) Entity-mean post-processing of the scores",
    "d_random_split": "(d) Random split instead of chronological",
    "abc_chronological": "(a) + (b) + (c), chronological",
    "abcd_random": "(a) + (b) + (c) + (d), random",
}

FAMILY_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}

IMBALANCE_LABELS = {"none": "none", "class_weight": "class weight", "smote": "SMOTE"}


def load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return dict(json.load(handle))


def f4(value: float) -> str:
    return f"{value:.4f}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def interval(record: dict[str, Any]) -> str:
    return f"{f4(record['low'])} to {f4(record['high'])}"


def signed_interval(record: dict[str, Any]) -> str:
    return f"{signed(record['point'])} ({signed(record['low'])} to {signed(record['high'])})"


def thousands(value: int) -> str:
    return f"{value:,}"


def negated(record: dict[str, Any]) -> dict[str, Any]:
    """The artifact records `first minus second`; the README reads shipped minus the other."""
    return {"point": -record["point"], "low": -record["high"], "high": -record["low"]}


def leakage_verdict(delta: dict[str, Any]) -> str:
    """The artifact's rule applied to its own numbers, read the way the README states it."""
    by_row = delta["pr_auc_by_row"]
    if not by_row["excludes_zero"]:
        return "no change"
    direction = "inflates" if by_row["difference"]["point"] > 0 else "deflates"
    if not delta["pr_auc_by_card"]["excludes_zero"]:
        return f"{direction} by row; by card the interval covers zero"
    return direction


def leakage_table(reports: Path) -> str:
    artifact = load(reports / "leakage_delta.json")
    variants = {v["id"]: v for v in artifact["variants"]}
    deltas = artifact["deltas"]["per_variant"]
    by_row = artifact["bootstrap"]["pr_auc_by_row"]
    lines = [
        "| Variant | Columns | Test PR-AUC (95 percent interval) | Against the baseline | Reading |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in LEAKAGE_LABELS.items():
        variant = variants[key]
        point = by_row[variant["split"]]["per_model"][key]
        if key == "baseline":
            against, reading = "", "reproduces the shipped model"
        else:
            delta = deltas[key]
            against = signed_interval(delta["pr_auc_by_row"]["difference"])
            reading = leakage_verdict(delta)
        lines.append(
            f"| {label} | {variant['n_columns']} | {f4(point['point'])} ({interval(point)}) "
            f"| {against} | {reading} |"
        )
    return "\n".join(lines)


def latency_table(reports: Path) -> str:
    artifact = load(reports / "label_latency.json")
    lines = [
        "| Labels still open for the last N days | Immature: test PR-AUC (against the reference) "
        "| Immature: implied fraud rate | Excluded: test PR-AUC (against the reference) |",
        "| --- | --- | --- | --- |",
    ]
    for entry in artifact["sweep"]:
        immature, excluded = entry["immature"], entry["excluded"]
        lines.append(
            f"| {entry['latency_days']} "
            f"| {f4(immature['test_pr_auc']['run']['point'])} "
            f"({signed_interval(immature['test_pr_auc']['difference'])}) "
            f"| {f4(immature['test_implied_fraud_rate']['run']['point'])} "
            f"| {f4(excluded['test_pr_auc']['run']['point'])} "
            f"({signed_interval(excluded['test_pr_auc']['difference'])}) |"
        )
    return "\n".join(lines)


def comparison_table(reports: Path) -> str:
    comparison = load(reports / "model_comparison.json")
    variance = load(reports / "metric_variance.json")
    per_model = variance["by_row"]["per_model"]
    lines = [
        "| Family | Imbalance | Columns | Validation PR-AUC | Test PR-AUC (95 percent interval) "
        "| Test ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in comparison["models"]:
        boot = per_model[model["name"]]
        lines.append(
            f"| {FAMILY_LABELS[model['family']]} | {IMBALANCE_LABELS[model['imbalance']]} "
            f"| {model['n_columns']} | {f4(model['val']['pr_auc'])} "
            f"| {f4(model['test']['pr_auc'])} ({interval(boot)}) | {f4(model['test']['roc_auc'])} |"
        )
    operating = load(reports / "operating_points.json")
    shipped = per_model["shipped"]
    n_columns = load(reports / "leakage_delta.json")["model"]["n_columns"]
    lines.append(
        f"| XGBoost, shipped stack | none | {n_columns} "
        f"| {f4(operating['primary_metric']['val']['pr_auc'])} "
        f"| {f4(shipped['point'])} ({interval(shipped)}) "
        f"| {f4(operating['primary_metric']['test']['roc_auc'])} |"
    )
    return "\n".join(lines)


def inline_numbers(reports: Path, loadtest: Path) -> dict[str, str]:
    """Every number the README quotes outside the three tables, named and formatted."""
    split = load(reports / "split_summary.json")
    audit = load(reports / "audit.json")
    variance = load(reports / "metric_variance.json")
    operating = load(reports / "operating_points.json")
    ablations = load(reports / "ablations.json")
    shap = load(reports / "shap" / "shap_global.json")
    leakage = load(reports / "leakage_delta.json")
    latency = load(reports / "latency.json")
    parity = load(reports / "parity.json")
    drift = load(reports / "monitoring" / "drift.json")
    decay = load(reports / "monitoring" / "decay.json")
    adversarial = load(reports / "adversarial_validation.json")
    lifecycle = load(reports / "lifecycle_demo.json")
    features = load(reports / "feature_summary.json")
    encoding = load(reports / "encoding_spec.json")
    graph = load(reports / "graph_summary.json")
    search = load(reports / "hyperparameter_search.json")
    univariate = load(reports / "eda" / "univariate.json")
    loadtest_results = load(loadtest / "results.json")

    shipped_row = variance["by_row"]["per_model"]["shipped"]
    shipped_card = variance["by_card"]["per_model"]["shipped"]
    against = {r["first"]: r for r in variance["shipped_against_each"]}
    test_vertex = operating["tuned_vertex"]["test"]
    candidates = audit["entity_key_candidates"]
    overlap = split["entity_overlap"]["test"]
    stacks = {s["id"]: s for s in ablations["stacks"]}
    causal_seeds = stacks["prepared+causal"]["against_reference"]["per_seed"]
    count_seeds = stacks["prepared+causal+graph|without=count"]["against_reference"]["per_seed"]
    graph_seeds = stacks["prepared+causal+graph"]["against_reference"]["per_seed"]
    gates = [
        c["challenger"]["gate"]
        for c in lifecycle["cycles"]
        if c.get("challenger") and c["challenger"].get("trained")
    ]
    promoted = [g["difference"]["point"] for g in gates if g["promote"]]
    refused_under_margin = [
        g["difference"]["point"] for g in gates if not g["promote"] and g["above_noise"]
    ]
    sweeps = {s["workers"]: s["saturation"] for s in loadtest_results["sweeps"]}
    addr2 = next(c for c in univariate["numeric"] if c["column"] == "addr2")
    default_columns = load(reports / "model_comparison.json")["n_columns"]
    random_delta = leakage["deltas"]["per_variant"]["d_random_split"]["pr_auc_by_row"]
    all_delta = leakage["deltas"]["per_variant"]["abcd_random"]["pr_auc_by_row"]

    return {
        "n_rows": thousands(split["totals"]["n_rows"]),
        "n_transaction_columns": str(audit["shape"]["n_columns"]),
        "n_identity_rows": thousands(audit["identity_join"]["n_identity_rows"]),
        "n_fraud": thousands(split["totals"]["n_fraud"]),
        "fraud_rate": f4(split["totals"]["fraud_rate"]),
        "span_days": f"{audit['transaction_dt']['span_days']:.0f}",
        "train_rows": thousands(split["splits"]["train"]["n_rows"]),
        "val_rows": thousands(split["splits"]["val"]["n_rows"]),
        "test_rows": thousands(split["splits"]["test"]["n_rows"]),
        "train_days": f"{split['splits']['train']['day_index_min']} to "
        f"{split['splits']['train']['day_index_max']}",
        "val_days": f"{split['splits']['val']['day_index_min']} to "
        f"{split['splits']['val']['day_index_max']}",
        "test_days": f"{split['splits']['test']['day_index_min']} to "
        f"{split['splits']['test']['day_index_max']}",
        "test_fraud_rate": f4(split["splits"]["test"]["fraud_rate"]),
        # derived: the test window less its fraud rows, and the accuracy of approving all of them
        "test_legitimate": thousands(
            split["splits"]["test"]["n_rows"] - split["splits"]["test"]["n_fraud"]
        ),
        "approve_all_accuracy": f4(1 - split["splits"]["test"]["fraud_rate"]),
        "identity_coverage_fraud": f4(audit["identity_join"]["coverage_fraud"]),
        "identity_coverage_non_fraud": f4(audit["identity_join"]["coverage_non_fraud"]),
        "purity_card1": f4(candidates["card1"]["label_purity_multi_entities"]),
        "purity_key": f4(candidates["card1_addr1_cardday"]["label_purity_multi_entities"]),
        "test_entities_unseen_share": f4(overlap["share_entities_unseen_in_train"]),
        "test_rows_unseen_share": f4(overlap["share_rows_on_unseen_entities"]),
        "test_fraud_rate_unseen": f4(overlap["fraud_rate_on_unseen_entity_rows"]),
        "test_fraud_rate_seen": f4(overlap["fraud_rate_on_seen_entity_rows"]),
        "label_lag_days": str(encoding["encoders"]["target"]["chosen_lag_days"]),
        "n_causal_features": str(features["n_features_derived"]),
        "test_pr_auc": f4(shipped_row["point"]),
        "test_pr_auc_interval": interval(shipped_row),
        "test_pr_auc_card_interval": interval(shipped_card),
        "n_boot": thousands(variance["n_boot"]),
        "shipped_minus_default_xgboost": signed_interval(
            negated(against["xgboost__none"]["difference"])
        ),
        "shipped_minus_best_forest": signed_interval(
            negated(against["random_forest__class_weight"]["difference"])
        ),
        "n_model_columns": str(leakage["model"]["n_columns"]),
        "n_default_columns": str(default_columns),
        # derived: the default stack less the shipped one
        "n_columns_removed": str(default_columns - leakage["model"]["n_columns"]),
        # derived: the random split's share of the whole gap, to one decimal
        "random_split_share_of_gap": f"{random_delta['difference']['point'] / all_delta['difference']['point']:.1f}",
        "search_default_val_pr_auc": f4(search["final"]["default"]["val"]["pr_auc"]),
        "search_tuned_val_pr_auc": f4(search["final"]["tuned"]["val"]["pr_auc"]),
        "threshold": f4(test_vertex["threshold"]),
        "test_precision": f4(test_vertex["precision"]),
        "test_recall": f4(test_vertex["recall"]),
        "test_f1": f4(test_vertex["f1"]),
        "alerts_per_day": f"{test_vertex['alerts_per_day']:.1f}",
        "band_low": str(operating["band_edges"]["low"]),
        "band_high": str(operating["band_edges"]["high"]),
        "causal_ablation_per_seed": ", ".join(
            signed(s["val_pr_auc_difference"]["point"]) for s in causal_seeds
        ),
        "count_ablation_range": f"{f4(min(s['val_pr_auc_difference']['point'] for s in count_seeds))} "
        f"to {f4(max(s['val_pr_auc_difference']['point'] for s in count_seeds))}",
        "graph_ablation_per_seed": ", ".join(
            signed(s["val_pr_auc_difference"]["point"]) for s in graph_seeds
        ),
        "shap_count_share": f4(shap["by_block"]["native_count"]["mean_abs_shap_share"]),
        "giant_component_share": f4(
            graph["end_of_train_graph"]["giant_component"]["share_of_transactions"]
        ),
        "addr2_top_value_share": f4(addr2["top_value_share"]),
        "path_p50_ms": f"{latency['summary']['serving_path_whole_no_store_p50_ms']['median']:.1f}",
        "booster_p50_ms": f"{latency['summary']['serving_path_p50_ms']['predict']['median']:.3f}",
        "parity_rows": thousands(parity["stream"]["n_rows"]),
        "parity_features": str(parity["backends"]["memory"]["n_features"]),
        "parity_max_difference": str(
            max(b["max_abs_difference_over_all_features"] for b in parity["backends"].values())
        ),
        "peak_rps_one_worker": f"{sweeps[1]['peak_throughput_rps']:.1f}",
        "peak_rps_four_workers": f"{sweeps[4]['peak_throughput_rps']:.1f}",
        "drift_batches": str(drift["calibration"]["n_batches"]),
        "drift_false_alert_rate": str(drift["in_control"]["leave_one_out"]["rate_any_alert"]),
        "textbook_alerts_per_week": f"{drift['in_control']['leave_one_out']['mean_textbook_alert_band']:.1f}",
        "decay_largest_drop": f4(decay["in_control"]["largest_drop"]),
        "decay_tolerance": str(decay["in_control"]["decay_tolerance"]),
        "promotion_margin": str(lifecycle["inputs"]["promotion_margin"]),
        "noise_half_width_by_card": f4(
            variance["by_card"]["widest_paired_half_width"]["half_width"]
        ),
        "maturity_days": str(lifecycle["inputs"]["maturity_days"]),
        "n_cycles": str(lifecycle["n_cycles"]),
        "n_challengers": str(lifecycle["n_challengers"]),
        "n_promoted": str(len(promoted)),
        "n_refused": str(len(gates) - len(promoted)),
        "gate_promoted": " and ".join(signed(g) for g in promoted),
        "gate_refused_under_margin": ", ".join(signed(g) for g in refused_under_margin),
        "adversarial_full_auc": f4(adversarial["fits"]["full"]["auc"]["out_of_fold"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--loadtest-dir", type=Path, default=Path("loadtest"))
    args = parser.parse_args()

    print("## leakage delta\n")
    print(leakage_table(args.reports_dir))
    print("\n## label latency\n")
    print(latency_table(args.reports_dir))
    print("\n## model comparison\n")
    print(comparison_table(args.reports_dir))
    print("\n## inline numbers\n")
    for name, value in inline_numbers(args.reports_dir, args.loadtest_dir).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
