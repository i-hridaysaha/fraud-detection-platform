"""Print every number the case study quotes, read from the committed artifacts.

Usage:
    make case-study-numbers
    .venv/bin/python scripts/case_study_numbers.py [--reports-dir reports] [--loadtest-dir loadtest]

The case study lives on the author's site, outside this repository, and cannot be held to the
artifacts by a test the way the README is. This script is the next best thing: one run prints
every number the article carries, named, in the spelling the article uses, so the article's
verification table can point at a key here rather than at a memory. It reuses the README's
formatting for the values the two documents share and adds the ones only the article needs.
Nothing here is computed from the data: the script reads JSON and formats it.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent


def readme_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("readme_numbers", HERE / "readme_numbers.py")
    if spec is None or spec.loader is None:
        sys.exit("scripts/readme_numbers.py is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_study_numbers(reports: Path, loadtest: Path) -> dict[str, str]:
    """The README's named values plus the ones only the case study quotes."""
    rn = readme_module()
    load, f4, signed, interval, signed_interval, thousands = (
        rn.load,
        rn.f4,
        rn.signed,
        rn.interval,
        rn.signed_interval,
        rn.thousands,
    )
    values: dict[str, str] = dict(rn.inline_numbers(reports, loadtest))

    audit = load(reports / "audit.json")
    split = load(reports / "split_summary.json")
    leakage = load(reports / "leakage_delta.json")
    latency = load(reports / "label_latency.json")
    encoding = load(reports / "encoding_spec.json")
    features = load(reports / "feature_summary.json")
    parity = load(reports / "parity.json")
    serving = load(reports / "latency.json")
    operating = load(reports / "operating_points.json")
    comparison = load(reports / "model_comparison.json")
    ablations = load(reports / "ablations.json")
    variance = load(reports / "metric_variance.json")
    lifecycle = load(reports / "lifecycle_demo.json")
    decay = load(reports / "monitoring" / "decay.json")
    adversarial = load(reports / "adversarial_validation.json")
    missingness = load(reports / "eda" / "missingness.json")
    column_plan = load(reports / "column_plan.json")
    schema = load(reports / "prep" / "schema.json")
    v_reduction = load(reports / "prep" / "v_reduction.json")
    transforms = load(reports / "prep" / "transforms.json")
    graph = load(reports / "graph_summary.json")
    duplicates = load(reports / "features" / "duplicate_content.json")
    velocity = load(reports / "features" / "velocity_vs_c.json")
    merchant = load(reports / "features" / "merchant_proxies.json")

    candidates = audit["entity_key_candidates"]
    start_day = audit["card_start_day_check"]["by_card1"]
    by_row = leakage["bootstrap"]["pr_auc_by_row"]
    deltas = leakage["deltas"]["per_variant"]
    variants = {v["id"]: v for v in leakage["variants"]}
    population = leakage["population"]
    sweep = {entry["latency_days"]: entry for entry in latency["sweep"]}
    runs = {run["id"]: run for run in latency["runs"]}
    target = encoding["encoders"]["target"]
    gap = target["lag_sweep"]["gap_test"]["per_lag"]
    entity_te = encoding["entity_target_encoding"]
    te_per_lag = {entry["lag_days"]: entry for entry in entity_te["per_lag"]}
    coverage = features["coverage"]["test"]["entity_history_against_training_overlap"]
    bands_test = operating["bands"]["test"]
    costs = operating["bands"]["costs"]
    models = {m["name"]: m for m in comparison["models"]}
    stacks = {s["id"]: s for s in ablations["stacks"]}
    against = {r["first"]: r for r in variance["shipped_against_each"]}
    gates = [
        (c["cycle"], c["challenger"]["gate"])
        for c in lifecycle["cycles"]
        if c.get("challenger") and c["challenger"].get("trained")
    ]
    drift_cycles = [
        c for c in lifecycle["cycles"] if c.get("drift") and c["drift"].get("drift_alert")
    ]
    decay_cycles = [
        c for c in lifecycle["cycles"] if c.get("decay") and any(b.get("decay") for b in c["decay"])
    ]
    store = serving["summary"]["store_p50_ms"]
    dup_1h = next(p for p in duplicates["per_feature"] if p["feature"] == "dup_prior_count_1h")[
        "by_split"
    ]

    def point(variant: str) -> dict[str, Any]:
        return by_row[variants[variant]["split"]]["per_model"][variant]

    def delta_row(variant: str) -> str:
        return signed_interval(deltas[variant]["pr_auc_by_row"]["difference"])

    def stack_seeds(stack: str) -> str:
        seeds = stacks[stack]["against_reference"]["per_seed"]
        return ", ".join(signed(s["val_pr_auc_difference"]["point"]) for s in seeds)

    values.update(
        {
            # the data and the split
            "n_joined_columns": str(column_plan["n_columns"]),
            "train_span_days": f"{split['splits']['train']['span_days']:.2f}",
            "n_test_fraud": thousands(split["splits"]["test"]["n_fraud"]),
            "n_val_fraud": thousands(split["splits"]["val"]["n_fraud"]),
            # the entity key
            "entities_card1": thousands(candidates["card1"]["n_entities"]),
            "entities_card1_addr1": thousands(candidates["card1_addr1"]["n_entities"]),
            "entities_key": thousands(candidates["card1_addr1_cardday"]["n_entities"]),
            "purity_card1_addr1": f4(candidates["card1_addr1"]["label_purity_multi_entities"]),
            "singleton_share_key": f4(
                candidates["card1_addr1_cardday"]["share_singleton_entities"]
            ),
            "card1_single_start_day_share": f4(start_day["fraction_single_valued"]),
            "card1_median_distinct_start_days": f"{start_day['distinct_values_per_group']['median']:.0f}",
            "card1_median_start_day_spread_days": f"{start_day['median_spread_days']:.0f}",
            # the leakage delta, beyond the table
            "random_split_pr_auc": f4(point("d_random_split")["point"]),
            "random_split_pr_auc_interval": interval(point("d_random_split")),
            "all_shortcuts_pr_auc": f4(point("abcd_random")["point"]),
            "all_shortcuts_pr_auc_interval": interval(point("abcd_random")),
            "delta_a_by_row": delta_row("a_full_fit_encoders"),
            "delta_a_by_card": signed_interval(
                deltas["a_full_fit_encoders"]["pr_auc_by_card"]["difference"]
            ),
            "delta_b": delta_row("b_full_data_entity_aggregates"),
            "delta_c": delta_row("c_entity_mean_postprocessing"),
            "delta_d": delta_row("d_random_split"),
            "delta_abc": delta_row("abc_chronological"),
            "delta_abcd": delta_row("abcd_random"),
            "n_aggregate_columns": str(
                variants["b_full_data_entity_aggregates"]["n_columns"]
                - variants["baseline"]["n_columns"]
            ),
            "random_seen_entity_row_share": f4(
                population["random"]["test"]["seen_entity_row_share"]
            ),
            "chrono_seen_entity_row_share": f4(
                population["chronological"]["test"]["seen_entity_row_share"]
            ),
            "random_seen_fraud_rate": f4(population["random"]["test"]["seen_entity_fraud_rate"]),
            "random_unseen_fraud_rate": f4(
                population["random"]["test"]["unseen_entity_fraud_rate"]
            ),
            # the label latency, beyond the table
            "latency_immature_15": signed_interval(
                sweep[15]["immature"]["test_pr_auc"]["difference"]
            ),
            "latency_excluded_15": signed_interval(
                sweep[15]["excluded"]["test_pr_auc"]["difference"]
            ),
            "latency_immature_60_pr_auc": f4(sweep[60]["immature"]["test_pr_auc"]["run"]["point"]),
            "latency_immature_60_implied_rate": f4(
                sweep[60]["immature"]["test_implied_fraud_rate"]["run"]["point"]
            ),
            "latency_immature_60_recall": f4(
                runs["immature_60"]["test"]["at_shipped_threshold"]["recall"]
            ),
            "latency_immature_60_alerts_per_day": f"{runs['immature_60']['test']['at_shipped_threshold']['alerts_per_day']:.1f}",
            "latency_reference_alerts_per_day": f"{runs['reference']['test']['at_shipped_threshold']['alerts_per_day']:.1f}",
            "latency_reference_implied_rate": f4(runs["reference"]["test"]["implied_fraud_rate"]),
            "latency_observed_rate": f4(runs["reference"]["test"]["observed_fraud_rate"]),
            "latency_excluded_60_implied_rate": f4(
                sweep[60]["excluded"]["test_implied_fraud_rate"]["run"]["point"]
            ),
            # the target encoding lag and the entity exclusion
            "te_gap_lag_0": signed(gap["0"]["mean"]),
            "te_gap_lag_14": signed(gap["14"]["mean"]),
            "te_gap_lag_28": signed(gap["28"]["mean"]),
            "te_val_cost_lag_14": f4(target["lag_sweep"]["cost_test"]["per_lag"]["14"]["mean"]),
            "te_smoothing": f"{target['chosen_smoothing']:.0f}",
            "entity_te_val_auc": f4(te_per_lag[14]["validation_auc"]),
            "entity_te_train_auc_lag_0": f4(te_per_lag[0]["train_auc"]),
            "entity_te_seen_auc": f4(entity_te["decomposition"]["auc_on_seen_rows"]),
            "entity_te_unseen_auc": f4(entity_te["decomposition"]["auc_on_unseen_rows"]),
            "entity_te_seen_share_val": f4(entity_te["share_validation_rows_on_a_seen_entity"]),
            # coverage of the two kinds of entity history on test
            "test_share_entity_seen_in_train": f4(coverage["share_entity_seen_in_train"]),
            "test_share_any_earlier_row": f4(coverage["share_with_any_earlier_row_in_the_stream"]),
            # the parity run
            "parity_days": str(parity["stream"]["n_days"]),
            "parity_ties_card": str(parity["stream"]["ties"]["card"]),
            "parity_ties_entity": str(parity["stream"]["ties"]["entity"]),
            "parity_ties_device": thousands(parity["stream"]["ties"]["device_node"]),
            # the store and the request
            "store_get_ms_memory": f"{store['memory']['get_features']['median']:.3f}",
            "store_commit_ms_memory": f"{store['memory']['commit']['median']:.3f}",
            "store_get_ms_redis": f"{store['redis']['get_features']['median']:.3f}",
            "store_commit_ms_redis": f"{store['redis']['commit']['median']:.2f}",
            # the cost bands
            "cost_missed_fraud": f"{costs['missed_fraud']:.0f}",
            "cost_false_decline": f"{costs['false_decline']:.0f}",
            "cost_review": f"{costs['review']:.1f}",
            "single_threshold_no_review": f4(operating["bands"]["single_threshold_without_review"]),
            "band_approve_per_day": f"{bands_test['bands']['approve']['per_day']:.1f}",
            "band_review_per_day": f"{bands_test['bands']['review']['per_day']:.1f}",
            "band_block_per_day": f"{bands_test['bands']['block']['per_day']:.1f}",
            "band_approve_fraud_share": f4(bands_test["bands"]["approve"]["share_of_all_fraud"]),
            "band_review_fraud_share": f4(bands_test["bands"]["review"]["share_of_all_fraud"]),
            "band_block_fraud_share": f4(bands_test["bands"]["block"]["share_of_all_fraud"]),
            "band_review_fraud_rate": f4(bands_test["bands"]["review"]["fraud_rate"]),
            "band_block_fraud_rate": f4(bands_test["bands"]["block"]["fraud_rate"]),
            "cost_per_row_bands": f4(bands_test["realised_cost_per_row"]),
            "cost_per_row_approve_all": f4(bands_test["approve_everything_cost_per_row"]),
            "cost_per_row_single_threshold": f4(
                bands_test["single_threshold_at_cost_optimum"]["cost_per_row"]
            ),
            # the model comparison, beyond the table
            "n_models_compared": str(len(comparison["models"])),
            "xgb_none_val_pr_auc": f4(models["xgboost__none"]["val"]["pr_auc"]),
            "xgb_weight_val_pr_auc": f4(models["xgboost__class_weight"]["val"]["pr_auc"]),
            "xgb_smote_val_pr_auc": f4(models["xgboost__smote"]["val"]["pr_auc"]),
            "forest_none_val_pr_auc": f4(models["random_forest__none"]["val"]["pr_auc"]),
            "logistic_none_val_pr_auc": f4(models["logistic_regression__none"]["val"]["pr_auc"]),
            "n_ablation_stacks": str(len(ablations["stacks"])),
            "n_ablation_seeds": str(len(ablations["seeds"])),
            "count_block_removal_per_seed": stack_seeds("prepared+causal+graph|without=count"),
            "shipped_minus_plain_forest": signed_interval(
                rn.negated(against["random_forest__none"]["difference"])
            ),
            # the gate and the monitors, beyond the README
            "noise_half_width_by_row": f4(
                variance["by_row"]["widest_paired_half_width"]["half_width"]
            ),
            "sustained_cycles": str(lifecycle["inputs"]["sustained_cycles"]),
            "decay_tolerance_headroom": f4(
                decay["in_control"]["decay_tolerance"] - decay["in_control"]["largest_drop"]
            ),
            "gate_refused_not_better": ", ".join(
                signed(g["difference"]["point"])
                for _, g in gates
                if not g["promote"] and not g["above_noise"]
            ),
            "gate_cycles_promoted": ", ".join(str(c) for c, g in gates if g["promote"]),
            "drift_flag_first_cycle": str(drift_cycles[0]["cycle"]) if drift_cycles else "none",
            "decay_flag_first_cycle": str(decay_cycles[0]["cycle"]) if decay_cycles else "none",
            "adversarial_without_te_auc": f4(
                adversarial["fits"]["without_target_encodings"]["auc"]["out_of_fold"]
            ),
            "adversarial_without_entity_and_clock_auc": f4(
                adversarial["fits"]["without_entity_and_clock"]["auc"]["out_of_fold"]
            ),
            # the frame
            "n_label_linked_columns": str(missingness["against_target"]["n_label_linked"]),
            "n_columns_with_nulls": str(missingness["n_columns_with_nulls"]),
            "n_columns_kept": str(column_plan["n_kept"]),
            "n_columns_dropped": str(column_plan["n_dropped"]),
            "n_dropped_constant": str(column_plan["n_dropped_by_reason"]["effectively_constant"]),
            "n_dropped_redundant": str(column_plan["n_dropped_by_reason"]["redundant"]),
            "n_prepared_columns": str(schema["prepared_shape"]["n_columns"]),
            "v_columns_before": str(v_reduction["comparison"]["n_pool_columns"]),
            "v_columns_after": str(v_reduction["comparison"]["block_mean"]["n_blocks"]),
            "graph_components_end_of_train": str(graph["end_of_train_graph"]["n_components"]),
            "dup_1h_ratio_by_split": ", ".join(
                f"{dup_1h[s]['lift']:.3f}" for s in ("train", "val", "test")
            ),
            "velocity_vs_c_max_rho": f4(velocity["summary"]["max_abs_rho"]),
            "n_merchant_name_matches": str(merchant["column_inventory"]["n_name_matches"]),
            "d_columns_psi_worsening": str(transforms["d_normalisation"]["n_worsening"]),
            "d_columns": str(transforms["d_normalisation"]["n_columns"]),
        }
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--loadtest-dir", type=Path, default=Path("loadtest"))
    args = parser.parse_args()
    rn = readme_module()

    print("## leakage delta\n")
    print(rn.leakage_table(args.reports_dir))
    print("\n## label latency\n")
    print(rn.latency_table(args.reports_dir))
    print("\n## model comparison\n")
    print(rn.comparison_table(args.reports_dir))
    print("\n## named numbers\n")
    for name, value in case_study_numbers(args.reports_dir, args.loadtest_dir).items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
