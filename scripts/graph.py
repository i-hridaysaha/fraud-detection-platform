"""Stage 5 graph features. Builds the structural features and writes what stage 6 reads.

Usage:
    make graph
    .venv/bin/python scripts/graph.py

One artifact, reports/graph_summary.json, in blocks:

    config              the graph, the switch, the lag and where the lag came from
    link_columns        cardinality, null rate and heaviest value of each link column, on train
    end_of_train_graph  the component-size distribution of the graph over the train split
    hub_sweep           the same with values above a share of train rows linking nothing
    link_variants       the giant component when one link column is dropped, or kept alone
    per_feature         every graph feature, point-in-time: null rate and drift per split, single
                        feature AUC per split, the zero-against-positive comparison stage 4 judged
                        its counts by, the fraud rate by decile per split, and the clock test
    component_fraud_rate  the label-derived features measured the way ADR 0015 measured entity
                        target encoding
    hub_excluded_rates  the same features rebuilt on each graph of the hub sweep
    transductive_leak   what a full-dataset component computation would have handed the model,
                        against the point-in-time reading of the same graph
    decision            what is in the candidate set, what the switch defaults to, and the rule

The features are computed over the whole stream for the reason stage 4 gave: every value is read
from rows strictly before the row being computed, so the whole frame is the correct input. The one
thing that learns from labels, the `LabelSource`, is built from the train split through
`assert_train_only`. `tests/test_causality.py` rebuilds the graph by brute force from
`TransactionDT < t` and asserts agreement; this script reports, it does not enforce.

The transductive block is the one construction here that is not point-in-time, and it is here to
be measured rather than used. It is not importable from the package.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from fraud_platform import config, data_loader, eda, encoders, graph_features
from fraud_platform.graph_features import GraphConfig

GRAPH_SUMMARY_PATH = config.REPORTS_DIR / "graph_summary.json"
ENCODING_SPEC_PATH = config.REPORTS_DIR / "encoding_spec.json"

INPUT_COLUMNS: tuple[str, ...] = (
    config.ID_COLUMN,
    config.TARGET,
    config.TIME_COLUMN,
    "card1",
    "addr1",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "dist1",
    "D1",
)

# The hub sweep. The first point is stage 3's rare-tail line, below which a categorical level is
# collapsed (`encoders.RARE_TAIL_SHARE`, taken from stage 2's measurement); the next two are a
# decade and two decades below it. A grid, and labelled as one: nothing in the repo derives the
# other two points, and the artifact records that no point on it is a decision.
HUB_SHARES: tuple[float, ...] = (
    encoders.RARE_TAIL_SHARE,
    encoders.RARE_TAIL_SHARE / 10,
    encoders.RARE_TAIL_SHARE / 100,
)

# The hub share the transductive comparison is also run at. The sweep shows the six-column graph
# is one component at every share down to this one, and a leak is only visible where components
# are smaller than the file, so the comparison is repeated on the most fragmented graph in the
# sweep. A grid point, not a choice about what to ship.
LEAK_HUB_SHARE = HUB_SHARES[-1]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def envelope(n_train_rows: int) -> dict[str, Any]:
    return {
        "section": "graph",
        "stage": 5,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make graph  # or: .venv/bin/python scripts/graph.py",
        "command": ".venv/bin/python scripts/graph.py",
        "computed_on": (
            "the whole time-sorted stream. Every feature reads only rows strictly before its own "
            "timestamp, so the whole frame is the correct input; the labels behind the component "
            "fraud rate come from the train split through assert_train_only and mature under "
            "the stage 3 lag. The end-of-train graph and the hub sweep are built over the train "
            "split only"
        ),
        "n_train_rows": n_train_rows,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
    }


def _clean(value: Any) -> Any:
    """NaN to null, numpy scalars to Python, recursively, so the JSON writer can refuse NaN."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")
    print(f"  wrote {path.relative_to(config.ROOT)}")


def load_inputs(transactions: str, identity: str) -> pd.DataFrame:
    frame = data_loader.add_entity_key(
        data_loader.load_raw(
            transactions_path=transactions, identity_path=identity, columns=list(INPUT_COLUMNS)
        )
    )
    return encoders.add_normalised_free_text(frame)


def read_lag() -> dict[str, Any]:
    """The label lag, read from stage 3's artifact and not restated."""
    spec = json.loads(ENCODING_SPEC_PATH.read_text())
    target = spec["encoders"]["target"]
    return {
        "lag_days": int(target["chosen_lag_days"]),
        "source": "reports/encoding_spec.json encoders.target.chosen_lag_days, read not retyped",
        "decided_in": "ADR 0014",
    }


def _auc(y: npt.NDArray[np.int64], score: npt.NDArray[np.float64]) -> dict[str, Any]:
    """AUC over the rows where the score is defined, the way stage 3's sweep reported it."""
    defined = np.isfinite(score)
    n = int(defined.sum())
    if n == 0 or len(np.unique(y[defined])) < 2:
        return {"auc": None, "n_scored": n, "n_undefined": int((~defined).sum())}
    return {
        "auc": float(roc_auc_score(y[defined], score[defined])),
        "n_scored": n,
        "n_undefined": int((~defined).sum()),
    }


# --- the static graph blocks -------------------------------------------------------------------


def build_end_of_train(train: pd.DataFrame, cfg: GraphConfig) -> dict[str, Any]:
    graph = graph_features.static_graph(train, cfg)
    out = graph_features.component_summary(graph)
    out["definition"] = (
        "the union-find over every train row, read after the last one was added. A component is "
        "a set of transactions joined through shared values of the link columns. Size is in "
        "transactions; the node count adds the value nodes"
    )
    out["n_value_nodes_by_column"] = {
        column: int(graph.value_degrees(column).size) for column in cfg.link_columns
    }
    return out


def build_hub_sweep(train: pd.DataFrame, cfg: GraphConfig) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for share in HUB_SHARES:
        hubs = graph_features.fit_hub_values(train, cfg.link_columns, share)
        graph = graph_features.static_graph(train, cfg, hubs)
        summary = graph_features.component_summary(graph)
        points.append(
            {
                "hub_share": share,
                "is_stage_3_rare_tail_line": share == encoders.RARE_TAIL_SHARE,
                "hubs_by_column": graph.hub_report,
                "n_components": summary["n_components"],
                "giant_component_share_of_transactions": summary["giant_component"][
                    "share_of_transactions"
                ],
                "giant_component_n_transactions": summary["giant_component"]["n_transactions"],
                "share_transactions_in_a_component_of_size_1": summary[
                    "share_transactions_in_a_component_of_size_1"
                ],
                "largest_ten_in_transactions": summary["largest_ten_in_transactions"],
                "size_in_transactions": summary["size_in_transactions"],
            }
        )
    return {
        "definition": (
            "a value that at least hub_share of train rows carry links nothing; every other edge "
            "stays. The first point is stage 3's rare-tail line and the other two are a decade "
            "and two decades below it. No point is a decision: the sweep measures whether there "
            "is a graph between one component and dust, and the artifact records what it found"
        ),
        "points": points,
    }


def build_link_variants(train: pd.DataFrame, cfg: GraphConfig) -> dict[str, Any]:
    def measure(columns: tuple[str, ...]) -> dict[str, Any]:
        graph = graph_features.static_graph(train, GraphConfig(enabled=True, link_columns=columns))
        summary = graph_features.component_summary(graph)
        return {
            "link_columns": list(columns),
            "n_components": summary["n_components"],
            "giant_component_share_of_transactions": summary["giant_component"][
                "share_of_transactions"
            ],
            "n_transactions_in_a_component_of_size_1": summary[
                "n_transactions_in_a_component_of_size_1"
            ],
            "largest_three_in_transactions": summary["largest_ten_in_transactions"][:3],
        }

    return {
        "definition": (
            "the end-of-train graph rebuilt with one link column removed, and with one link "
            "column alone. Which columns the collapse depends on"
        ),
        "leave_one_out": {
            column: measure(tuple(c for c in cfg.link_columns if c != column))
            for column in cfg.link_columns
        },
        "single_column": {column: measure((column,)) for column in cfg.link_columns},
        "pairs_with_the_anchor": {
            column: measure((graph_features.ANCHOR_COLUMN, column))
            for column in cfg.link_columns
            if column != graph_features.ANCHOR_COLUMN
        },
    }


# --- the point-in-time features ------------------------------------------------------------------


def build_per_feature(
    frame: pd.DataFrame,
    built: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    cfg: GraphConfig,
) -> list[dict[str, Any]]:
    labels = frame[config.TARGET].to_numpy(dtype="int64")
    train_mask = masks["train"].to_numpy(dtype=bool)
    ts_train = frame.loc[masks["train"], config.TIME_COLUMN].to_numpy(dtype="float64")
    out: list[dict[str, Any]] = []
    for spec in graph_features.catalogue(cfg):
        name = spec.name
        values = built[name]
        row = spec.to_dict()
        row["in_candidate_set"] = name in graph_features.CANDIDATE_FEATURES
        row["null_rate_by_split"] = {
            split: float(values[mask].isna().mean()) for split, mask in masks.items()
        }
        train_values = values[masks["train"]]
        row["drift"] = {
            split: eda.population_stability_index(train_values, values[mask])["psi"]
            for split, mask in masks.items()
            if split != "train"
        }
        row["single_feature_auc"] = {
            split: _auc(labels[mask.to_numpy(dtype=bool)], values[mask].to_numpy(dtype="float64"))
            for split, mask in masks.items()
        }
        row["zero_against_positive_by_split"] = eda.zero_against_positive_by_split(
            values, labels, masks
        )
        row["fraud_rate_by_decile"] = {
            split: eda.decile_table(values[mask], frame.loc[mask, config.TARGET])
            for split, mask in masks.items()
        }
        finite = np.isfinite(train_values.to_numpy(dtype="float64"))
        rho = (
            stats.spearmanr(train_values.to_numpy(dtype="float64")[finite], ts_train[finite])
            if finite.sum() > 2
            else None
        )
        row["clock_test"] = {
            "definition": (
                "Spearman correlation between the feature and TransactionDT on the train split. "
                "A feature that grows with the stream is a clock whatever else it is, and a "
                "clock's fraud rate by decile is the weekly fraud rate stage 2 measured"
            ),
            "spearman_with_transaction_dt_train": float(rho.statistic) if rho is not None else None,
            "n": int(finite.sum()),
        }
        comparison = row["zero_against_positive_by_split"]
        row["stage_4_ship_rule"] = {
            "rule": (
                "same sign on all three splits with disjoint Wilson intervals on each, the rule "
                "ADR 0022 wrote for a count feature"
            ),
            "direction_stable": comparison["direction_stable"],
            "disjoint_on_every_split": comparison["disjoint_on_every_split"],
            "passes": bool(
                comparison["direction_stable"] and comparison["disjoint_on_every_split"]
            ),
        }
        out.append(row)
    del train_mask
    return out


def _label_feature_evaluation(
    values: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    masks: Mapping[str, pd.Series[bool]],
    seen: npt.NDArray[np.bool_],
) -> dict[str, Any]:
    """AUC per split, and on the later splits the ADR 0015 decomposition on whether the training
    window saw the row's entity."""
    block: dict[str, Any] = {
        "max": float(np.nanmax(values)) if np.isfinite(values).any() else None,
        "null_rate_by_split": {
            split: float(np.isnan(values[mask.to_numpy(dtype=bool)]).mean())
            for split, mask in masks.items()
        },
        "by_split": {
            split: _auc(y[mask.to_numpy(dtype=bool)], values[mask.to_numpy(dtype=bool)])
            for split, mask in masks.items()
        },
    }
    for split in ("val", "test"):
        mask = masks[split].to_numpy(dtype=bool)
        on_seen = mask & seen
        on_unseen = mask & ~seen
        block[f"{split}_decomposition"] = {
            "auc_on_seen_rows": _auc(y[on_seen], values[on_seen]),
            "auc_on_unseen_rows": _auc(y[on_unseen], values[on_unseen]),
            "share_rows_on_a_seen_entity": float(on_seen.sum() / mask.sum()),
        }
    return block


def build_component_fraud_rate(
    frame: pd.DataFrame,
    built: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    labels: graph_features.LabelSource,
    n_labels_attached: int,
) -> dict[str, Any]:
    """The label-derived features measured the way ADR 0015 measured entity target encoding.

    Validation and test are split on whether the training window ever saw the row's entity. A
    rate that separates seen rows and not unseen ones is carrying the entity's own labels, which
    is what ADR 0015 refused, whatever the lag.
    """
    y = frame[config.TARGET].to_numpy(dtype="int64")
    seen = (
        frame[config.ENTITY_ID_COLUMN]
        .isin(set(frame.loc[masks["train"], config.ENTITY_ID_COLUMN].unique()))
        .to_numpy(dtype=bool)
    )
    out: dict[str, Any] = {
        "labels": labels.to_dict(),
        "n_labels_attached_by_end_of_stream": n_labels_attached,
        "why_measured_this_way": (
            "ADR 0015 split validation on whether the training window saw the entity and found "
            "an entity target encoding at 0.9877 on seen rows and 0.4915 on unseen. The same "
            "decomposition is the test for a component rate: the component contains the entity's "
            "own earlier transactions, and a rate that only works where they exist is their "
            "labels wearing a graph's clothing"
        ),
    }
    for name in graph_features.LABEL_FEATURES:
        out[name] = _label_feature_evaluation(built[name].to_numpy(dtype="float64"), y, masks, seen)
    rate = built[graph_features.COMPONENT_FRAUD_RATE].to_numpy(dtype="float64")
    others = built[graph_features.COMPONENT_FRAUD_RATE_OTHERS].to_numpy(dtype="float64")
    both = np.isfinite(rate) & np.isfinite(others)
    out["rate_against_others_rate"] = {
        "n_both_defined": int(both.sum()),
        "share_equal": float((rate[both] == others[both]).mean()) if both.any() else None,
        "max_abs_difference": float(np.abs(rate[both] - others[both]).max())
        if both.any()
        else None,
    }
    return out


def build_hub_excluded_rates(
    frame: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    labels: graph_features.LabelSource,
    cfg: GraphConfig,
) -> dict[str, Any]:
    """The same label-derived features on the graphs of the hub sweep, point-in-time.

    The six-column graph is one component, so its rate is the lagged training rate and the
    decomposition above says nothing about what a component rate would be worth on a graph that
    had components. The sweep's graphs do, and this measures the causal rate on each of them
    with the same decomposition. Measured so the ADR has the number; not shipped, because no hub
    share on the grid is derived from anything.
    """
    y = frame[config.TARGET].to_numpy(dtype="int64")
    train = frame[masks["train"]]
    seen = (
        frame[config.ENTITY_ID_COLUMN]
        .isin(set(train[config.ENTITY_ID_COLUMN].unique()))
        .to_numpy(dtype=bool)
    )
    points: list[dict[str, Any]] = []
    all_features = cfg.with_features(*graph_features.ALL_FEATURES)
    for share in HUB_SHARES:
        hubs = graph_features.fit_hub_values(train, cfg.link_columns, share)
        causal = graph_features.build_graph_features(frame, labels, all_features, hubs)
        point: dict[str, Any] = {
            "hub_share": share,
            "n_hub_values_by_column": {
                column: len(values) for column, values in hubs["values"].items()
            },
        }
        for name in (graph_features.COMPONENT_SIZE, *graph_features.LABEL_FEATURES):
            values = causal[name].to_numpy(dtype="float64")
            point[name] = _label_feature_evaluation(values, y, masks, seen)
            train_values = causal.loc[masks["train"], name]
            point[name]["drift"] = {
                split: eda.population_stability_index(train_values, causal.loc[mask, name])["psi"]
                for split, mask in masks.items()
                if split != "train"
            }
        points.append(point)
        del causal
        gc.collect()
    return {
        "definition": (
            "values that at least hub_share of train rows carry link nothing on any split, the "
            "hub set being fitted on train through the guard, and the point-in-time features are "
            "rebuilt on what remains. Same lag, same label source, same decomposition as the "
            "block above"
        ),
        "points": points,
    }


# --- the transductive comparison, ADR 0024 --------------------------------------------------------


def _transductive(
    frame: pd.DataFrame, cfg: GraphConfig, hubs: Mapping[str, Any] | None
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64], dict[str, Any]]:
    """Component size and leave-one-out component fraud rate over the WHOLE frame, every split.

    This is the construction the module refuses: one graph over every row, val and test included,
    each row reading the component it ends up in and the labels of everything in it except
    itself. It exists so the leak has a number.
    """
    graph = graph_features.static_graph(frame, cfg, hubs)
    stream = graph.stream
    n = stream.n_rows
    roots = np.array([graph.uf.find(graph.tx_base + i) for i in range(n)], dtype="int64")
    _, root_codes = np.unique(roots, return_inverse=True)
    y_sorted = frame[config.TARGET].to_numpy(dtype="int64")[stream.order]
    y_root = np.bincount(root_codes, weights=y_sorted.astype("float64"))
    n_root = np.bincount(root_codes).astype("float64")
    size_sorted = n_root[root_codes]
    with np.errstate(invalid="ignore", divide="ignore"):
        loo_sorted = np.where(
            size_sorted > 1, (y_root[root_codes] - y_sorted) / (size_sorted - 1), np.nan
        )
    size = np.empty(n, dtype="int64")
    size[stream.order] = size_sorted.astype("int64")
    loo = np.empty(n, dtype="float64")
    loo[stream.order] = loo_sorted
    summary = graph_features.component_summary(graph)
    return (
        size,
        loo,
        {
            key: summary[key]
            for key in (
                "n_transactions",
                "n_components",
                "giant_component",
                "n_transactions_in_a_component_of_size_1",
            )
        },
    )


def build_transductive_leak(
    frame: pd.DataFrame,
    built: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    labels: graph_features.LabelSource,
    cfg: GraphConfig,
) -> dict[str, Any]:
    y = frame[config.TARGET].to_numpy(dtype="int64")
    out: dict[str, Any] = {
        "definition": (
            "a full-dataset component computation: one graph over all 590,540 rows, every row "
            "reading the size of the component it finally sits in and the fraud rate of every "
            "other transaction in it, val and test labels included, no lag. Against the "
            "point-in-time reading of the same graph: the component as it stood before the row, "
            "and the train-only labels that had matured by then"
        ),
        "graphs": [],
    }
    train = frame[masks["train"]]
    for label, hub_share in (("six_column_graph", None), ("hub_excluded_graph", LEAK_HUB_SHARE)):
        if hub_share is None:
            causal = built
            hubs: dict[str, Any] | None = None
        else:
            hubs = graph_features.fit_hub_values(train, cfg.link_columns, hub_share)
            causal = graph_features.build_graph_features(
                frame, labels, cfg.with_features(*graph_features.ALL_FEATURES), hubs
            )
        size_t, loo_t, final_graph = _transductive(frame, cfg, hubs)
        size_c = causal[graph_features.COMPONENT_SIZE].to_numpy(dtype="int64")
        rate_c = causal[graph_features.COMPONENT_FRAUD_RATE].to_numpy(dtype="float64")
        entry: dict[str, Any] = {
            "graph": label,
            "hub_share": hub_share,
            "n_hub_values_by_column": {
                column: len(values) for column, values in hubs["values"].items()
            }
            if hubs is not None
            else {},
            "final_graph_over_every_row": final_graph,
            "component_size": {
                "share_rows_where_transductive_exceeds_causal": float((size_t > size_c).mean()),
                "median_transductive_over_causal_plus_one": float(np.median(size_t / (size_c + 1))),
                "auc_by_split": {
                    split: {
                        "transductive": _auc(
                            y[m.to_numpy(dtype=bool)],
                            size_t[m.to_numpy(dtype=bool)].astype("float64"),
                        ),
                        "causal": _auc(
                            y[m.to_numpy(dtype=bool)],
                            size_c[m.to_numpy(dtype=bool)].astype("float64"),
                        ),
                    }
                    for split, m in masks.items()
                },
            },
            "component_fraud_rate": {
                "transductive_is": (
                    "the fraud rate of every other transaction in the row's final component, "
                    "all splits, no lag"
                ),
                "causal_is": graph_features.COMPONENT_FRAUD_RATE,
                "auc_by_split": {
                    split: {
                        "transductive_leave_one_out": _auc(
                            y[m.to_numpy(dtype=bool)], loo_t[m.to_numpy(dtype=bool)]
                        ),
                        "causal": _auc(y[m.to_numpy(dtype=bool)], rate_c[m.to_numpy(dtype=bool)]),
                    }
                    for split, m in masks.items()
                },
            },
        }
        out["graphs"].append(entry)
        del size_t, loo_t
        gc.collect()
    return out


# --- entry point ----------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 graph features.")
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    args = parser.parse_args()

    started = time.perf_counter()
    print("loading the columns the graph reads")
    frame = load_inputs(args.transactions, args.identity)
    print(f"  {len(frame):,} rows by {frame.shape[1]} columns")

    masks = data_loader.split_masks(frame)
    train = frame[masks["train"]]
    data_loader.assert_train_only(train, what="stage 5 label source and end-of-train graph")
    print(
        f"  train {int(masks['train'].sum()):,} rows, val {int(masks['val'].sum()):,}, "
        f"test {int(masks['test'].sum()):,}"
    )

    lag = read_lag()
    labels = graph_features.label_source(train, lag_days=lag["lag_days"])
    cfg = GraphConfig(enabled=True).with_features(*graph_features.ALL_FEATURES)

    print("building every graph feature over the whole stream")
    build_started = time.perf_counter()
    built = graph_features.build_graph_features(frame, labels, cfg)
    build_seconds = time.perf_counter() - build_started
    print(f"  {built.shape[1]} features on {len(built):,} rows in {build_seconds:.1f}s")
    n_labels_attached = int(built.attrs["n_labels_attached"])

    report = envelope(int(masks["train"].sum()))
    report["config"] = {**graph_features.DEFAULT_CONFIG.to_dict(), "built_here_with": cfg.to_dict()}
    report["lag"] = lag
    report["build_seconds"] = build_seconds

    print("profiling the link columns and the end-of-train graph")
    report["link_columns"] = graph_features.link_column_profile(train, cfg.link_columns)
    report["end_of_train_graph"] = build_end_of_train(train, cfg)
    print("hub sweep")
    report["hub_sweep"] = build_hub_sweep(train, cfg)
    print("link variants")
    report["link_variants"] = build_link_variants(train, cfg)
    print("per feature")
    report["per_feature"] = build_per_feature(frame, built, masks, cfg)
    print("component fraud rate")
    report["component_fraud_rate"] = build_component_fraud_rate(
        frame, built, masks, labels, n_labels_attached
    )
    print("hub-excluded rates")
    report["hub_excluded_rates"] = build_hub_excluded_rates(frame, masks, labels, cfg)
    print("transductive leak")
    report["transductive_leak"] = build_transductive_leak(frame, built, masks, labels, cfg)

    passing = [
        row["feature"] for row in report["per_feature"] if row["stage_4_ship_rule"]["passes"]
    ]
    report["decision"] = {
        "default_enabled": graph_features.DEFAULT_ENABLED,
        "candidate_features": list(graph_features.CANDIDATE_FEATURES),
        "features_passing_the_stage_4_rule": passing,
        "rule": (
            "a graph feature joins the default set when its zero-against-positive separation has "
            "the same sign on all three splits with disjoint Wilson intervals on each, the rule "
            "ADR 0022 wrote for stage 4's counts. The switch defaults to off when none passes. "
            "The component features stay out of the candidate set on the clock test whatever "
            "the rule says, because a feature that is a count of the transactions before it "
            "carries the weekly fraud rate and nothing structural"
        ),
        "provided_by_stage_4": dict(graph_features.PROVIDED_BY_STAGE_4),
        "not_reachable": dict(graph_features.not_reachable()),
    }
    report["seconds"] = time.perf_counter() - started
    write(report, GRAPH_SUMMARY_PATH)
    print(f"\ndone in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
