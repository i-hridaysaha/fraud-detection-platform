"""Draw the stage 4 figures from the stage 4 artifacts. Writes PNG at 200 DPI to figures/.

Usage:
    make features-figures
    .venv/bin/python scripts/feature_figures.py

Same rule as stages 2 and 3: every figure reads a JSON artifact and never the data, so a figure
and the prose beside it quote the same number and a figure cannot be redrawn from a different
slice than the one it claims.

Figures:
    velocity_vs_native_c.png   reports/features/velocity_vs_c.json     this repo's point-in-time
                                                                        velocity against the
                                                                        native C columns
    feature_coverage.png       reports/feature_summary.json            what a point-in-time
                                                                        feature covers against
                                                                        what a train-fitted
                                                                        entity statistic covers
    duplicate_content.png      reports/features/duplicate_content.json fraud rate by time since
                                                                        the last identical
                                                                        purchase
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from fraud_platform import config

DPI = 200
GRID = {"color": "0.85", "linewidth": 0.6}
OWN = "#2166ac"
NATIVE = "#b2182b"
NEUTRAL = "0.35"


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def velocity_vs_native_c(report: dict[str, Any], out: Path) -> None:
    """Correlation between the two blocks, and the single-feature lift of each.

    The left panel is the answer to "is this the same thing measured twice". The right panel is
    the answer to "and which one is stronger", and the two answers point in different directions,
    which is the reason both blocks go into stage 6.
    """
    own = report["per_feature"]
    native = report["native_c_columns"]
    own_names = [row["feature"] for row in own]
    native_names = [row["column"] for row in native]

    matrix = np.array(
        [
            [abs(row["spearman_against_c"].get(column) or 0.0) for column in native_names]
            for row in own
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    image = ax.imshow(matrix, cmap="magma_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(native_names)), native_names, fontsize=8)
    ax.set_yticks(
        range(len(own_names)),
        [name.replace("vel_count_", "") for name in own_names],
        fontsize=8,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}".lstrip("0"),
                ha="center",
                va="center",
                fontsize=6.5,
                color="0.15" if matrix[i, j] < 0.6 else "white",
            )
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="|Spearman rho|")
    ax.set_title(
        f"Own velocity against the native C columns\n"
        f"largest |rho| {report['summary']['max_abs_rho']:.4f}, median "
        f"{report['summary']['median_abs_rho']:.4f}, none above 0.5",
        fontsize=10,
    )

    ax = axes[1]
    own_lift = [row["zero_against_positive"]["lift"] for row in own]
    native_lift = [row["zero_against_positive"]["lift"] for row in native]
    own_share = [row["zero_against_positive"]["share_positive"] for row in own]
    native_share = [row["zero_against_positive"]["share_positive"] for row in native]
    ax.scatter(own_share, own_lift, s=55, color=OWN, label="this repo's velocity", zorder=3)
    ax.scatter(
        native_share, native_lift, s=55, color=NATIVE, marker="s", label="native C", zorder=3
    )
    labelled = [
        (share, lift, name)
        for share, lift, name in zip(native_share, native_lift, native_names, strict=True)
        if lift is not None and lift >= 3.0
    ]
    for i, (share, lift, name) in enumerate(sorted(labelled, key=lambda row: -row[1])):
        ax.annotate(
            name,
            (share, lift),
            fontsize=7,
            xytext=(6, 4 if i % 2 == 0 else -9),
            textcoords="offset points",
        )
    ax.axhline(1.0, color=NEUTRAL, linewidth=0.9, linestyle="--")
    ax.set_xlabel("share of train rows where the count is positive")
    ax.set_ylabel("fraud rate ratio, positive against zero")
    ax.set_title(
        "Single-feature lift of each block\nthe native counters are stronger; their causality "
        "is not established",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def feature_coverage(report: dict[str, Any], out: Path) -> None:
    """What an entity-history feature covers, against what a train-fitted entity statistic does.

    The two bars per split are the whole argument for building behaviour instead of encoding
    labels. One is bounded by how much of the split the training window happened to have seen.
    The other is bounded only by whether the entity has transacted before, which an online store
    knows whatever window the model was fitted on.
    """
    coverage = report["coverage"]
    splits = [name for name in ("val", "test") if name in coverage]
    overlap = [
        coverage[name]["entity_history_against_training_overlap"]["share_entity_seen_in_train"]
        for name in splits
    ]
    history = [
        coverage[name]["entity_history_against_training_overlap"][
            "share_with_any_earlier_row_in_the_stream"
        ]
        for name in splits
    ]
    positions = np.arange(len(splits))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.bar(
        positions - width / 2,
        overlap,
        width,
        color=NATIVE,
        label="entity seen in the training window",
    )
    ax.bar(
        positions + width / 2,
        history,
        width,
        color=OWN,
        label="has an earlier row in the stream",
    )
    for x, value in zip(positions - width / 2, overlap, strict=True):
        ax.text(x, value + 0.012, f"{value:.4f}", ha="center", fontsize=8)
    for x, value in zip(positions + width / 2, history, strict=True):
        ax.text(x, value + 0.012, f"{value:.4f}", ha="center", fontsize=8)
    ax.set_xticks(positions, splits)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("share of rows")
    ax.set_title(
        "Coverage of an entity feature\na train-fitted statistic against a point-in-time one",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style(ax)

    ax = axes[1]
    families: dict[str, list[float]] = {}
    for row in report["per_feature"]:
        rate = row["null_rate_by_split"]["train"]["null_rate"]
        if rate is not None:
            families.setdefault(str(row["family"]), []).append(float(rate))
    names = sorted(families, key=lambda name: -float(np.mean(families[name])))
    ax.barh(
        range(len(names)),
        [float(np.mean(families[name])) for name in names],
        color=NEUTRAL,
    )
    for i, name in enumerate(names):
        ax.text(
            float(np.mean(families[name])) + 0.01,
            i,
            f"{np.mean(families[name]):.3f} over {len(families[name])}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(range(len(names)), names, fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("mean null rate on train, per family")
    ax.set_title(
        "Where the nulls are\na null means the stream held no earlier event of that kind",
        fontsize=10,
    )
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def duplicate_content(report: dict[str, Any], out: Path) -> None:
    """Fraud rate by time since the last identical purchase, and the causal against the groupby."""
    entry = next(row for row in report["per_feature"] if row["feature"] == "dup_seconds_since_prev")
    bins = [row for row in entry["deciles"] if row["bin"] != "(missing)"]
    missing = next((row for row in entry["deciles"] if row["bin"] == "(missing)"), None)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    positions = np.arange(len(bins))
    rates = [row["fraud_rate"] for row in bins]
    low = [row["fraud_rate"] - row["interval"]["low"] for row in bins]
    high = [row["interval"]["high"] - row["fraud_rate"] for row in bins]
    ax.errorbar(
        positions, rates, yerr=[low, high], fmt="o-", color=OWN, capsize=3, markersize=5, zorder=3
    )
    if missing is not None:
        ax.axhline(
            missing["fraud_rate"],
            color=NATIVE,
            linestyle="--",
            linewidth=1.1,
            label=f"no earlier identical purchase, {missing['fraud_rate']:.5f}",
        )
    ax.set_xticks(
        positions,
        [f"{row['x_min']:.0f}\nto\n{row['x_max']:.0f}" for row in bins],
        fontsize=6.5,
    )
    ax.set_xlabel("seconds since the last identical purchase, train deciles")
    ax.set_ylabel("fraud rate")
    ax.set_title(
        f"dup_seconds_since_prev, information value {entry['information_value']:.4f}\n"
        "the signal is recency, not repetition",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    ax = axes[1]
    hour = next(
        row["by_split"] for row in report["per_feature"] if row["feature"] == "dup_prior_count_1h"
    )
    ever = report["causal_grouping_by_split"]
    splits = ["train", "val", "test"]
    positions = np.arange(len(splits))
    width = 0.2

    series = (
        ("any earlier identical, marked", ever, "fraud_rate_positive", NATIVE, "o"),
        ("any earlier identical, none", ever, "fraud_rate_zero", "0.65", "o"),
        ("identical inside 3,600s, marked", hour, "fraud_rate_positive", OWN, "s"),
        ("identical inside 3,600s, none", hour, "fraud_rate_zero", "0.4", "s"),
    )
    for i, (label, block, key, color, marker) in enumerate(series):
        values = [block[split][key] for split in splits]
        interval = "interval_positive" if key.endswith("positive") else "interval_zero"
        low = [block[split][key] - block[split][interval]["low"] for split in splits]
        high = [block[split][interval]["high"] - block[split][key] for split in splits]
        ax.errorbar(
            positions + (i - 1.5) * width,
            values,
            yerr=[low, high],
            fmt=marker,
            color=color,
            capsize=3,
            markersize=6,
            linestyle="none",
            label=label,
            zorder=3,
        )
    ax.set_xticks(positions, splits)
    ax.set_ylabel("fraud rate")
    ax.set_title(
        "The all-history count reverses, the hour window does not\n"
        f"ratio {ever['train']['lift']:.2f}, {ever['val']['lift']:.2f}, {ever['test']['lift']:.2f} "
        f"against {hour['train']['lift']:.2f}, {hour['val']['lift']:.2f}, {hour['test']['lift']:.2f}",
        fontsize=10,
    )
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 figures.")
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()

    reports, out_dir = Path(args.reports_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        (
            reports / "features" / "velocity_vs_c.json",
            velocity_vs_native_c,
            "velocity_vs_native_c.png",
        ),
        (reports / "feature_summary.json", feature_coverage, "feature_coverage.png"),
        (
            reports / "features" / "duplicate_content.json",
            duplicate_content,
            "duplicate_content.png",
        ),
    )
    for source, draw, name in jobs:
        path = out_dir / name
        draw(load(source), path)
        print(f"wrote {path} from {source.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
