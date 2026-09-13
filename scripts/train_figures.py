"""Draw the stage 6 figures from the stage 6 artifacts. Writes PNG at 200 DPI to figures/.

Usage:
    make train-figures
    .venv/bin/python scripts/train_figures.py

Same rule as stages 2 to 5: each figure reads one artifact under reports/ and never the data or
the score cache, so it and the prose beside it quote the same numbers.

Figures:
    model_comparison.png    left, test PR-AUC per model with its paired-bootstrap interval;
                            right, the shipped model's paired difference against each of the
                            others, with the interval. From reports/metric_variance.json.
    ablations.png           left, validation PR-AUC per feature stack, three seeds each, test
                            beside it; right, each stack's paired-bootstrap difference against
                            its reference per seed, the interval the ship rule reads. From
                            reports/ablations.json.
    threshold_curve.png     precision, recall and alerts per day against the calibrated
                            threshold on the validation split, the band edges marked. From
                            reports/threshold_curve.json.
    calibration.png         observed fraud rate against predicted probability on the test split,
                            raw score and calibrated. From reports/operating_points.json.
    shap_global.png         the twenty largest mean absolute SHAP values and the share per
                            block. From reports/shap/shap_global.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fraud_platform import config

DPI = 200
GRID = {"color": "0.85", "linewidth": 0.6}
RED = "#b2182b"
BLUE = "#2166ac"
GREY = "0.35"
FAMILY_COLOURS = {
    "logistic_regression": "#4d9221",
    "random_forest": BLUE,
    "xgboost": RED,
    "shipped": "0.15",
}


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def short(name: str) -> str:
    return (
        name.replace("logistic_regression", "LR")
        .replace("random_forest", "RF")
        .replace("xgboost", "XGB")
        .replace("__", " / ")
    )


def model_comparison(report: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    by_row = report["by_row"]["per_model"]
    names = list(report["models"])

    ax = axes[0]
    ys = list(range(len(names)))[::-1]
    for y, name in zip(ys, names, strict=True):
        interval = by_row[name]
        family = name.split("__")[0] if name != "shipped" else "shipped"
        ax.plot([interval["low"], interval["high"]], [y, y], color=FAMILY_COLOURS[family], lw=2)
        ax.plot(interval["point"], y, "o", color=FAMILY_COLOURS[family], ms=5)
    ax.set_yticks(ys)
    ax.set_yticklabels([short(n) for n in names], fontsize=8)
    ax.set_xlabel("test PR-AUC, 95 percent interval over 1,000 row resamples")
    ax.set_title("Every model on the same resamples of the test split", fontsize=10)
    style(ax)

    ax = axes[1]
    pairs = report["shipped_against_each"]
    ys = list(range(len(pairs)))[::-1]
    for y, pair in zip(ys, pairs, strict=True):
        sign = 1.0 if pair["first"] == "shipped" else -1.0
        diff = pair["difference"]
        low, high = sorted([sign * diff["low"], sign * diff["high"]])
        colour = RED if pair["excludes_zero"] else GREY
        ax.plot([low, high], [y, y], color=colour, lw=2)
        ax.plot(sign * diff["point"], y, "o", color=colour, ms=5)
        ax.text(
            high,
            y,
            f"  {pair['sign_agreement']:.3f}" if pair["sign_agreement"] is not None else "",
            va="center",
            fontsize=7,
        )
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels(
        [short(p["second"] if p["first"] == "shipped" else p["first"]) for p in pairs], fontsize=8
    )
    ax.set_xlabel("shipped minus the other, test PR-AUC, paired 95 percent interval")
    noise = report["noise_band"]
    origin = "card resample" if noise["from"] == "by_card" else "row resample"
    ax.set_title(
        "Paired differences on the row resample; label is the sign agreement\n"
        f"noise band {noise['half_width']:.4f}, the widest paired half-width, from the {origin}",
        fontsize=10,
    )
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def ablations(report: dict[str, Any], out: Path) -> None:
    stacks = report["stacks"]
    fig, axes = plt.subplots(
        1, 2, figsize=(15, 6.0), sharey=True, gridspec_kw={"width_ratios": [3, 2]}
    )
    ys = list(range(len(stacks)))[::-1]

    ax = axes[0]
    for y, entry in zip(ys, stacks, strict=True):
        vals = [row["val_pr_auc"] for row in entry["per_seed"]]
        tests = [row["test_pr_auc"] for row in entry["per_seed"]]
        ax.plot(vals, [y] * len(vals), "o", color=BLUE, ms=5, alpha=0.7)
        ax.plot(tests, [y - 0.25] * len(tests), "o", color=RED, ms=4, alpha=0.5, mfc="none")
        ax.plot(entry["summary"]["val_pr_auc"]["mean"], y, "|", color="0.1", ms=14, mew=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([entry["id"] for entry in stacks], fontsize=8)
    ax.set_xlabel("PR-AUC, one marker per seed; filled val, hollow test, bar the val mean")
    decision = report["decision"]
    ax.set_title(
        f"{report['family']} with {report['imbalance']}, seeds {report['seeds']}\n"
        f"shipped stack: {decision['shipped_stack']['name']}",
        fontsize=10,
    )
    ax.plot([], [], "o", color=BLUE, label="validation")
    ax.plot([], [], "o", color=RED, mfc="none", label="test")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    style(ax)

    # The interval the ship rule reads: each stack's validation PR-AUC minus its reference's,
    # paired on identical resamples, one interval per seed. Red when the interval excludes zero.
    ax = axes[1]
    n_seeds = len(report["seeds"])
    offsets = [(i - (n_seeds - 1) / 2) * 0.22 for i in range(n_seeds)]
    for y, entry in zip(ys, stacks, strict=True):
        against = entry.get("against_reference")
        if against is None:
            ax.text(0, y, "  reference stack", va="center", fontsize=7, color=GREY)
            continue
        highs = []
        for offset, row in zip(offsets, against["per_seed"], strict=True):
            diff = row["val_pr_auc_difference"]
            colour = RED if row["excludes_zero"] else GREY
            ax.plot([diff["low"], diff["high"]], [y + offset, y + offset], color=colour, lw=1.6)
            ax.plot(diff["point"], y + offset, "o", color=colour, ms=3.5)
            highs.append(diff["high"])
        ax.text(
            max(highs) + 0.004,
            y,
            f"vs {against['reference']}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=GREY,
        )
    ax.axvline(0, color="0.2", lw=0.8)
    left, right = ax.get_xlim()
    ax.set_xlim(left, right + 0.06)  # room for the reference labels
    ax.set_xlabel("stack minus its reference, val PR-AUC, paired 95 percent interval per seed")
    ax.set_title(
        "What the ship rule reads: red where the interval excludes zero\n"
        "a built block ships only if every seed is red and above zero",
        fontsize=10,
    )
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def threshold_curve(report: dict[str, Any], out: Path) -> None:
    sweep = report["calibrated"]["val"]
    xs = [row["threshold"] for row in sweep]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(xs, [row["precision"] or 0.0 for row in sweep], color=BLUE, label="precision")
    ax.plot(xs, [row["recall"] or 0.0 for row in sweep], color=RED, label="recall")
    ax.plot(xs, [row["f1"] or 0.0 for row in sweep], color="0.3", ls="--", label="F1")
    ax.set_xlabel("threshold on the calibrated probability, validation split")
    ax.set_ylabel("precision, recall, F1")
    ax.set_ylim(0, 1)
    edges = report["band_edges"]
    for name in ("low", "high"):
        ax.axvline(edges[name], color="0.5", lw=0.8, ls=":")
        ax.text(edges[name], 0.97, f" {name} {edges[name]:.3g}", fontsize=8, color="0.3", va="top")
    twin = ax.twinx()
    twin.plot(
        xs,
        [row["alerts_per_day"] for row in sweep],
        color="#4d9221",
        lw=1.2,
        label="alerts per day",
    )
    twin.set_yscale("log")
    twin.set_ylabel("alerts per day (log)")
    twin.spines["top"].set_visible(False)
    ax.set_title(
        "Threshold sweep on validation; approve below low, review between, block above high",
        fontsize=10,
    )
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, frameon=False, loc="center right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def calibration(report: dict[str, Any], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, key, colour in (
        ("raw score", "raw_score", GREY),
        ("calibrated on val", "calibrated", RED),
    ):
        table = report["calibration"][key]["test"]
        ax.plot(
            [row["mean_predicted"] for row in table["bins"]],
            [row["observed_rate"] for row in table["bins"]],
            marker="o",
            color=colour,
            label=f"{label}: Brier {table['brier']:.4f}, ECE {table['expected_calibration_error']:.4f}",
        )
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean predicted probability in the bin (log)")
    ax.set_ylabel("observed fraud rate in the bin (log)")
    ax.set_title("Reliability on the test split, ten equal-count bins", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def shap_global(report: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    top = report["mean_abs_shap"][:20][::-1]
    ax.barh([row["column"] for row in top], [row["mean_abs_shap"] for row in top], color=BLUE)
    ax.set_xlabel("mean absolute SHAP value, log-odds")
    ax.set_title(
        f"Twenty largest of {len(report['mean_abs_shap'])} on {report['method']['n_rows']:,} test rows",
        fontsize=10,
    )
    ax.tick_params(axis="y", labelsize=8)
    style(ax)

    ax = axes[1]
    blocks = report["by_block"]
    names = list(blocks)
    ax.barh(names[::-1], [blocks[n]["mean_abs_shap_share"] for n in names][::-1], color=RED)
    for i, name in enumerate(names[::-1]):
        ax.text(
            blocks[name]["mean_abs_shap_share"],
            i,
            f"  {blocks[name]['n_columns']} cols",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("share of the summed mean absolute SHAP")
    ax.set_title("By block; own velocity is a subset of stage 4", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6 figures.")
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()

    reports, out_dir = Path(args.reports_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("metric_variance.json", "model_comparison.png", model_comparison),
        ("ablations.json", "ablations.png", ablations),
        ("threshold_curve.json", "threshold_curve.png", threshold_curve),
        ("operating_points.json", "calibration.png", calibration),
        ("shap/shap_global.json", "shap_global.png", shap_global),
    )
    for source_name, out_name, draw in jobs:
        source = reports / source_name
        path = out_dir / out_name
        draw(load(source), path)
        print(f"wrote {path} from {source_name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
