"""Draw the stage 2 figures from the stage 2 artifacts. Writes PNG at 200 DPI to figures/.

Usage:
    make eda-figures
    .venv/bin/python scripts/eda_figures.py

Every figure reads a JSON file under reports/eda/ and nothing else. The data files are never
opened here. That is the point: a figure and the prose beside it quote the same artifact, so a
number in a caption cannot drift from the number in the text, and a figure cannot be redrawn
from a different slice of the data than the one it claims.

Figures:
    fraud_rate_over_time.png     target.json        daily and weekly rate across the train span
    amount_distribution.png      univariate.json    TransactionAmt raw against log1p
    missingness_blocks.png       missingness.json   which column blocks go missing together
    fraud_rate_by_decile.png     bivariate.json     shape of the relationship for eight columns
    psi_train_vs_test.png        temporal.json      drift per feature, val and test
    time_consistency.png         temporal.json      early window AUC against late window AUC
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
FRAUD = "#b2182b"
CLEAN = "#2166ac"
NEUTRAL = "0.35"


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fraud_rate_over_time(target: dict[str, Any], out: Path) -> None:
    daily = target["by_day"]
    weekly = target["by_week"]
    overall = target["overall_train"]["fraud_rate"]
    trend = target["stationarity"]["weekly"]

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(
        [row["period"] for row in daily],
        [row["fraud_rate"] for row in daily],
        color=NEUTRAL,
        linewidth=0.9,
        alpha=0.55,
        label="daily",
    )
    ax.plot(
        [row["period"] * 7 + 4 for row in weekly],
        [row["fraud_rate"] for row in weekly],
        color=FRAUD,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="weekly",
    )
    ax.axhline(overall, color=CLEAN, linewidth=1.2, linestyle="--", label="train mean")
    ax.set_xlabel("day index within the train split (relative, day 1 is the first day observed)")
    ax.set_ylabel("fraud rate")
    ax.set_title(
        "Fraud rate over the train span\n"
        f"weekly range {trend['rate_min']:.4f} to {trend['rate_max']:.4f}, "
        f"Mann-Kendall tau {trend['kendall_tau']:.3f} (p = {trend['kendall_p_value']:.3g})",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def amount_distribution(univariate: dict[str, Any], out: Path) -> None:
    amount = univariate["amount"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    for ax, key, title, skew in (
        (axes[0], "histogram_raw", "TransactionAmt", amount["skew_raw"]),
        (axes[1], "histogram_log1p", "log1p(TransactionAmt)", amount["skew_log1p"]),
    ):
        hist = amount[key]
        edges = np.array(hist["edges"])
        counts = np.array(hist["counts"])
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=CLEAN,
            edgecolor="none",
        )
        ax.set_yscale("log")
        ax.set_xlabel(title)
        ax.set_ylabel("rows (log scale)")
        ax.set_title(f"{title}\nskew {skew:.3f}", fontsize=10)
        style(ax)
    fig.suptitle("Amount before and after log1p, train split", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def missingness_blocks(missingness: dict[str, Any], out: Path) -> None:
    overlap = missingness["block_overlap"]
    blocks = overlap["blocks"]
    matrix = np.array(overlap["share_missing_in_both"])
    labels = [f"{b['representative']} ({b['n_columns']})" for b in blocks]

    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    image = ax.imshow(matrix, cmap="magma_r", vmin=0.0, vmax=float(matrix.max()))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(
        "Missingness blocks: share of train rows missing in both blocks\n"
        "diagonal is the block's own missing rate, label gives its representative "
        "column and column count",
        fontsize=10,
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.045)
    bar.set_label("share of train rows", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def fraud_rate_by_decile(bivariate: dict[str, Any], out: Path) -> None:
    tables = bivariate["deciles_reference"]
    columns = list(tables)[:8]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2))
    for ax, column in zip(axes.ravel(), columns, strict=False):
        rows = tables[column]
        positions = np.arange(len(rows))
        rates = [row["fraud_rate"] for row in rows]
        low = [row["fraud_rate"] - row["interval"]["low"] for row in rows]
        high = [row["interval"]["high"] - row["fraud_rate"] for row in rows]
        colours = ["0.6" if row["bin"] == "(missing)" else FRAUD for row in rows]
        ax.bar(positions, rates, color=colours, edgecolor="none")
        ax.errorbar(positions, rates, yerr=[low, high], fmt="none", ecolor="0.25", elinewidth=0.8)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            ["miss" if row["bin"] == "(missing)" else row["bin"][1:] for row in rows],
            fontsize=6,
        )
        ax.set_title(column, fontsize=9)
        ax.set_ylabel("fraud rate", fontsize=8)
        ax.tick_params(labelsize=7)
        style(ax)
    for ax in axes.ravel()[len(columns) :]:
        ax.axis("off")
    fig.suptitle(
        "Fraud rate by decile on the train split, grey bar is the missing level, "
        "whiskers are 95 percent Wilson intervals",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def psi_train_vs_test(temporal: dict[str, Any], out: Path) -> None:
    rows = temporal["psi"]["per_feature"]
    top = rows[:25]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})

    positions = np.arange(len(top))[::-1]
    axes[0].barh(
        positions + 0.2,
        [r["psi_test"] for r in top],
        height=0.38,
        color=FRAUD,
        label="train against test",
    )
    axes[0].barh(
        positions - 0.2,
        [r["psi_val"] for r in top],
        height=0.38,
        color=CLEAN,
        label="train against val",
    )
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([r["column"] for r in top], fontsize=7)
    axes[0].set_xlabel("population stability index")
    axes[0].set_title("The 25 features that drift most against test", fontsize=10)
    axes[0].axvline(0.10, color="0.4", linestyle=":", linewidth=1)
    axes[0].axvline(0.25, color="0.4", linestyle="--", linewidth=1)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    style(axes[0])

    values = np.array([r["psi_test"] for r in rows])
    axes[1].hist(np.clip(values, 0, 0.6), bins=60, color=NEUTRAL)
    axes[1].axvline(0.10, color="0.4", linestyle=":", linewidth=1)
    axes[1].axvline(0.25, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("PSI, train against test (clipped at 0.6 for display)")
    axes[1].set_ylabel("features (log scale)")
    summary = temporal["psi"]["summary"]
    axes[1].set_title(
        f"All {len(rows)} features: {summary['n_psi_test_above_0_10']} above 0.10, "
        f"{summary['n_psi_test_above_0_25']} above 0.25",
        fontsize=10,
    )
    style(axes[1])

    fig.suptitle("Distribution drift from the train window, bins fixed on train", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def time_consistency(temporal: dict[str, Any], out: Path) -> None:
    block = temporal["time_consistency"]
    rows = [r for r in block["per_feature"] if r["early_auc"] is not None]
    flagged = {r["column"] for r in block["flagged"]}
    early = np.array([r["early_auc"] for r in rows])
    late = np.array([r["late_auc"] for r in rows])
    is_flagged = np.array([r["column"] in flagged for r in rows])

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(early[~is_flagged], late[~is_flagged], s=14, color=CLEAN, alpha=0.6, label="kept")
    ax.scatter(early[is_flagged], late[is_flagged], s=20, color=FRAUD, label="flagged")
    ax.axhline(0.5, color="0.4", linewidth=1, linestyle="--")
    ax.axvline(block["flag_rule"]["early_auc_at_least"], color="0.4", linewidth=1, linestyle=":")
    ax.set_xlabel("AUC on the first 30 days of train, in sample")
    ax.set_ylabel("AUC on the last 30 days of train")
    ax.set_title(
        "Time consistency screen, one single-feature model per column\n"
        f"{block['n_flagged']} of {len(rows)} columns separate early and invert late",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default=str(config.REPORTS_DIR / "eda"))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        ("target.json", fraud_rate_over_time, "fraud_rate_over_time.png"),
        ("univariate.json", amount_distribution, "amount_distribution.png"),
        ("missingness.json", missingness_blocks, "missingness_blocks.png"),
        ("bivariate.json", fraud_rate_by_decile, "fraud_rate_by_decile.png"),
        ("temporal.json", psi_train_vs_test, "psi_train_vs_test.png"),
        ("temporal.json", time_consistency, "time_consistency.png"),
    )
    for source, draw, name in jobs:
        path = out_dir / name
        draw(load(in_dir / source), path)
        print(f"wrote {path} from {source} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
