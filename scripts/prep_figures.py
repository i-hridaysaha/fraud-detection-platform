"""Draw the stage 3 figures from the stage 3 artifacts. Writes PNG at 200 DPI to figures/.

Usage:
    make prep-figures
    .venv/bin/python scripts/prep_figures.py

Same rule as stage 2: every figure reads a JSON artifact and never the data. A caption and the
prose beside it therefore quote the same number, and a figure cannot be redrawn from a different
slice than the one it claims.

Figures:
    d_column_psi.png         reports/prep/transforms.json   D column PSI before and after
                                                            normalisation, per column
    target_encoding_lag.png  reports/encoding_spec.json     what the label lag costs and what it
                                                            removes
    v_reduction.png          reports/prep/v_reduction.json  column count against validation AUC
                                                            for the three strategies and the
                                                            unreduced baseline
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
BEFORE = "#2166ac"
AFTER = "#b2182b"
NEUTRAL = "0.35"


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def d_column_psi(report: dict[str, Any], out: Path) -> None:
    """PSI per D column, raw against normalised, with the validation AUC beside it."""
    block = report["d_normalisation"]
    rows = sorted(block["per_column"], key=lambda r: int(str(r["column"])[1:]))
    labels = [r["column"] for r in rows]
    before = [r["psi_before"] for r in rows]
    after = [r["psi_after"] for r in rows]
    auc_before = [r["validation_auc_before"] for r in rows]
    auc_after = [r["validation_auc_after"] for r in rows]
    positions = np.arange(len(rows))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)

    ax = axes[0]
    ax.bar(positions - width / 2, before, width, color=BEFORE, label="raw delta")
    ax.bar(positions + width / 2, after, width, color=AFTER, label="normalised to origin")
    for level, text in ((0.10, "0.10"), (0.25, "0.25")):
        ax.axhline(level, color=NEUTRAL, linewidth=0.8, linestyle=":")
        ax.text(len(rows) - 0.4, level, f" PSI {text}", fontsize=7, color=NEUTRAL, va="bottom")
    ax.set_ylabel("PSI, train against test")
    ax.set_title(
        f"D-column normalisation raises PSI on {block['n_worsening']} of "
        f"{block['n_columns']} columns and lowers it on {block['n_improving']}",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = axes[1]
    ax.bar(positions - width / 2, auc_before, width, color=BEFORE, label="raw delta")
    ax.bar(positions + width / 2, auc_after, width, color=AFTER, label="normalised to origin")
    ax.axhline(0.5, color=NEUTRAL, linewidth=0.8, linestyle=":")
    ax.set_ylim(0.45, max(auc_before + auc_after) + 0.03)
    ax.set_ylabel("single-feature validation AUC")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    improves = block["auc_probe"]["n_columns_where_validation_auc_improves"]
    ax.set_title(
        f"and improves validation AUC on {improves} of {block['n_columns']}, none of them by "
        "more than the split resolves",
        fontsize=11,
    )
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def target_encoding_lag(report: dict[str, Any], out: Path) -> None:
    """What the label lag costs on validation, and what it removes from the train gap."""
    sweep = report["encoders"]["target"]["lag_sweep"]
    lags = sorted(int(lag) for lag in sweep["per_lag"])
    val = [sweep["per_lag"][str(lag)]["mean_validation_auc"] for lag in lags]
    train = [sweep["per_lag"][str(lag)]["mean_train_auc"] for lag in lags]
    gap = [sweep["gap_test"]["per_lag"][str(lag)]["mean"] for lag in lags]
    half = [sweep["gap_test"]["per_lag"][str(lag)]["half_width_95"] for lag in lags]
    chosen = int(report["encoders"]["target"]["chosen_lag_days"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(lags, train, marker="o", color=AFTER, label="train, in sample")
    ax.plot(lags, val, marker="o", color=BEFORE, label="validation")
    ax.axvline(chosen, color=NEUTRAL, linewidth=0.8, linestyle=":")
    ax.text(chosen, min(val), f" lag {chosen}", fontsize=8, color=NEUTRAL, va="bottom")
    ax.set_xlabel("label lag, days")
    ax.set_ylabel("mean AUC of the encoded column alone")
    ax.set_title("What the lag costs", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    style(ax)

    ax = axes[1]
    ax.errorbar(lags, gap, yerr=half, marker="o", color=AFTER, capsize=3, linewidth=1.2)
    ax.axhline(0.0, color=NEUTRAL, linewidth=0.8, linestyle=":")
    ax.axvline(chosen, color=NEUTRAL, linewidth=0.8, linestyle=":")
    ax.set_xlabel("label lag, days")
    ax.set_ylabel("train AUC minus validation AUC")
    ax.set_title("What the lag removes", fontsize=11)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def v_reduction(report: dict[str, Any], out: Path) -> None:
    """Column count against validation AUC, with the seed spread each strategy shows."""
    results = report["comparison"]["results"]
    variance = {v["strategy"]: v for v in report["comparison"]["seed_variance"]["per_strategy"]}
    chosen = report["decision"]["chosen"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in results:
        name = row["strategy"]
        spread = variance.get(name, {})
        aucs = spread.get("validation_aucs") or [row["validation_auc"]]
        low, high = min(aucs), max(aucs)
        colour = AFTER if name == chosen else BEFORE
        ax.plot([row["n_columns"], row["n_columns"]], [low, high], color=colour, linewidth=2.0)
        ax.scatter(
            [row["n_columns"]],
            [row["validation_auc"]],
            color=colour,
            s=60 if name == chosen else 36,
            zorder=3,
        )
        ax.annotate(
            f"{name}\n{row['n_columns']} columns",
            (row["n_columns"], low),
            textcoords="offset points",
            xytext=(0, -22),
            ha="center",
            fontsize=8,
            color=colour,
        )
    ax.set_xscale("log")
    ax.margins(x=0.25, y=0.30)
    ax.set_xlabel("columns in the V block after the strategy (log scale)")
    ax.set_ylabel("validation AUC, single probe")
    observed = report["decision"]["resolution"]["observed_spread_between_strategies"]
    seed_spread = report["decision"]["resolution"]["largest_seed_spread_within_a_strategy"]
    ax.set_title(
        f"Four strategies spread {observed:.4f} in validation AUC; one strategy spreads "
        f"{seed_spread:.4f} across seeds",
        fontsize=11,
    )
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()

    reports, out_dir = Path(args.reports_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = (
        (reports / "prep" / "transforms.json", d_column_psi, "d_column_psi.png"),
        (reports / "encoding_spec.json", target_encoding_lag, "target_encoding_lag.png"),
        (reports / "prep" / "v_reduction.json", v_reduction, "v_reduction.png"),
    )
    for source, draw, name in jobs:
        path = out_dir / name
        draw(load(source), path)
        print(f"wrote {path} from {source.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
