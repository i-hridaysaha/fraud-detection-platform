"""Draw the stage 7 figures from the stage 7 artifacts. Writes PNG at 200 DPI to figures/.

Usage:
    make experiment-figures
    .venv/bin/python scripts/experiment_figures.py

Same rule as stages 2 to 6: each figure reads one artifact under reports/ and never the data,
so it and the prose beside it quote the same numbers.

Figures:
    leakage_delta.png     left, test PR-AUC per variant with its bootstrap interval; right, each
                          variant's difference against the causal baseline with the interval,
                          paired where the rows are shared and independent where they are not.
                          From reports/leakage_delta.json.
    label_latency.png     left, test PR-AUC against the simulated latency for the immature and
                          the excluded treatment, the reference beside them; right, the fraud
                          rate each model implies on test against the observed rate. From
                          reports/label_latency.json.
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
DARK = "0.15"
GREY = "0.5"

VARIANT_LABELS = {
    "baseline": "causal baseline",
    "a_full_fit_encoders": "(a) encoders fitted on all rows",
    "b_full_data_entity_aggregates": "(b) entity aggregates over all rows",
    "c_entity_mean_postprocessing": "(c) entity-mean post-processing",
    "d_random_split": "(d) random split",
    "abc_chronological": "(a)+(b)+(c), chronological",
    "abcd_random": "(a)+(b)+(c)+(d), random",
}


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def leakage_delta(report: dict[str, Any], out: Path) -> None:
    variants = report["variants"]
    deltas = report["deltas"]["per_variant"]
    names = [v["id"] for v in variants]
    ys = list(range(len(names)))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    def colour(variant_id: str) -> str:
        if variant_id == "baseline":
            return DARK
        return RED if deltas[variant_id]["split"] == "random" else BLUE

    ax = axes[0]
    for y, variant in zip(ys, variants, strict=True):
        vid = variant["id"]
        kind = "chronological" if variant["split"] == "chronological" else "random"
        interval = report["bootstrap"]["pr_auc_by_row"][kind]["per_model"][vid]
        ax.plot([interval["low"], interval["high"]], [y, y], color=colour(vid), lw=2)
        ax.plot(interval["point"], y, "o", color=colour(vid), ms=5)
        ax.text(
            interval["high"] + 0.006,
            y,
            f"{interval['point']:.4f}",
            va="center",
            fontsize=8,
            color="0.25",
        )
    ax.axvline(
        report["shipped_model_test_pr_auc"],
        color=GREY,
        lw=0.8,
        ls="--",
        label="shipped model, stage 6",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([VARIANT_LABELS[n] for n in names], fontsize=9)
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_xlabel("test PR-AUC, 95 percent interval over 1,000 row resamples")
    ax.set_title("What the same model scores with each shortcut on", fontsize=10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    style(ax)

    ax = axes[1]
    for y, variant in zip(ys, variants, strict=True):
        vid = variant["id"]
        if vid == "baseline":
            continue
        d = deltas[vid]["pr_auc_by_row"]["difference"]
        ax.plot([d["low"], d["high"]], [y, y], color=colour(vid), lw=2)
        ax.plot(d["point"], y, "o", color=colour(vid), ms=5)
        tag = "" if deltas[vid]["pr_auc_by_row"]["excludes_zero"] else "  inside the noise"
        ax.text(
            max(d["high"], 0) + 0.006,
            y,
            f"{d['point']:+.4f}{tag}",
            va="center",
            fontsize=8,
            color="0.25",
        )
    ax.axvline(0, color="0.2", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([""] * len(ys))
    ax.set_ylim(-0.5, len(names) - 0.5)
    ax.set_xlabel("test PR-AUC, variant minus baseline, 95 percent interval")
    ax.set_title("The leakage delta", fontsize=10)
    ax.plot([], [], color=BLUE, lw=2, label="chronological test rows, paired resamples")
    ax.plot([], [], color=RED, lw=2, label="random test rows, independent resamples")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def label_latency(report: dict[str, Any], out: Path) -> None:
    sweep = report["sweep"]
    latencies = [row["latency_days"] for row in sweep]
    reference = next(r for r in report["runs"] if r["id"] == "reference")
    per_model = report["bootstrap"]["pr_auc"]["per_model"]
    rate_model = report["bootstrap"]["mean_score"]["per_model"]
    observed = reference["test"]["observed_fraud_rate"]
    colours = {"immature": RED, "excluded": BLUE}
    labels = {
        "immature": "immature: recent fraud labelled legitimate",
        "excluded": "excluded: recent rows dropped from the fit",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    ref = per_model["reference"]
    ax.axhspan(ref["low"], ref["high"], color="0.9", lw=0)
    ax.axhline(ref["point"], color=GREY, lw=0.8, ls="--", label="reference: every label final")
    for treatment in ("immature", "excluded"):
        points = [per_model[row[treatment]["id"]] for row in sweep]
        ax.errorbar(
            latencies,
            [p["point"] for p in points],
            yerr=[
                [p["point"] - p["low"] for p in points],
                [p["high"] - p["point"] for p in points],
            ],
            color=colours[treatment],
            marker="o",
            ms=5,
            lw=1.5,
            capsize=3,
            label=labels[treatment],
        )
    ax.set_xticks(latencies)
    ax.set_xlabel("simulated latency N, days of the training window with immature labels")
    ax.set_ylabel("test PR-AUC, 95 percent interval over 1,000 row resamples")
    ax.set_title("Test PR-AUC against the simulated latency", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    style(ax)

    ax = axes[1]
    ax.axhline(
        observed, color="0.2", lw=0.8, ls="-", label=f"observed test fraud rate {observed:.4f}"
    )
    ax.axhline(
        rate_model["reference"]["point"],
        color=GREY,
        lw=0.8,
        ls="--",
        label=f"reference model implies {rate_model['reference']['point']:.4f}",
    )
    for treatment in ("immature", "excluded"):
        points = [rate_model[row[treatment]["id"]] for row in sweep]
        ax.errorbar(
            latencies,
            [p["point"] for p in points],
            yerr=[
                [p["point"] - p["low"] for p in points],
                [p["high"] - p["point"] for p in points],
            ],
            color=colours[treatment],
            marker="o",
            ms=5,
            lw=1.5,
            capsize=3,
            label=labels[treatment],
        )
    ax.set_xticks(latencies)
    ax.set_ylim(bottom=-0.0015)
    ax.set_xlabel("simulated latency N, days")
    ax.set_ylabel("fraud rate the model implies on test: its mean score")
    ax.set_title("The bias in the implied fraud rate", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 figures.")
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()
    reports = Path(args.reports_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    leakage_delta(load(reports / "leakage_delta.json"), out / "leakage_delta.png")
    print(f"  wrote {out / 'leakage_delta.png'}")
    label_latency(load(reports / "label_latency.json"), out / "label_latency.png")
    print(f"  wrote {out / 'label_latency.png'}")


if __name__ == "__main__":
    main()
