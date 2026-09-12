"""Draw the stage 9 figures from the stage 9 artifacts. Writes PNG at 200 DPI to figures/.

Usage:
    make monitoring-figures
    .venv/bin/python scripts/monitoring_figures.py

Same rule as stages 2 to 7: each figure reads one artifact under reports/ and never the data,
so it and the prose beside it quote the same numbers.

Figures:
    drift_calibration.png       left, how many of the shipped model's columns the textbook 0.20
                                band puts in alert on each in-control weekly batch against how
                                many the calibrated rule does with that batch held out; right,
                                the score PSI per batch against its alert edge. From
                                reports/monitoring/drift.json.
    decay_in_control.png        the shipped model's PR-AUC on each in-control weekly batch with
                                its bootstrap interval, the reference it was accepted at, and
                                the decay tolerance below it. From reports/monitoring/decay.json.
    adversarial_validation.png  the gain share of the twenty columns that best separate train
                                from test once the target encodings are set aside, coloured by
                                whether the column names who and when or what. From
                                reports/adversarial_validation.json.
    lifecycle_demo.png          top, the drift job per cycle (score PSI and columns past their
                                edges); bottom, the decay job per batch at the cycle its labels
                                matured, with the gate's verdicts. From reports/lifecycle_demo.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fraud_platform import config

DPI = 200
GRID = {"color": "0.85", "linewidth": 0.6}
RED = "#b2182b"
BLUE = "#2166ac"
DARK = "0.15"
GREY = "0.5"
LIGHT = "0.75"

ENTITY_AND_CLOCK = ("identifier", "identifier_encoding", "target_encoding", "timedelta")


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def batch_labels(batches: list[dict[str, Any]]) -> list[str]:
    return [f"{b['window']} {b['batch']}" for b in batches]


def drift_calibration(report: dict[str, Any], out: Path) -> None:
    batches = report["batches"]
    labels = batch_labels(batches)
    xs = list(range(len(batches)))
    textbook = [b["aggregate"]["n_alert_band"] for b in batches]
    # The calibrated count on a batch the edges were fitted on is zero by construction; the
    # held-out count is the one that says something.
    calibrated = list(report["in_control"]["leave_one_out"]["counts"])
    n_columns = report["calibration"]["n_columns"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    ax = axes[0]
    ax.bar([x - 0.2 for x in xs], textbook, width=0.4, color=LIGHT, label="PSI above 0.20")
    ax.bar(
        [x + 0.2 for x in xs],
        calibrated,
        width=0.4,
        color=BLUE,
        label="past the calibrated edge, batch held out of the calibration",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"columns, of {n_columns}")
    ax.set_title("columns in alert per in-control weekly batch", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)

    ax = axes[1]
    psi = [b["score"]["psi"] for b in batches]
    edge = report["calibration"]["score_alert_edge"]
    ax.plot(xs, psi, "o-", color=BLUE, ms=5, lw=1.5, label="score PSI against train")
    ax.axhline(edge, color=RED, lw=1, ls="--", label=f"alert edge {edge:.2f}")
    ax.axhline(
        report["calibration"]["worst_batch_score_psi"],
        color=GREY,
        lw=0.8,
        ls=":",
        label=f"worst in-control batch {report['calibration']['worst_batch_score_psi']:.3f}",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PSI")
    ax.set_ylim(0, max(edge * 1.25, max(psi) * 1.25))
    ax.set_title("the score distribution against the training window", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def decay_in_control(report: dict[str, Any], out: Path) -> None:
    batches = [b for b in report["batches"] if b["status"] == "evaluated"]
    labels = batch_labels(batches)
    xs = list(range(len(batches)))
    reference = report["reference"]["pr_auc"]
    tolerance = report["in_control"]["decay_tolerance"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axhspan(reference["low"], reference["high"], color="0.92", lw=0)
    ax.axhline(
        reference["point"],
        color=GREY,
        lw=0.8,
        ls="--",
        label=f"reference, test window: {reference['point']:.4f}",
    )
    ax.axhline(
        reference["point"] - tolerance,
        color=RED,
        lw=1,
        ls="--",
        label=f"reference less the tolerance {tolerance:.2f}",
    )
    for x, b in zip(xs, batches, strict=True):
        colour = RED if b["decay"] else BLUE
        ax.plot([x, x], [b["pr_auc"]["low"], b["pr_auc"]["high"]], color=colour, lw=2)
        ax.plot(x, b["pr_auc"]["point"], "o", color=colour, ms=5)
    ax.plot([], [], "o-", color=BLUE, label="weekly batch, 95 percent row bootstrap")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("PR-AUC")
    ax.set_title(
        "the shipped model on the in-control weekly batches, labels mature", fontsize=10, loc="left"
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def adversarial_validation(report: dict[str, Any], out: Path, n_top: int = 20) -> None:
    fit = report["fits"]["without_target_encodings"]
    top = fit["top"][:n_top]
    ys = list(range(len(top)))[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for y, row in zip(ys, top, strict=True):
        colour = BLUE if row["group"] in ENTITY_AND_CLOCK else DARK
        ax.barh(y, row["gain_share"], color=colour, height=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["column"] for r in top], fontsize=8)
    ax.set_xlabel("share of gain, train against test, out of fold")
    handles = [
        Patch(color=BLUE, label="who or when: identifiers, their encodings, timedeltas"),
        Patch(color=DARK, label="what: counts, amounts, V blocks, identity fields, the rest"),
    ]
    aucs = {name: f["auc"]["out_of_fold"] for name, f in report["fits"].items()}
    ax.set_title(
        "what separates train from test once the target encodings are set aside\n"
        f"out-of-fold AUC {aucs['without_target_encodings']:.3f}; with the target encodings "
        f"{aucs['full']:.3f}; without identifiers, encodings and timedeltas "
        f"{aucs['without_entity_and_clock']:.3f}",
        fontsize=9,
        loc="left",
    )
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def lifecycle_demo(report: dict[str, Any], out: Path) -> None:
    cycles = report["cycles"]
    drift_from = report["inputs"]["drift_from_day"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)

    ax = axes[0]
    with_batch = [c for c in cycles if c["drift"]]
    days = [c["as_of_day"] for c in with_batch]
    ax.plot(
        days,
        [c["drift"]["score_psi"] for c in with_batch],
        "o-",
        color=BLUE,
        ms=5,
        lw=1.5,
        label="score PSI against the champion's training window",
    )
    edge = report["champion"]["drift_reference"]["score_alert_edge"]
    ax.axhline(
        edge, color=RED, lw=1, ls="--", label=f"score alert edge of the shipped model {edge:.2f}"
    )
    ax.axvline(drift_from, color=GREY, lw=0.8, ls=":")
    ax.text(drift_from, edge * 2.5, " drift injected", color=GREY, fontsize=8, va="bottom")
    for c in with_batch:
        ax.annotate(
            f"{c['drift']['n_alert']} col",
            (c["as_of_day"], c["drift"]["score_psi"]),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=7,
            color="0.3",
        )
    ax.set_ylabel("PSI")
    ax.set_title(
        "the drift job at the close of each weekly batch (labels not read); the annotation is "
        "the number of columns past their calibrated edge",
        fontsize=9,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=8, loc="center right")
    style(ax)

    ax = axes[1]
    first = True
    for c in cycles:
        for d in c["decay"]:
            colour = RED if d["decay"] else BLUE
            ax.plot([c["as_of_day"]] * 2, [d["pr_auc_low"], d["pr_auc_high"]], color=colour, lw=2)
            ax.plot(c["as_of_day"], d["pr_auc"], "o", color=colour, ms=5)
            ax.annotate(
                f"batch {d['batch']}",
                (c["as_of_day"], d["pr_auc_high"]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=7,
                color="0.3",
            )
            if first:
                ax.plot([], [], "o", color=BLUE, label="matured batch, PR-AUC with interval")
                ax.plot([], [], "o", color=RED, label="decay: drop clears the tolerance")
                first = False
    references: dict[float, float] = {}
    for c in cycles:
        for d in c["decay"]:
            references[c["as_of_day"]] = d["reference_pr_auc"]
    if references:
        xs = sorted(references)
        ax.step(
            xs,
            [references[x] for x in xs],
            where="post",
            color=GREY,
            lw=0.8,
            ls="--",
            label="the champion's reference PR-AUC",
        )
        ax.step(
            xs,
            [references[x] - report["inputs"]["decay_tolerance"] for x in xs],
            where="post",
            color=RED,
            lw=0.8,
            ls="--",
            label="reference less the tolerance",
        )
    for c in cycles:
        challenger = c.get("challenger")
        if not challenger or not challenger["trained"]:
            continue
        verdict = challenger["gate"]
        marker = "^" if verdict["promote"] else "v"
        ax.plot(c["as_of_day"], verdict["challenger"]["point"], marker, color=DARK, ms=8)
        ax.annotate(
            "promoted" if verdict["promote"] else "refused",
            (c["as_of_day"], verdict["challenger"]["point"]),
            textcoords="offset points",
            xytext=(8, -3),
            fontsize=7,
            color=DARK,
        )
    ax.plot([], [], "^", color=DARK, label="challenger at the gate, promoted")
    ax.plot([], [], "v", color=DARK, label="challenger at the gate, refused")
    ax.set_xlabel("day of the clock (TransactionDT / 86,400)")
    ax.set_ylabel("PR-AUC")
    ax.set_title(
        "the decay job when a batch's labels mature (30 days after it closes), and the gate",
        fontsize=9,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()
    reports = Path(args.reports_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    drift_calibration(load(reports / "monitoring" / "drift.json"), out / "drift_calibration.png")
    decay_in_control(load(reports / "monitoring" / "decay.json"), out / "decay_in_control.png")
    adversarial_validation(
        load(reports / "adversarial_validation.json"), out / "adversarial_validation.png"
    )
    demo = reports / "lifecycle_demo.json"
    if demo.exists():
        lifecycle_demo(load(demo), out / "lifecycle_demo.png")
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
