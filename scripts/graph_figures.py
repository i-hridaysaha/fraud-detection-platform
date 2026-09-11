"""Draw the stage 5 figure from the stage 5 artifact. Writes PNG at 200 DPI to figures/.

Usage:
    make graph-figures
    .venv/bin/python scripts/graph_figures.py

Same rule as stages 2 to 4: the figure reads reports/graph_summary.json and never the data, so it
and the prose beside it quote the same numbers.

Figure:
    graph_components.png    left, the hub sweep: how much of the training window sits in the
                            largest component, and how much sits alone, as values above a share
                            of train rows are stopped from linking. Right, the fraud rate by
                            decile of the point-in-time component size on each split, which is
                            the shape of a clock and not of a graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from fraud_platform import config, graph_features

DPI = 200
GRID = {"color": "0.85", "linewidth": 0.6}
GIANT = "#b2182b"
LONE = "#2166ac"
SPLIT_COLOURS = {"train": "0.25", "val": "#2166ac", "test": "#b2182b"}


def load(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text()))


def style(ax: Any) -> None:
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def graph_components(report: dict[str, Any], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    end = report["end_of_train_graph"]
    points = report["hub_sweep"]["points"]
    shares = [1.0, *[point["hub_share"] for point in points]]
    giant = [
        end["giant_component"]["share_of_transactions"],
        *[point["giant_component_share_of_transactions"] for point in points],
    ]
    lone = [
        end["share_transactions_in_a_component_of_size_1"],
        *[point["share_transactions_in_a_component_of_size_1"] for point in points],
    ]
    ax.plot(shares, giant, marker="o", color=GIANT, label="in the largest component")
    ax.plot(shares, lone, marker="s", color=LONE, label="alone in a component of size 1")
    for share, g, lo in zip(shares, giant, lone, strict=True):
        # The label sits on the far side of the other curve, so the two never collide.
        ax.annotate(
            f"{g:.4f}",
            (share, g),
            fontsize=7,
            xytext=(0, 6 if g >= lo else -11),
            textcoords="offset points",
            ha="center",
        )
        ax.annotate(
            f"{lo:.4f}",
            (share, lo),
            fontsize=7,
            xytext=(0, -11 if g >= lo else 6),
            textcoords="offset points",
            ha="center",
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(shares, ["no exclusion", *[f"{share:g}" for share in shares[1:]]], fontsize=8)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlabel("values carried by at least this share of train rows link nothing")
    ax.set_ylabel("share of train transactions")
    ax.set_title(
        f"The end-of-train graph over {len(report['config']['link_columns'])} link columns\n"
        f"{end['n_components']} components, the largest holding "
        f"{end['giant_component']['share_of_transactions']:.4f} of {end['n_transactions']:,} rows",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="center left")
    style(ax)

    ax = axes[1]
    size_row = next(
        row for row in report["per_feature"] if row["feature"] == graph_features.COMPONENT_SIZE
    )
    for split in config.SPLIT_NAMES:
        table = [
            bin_ for bin_ in size_row["fraud_rate_by_decile"][split] if bin_["x_min"] is not None
        ]
        xs = list(range(1, len(table) + 1))
        ax.plot(
            xs,
            [bin_["fraud_rate"] for bin_ in table],
            marker="o",
            color=SPLIT_COLOURS[split],
            label=f"{split}, PSI against train {size_row['drift'].get(split, 0.0):.2f}"
            if split != "train"
            else "train",
        )
    ax.set_xticks(list(range(1, 11)))
    ax.set_xlabel(f"decile of {graph_features.COMPONENT_SIZE} within the split")
    ax.set_ylabel("fraud rate")
    rho = size_row["clock_test"]["spearman_with_transaction_dt_train"]
    ax.set_title(
        f"Fraud rate by decile of the point-in-time component size\n"
        f"Spearman with TransactionDT on train {rho:.4f}: the deciles are time",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False)
    style(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 figure.")
    parser.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    parser.add_argument("--out-dir", default=str(config.FIGURES_DIR))
    args = parser.parse_args()

    reports, out_dir = Path(args.reports_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = reports / "graph_summary.json"
    path = out_dir / "graph_components.png"
    graph_components(load(source), path)
    print(f"wrote {path} from {source.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
