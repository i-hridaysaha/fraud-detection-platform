"""Describe the chronological split and write reports/split_summary.json.

Usage:
    make split-summary
    .venv/bin/python scripts/split_summary.py --out reports/split_summary.json

Rows, time span, fraud rate and entity overlap per split. The split itself comes from
fraud_platform.data_loader.time_based_split, so this script measures the code that later stages
use rather than a second implementation that could drift from it.

Only the columns the summary needs are read: the entity key sources plus the target. The full
434 column frame is loaded and profiled by scripts/memory_profile.py, not here.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, data_loader

SUMMARY_COLUMNS = [
    config.ID_COLUMN,
    config.TARGET,
    config.TIME_COLUMN,
    *config.ENTITY_KEY_SOURCE_COLUMNS,
]


def describe_split(name: str, part: pd.DataFrame, n_total_rows: int) -> dict[str, Any]:
    """Rows, span and fraud rate for one split."""
    dt = part[config.TIME_COLUMN]
    span_seconds = int(dt.max() - dt.min())
    return {
        "name": name,
        "n_rows": len(part),
        "row_fraction": len(part) / n_total_rows,
        "dt_min": int(dt.min()),
        "dt_max": int(dt.max()),
        "span_seconds": span_seconds,
        "span_days": span_seconds / config.SECONDS_PER_DAY,
        "day_index_min": int(np.floor(int(dt.min()) / config.SECONDS_PER_DAY)),
        "day_index_max": int(np.floor(int(dt.max()) / config.SECONDS_PER_DAY)),
        "n_fraud": int(part[config.TARGET].sum()),
        "fraud_rate": float(part[config.TARGET].mean()),
        "n_entities": int(part[config.ENTITY_ID_COLUMN].nunique()),
    }


def describe_overlap(part: pd.DataFrame, train_entities: set[str]) -> dict[str, Any]:
    """How much of a later split sits on entities the training window never saw.

    The share of rows matters more than the share of entities: an entity-history feature is
    missing for exactly the rows counted here, and stage 0 measured that those rows carry a
    higher fraud rate, so they are not a tail that can be ignored.
    """
    entities = part[config.ENTITY_ID_COLUMN]
    unseen_rows = ~entities.isin(train_entities)
    unique_entities = set(entities)
    unseen_entities = unique_entities - train_entities
    seen_mask = ~unseen_rows

    return {
        "n_entities": len(unique_entities),
        "n_entities_unseen_in_train": len(unseen_entities),
        "share_entities_unseen_in_train": len(unseen_entities) / len(unique_entities),
        "n_rows_on_unseen_entities": int(unseen_rows.sum()),
        "share_rows_on_unseen_entities": float(unseen_rows.mean()),
        "fraud_rate_on_unseen_entity_rows": float(part.loc[unseen_rows, config.TARGET].mean()),
        "fraud_rate_on_seen_entity_rows": float(part.loc[seen_mask, config.TARGET].mean()),
    }


def build_report(frame: pd.DataFrame) -> dict[str, Any]:
    """Everything the artifact holds, from one already-keyed frame."""
    train, val, test = data_loader.time_based_split(frame)
    n_total = len(frame)

    splits = {
        "train": describe_split("train", train, n_total),
        "val": describe_split("val", val, n_total),
        "test": describe_split("test", test, n_total),
    }

    train_entities = set(train[config.ENTITY_ID_COLUMN])
    overlap = {
        "key": list(config.ENTITY_KEY_COLUMNS),
        "n_train_entities": len(train_entities),
        "val": describe_overlap(val, train_entities),
        "test": describe_overlap(test, train_entities),
    }

    fraud_rates = [splits[name]["fraud_rate"] for name in config.SPLIT_NAMES]

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make split-summary",
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "seed": config.SEED,
        "boundaries": {
            "train_end_exclusive_dt": config.TRAIN_END_DT,
            "val_end_exclusive_dt": config.VAL_END_DT,
            "rule": "train: dt < train_end; val: train_end <= dt < val_end; test: dt >= val_end",
            "source": "ADR 0002, quantiles 0.70 and 0.85 of TransactionDT measured in stage 0",
        },
        "totals": {
            "n_rows": n_total,
            "n_fraud": int(frame[config.TARGET].sum()),
            "fraud_rate": float(frame[config.TARGET].mean()),
            "dt_min": int(frame[config.TIME_COLUMN].min()),
            "dt_max": int(frame[config.TIME_COLUMN].max()),
        },
        "splits": splits,
        "entity_overlap": overlap,
        "checks": {
            "rows_sum_to_total": sum(splits[n]["n_rows"] for n in config.SPLIT_NAMES) == n_total,
            "train_max_dt_below_val_min_dt": splits["train"]["dt_max"] < splits["val"]["dt_min"],
            "val_max_dt_below_test_min_dt": splits["val"]["dt_max"] < splits["test"]["dt_min"],
            "max_fraud_rate_gap": max(fraud_rates) - min(fraud_rates),
        },
        "command": (
            "frame = data_loader.add_entity_key(data_loader.load_raw(columns=SUMMARY_COLUMNS)); "
            "train, val, test = data_loader.time_based_split(frame)"
        ),
    }


def print_report(report: dict[str, Any]) -> None:
    """A table a reader can check against the JSON without opening it."""
    print(f"rows {report['totals']['n_rows']:,}  fraud rate {report['totals']['fraud_rate']:.5f}")
    print(
        f"boundaries: train < {report['boundaries']['train_end_exclusive_dt']:,} "
        f"<= val < {report['boundaries']['val_end_exclusive_dt']:,} <= test"
    )
    print()
    header = (
        f"{'split':<6}{'rows':>10}{'share':>8}{'days':>9}{'fraud':>8}{'rate':>10}{'entities':>10}"
    )
    print(header)
    print("-" * len(header))
    for name in config.SPLIT_NAMES:
        s = report["splits"][name]
        print(
            f"{name:<6}{s['n_rows']:>10,}{s['row_fraction']:>8.4f}{s['span_days']:>9.3f}"
            f"{s['n_fraud']:>8,}{s['fraud_rate']:>10.5f}{s['n_entities']:>10,}"
        )
    print()
    overlap = report["entity_overlap"]
    print(f"entity key: {' + '.join(overlap['key'])}, {overlap['n_train_entities']:,} in train")
    for name in ("val", "test"):
        o = overlap[name]
        print(
            f"  {name:<5} {o['share_entities_unseen_in_train']:.4f} of entities unseen, "
            f"{o['share_rows_on_unseen_entities']:.4f} of rows, "
            f"fraud rate {o['fraud_rate_on_unseen_entity_rows']:.4f} unseen "
            f"against {o['fraud_rate_on_seen_entity_rows']:.4f} seen"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    parser.add_argument("--out", default=str(config.SPLIT_SUMMARY_PATH))
    args = parser.parse_args()

    frame = data_loader.load_raw(
        transactions_path=args.transactions,
        identity_path=args.identity,
        columns=SUMMARY_COLUMNS,
    )
    report = build_report(data_loader.add_entity_key(frame))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False so an empty slice that became NaN fails here rather than writing invalid
    # JSON that a downstream reader silently accepts.
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print_report(report)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
