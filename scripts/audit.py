"""Run the stage 0 audit and write reports/audit.json.

Usage:
    make audit
    .venv/bin/python scripts/audit.py --transactions data/train_transaction.csv \
        --identity data/train_identity.csv --out reports/audit.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import audit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", default="data/train_transaction.csv")
    parser.add_argument("--identity", default="data/train_identity.csv")
    parser.add_argument("--out", default="reports/audit.json")
    args = parser.parse_args()

    tx_path = Path(args.transactions)
    id_path = Path(args.identity)

    df = audit.load_transactions(str(tx_path))
    df = audit.add_card_start_day(df)
    identity_ids = pd.read_csv(id_path, usecols=["TransactionID"])["TransactionID"]
    n_identity_columns = audit.count_columns(str(id_path))

    split = audit.chronological_split(df)

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "regenerate_with": "make audit",
        "inputs": {
            "transactions": {
                "path": str(tx_path),
                "bytes": tx_path.stat().st_size,
                "sha256": sha256(tx_path),
            },
            "identity": {
                "path": str(id_path),
                "bytes": id_path.stat().st_size,
                "sha256": sha256(id_path),
                "n_columns": n_identity_columns,
            },
            "note": (
                "Kaggle competition ieee-fraud-detection, training files only. The test files "
                "are unlabelled and are not downloaded. The data is not committed."
            ),
        },
        "shape": audit.basic_shape(df, audit.count_columns(str(tx_path))),
        "transaction_dt": audit.transaction_dt_profile(df),
        "label_granularity_card1": audit.label_granularity(df, ["card1"]),
        "card1_profile": audit.card1_profile(df),
        "d1_profile": audit.d1_profile(df),
        "card_start_day_check": {
            "claim": (
                "D1 is days since the card began, so floor(TransactionDT / 86400 - D1) is near "
                "constant per card and can act as part of an entity key."
            ),
            "by_card1": audit.verify_card_start_day(df, ["card1"]),
            "by_card1_addr1": audit.verify_card_start_day(df, ["card1", "addr1"]),
        },
        "entity_key_candidates": {
            name: audit.entity_key_stats(df, name, key) for name, key in audit.ENTITY_KEYS.items()
        },
        "label_granularity_by_candidate": {
            name: audit.label_granularity(df, key) for name, key in audit.ENTITY_KEYS.items()
        },
        "identity_join": audit.identity_join_coverage(df, identity_ids),
        "chronological_split": split,
        "entity_overlap": {
            name: audit.entity_overlap(df, split, key) for name, key in audit.ENTITY_KEYS.items()
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False so an empty slice that became NaN fails here rather than writing
    # invalid JSON that a downstream reader silently accepts.
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
