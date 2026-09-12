"""Stage 8: the parity measurement on the real stream, as an artifact.

`tests/test_parity.py` asserts that the batch pipeline and the online store agree on every
feature of every row, on a generated stream and on a short prefix of the file. This script runs
the same comparison over a longer prefix and writes what it found: how many rows, how many
features, the largest absolute difference per feature and overall, the tie counts per grain
that make the exclusion rule matter, and the seconds per row on each backend. The numbers the
stage quotes about parity come from here.

    .venv/bin/python scripts/parity.py --days 30

Writes reports/parity.json. Uses the in-process backend always and Redis when a server answers
at FRAUD_REDIS_URL (default redis://localhost:6379/0).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, data_loader, encoders, features, graph_features, online_store

OUT_PATH = config.REPORTS_DIR / "parity.json"
REDIS_URL = os.environ.get("FRAUD_REDIS_URL", "redis://localhost:6379/0")
DAY = config.SECONDS_PER_DAY

STREAM_COLUMNS: tuple[str, ...] = (
    config.ID_COLUMN,
    config.TIME_COLUMN,
    config.AMOUNT_COLUMN,
    "card1",
    "card2",
    "addr1",
    "addr2",
    "dist1",
    "D1",
    "ProductCD",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
)

# The parity test's tolerance, restated here so the artifact records the rule it was held to.
FLOAT_TOLERANCE = 1e-9
TOLERANT_COLUMNS: frozenset[str] = frozenset(
    {
        "amt_entity_mean",
        "amt_entity_std",
        "amt_deviation_from_entity_mean",
        "amt_zscore_within_entity",
    }
)
TOLERANT_PREFIXES: tuple[str, ...] = ("vel_amount_sum_", "vel_amount_mean_")


def load_stream(n_days: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = data_loader.add_entity_key(data_loader.load_raw(columns=list(STREAM_COLUMNS)))
    frame = encoders.add_normalised_free_text(frame)
    buckets = features.fit_dist_buckets(frame[data_loader.split_masks(frame)["train"]])
    frame = frame[frame[config.TIME_COLUMN] < (1 + n_days) * DAY].reset_index(drop=True)
    return frame, buckets


def ties(frame: pd.DataFrame) -> dict[str, int]:
    """Rows sharing a key and a timestamp with an earlier row, per grain."""
    content = pd.Series(
        [online_store.content_key(t) for t in online_store.transactions(frame)],
        index=frame.index[online_store.stream_order(frame)],
    ).reindex(frame.index)
    keys = {
        "entity": frame[config.ENTITY_ID_COLUMN],
        "card": frame["card1"],
        "content": content,
        "addr1_node": frame["addr1"],
        "device_node": frame[graph_features.DEVICE_COLUMN],
    }
    out = {}
    for grain, key in keys.items():
        pair = pd.DataFrame({"k": key, "t": frame[config.TIME_COLUMN]})
        out[grain] = int(pair.duplicated().sum())
    return out


def compare(batch: pd.DataFrame, online: pd.DataFrame) -> dict[str, Any]:
    per_feature: dict[str, dict[str, Any]] = {}
    worst = 0.0
    n_off = 0
    for name in batch.columns:
        expected = batch[name].to_numpy(dtype="float64")
        got = online[name].to_numpy(dtype="float64")
        null_mismatch = int((np.isnan(expected) != np.isnan(got)).sum())
        present = ~np.isnan(expected)
        gap = np.abs(expected[present] - got[present]) if present.any() else np.zeros(0)
        largest = float(gap.max()) if gap.size else 0.0
        tolerant = name in TOLERANT_COLUMNS or name.startswith(TOLERANT_PREFIXES)
        if tolerant:
            allowed = FLOAT_TOLERANCE * np.maximum(np.abs(expected[present]), 1.0)
            off = int((gap > allowed).sum())
        else:
            off = int((gap != 0.0).sum())
        per_feature[name] = {
            "max_abs_difference": largest,
            "null_mask_mismatches": null_mismatch,
            "rows_outside_tolerance": off,
            "tolerance": f"relative {FLOAT_TOLERANCE}" if tolerant else "exact",
            "n_present": int(present.sum()),
        }
        worst = max(worst, largest)
        n_off += off + null_mismatch
    return {
        "n_features": len(per_feature),
        "max_abs_difference_over_all_features": worst,
        "rows_outside_tolerance_over_all_features": n_off,
        "bit_identical": worst == 0.0 and n_off == 0,
        "per_feature": per_feature,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    started = time.perf_counter()

    frame, buckets = load_stream(args.days)
    print(f"{len(frame):,} rows in the first {args.days} days")
    t0 = time.perf_counter()
    batch = pd.concat(
        [
            features.build_features(frame, buckets),
            graph_features.build_graph_features(frame, None, online_store.GRAPH_ON),
        ],
        axis=1,
    )
    batch_seconds = time.perf_counter() - t0

    backends: dict[str, online_store.StateBackend] = {"memory": online_store.MemoryBackend()}
    redis_note = None
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.2)
        client.ping()
        backends["redis"] = online_store.RedisBackend(client, prefix="parity-script")
    except Exception as error:
        redis_note = f"no Redis server at {REDIS_URL}: {error}"

    results: dict[str, Any] = {}
    for name, backend in backends.items():
        print(f"replaying through the {name} backend")
        backend.flush()
        store = online_store.OnlineFeatureStore(backend, buckets)
        t0 = time.perf_counter()
        online = online_store.replay(store, frame)
        seconds = time.perf_counter() - t0
        results[name] = {
            **compare(batch, online),
            "seconds": seconds,
            "milliseconds_per_row": 1000.0 * seconds / len(frame),
            "n_keys_held": backend.n_keys
            if isinstance(backend, online_store.MemoryBackend)
            else None,
        }
        backend.flush()
        print(
            f"  {results[name]['n_features']} features, max abs difference "
            f"{results[name]['max_abs_difference_over_all_features']}, "
            f"{results[name]['milliseconds_per_row']:.2f} ms per row"
        )

    report = {
        "section": "parity",
        "stage": 8,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make parity  # or: .venv/bin/python scripts/parity.py --days 30",
        "command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
        "stream": {
            "n_days": args.days,
            "n_rows": len(frame),
            "first_dt": int(frame[config.TIME_COLUMN].min()),
            "last_dt": int(frame[config.TIME_COLUMN].max()),
            "columns": list(STREAM_COLUMNS),
            "rule": "a prefix of the file, so every key's history is complete from its first row",
            "ties": ties(frame),
            "n_keys": {
                "entity": int(frame[config.ENTITY_ID_COLUMN].nunique()),
                "card": int(frame["card1"].nunique()),
            },
        },
        "batch": {
            "seconds": batch_seconds,
            "n_features": int(batch.shape[1]),
            "stage_4": features.feature_names(),
            "stage_5": list(graph_features.graph_feature_names(online_store.GRAPH_ON)),
        },
        "tolerance": {
            "exact": "every count, flag, timestamp difference and bucket, null mask included",
            "relative": FLOAT_TOLERANCE,
            "relative_columns": sorted(TOLERANT_COLUMNS) + [f"{p}*" for p in TOLERANT_PREFIXES],
        },
        "backends": results,
        "redis": {"url": REDIS_URL, "note": redis_note},
        "seconds": time.perf_counter() - started,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT_PATH.relative_to(config.ROOT)} in {report['seconds']:.0f} s")


if __name__ == "__main__":
    main()
