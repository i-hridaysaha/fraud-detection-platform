"""Measure what the typing strategy costs and saves, and write reports/memory_profile.json.

Usage:
    make memory-profile
    .venv/bin/python scripts/memory_profile.py --out reports/memory_profile.json

Two numbers are reported per strategy and they are not the same thing.

`frame_bytes` is df.memory_usage(deep=True).sum(): the payload of the frame itself. It is
deterministic and is the number the typing decision is about.

`peak_rss_bytes` is the high-water resident set of the process that produced the frame. It
includes the parser's own buffers, which for a 683 MB CSV are large, so it is always well above
frame_bytes and it moves less than frame_bytes does. Peak RSS is a high-water mark for the whole
process, so each strategy runs in its own subprocess; measuring both in one process would report
the larger of the two twice.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, data_loader

STRATEGIES = ("naive", "typed", "typed_pyarrow_engine", "typed_chunked")


def peak_rss_bytes() -> int:
    """Peak resident set of this process.

    ru_maxrss is bytes on macOS and kilobytes on Linux. getrusage does not say which, so the
    platform decides. On macOS the kernel compresses inactive pages, so this can read below the
    logical size of the objects held; it is a floor on what the process needed, not a ceiling.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def run_strategy(strategy: str) -> dict[str, Any]:
    """Load the joined frame one way and report what it cost. Runs as a subprocess child."""
    started = time.perf_counter()

    if strategy == "naive":
        transactions = pd.read_csv(config.TRANSACTIONS_PATH)
        identity = pd.read_csv(config.IDENTITY_PATH)
        frame = transactions.merge(identity, on=config.ID_COLUMN, how="left")
        del transactions, identity
    elif strategy == "typed":
        frame = data_loader.load_raw()
    elif strategy == "typed_pyarrow_engine":
        frame = _read_with_engine("pyarrow")
    elif strategy == "typed_chunked":
        frame = _read_chunked()
    else:
        raise ValueError(f"unknown strategy {strategy}")

    dtypes = frame.dtypes
    return {
        "strategy": strategy,
        "n_rows": len(frame),
        "n_columns": int(frame.shape[1]),
        "frame_bytes": data_loader.frame_bytes(frame),
        "peak_rss_bytes": peak_rss_bytes(),
        "seconds": round(time.perf_counter() - started, 3),
        "n_category_columns": int(sum(isinstance(dtype, pd.CategoricalDtype) for dtype in dtypes)),
        "dtype_counts": _dtype_counts(dtypes),
    }


def _dtype_counts(dtypes: pd.Series) -> dict[str, int]:
    """Column count per dtype, with every categorical column counted under one key.

    dtypes.value_counts() treats two categoricals with different category sets as different
    dtypes, which would give 31 one-column entries instead of one entry of 31.
    """
    counts: dict[str, int] = {}
    for dtype in dtypes:
        name = "category" if isinstance(dtype, pd.CategoricalDtype) else str(dtype)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _read_with_engine(engine: str) -> pd.DataFrame:
    """The same dtype map, read through a different parser."""
    frames = []
    for path in (config.TRANSACTIONS_PATH, config.IDENTITY_PATH):
        columns = list(pd.read_csv(path, nrows=0).columns)
        frames.append(pd.read_csv(path, dtype=data_loader.build_dtype_map(columns), engine=engine))
    return frames[0].merge(frames[1], on=config.ID_COLUMN, how="left")


def _read_chunked(chunk_rows: int = 100_000) -> pd.DataFrame:
    """The same dtype map, read in chunks and concatenated."""
    frames = []
    for path in (config.TRANSACTIONS_PATH, config.IDENTITY_PATH):
        columns = list(pd.read_csv(path, nrows=0).columns)
        chunks = list(
            pd.read_csv(path, dtype=data_loader.build_dtype_map(columns), chunksize=chunk_rows)
        )
        frames.append(pd.concat(chunks, ignore_index=True))
        del chunks
    return frames[0].merge(frames[1], on=config.ID_COLUMN, how="left")


def float32_roundtrip_error() -> dict[str, Any]:
    """How far float32 moves each numeric column, measured on the real values.

    This is the evidence for the one exception in the typing spec. Each file is read once at
    float64, which is the only way to see what float32 costs: comparing a float32 column against
    itself cast back up would report zero for everything. Reported as the largest absolute and
    relative move over the non-null values, with the amount column separated out because it is
    the only one denominated in money.
    """
    worst_column = ""
    worst_absolute = 0.0
    worst_relative = 0.0
    amount: dict[str, float] = {}
    n_tested = 0

    for path in (config.TRANSACTIONS_PATH, config.IDENTITY_PATH):
        columns = [
            column
            for column in pd.read_csv(path, nrows=0).columns
            if column not in config.CATEGORICAL_COLUMNS
            and column not in config.INTEGER_COLUMN_DTYPES
        ]
        frame = pd.read_csv(path, usecols=columns, dtype="float64")
        for column in columns:
            values = frame[column].dropna().to_numpy(dtype="float64")
            if values.size == 0:
                continue
            n_tested += 1
            moved = np.abs(values.astype(np.float32).astype(np.float64) - values)
            scale = np.where(np.abs(values) > 0, np.abs(values), 1.0)
            absolute = float(moved.max())
            relative = float((moved / scale).max())
            if column == config.AMOUNT_COLUMN:
                amount = {
                    "max_absolute_move": absolute,
                    "max_relative_move": relative,
                    "max_absolute_value": float(np.abs(values).max()),
                }
                continue
            if absolute > worst_absolute:
                worst_absolute, worst_column = absolute, column
            worst_relative = max(worst_relative, relative)
        del frame

    return {
        "n_numeric_columns_tested": n_tested,
        "amount_column": config.AMOUNT_COLUMN,
        "amount": amount,
        "worst_other_column": worst_column,
        "max_absolute_move_excluding_amount": worst_absolute,
        "max_relative_move_excluding_amount": worst_relative,
        "command": (
            "v = pd.read_csv(path, usecols=[c], dtype='float64')[c].dropna().to_numpy(); "
            "abs(v.astype('float32').astype('float64') - v).max()"
        ),
    }


def integer_column_ranges() -> dict[str, Any]:
    """The measured range and null count behind every integer dtype in the map.

    The dtype map claims 18 columns hold integers with no nulls. This is the measurement that
    claim rests on, so the ADR can cite a file rather than a memory.
    """
    columns = list(config.INTEGER_COLUMN_DTYPES)
    frame = data_loader.load_raw(columns=columns)
    return {
        column: {
            "dtype": config.INTEGER_COLUMN_DTYPES[column],
            "min": int(frame[column].min()),
            "max": int(frame[column].max()),
            "n_null": int(frame[column].isna().sum()),
        }
        for column in columns
    }


def numeric_column_shape() -> dict[str, Any]:
    """How many numeric columns hold whole numbers and how many hold fractions.

    The split matters because the two halves are lossless in float32 for different reasons: whole
    numbers because they are small enough to be exact, fractions because the values in this file
    happen to round-trip. Only the fractional half can move at all.
    """
    integral = 0
    fractional = 0
    for path in (config.TRANSACTIONS_PATH, config.IDENTITY_PATH):
        columns = [
            column
            for column in pd.read_csv(path, nrows=0).columns
            if column not in config.CATEGORICAL_COLUMNS
            and column not in config.INTEGER_COLUMN_DTYPES
        ]
        frame = pd.read_csv(path, usecols=columns, dtype="float64")
        for column in columns:
            values = frame[column].dropna().to_numpy(dtype="float64")
            if values.size == 0:
                continue
            if bool(np.all(np.floor(values) == values)):
                integral += 1
            else:
                fractional += 1
        del frame
    return {
        "n_whole_number_columns": integral,
        "n_fractional_columns": fractional,
        "command": "v = frame[c].dropna().to_numpy('float64'); (np.floor(v) == v).all()",
    }


def categorical_cardinality() -> dict[str, int]:
    """Distinct values per string column, the ground for making all of them categorical."""
    frame = data_loader.load_raw(columns=list(config.CATEGORICAL_COLUMNS))
    return {
        column: int(frame[column].nunique(dropna=True)) for column in config.CATEGORICAL_COLUMNS
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse repeated runs of one strategy.

    frame_bytes is deterministic and must agree across runs, so it is reported once and a
    disagreement raises. peak_rss_bytes is not deterministic on this platform, so it is reported
    as the spread rather than as a value.
    """
    frame_bytes = {run["frame_bytes"] for run in runs}
    if len(frame_bytes) != 1:
        raise RuntimeError(f"frame_bytes differed across runs: {sorted(frame_bytes)}")

    peaks = sorted(run["peak_rss_bytes"] for run in runs)
    seconds = sorted(run["seconds"] for run in runs)
    first = runs[0]
    return {
        "strategy": first["strategy"],
        "n_runs": len(runs),
        "n_rows": first["n_rows"],
        "n_columns": first["n_columns"],
        "frame_bytes": first["frame_bytes"],
        "n_category_columns": first["n_category_columns"],
        "dtype_counts": first["dtype_counts"],
        "peak_rss_bytes": {
            "min": peaks[0],
            "median": peaks[len(peaks) // 2],
            "max": peaks[-1],
            "spread": peaks[-1] - peaks[0],
            "spread_over_median": (peaks[-1] - peaks[0]) / peaks[len(peaks) // 2],
        },
        "seconds": {"min": seconds[0], "median": seconds[len(seconds) // 2], "max": seconds[-1]},
    }


def peak_rss_comparisons(measurements: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Which strategies can be ordered on peak resident set, and which cannot.

    A median is not a comparison when the runs behind it spread further than the gap between two
    strategies. So each pair is reported as disjoint or overlapping, and only a disjoint pair is
    given a direction. An overlapping pair gets `higher: null`, which is the honest answer.
    """
    names = list(measurements)
    out = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a = measurements[left]["peak_rss_bytes"]
            b = measurements[right]["peak_rss_bytes"]
            disjoint = a["min"] > b["max"] or b["min"] > a["max"]
            out[f"{left}_vs_{right}"] = {
                "ranges_disjoint": disjoint,
                "higher": (left if a["min"] > b["max"] else right) if disjoint else None,
                "median_ratio": a["median"] / b["median"],
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(config.MEMORY_PROFILE_PATH))
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="runs per strategy; peak RSS varies run to run so the artifact carries the spread",
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        help="internal: run one strategy and print its result as JSON, for the parent to collect",
    )
    args = parser.parse_args()

    if args.strategy:
        print(json.dumps(run_strategy(args.strategy)))
        return

    measurements = {}
    for strategy in STRATEGIES:
        runs = []
        for _ in range(args.repeats):
            completed = subprocess.run(
                [sys.executable, __file__, "--strategy", strategy],
                capture_output=True,
                text=True,
                check=True,
            )
            runs.append(json.loads(completed.stdout))
        measurements[strategy] = summarise(runs)

    naive = measurements["naive"]
    typed = measurements["typed"]

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make memory-profile",
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "note": (
            "frame_bytes is the frame payload from memory_usage(deep=True) and is deterministic. "
            "peak_rss_bytes is the high-water resident set of the process that built the frame, "
            "parser buffers included. Each run is its own subprocess, because peak RSS is a "
            "high-water mark for a whole process. On darwin the kernel compresses inactive pages, "
            "so peak RSS varies run to run on identical work; the spread across runs is reported "
            "for that reason and the typing decision rests on frame_bytes."
        ),
        "strategies": measurements,
        "peak_rss_comparisons": peak_rss_comparisons(measurements),
        "typed_against_naive": {
            "frame_bytes_saved": naive["frame_bytes"] - typed["frame_bytes"],
            "frame_bytes_ratio": typed["frame_bytes"] / naive["frame_bytes"],
            "peak_rss_median_ratio": (
                typed["peak_rss_bytes"]["median"] / naive["peak_rss_bytes"]["median"]
            ),
            "peak_rss_ranges_overlap": (
                typed["peak_rss_bytes"]["min"] <= naive["peak_rss_bytes"]["max"]
                and naive["peak_rss_bytes"]["min"] <= typed["peak_rss_bytes"]["max"]
            ),
        },
        "float32_roundtrip": float32_roundtrip_error(),
        "categorical_cardinality": categorical_cardinality(),
        "integer_column_ranges": integer_column_ranges(),
        "numeric_column_shape": numeric_column_shape(),
        "n_rows": typed["n_rows"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
