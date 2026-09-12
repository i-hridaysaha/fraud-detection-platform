"""Stage 8: single-row inference latency, with repeats, the spread, and the pinning comparison.

Single-row latency on a laptop moves by roughly a factor of two between invocations while the
ratio between models does not, so the number this script reports is never one point. The
parent process fits the two families stage 6 did not ship (random forest and logistic
regression, at the shipped configuration: module-default hyperparameters, no reweighting, the
186 column stack, seed 42, from the stage 6 cache) beside the shipped booster, draws the rows,
and then runs the timing loop in `--repeats` fresh interpreter processes. Every process times
every model on the same rows; the artifact carries each process's percentiles and the spread
across processes, and the ratios between models per process.

Like for like: every model is pinned the way the serving path pins the booster, one thread
(`serving.INFERENCE_THREADS`), and the comparison the pinning rests on is the same booster
pinned and unpinned on the same rows in the same process. The serving path's own steps
(payload to frame, the stage 3 pipeline, the schema, the matrix, the booster, the calibrator,
the factors, the store on both backends) are timed one by one, because the model call is a
small part of the request and the artifact should say so.

    .venv/bin/python scripts/serving_latency.py

Writes reports/latency.json. Needs reports/cache from `make train` and the registered bundle
from `make register`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import xgboost
from threadpoolctl import threadpool_limits

from fraud_platform import config, data_loader, modelling, prepare, schema, serving
from fraud_platform.online_store import MemoryBackend, OnlineFeatureStore, RedisBackend

CACHE_DIR = config.REPORTS_DIR / "cache"
MODELS_DIR = config.ROOT / "models"
OUT_PATH = config.REPORTS_DIR / "latency.json"

N_ROWS = 300
N_WARMUP = 20
N_REPEATS = 5
N_PATH_ROWS = 100
BATCH_ROWS = 1_000
TOP_FACTORS = 5
FAMILIES: tuple[str, ...] = ("xgboost", "random_forest", "logistic_regression")
REDIS_URL = os.environ.get("FRAUD_REDIS_URL", "redis://localhost:6379/0")


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype="float64")
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def timed(fn: Any, *args: Any, n_warmup: int = N_WARMUP) -> list[float]:
    """Milliseconds per call over the argument sets, after a warm-up on the first ones."""
    for i in range(min(n_warmup, len(args[0]))):
        fn(*(a[i] for a in args))
    out: list[float] = []
    for i in range(len(args[0])):
        started = time.perf_counter_ns()
        fn(*(a[i] for a in args))
        out.append((time.perf_counter_ns() - started) / 1e6)
    return out


# --- the parent: fit, draw, spawn ---------------------------------------------------------------------


def shipped_columns(cache: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    description = json.loads((MODELS_DIR / "shipped_model.json").read_text())
    stack = modelling.FeatureStack(
        causal=bool(description["stack"]["causal"]),
        graph=bool(description["stack"]["graph"]),
        native_groups=tuple(description["stack"]["native_groups"]),
    )
    columns = modelling.stack_columns(cache["prepared"], cache["causal"], cache["graph"], stack)
    if columns != list(description["columns"]):
        raise SystemExit("the cache's column list is not the shipped model's; run make train")
    return columns, description


def fit_families(work: Path) -> dict[str, Any]:
    """The shipped booster and the two refits, each pickled under `work`, plus the rows."""
    if not (CACHE_DIR / "columns.json").exists():
        raise SystemExit("reports/cache is missing; run make train first")
    cache = json.loads((CACHE_DIR / "columns.json").read_text())
    frames = {
        split: pd.read_parquet(CACHE_DIR / f"prepared_{split}.parquet")
        for split in ("train", "test")
    }
    later = [*cache["causal"], *cache["graph"]]
    cache["prepared"] = modelling.prepared_feature_columns(frames["train"].drop(columns=later))
    columns, description = shipped_columns(cache)
    y_train = frames["train"][config.TARGET].to_numpy(dtype="int64")
    rng = np.random.default_rng(config.SEED)
    positions = np.sort(rng.choice(len(frames["test"]), size=N_ROWS, replace=False))

    tree = {split: modelling.tree_matrix(frame, columns) for split, frame in frames.items()}
    x_tree = {split: matrix.to_numpy(dtype="float32") for split, matrix in tree.items()}
    print("fitting the complete-data path for the logistic regression")
    complete = modelling.fit_complete_data_path(frames["train"], columns)
    x_complete = {
        split: modelling.apply_complete_data_path(frame, complete)[0]
        for split, frame in frames.items()
    }

    models: dict[str, Any] = {}
    fit_seconds: dict[str, float] = {}
    shipped = xgboost.XGBClassifier()
    shipped.load_model(MODELS_DIR / description["model_file"])
    models["xgboost"] = shipped
    fit_seconds["xgboost"] = 0.0
    for family in ("random_forest", "logistic_regression"):
        print(f"fitting {family} at the shipped configuration")
        x_fit = (
            x_tree["train"] if modelling.PATH_BY_FAMILY[family] == "tree" else x_complete["train"]
        )
        model = modelling.make_model(family, "none", y_train, config.SEED)
        started = time.perf_counter()
        model.fit(x_fit, y_train)
        fit_seconds[family] = time.perf_counter() - started
        models[family] = model

    rows = {
        "tree": x_tree["test"][positions],
        "complete": np.asarray(x_complete["test"][positions], dtype="float64"),
        "batch_tree": x_tree["test"][:BATCH_ROWS],
    }
    np.savez(work / "rows.npz", **rows)
    for family, model in models.items():
        (work / f"{family}.pkl").write_bytes(pickle.dumps(model))
    return {
        "columns": columns,
        "n_columns": len(columns),
        "n_complete_columns": int(x_complete["test"].shape[1]),
        "stack": description["stack"],
        "hyperparameters": {
            family: (
                description["hyperparameters"]
                if family == "xgboost"
                else modelling.DEFAULT_HYPERPARAMETERS[family]
            )
            for family in FAMILIES
        },
        "fit_seconds": fit_seconds,
        "convergence": modelling.convergence(models["logistic_regression"]),
        "rows": {"n": int(positions.size), "split": "test", "positions": positions.tolist()},
        "n_test_rows": len(frames["test"]),
    }


def draw_payloads(positions: list[int], raw_columns: list[str]) -> list[dict[str, Any]]:
    """Raw test rows for the serving-path timing, as the service would receive them."""
    frame = data_loader.load_raw(columns=raw_columns)
    test = frame[frame[config.TIME_COLUMN] >= config.VAL_END_DT].reset_index(drop=True)
    payloads = []
    for position in positions[:N_PATH_ROWS]:
        record = test.iloc[int(position)]
        payloads.append(
            {
                str(k): (v.item() if isinstance(v, np.generic) else v)
                for k, v in record.items()
                if str(k) != config.TARGET and not pd.isna(v)
            }
        )
    return payloads


# --- the worker: time everything in a fresh process ------------------------------------------------------


def pin(model: Any, threads: int | None) -> None:
    """Pin a model the way the serving path pins the booster, or unpin it (None)."""
    if isinstance(model, xgboost.XGBClassifier):
        n = threads if threads is not None else 0
        model.set_params(n_jobs=threads if threads is not None else -1)
        model.get_booster().set_param({"nthread": n})
    elif hasattr(model, "n_jobs"):
        model.set_params(n_jobs=threads if threads is not None else -1)


def time_model(model: Any, rows: np.ndarray, threads: int | None) -> dict[str, Any]:
    pin(model, threads)
    single = [rows[i : i + 1] for i in range(len(rows))]
    with threadpool_limits(limits=threads):
        samples = timed(lambda x: model.predict_proba(x), single)
    return percentiles(samples)


def time_batch(model: Any, batch: np.ndarray, threads: int | None) -> dict[str, Any]:
    pin(model, threads)
    with threadpool_limits(limits=threads):
        model.predict_proba(batch)
        samples = []
        for _ in range(5):
            started = time.perf_counter_ns()
            model.predict_proba(batch)
            samples.append((time.perf_counter_ns() - started) / 1e6)
    return {"rows": len(batch), **percentiles(samples)}


def time_serving_path(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    bundle, about = serving.load_from_registry()
    raw = [bundle.raw_columns] * len(payloads)
    frames = [serving.raw_frame(p, bundle.raw_columns) for p in payloads]
    prepared = [prepare.apply_preparation(f, bundle.fitted, bundle.plan) for f in frames]
    matrices = [serving.matrix(bundle, p, None) for p in prepared]
    scores = [serving.score(bundle, x) for x in matrices]
    steps: dict[str, dict[str, Any]] = {
        "raw_frame": percentiles(timed(serving.raw_frame, payloads, raw)),
        "apply_preparation": percentiles(
            timed(lambda f: prepare.apply_preparation(f, bundle.fitted, bundle.plan), frames)
        ),
        "schema_validate": percentiles(
            timed(lambda p: schema.validate(p, bundle.schema), prepared)
        ),
        "matrix": percentiles(timed(lambda p: serving.matrix(bundle, p, None), prepared)),
        "predict": percentiles(timed(lambda x: serving.score(bundle, x), matrices)),
        "calibrate_and_band": percentiles(
            timed(lambda s: serving.band(bundle, serving.calibrate(bundle, s)), scores)
        ),
        f"explain_top_{TOP_FACTORS}": percentiles(
            timed(lambda x: serving.explain(bundle, x, TOP_FACTORS), matrices)
        ),
    }
    rows = [{str(k): v for k, v in f.iloc[0].to_dict().items()} for f in frames]
    stores: dict[str, dict[str, Any]] = {}
    backends: dict[str, Any] = {"memory": MemoryBackend()}
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL, socket_connect_timeout=0.2)
        client.ping()
        backends["redis"] = RedisBackend(client, prefix="latency-bench")
    except Exception:
        stores["redis"] = {"available": False, "url": REDIS_URL}
    for name, backend in backends.items():
        backend.flush()
        store = OnlineFeatureStore(backend, bundle.dist_buckets)
        gets: list[float] = []
        commits: list[float] = []
        for row in rows:
            started = time.perf_counter_ns()
            store.get_features(row)
            gets.append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            store.commit(row)
            commits.append((time.perf_counter_ns() - started) / 1e6)
        backend.flush()
        stores[name] = {
            "available": True,
            "get_features": percentiles(gets[N_WARMUP:]),
            "commit": percentiles(commits[N_WARMUP:]),
        }
    whole = percentiles(
        timed(lambda p: serving.score_transaction(bundle, p, None, TOP_FACTORS), payloads)
    )
    return {"model": about, "steps": steps, "store": stores, "whole_path_no_store": whole}


def worker(work: Path, payload_path: Path, out: Path) -> None:
    rows = np.load(work / "rows.npz")
    models = {family: pickle.loads((work / f"{family}.pkl").read_bytes()) for family in FAMILIES}
    threads = serving.INFERENCE_THREADS
    result: dict[str, Any] = {
        "pid": os.getpid(),
        "pinned": {
            family: time_model(
                model, rows["complete" if family == "logistic_regression" else "tree"], threads
            )
            for family, model in models.items()
        },
        "unpinned": {
            family: time_model(
                model, rows["complete" if family == "logistic_regression" else "tree"], None
            )
            for family, model in models.items()
            if family != "logistic_regression"
        },
        "batch": {
            "pinned": time_batch(models["xgboost"], rows["batch_tree"], threads),
            "unpinned": time_batch(models["xgboost"], rows["batch_tree"], None),
        },
    }
    payloads = json.loads(payload_path.read_text())
    result["serving_path"] = time_serving_path(payloads)
    out.write_text(json.dumps(result))


# --- aggregation -------------------------------------------------------------------------------------------


def spread(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype="float64")
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
        "max_over_min": float(array.max() / array.min()) if array.min() > 0 else float("nan"),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"n_repeats": len(runs)}
    out["pinned_p50_by_family"] = {
        family: spread([run["pinned"][family]["p50"] for run in runs]) for family in FAMILIES
    }
    out["pinned_p99_by_family"] = {
        family: spread([run["pinned"][family]["p99"] for run in runs]) for family in FAMILIES
    }
    out["ratio_p50_to_xgboost"] = {
        family: spread(
            [run["pinned"][family]["p50"] / run["pinned"]["xgboost"]["p50"] for run in runs]
        )
        for family in FAMILIES
        if family != "xgboost"
    }
    out["pinning"] = {
        family: {
            "pinned_p50": spread([run["pinned"][family]["p50"] for run in runs]),
            "unpinned_p50": spread([run["unpinned"][family]["p50"] for run in runs]),
            "unpinned_over_pinned_p50": spread(
                [run["unpinned"][family]["p50"] / run["pinned"][family]["p50"] for run in runs]
            ),
        }
        for family in ("xgboost", "random_forest")
    }
    out["batch_xgboost"] = {
        "rows": runs[0]["batch"]["pinned"]["rows"],
        "pinned_p50_ms": spread([run["batch"]["pinned"]["p50"] for run in runs]),
        "unpinned_p50_ms": spread([run["batch"]["unpinned"]["p50"] for run in runs]),
        "pinned_over_unpinned_p50": spread(
            [run["batch"]["pinned"]["p50"] / run["batch"]["unpinned"]["p50"] for run in runs]
        ),
    }
    steps = runs[0]["serving_path"]["steps"]
    out["serving_path_p50_ms"] = {
        name: spread([run["serving_path"]["steps"][name]["p50"] for run in runs]) for name in steps
    }
    out["serving_path_whole_no_store_p50_ms"] = spread(
        [run["serving_path"]["whole_path_no_store"]["p50"] for run in runs]
    )
    out["store_p50_ms"] = {}
    for name in runs[0]["serving_path"]["store"]:
        if not runs[0]["serving_path"]["store"][name].get("available"):
            out["store_p50_ms"][name] = {"available": False}
            continue
        out["store_p50_ms"][name] = {
            "get_features": spread(
                [run["serving_path"]["store"][name]["get_features"]["p50"] for run in runs]
            ),
            "commit": spread([run["serving_path"]["store"][name]["commit"]["p50"] for run in runs]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--worker", nargs=3, metavar=("WORK", "PAYLOADS", "OUT"))
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), Path(args.worker[1]), Path(args.worker[2]))
        return

    started = time.perf_counter()
    work = Path(tempfile.mkdtemp(prefix="latency-"))
    setup = fit_families(work)
    bundle, _ = serving.load_from_registry()
    payloads = draw_payloads(setup["rows"]["positions"], bundle.raw_columns)
    payload_path = work / "payloads.json"
    payload_path.write_text(json.dumps(payloads))

    runs: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        out = work / f"run-{repeat}.json"
        print(f"repeat {repeat + 1} of {args.repeats}: a fresh process")
        subprocess.run(
            [sys.executable, __file__, "--worker", str(work), str(payload_path), str(out)],
            check=True,
            cwd=config.ROOT,
        )
        runs.append(json.loads(out.read_text()))

    summary = aggregate(runs)
    report = {
        "section": "latency",
        "stage": 8,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make serving-latency  # or: .venv/bin/python scripts/serving_latency.py",
        "command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "seed": config.SEED,
        "method": {
            "unit": "milliseconds per single-row predict_proba call, one call per row",
            "rows": setup["rows"],
            "n_warmup_calls": N_WARMUP,
            "repeats": "each repeat is a fresh interpreter process timing every model on the same rows",
            "pinned": f"{serving.INFERENCE_THREADS} thread: xgboost nthread and n_jobs, sklearn n_jobs, "
            "and threadpoolctl over BLAS and OpenMP for every model",
            "unpinned": "the library default: xgboost nthread 0 (every core), sklearn n_jobs -1; the "
            "logistic regression has no thread parameter and is timed pinned only",
            "like_for_like": "every family at the shipped configuration on the shipped stack: "
            "module-default hyperparameters, no reweighting, seed 42, refitted from the stage 6 "
            "cache; the tree families on the tree path, the logistic regression on the "
            "complete-data path",
            "serving_path": f"the first {N_PATH_ROWS} of the same test rows as raw payloads, each "
            "step timed on its own, then the whole path without the store",
        },
        "models": {
            k: setup[k]
            for k in (
                "stack",
                "n_columns",
                "n_complete_columns",
                "hyperparameters",
                "fit_seconds",
                "convergence",
            )
        },
        "pinning": {
            "inference_threads": serving.INFERENCE_THREADS,
            "comparison": summary["pinning"],
            "batch": summary["batch_xgboost"],
        },
        "summary": {k: v for k, v in summary.items() if k not in ("pinning", "batch_xgboost")},
        "runs": runs,
        "seconds": time.perf_counter() - started,
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT_PATH.relative_to(config.ROOT)} in {report['seconds']:.0f} s")
    print(
        json.dumps(
            {
                "pinning": summary["pinning"],
                "families": summary["pinned_p50_by_family"],
                "ratios": summary["ratio_p50_to_xgboost"],
                "path": summary["serving_path_p50_ms"],
                "store": summary["store_p50_ms"],
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
