"""Stage 8: an end-to-end load test through the store-backed scoring endpoint.

One uvicorn worker, the Redis store, and the first rows of the test split replayed in
chronological order as HTTP requests, at each concurrency level in a sweep. The sweep is meant
to include the point where the one worker saturates: throughput stops growing and latency
starts growing with the number of clients instead. The artifact reports the point rather than
the best number.

**Ordering.** The store's definitions are "strictly before t" and it refuses an older row on a
key it has already committed (409). Concurrent clients replaying one stream would produce
exactly that whenever two in-flight rows share a key, so the dispatcher hands rows out in
stream order and holds the next row back while any in-flight row shares one of its keys
(entity, card, content). That is the per-key ordering a deployment would get from a partitioned
queue, done by the client here, and the artifact counts how often it had to wait.

**What this is.** A single-machine replay of a historical file against a laptop, with the
client and the server sharing the cores. The latency profile and the shape of the sweep are
real; the throughput number is a laptop number and says nothing about a machine it was not
measured on.

    .venv/bin/python loadtest/run_loadtest.py

Starts the server itself unless --url points at one. Writes loadtest/results.json and renders
loadtest/results.md from it.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import os
import platform
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd

from fraud_platform import config, data_loader, online_store, serving

ROOT = config.ROOT
OUT_JSON = ROOT / "loadtest" / "results.json"
OUT_MD = ROOT / "loadtest" / "results.md"

DEFAULT_ROWS = 2_000
DEFAULT_SWEEP: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
# One worker is the sweep the stage asks for; four shows what a process per core buys.
DEFAULT_WORKERS: tuple[int, ...] = (1, 4)
PORT = 8901
REDIS_URL = os.environ.get("FRAUD_REDIS_URL", "redis://localhost:6379/0")
REDIS_PREFIX = "loadtest"
# The gain in throughput over the previous level below which the worker counts as saturated.
SATURATION_GAIN = 0.10


# --- the rows ------------------------------------------------------------------------------------------


def draw_rows(n: int) -> list[dict[str, Any]]:
    """The first `n` test rows as payloads, each with the keys the store will touch."""
    bundle, _ = serving.load_from_registry()
    frame = data_loader.load_raw(columns=bundle.raw_columns)
    test = frame[frame[config.TIME_COLUMN] >= config.VAL_END_DT]
    test = test.sort_values(config.TIME_COLUMN, kind="stable").head(n)
    keyed = data_loader.add_entity_key(test)
    rows: list[dict[str, Any]] = []
    for _, record in keyed.iterrows():
        payload = {
            str(k): (v.item() if isinstance(v, np.generic) else v)
            for k, v in record.items()
            if str(k) in set(bundle.raw_columns) and str(k) != config.TARGET and not pd.isna(v)
        }
        named = online_store.keys_of({str(k): v for k, v in record.items()})
        keys = {(grain, str(named[grain])) for grain in ("entity", "card", "content")}
        rows.append({"payload": payload, "keys": keys})
    return rows


# --- the dispatcher: stream order, one row per key in flight -----------------------------------------------


class Dispatcher:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.queue = deque(rows)
        self.in_flight: set[tuple[str, str]] = set()
        self.lock = threading.Condition()
        self.stalls = 0
        self.stall_seconds = 0.0

    def take(self) -> dict[str, Any] | None:
        with self.lock:
            waited = False
            started = time.perf_counter()
            while True:
                if not self.queue:
                    return None
                head = self.queue[0]
                if not (head["keys"] & self.in_flight):
                    self.queue.popleft()
                    self.in_flight |= head["keys"]
                    if waited:
                        self.stalls += 1
                        self.stall_seconds += time.perf_counter() - started
                    return head
                waited = True
                self.lock.wait(timeout=1.0)

    def release(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.in_flight -= row["keys"]
            self.lock.notify_all()


# --- the client ----------------------------------------------------------------------------------------------


def post(conn: http.client.HTTPConnection, path: str, body: bytes) -> tuple[int, dict[str, Any]]:
    conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    raw = response.read()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"detail": raw.decode(errors="replace")}
    return response.status, parsed


def run_level(url: str, rows: list[dict[str, Any]], concurrency: int) -> dict[str, Any]:
    parsed = urlparse(url)
    dispatcher = Dispatcher(rows)
    samples: list[tuple[float, int, float | None]] = []
    samples_lock = threading.Lock()

    def worker() -> None:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
        while True:
            row = dispatcher.take()
            if row is None:
                break
            body = json.dumps({"transaction": row["payload"]}).encode()
            started = time.perf_counter()
            try:
                status, out = post(conn, "/score", body)
            except (http.client.HTTPException, OSError):
                status, out = 599, {}
                conn.close()
                conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
            elapsed = (time.perf_counter() - started) * 1000.0
            dispatcher.release(row)
            with samples_lock:
                samples.append((elapsed, status, out.get("latency_ms") if status == 200 else None))
        conn.close()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    wall_started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall = time.perf_counter() - wall_started

    client_ms = np.array([s[0] for s in samples], dtype="float64")
    ok = np.array([s[1] == 200 for s in samples])
    server_ms = np.array([s[2] for s in samples if s[2] is not None], dtype="float64")
    statuses: dict[str, int] = {}
    for _, status, _ in samples:
        statuses[str(status)] = statuses.get(str(status), 0) + 1
    return {
        "concurrency": concurrency,
        # Requests actually in flight on average: a client waiting on key ordering is not.
        "effective_concurrency": float(client_ms.sum() / 1000.0 / wall),
        "n_requests": len(samples),
        "n_ok": int(ok.sum()),
        "n_errors": int((~ok).sum()),
        "status_counts": statuses,
        "wall_seconds": wall,
        "throughput_rps": len(samples) / wall,
        "client_latency_ms": {
            "p50": float(np.percentile(client_ms, 50)),
            "p95": float(np.percentile(client_ms, 95)),
            "p99": float(np.percentile(client_ms, 99)),
            "mean": float(client_ms.mean()),
            "max": float(client_ms.max()),
        },
        "server_latency_ms": {
            "p50": float(np.percentile(server_ms, 50)) if server_ms.size else None,
            "p95": float(np.percentile(server_ms, 95)) if server_ms.size else None,
            "p99": float(np.percentile(server_ms, 99)) if server_ms.size else None,
        },
        "dispatch_stalls": dispatcher.stalls,
        "dispatch_stall_seconds": dispatcher.stall_seconds,
    }


# --- the server -------------------------------------------------------------------------------------------------


def start_server(port: int, log: Path, workers: int) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env["FRAUD_REDIS_URL"] = REDIS_URL
    env["FRAUD_REDIS_PREFIX"] = REDIS_PREFIX
    env.setdefault(serving.ENV_TRACKING_URI, serving.tracking_uri())
    process = subprocess.Popen(
        [
            str(ROOT / ".venv" / "bin" / "uvicorn"),
            "fraud_platform.service:app",
            "--port",
            str(port),
            "--workers",
            str(workers),
            # A client stalled on key ordering holds its connection idle for longer than the
            # default five seconds; a closed keep-alive would count as a client error here.
            "--timeout-keep-alive",
            "300",
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=log.open("wb"),
        stderr=subprocess.STDOUT,
        cwd=ROOT,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("localhost", port, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return process
        except OSError:
            time.sleep(0.5)
    process.kill()
    raise SystemExit(f"the server did not come up; see {log}")


def health(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    conn.request("GET", "/health")
    return dict(json.loads(conn.getresponse().read()))


def flush_store() -> None:
    import redis

    online_store.RedisBackend(redis.Redis.from_url(REDIS_URL), prefix=REDIS_PREFIX).flush()


# --- the report -------------------------------------------------------------------------------------------------


def saturation(levels: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """The first level whose throughput gain over the previous one falls under the threshold."""
    ordered = sorted(levels, key=lambda level: level["concurrency"])
    gains: list[dict[str, Any]] = []
    point: int | None = None
    for previous, current in itertools.pairwise(ordered):
        gain = current["throughput_rps"] / previous["throughput_rps"] - 1.0
        gains.append({"from": previous["concurrency"], "to": current["concurrency"], "gain": gain})
        if point is None and gain < SATURATION_GAIN:
            point = current["concurrency"]
    best = max(ordered, key=lambda level: level["throughput_rps"])
    return {
        "rule": f"the first level whose throughput is under {1 + SATURATION_GAIN:.2f} times "
        "the previous level's",
        "gains": gains,
        "saturated_at_concurrency": point,
        "peak_throughput_rps": best["throughput_rps"],
        "peak_at_concurrency": best["concurrency"],
        "workers": workers,
    }


def render(report: dict[str, Any]) -> str:
    one = report["sweeps"][0]
    lines = [
        "# Load test: the store-backed scoring endpoint",
        "",
        f"Rendered from `loadtest/results.json` by `{report['regenerate_with']}`. Not edited by "
        "hand; every number below is the artifact's.",
        "",
        "## Setup",
        "",
        f"- **Server:** uvicorn, `fraud_platform.service:app`, model "
        f"`{one['server']['model']['uri']}` version {one['server']['model']['version']}, "
        f"inference pinned to {one['server']['inference_threads']} thread, store "
        f"`{one['server']['store']}`. One sweep per worker count: "
        f"{', '.join(str(sweep['workers']) for sweep in report['sweeps'])}.",
        f"- **Client:** the first {report['rows']['n']} rows of the test split in chronological "
        "order, one HTTP connection per client, stream-order dispatch that holds a row back "
        "while an in-flight row shares its entity, card or content key. Effective concurrency "
        "is the mean number of requests actually in flight.",
        f"- **Machine:** {report['environment']['platform']}, {report['environment']['cpu_count']} "
        "cores, client and server on the same machine.",
        "- **Errors:** any non-200 response, counted per level; the status counts are in the JSON.",
        "",
    ]
    for sweep in report["sweeps"]:
        levels = sorted(sweep["levels"], key=lambda level: level["concurrency"])
        sat = sweep["saturation"]
        lines += [
            f"## {sweep['workers']} worker{'s' if sweep['workers'] != 1 else ''}",
            "",
            "| Clients | In flight | Requests | Errors | Throughput, req/s | Client p50, ms | "
            "Client p95, ms | Client p99, ms | Server p50, ms | Dispatch stalls |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for level in levels:
            client = level["client_latency_ms"]
            server = level["server_latency_ms"]
            lines.append(
                f"| {level['concurrency']} | {level['effective_concurrency']:.1f} | "
                f"{level['n_requests']} | {level['n_errors']} | {level['throughput_rps']:.1f} | "
                f"{client['p50']:.1f} | {client['p95']:.1f} | {client['p99']:.1f} | "
                f"{server['p50']:.1f} | {level['dispatch_stalls']} |"
            )
        lines += [
            "",
            f"- **Rule:** {sat['rule']}.",
            f"- **Peak throughput:** {sat['peak_throughput_rps']:.1f} req/s at "
            f"{sat['peak_at_concurrency']} clients.",
            f"- **Saturated at:** {sat['saturated_at_concurrency']} clients"
            + (
                ", the first level whose throughput gain over the previous level fell under the "
                "rule; from there latency grows with the number of clients and throughput does "
                "not."
                if sat["saturated_at_concurrency"] is not None
                else ": not reached inside the sweep."
            ),
            "- **Gains:** "
            + ", ".join(f"{g['from']} to {g['to']}: {100 * g['gain']:+.0f}%" for g in sat["gains"])
            + ".",
            "",
        ]
    lines += [
        "## What this does and does not show",
        "",
        "- One laptop, the client on the same cores as the server, a historical file replayed "
        "as fast as the server accepts it: the throughput is a number about this machine and "
        "this replay, not a capacity statement.",
        "- The latency profile per request (`reports/latency.json`) and the shape of each sweep "
        "(where a worker stops scaling, and what more workers buy) are what transfer; the "
        "absolute numbers do not.",
        "- The per-key ordering the store needs is supplied by the client's dispatcher here. A "
        "deployment gets it from a partitioned queue or pays for it with 409s; the stall counts "
        "are what that ordering cost this replay, and the in-flight column is the concurrency "
        "the server actually saw.",
        "",
    ]
    return "\n".join(lines)


# --- main ---------------------------------------------------------------------------------------------------------


def run_sweep(
    url: str | None,
    port: int,
    workers: int,
    rows: list[dict[str, Any]],
    sweep: list[int],
    log: Path,
) -> dict[str, Any]:
    process: subprocess.Popen[bytes] | None = None
    if url is None:
        process = start_server(port, log, workers)
        url = f"http://localhost:{port}"
    try:
        server = health(url)
        print(f"workers {workers}: {server}")
        levels = []
        for concurrency in sweep:
            flush_store()
            level = run_level(url, rows, concurrency)
            print(
                f"  concurrency {concurrency}: {level['throughput_rps']:.1f} req/s, p50 "
                f"{level['client_latency_ms']['p50']:.1f} ms, p99 "
                f"{level['client_latency_ms']['p99']:.1f} ms, errors {level['n_errors']}, "
                f"stalls {level['dispatch_stalls']}, effective concurrency "
                f"{level['effective_concurrency']:.1f}"
            )
            levels.append(level)
        flush_store()
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=30)
    return {
        "workers": workers,
        "server": {**server, "workers": workers, "redis_url": REDIS_URL},
        "levels": levels,
        "saturation": saturation(levels, workers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=None, help="a running server; default: start one")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--concurrency", type=int, nargs="+", default=list(DEFAULT_SWEEP))
    parser.add_argument("--workers", type=int, nargs="+", default=list(DEFAULT_WORKERS))
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    started = time.perf_counter()

    rows = draw_rows(args.rows)
    print(f"{len(rows)} rows drawn")
    log = ROOT / "loadtest" / "server.log"
    sweeps = [
        run_sweep(args.url, args.port, workers, rows, args.concurrency, log)
        for workers in (args.workers if args.url is None else [0])
    ]
    report = {
        "section": "loadtest",
        "stage": 8,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make loadtest  # or: .venv/bin/python loadtest/run_loadtest.py",
        "command": " ".join(sys.argv),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "rows": {
            "n": len(rows),
            "split": "test",
            "order": "chronological, from the split's first row",
        },
        "sweeps": sweeps,
        "seconds": time.perf_counter() - started,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    OUT_MD.write_text(render(report))
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")
    for sweep in sweeps:
        print(json.dumps({"workers": sweep["workers"], **sweep["saturation"]}, indent=1))


if __name__ == "__main__":
    main()
