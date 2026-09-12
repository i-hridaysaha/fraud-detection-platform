# Load test: the store-backed scoring endpoint

Rendered from `loadtest/results.json` by `make loadtest  # or: .venv/bin/python loadtest/run_loadtest.py`. Not edited by hand; every number below is the artifact's.

## Setup

- **Server:** uvicorn, `fraud_platform.service:app`, model `models:/fraud-detector@production` version 1, inference pinned to 1 thread, store `redis (redis://localhost:6379/0)`. One sweep per worker count: 1, 4.
- **Client:** the first 2000 rows of the test split in chronological order, one HTTP connection per client, stream-order dispatch that holds a row back while an in-flight row shares its entity, card or content key. Effective concurrency is the mean number of requests actually in flight.
- **Machine:** macOS-27.0-arm64-arm-64bit, 10 cores, client and server on the same machine.
- **Errors:** any non-200 response, counted per level; the status counts are in the JSON.

## 1 worker

| Clients | In flight | Requests | Errors | Throughput, req/s | Client p50, ms | Client p95, ms | Client p99, ms | Server p50, ms | Dispatch stalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 2000 | 0 | 25.2 | 38.1 | 40.4 | 84.2 | 37.4 | 0 |
| 2 | 2.0 | 2000 | 0 | 22.6 | 83.9 | 129.9 | 133.5 | 82.5 | 43 |
| 4 | 3.8 | 2000 | 0 | 21.6 | 174.7 | 232.6 | 261.7 | 170.1 | 233 |
| 8 | 6.9 | 2000 | 0 | 20.3 | 365.4 | 457.4 | 526.4 | 359.1 | 755 |
| 16 | 10.1 | 2000 | 0 | 20.2 | 495.2 | 846.7 | 924.4 | 485.5 | 1439 |
| 32 | 11.1 | 2000 | 0 | 20.1 | 508.5 | 1108.1 | 1322.9 | 500.3 | 1636 |

- **Rule:** the first level whose throughput is under 1.10 times the previous level's.
- **Peak throughput:** 25.2 req/s at 1 clients.
- **Saturated at:** 2 clients, the first level whose throughput gain over the previous level fell under the rule; from there latency grows with the number of clients and throughput does not.
- **Gains:** 1 to 2: -10%, 2 to 4: -5%, 4 to 8: -6%, 8 to 16: -0%, 16 to 32: -0%.

## 4 workers

| Clients | In flight | Requests | Errors | Throughput, req/s | Client p50, ms | Client p95, ms | Client p99, ms | Server p50, ms | Dispatch stalls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.0 | 2000 | 0 | 24.8 | 38.4 | 41.0 | 85.6 | 37.7 | 0 |
| 2 | 2.0 | 2000 | 0 | 46.6 | 40.0 | 43.1 | 89.8 | 39.3 | 47 |
| 4 | 3.7 | 2000 | 0 | 59.7 | 44.7 | 99.9 | 144.0 | 43.9 | 271 |
| 8 | 6.0 | 2000 | 0 | 62.1 | 97.5 | 173.2 | 217.6 | 95.2 | 789 |
| 16 | 8.6 | 2000 | 0 | 52.1 | 151.7 | 333.5 | 393.4 | 148.3 | 1442 |
| 32 | 9.7 | 2000 | 0 | 51.8 | 154.8 | 394.3 | 574.0 | 151.1 | 1638 |

- **Rule:** the first level whose throughput is under 1.10 times the previous level's.
- **Peak throughput:** 62.1 req/s at 8 clients.
- **Saturated at:** 8 clients, the first level whose throughput gain over the previous level fell under the rule; from there latency grows with the number of clients and throughput does not.
- **Gains:** 1 to 2: +88%, 2 to 4: +28%, 4 to 8: +4%, 8 to 16: -16%, 16 to 32: -0%.

## What this does and does not show

- One laptop, the client on the same cores as the server, a historical file replayed as fast as the server accepts it: the throughput is a number about this machine and this replay, not a capacity statement.
- The latency profile per request (`reports/latency.json`) and the shape of each sweep (where a worker stops scaling, and what more workers buy) are what transfer; the absolute numbers do not.
- The per-key ordering the store needs is supplied by the client's dispatcher here. A deployment gets it from a partitioned queue or pays for it with 409s; the stall counts are what that ordering cost this replay, and the in-flight column is the concurrency the server actually saw.
