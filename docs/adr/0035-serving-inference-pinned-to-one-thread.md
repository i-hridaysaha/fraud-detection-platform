# 0035: Serving inference is pinned to one thread, for the measured reason

- **Status:** accepted
- **Date:** 2026-09-12
- **Stage:** 8

## Context

The shipped booster was trained with `n_jobs=-1` and loads with `nthread` unset, which XGBoost
reads as every core. The service scores one row per request. The folklore reason to pin
inference to one thread is that a server handling many requests should not let each of them
grab every core; that may be true, and it is not what the benchmark measures. This ADR records
the reason the artifact supports and leaves the rest as unestablished.

## Options considered

**Leave the booster at its default, every core.** Rejected on the measurement below: the
single-row call is slower that way in every one of five fresh processes, by a factor between
1.49 and 1.59. A single row gives the thread pool nothing to divide and the fork-join is pure
cost.

**Pin to a small number of threads, two or four.** Not measured. Rejected because the question
it answers, where between one and ten the single-row optimum lies, has a measured answer at one
end and an assumed one everywhere else, and the artifact would then carry a number nothing
supports.

**Pin to one thread, and pin every model the same way when they are compared.** Chosen.
`serving.load_bundle` sets `n_jobs=1` and `nthread=1` on the loaded booster, and the latency
benchmark pins the random forest (`n_jobs=1`) and the logistic regression (BLAS and OpenMP
limited to one thread through `threadpoolctl`) the same way before the families are compared,
so the comparison is between models and not between two deployments.

## Decision

`serving.INFERENCE_THREADS = 1`, applied in `load_bundle` to the booster and to the DMatrix the
factors are computed on. The comment beside it points at `reports/latency.json`, block
`pinning`, and states the direction of the difference and the counterpoint; the numbers stay
in the artifact so that a new machine changes the artifact and not the text.

## Evidence

`reports/latency.json`, regenerated with `make serving-latency`. Five fresh interpreter
processes, each timing every model on the same 300 test rows, single-row `predict_proba`, 20
warm-up calls, milliseconds; the summary is the spread across the five processes.

| Model, pinned to one thread | p50, five-process range | p99 range |
| --- | --- | --- |
| xgboost, the shipped booster | 0.077 to 0.081 | 0.146 to 0.158 |
| random forest, 200 trees | 3.55 to 3.59 | 3.76 to 4.41 |
| logistic regression, complete-data path, 500 columns | 0.033 to 0.034 | 0.041 to 0.043 |

Ratios per process, random forest over xgboost 43.9 to 46.2, logistic regression over xgboost
0.410 to 0.430.

The pinning comparison, same rows, same process, pinned against the library default:

| Model | pinned p50 | unpinned p50 | unpinned over pinned, per process |
| --- | --- | --- | --- |
| xgboost, one row | 0.077 to 0.081 | 0.119 to 0.123 | 1.49 to 1.59 |
| random forest, one row | 3.55 to 3.59 | 15.7 to 15.9 | 4.37 to 4.47 |
| xgboost, 1,000 rows | 10.6 to 11.0 | 2.28 to 2.70 | pinned over unpinned 4.00 to 4.75 |

The between-process spread of the single-row medians was 5 percent at most
(`max_over_min` 1.05 for xgboost), not the factor of two the brief warned of; the ratios moved
less than that. The random forest's fit took 42.0 seconds and the logistic regression's 9.5
(311 iterations of 5,000, converged); both were fit from the stage 6 cache at the module-default
hyperparameters, no reweighting, seed 42, on the shipped 186-column stack.

## Consequences

- On a single row the booster is 0.08 ms of a 37.6 ms request (`summary.serving_path_p50_ms`
  and `summary.serving_path_whole_no_store_p50_ms`, medians of five processes, the path without
  the store and with five factors). Pinning saves about 0.04 ms per request. The decision is
  right and it is not where the latency is.
- The same booster on a batch of a thousand rows is 4.7 times faster unpinned. The pin is a
  single-row setting; a batch scorer built on this bundle must unpin.
- What the pin does under concurrent requests is not isolated. The load test runs pinned and
  reports a one-worker ceiling that the Python path, not the booster, sets
  (`loadtest/results.json`); a pinned-against-unpinned sweep under load would be the
  measurement, and it was not run.
- The families are compared at 0.08, 3.57 and 0.033 ms. A random forest of this size cannot
  serve inside the same budget as the booster on one thread, and the logistic regression is
  faster still on its 500-column dense row; the family stage 6 shipped is the cheapest of the
  two that can score this problem, not the cheapest model.
