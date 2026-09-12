# Every target runs against the project venv at .venv/ so that a shell without the venv
# activated still gets the pinned interpreter.
PY := .venv/bin/python
PIP := .venv/bin/pip
KAGGLE := .venv/bin/kaggle

.PHONY: setup data audit split-summary memory-profile eda eda-figures prep prep-figures features features-figures graph graph-figures train train-figures experiments leakage latency experiment-figures register parity serving-latency serve loadtest test lint reproduce clean

setup:
	$(PY) -m pip install --upgrade pip
	$(PIP) install -e ".[dev,data]"
	.venv/bin/pre-commit install
	install -m 0755 scripts/git-hooks/pre-push .git/hooks/pre-push

data:
	bash scripts/download_data.sh

audit:
	$(PY) scripts/audit.py --transactions data/train_transaction.csv --identity data/train_identity.csv --out reports/audit.json

split-summary:
	$(PY) scripts/split_summary.py --transactions data/train_transaction.csv --identity data/train_identity.csv --out reports/split_summary.json

memory-profile:
	$(PY) scripts/memory_profile.py --out reports/memory_profile.json

eda:
	$(PY) scripts/eda.py --transactions data/train_transaction.csv --identity data/train_identity.csv --out-dir reports/eda
	$(MAKE) eda-figures

eda-figures:
	$(PY) scripts/eda_figures.py --in-dir reports/eda --out-dir figures

prep:
	$(PY) scripts/prepare.py --transactions data/train_transaction.csv --identity data/train_identity.csv
	$(MAKE) prep-figures

prep-figures:
	$(PY) scripts/prep_figures.py --reports-dir reports --out-dir figures

features:
	$(PY) scripts/features.py --transactions data/train_transaction.csv --identity data/train_identity.csv
	$(MAKE) features-figures

features-figures:
	$(PY) scripts/feature_figures.py --reports-dir reports --out-dir figures

graph:
	$(PY) scripts/graph.py --transactions data/train_transaction.csv --identity data/train_identity.csv
	$(MAKE) graph-figures

graph-figures:
	$(PY) scripts/graph_figures.py --reports-dir reports --out-dir figures

# The whole of stage 6: the cached matrices, the family comparison, the ablations, the search, the
# bootstrap, the thresholds and bands, and the SHAP artifacts, then the figures. Every number the
# stage quotes comes out of this one target.
train:
	$(PY) scripts/train.py --transactions data/train_transaction.csv --identity data/train_identity.csv
	$(MAKE) train-figures

train-figures:
	$(PY) scripts/train_figures.py --reports-dir reports --out-dir figures

# The model-selection half of stage 6 on its own: which family, which strategy, which stack,
# which hyperparameters. Reads the cached matrices, so run `make train` once first.
experiments:
	$(PY) scripts/train.py --sections comparison ablations search

# Stage 7. The leakage delta rebuilds its variants from the raw files; the label latency reads
# the stage 6 cache, so `make train` runs first. Both write one artifact each under reports/.
leakage:
	$(PY) scripts/leakage_delta.py --transactions data/train_transaction.csv --identity data/train_identity.csv

latency:
	$(PY) scripts/label_latency.py

experiment-figures:
	$(PY) scripts/experiment_figures.py --reports-dir reports --out-dir figures

# Stage 8. `register` refits the stage 3 pipeline on train and logs the shipped booster with it
# as one model version under mlruns/ (gitignored), alias production; the service loads that alias
# and nothing else. `parity` and `serving-latency` write the two artifacts the stage quotes.
# `loadtest` starts one uvicorn worker on the Redis store and drives the concurrency sweep.
register:
	$(PY) scripts/register_model.py --transactions data/train_transaction.csv --identity data/train_identity.csv

parity:
	$(PY) scripts/parity.py --days 30

serving-latency:
	$(PY) scripts/serving_latency.py

# One worker. FRAUD_REDIS_URL selects the Redis store (redis://localhost:6379/0); unset, the
# store is in-process. The registry location comes from MLFLOW_TRACKING_URI or mlruns/.
serve:
	.venv/bin/uvicorn fraud_platform.service:app --host 127.0.0.1 --port 8000 --workers 1

loadtest:
	$(PY) loadtest/run_loadtest.py

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts
	.venv/bin/mypy
	bash scripts/check_no_em_dash.sh

# The single entry point the README promises: raw data in, every committed artifact out.
# Stages append their steps here as they land, so the chain is never retrofitted.
reproduce: data audit split-summary memory-profile eda prep features graph train leakage latency experiment-figures register parity serving-latency
	@echo "reproduce: stages 0 to 8 complete (data, audit, split-summary, memory-profile, eda, prep, features, graph, train, leakage, latency, experiment-figures, register, parity, serving-latency)"
	@echo "the load test needs a Redis server and is run on its own: make loadtest"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
