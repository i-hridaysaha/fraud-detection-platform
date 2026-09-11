# Every target runs against the project venv at .venv/ so that a shell without the venv
# activated still gets the pinned interpreter.
PY := .venv/bin/python
PIP := .venv/bin/pip
KAGGLE := .venv/bin/kaggle

.PHONY: setup data audit split-summary memory-profile eda eda-figures prep prep-figures features features-figures graph graph-figures train train-figures experiments serve test lint reproduce clean

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

serve:
	@echo "serve: not implemented until stage 7"
	@exit 1

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts
	.venv/bin/mypy
	bash scripts/check_no_em_dash.sh

# The single entry point the README promises: raw data in, every committed artifact out.
# Stages append their steps here as they land, so the chain is never retrofitted.
reproduce: data audit split-summary memory-profile eda prep features graph train
	@echo "reproduce: stages 0 to 6 complete (data, audit, split-summary, memory-profile, eda, prep, features, graph, train)"
	@echo "later stages append serve here"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
