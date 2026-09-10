# Every target runs against the project venv at .venv/ so that a shell without the venv
# activated still gets the pinned interpreter.
PY := .venv/bin/python
PIP := .venv/bin/pip
KAGGLE := .venv/bin/kaggle

.PHONY: setup data audit split-summary memory-profile eda prep features train experiments serve test lint reproduce clean

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
	@echo "eda: not implemented until stage 2"
	@exit 1

prep:
	@echo "prep: not implemented until stage 3"
	@exit 1

features:
	@echo "features: not implemented until stage 4"
	@exit 1

train:
	@echo "train: not implemented until stage 5"
	@exit 1

experiments:
	@echo "experiments: not implemented until stage 6"
	@exit 1

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
reproduce: data audit split-summary memory-profile
	@echo "reproduce: stages 0 and 1 complete (data, audit, split-summary, memory-profile)"
	@echo "later stages append prep, features, train, experiments here"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
