# 0004: Tooling baseline

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 0

## Context

This repository is meant to be read by someone deciding whether its numbers can be trusted. That
puts the burden on the checks that run before anything is merged, and it makes the choice of
checks a decision worth recording rather than a default worth inheriting quietly.

Four failure modes are worth spending CI time on. Code that does not agree with itself on style,
which makes diffs unreadable. Type errors in the data path, which in pandas code usually surface
as a silent wrong answer rather than a crash. Behaviour changes in the audit and feature logic,
which is what the tests are for. And drift between a committed artifact and the code that claims
to produce it, which is the one that would quietly invalidate every number in the README.

## Options considered

**Nothing but tests.** Rejected. Tests catch behaviour, not the class of mistake where a
DataFrame column is silently object dtype and every downstream number is wrong but plausible.

**black plus flake8 plus isort.** Three tools, three configs, three failure messages for the same
line. Rejected in favour of one tool that does lint and format, which keeps the pre-commit hook
fast enough that it is not disabled.

**ruff plus mypy plus pytest, all gated in CI and pre-commit.** Chosen.

**mypy in non-strict mode.** Rejected. Non-strict mypy passes an unannotated function without
comment, which in practice means the annotations stop at whatever was convenient. Strict mode on
src/ only keeps the discipline where the numbers are produced without forcing it on throwaway
scripts.

## Decision

| Tool | Version | Scope | What it is there to catch |
| --- | --- | --- | --- |
| ruff | 0.16.6 | src, tests, scripts | Lint and format in one pass: unused names, shadowed builtins, import order, pathlib misuse, formatting. |
| mypy | 2.3.1 | src only, strict | Unannotated or wrongly annotated data-path functions, the dtype and None mistakes that produce a plausible wrong number. |
| pytest | 9.1.1 with pytest-cov 7.1.0 | tests | Behaviour of the audit logic, plus internal consistency of the committed artifact. |
| pre-commit | 4.6.2 | staged files | Runs all three locally before a commit, plus the em dash check and a 2 MB file size guard so the CSVs cannot be committed by accident. |
| GitHub Actions | workflow .github/workflows/ci.yml | push and pull request | The same three checks on a clean checkout with no local state. |

Python is pinned to 3.12 in CI and in the local venv, and every runtime dependency is pinned to
an exact version in pyproject.toml.

The em dash check (`scripts/check_no_em_dash.sh`) is part of lint rather than a manual step,
because a house style rule that depends on remembering to grep is not a rule.

Tests that read the IEEE-CIS CSVs are marked `needs_data` and deselected in CI, since the data is
not redistributable and never reaches the runner. The gap that leaves is covered by
tests/test_audit_report.py, which runs against the committed reports/audit.json in CI and checks
that the splits reconcile with the row count, that the windows are ordered and disjoint, that
every reported rate is a rate, and that every section carries the command that produced it.

## Evidence

- **Artifact:** pyproject.toml, .pre-commit-config.yaml, .github/workflows/ci.yml.
- **Command:** `make lint && make test`, and `.venv/bin/python -m pip list` for the versions in
  the table above.
- Versions were read from the installed environment, not chosen from memory. The full test suite
  at the time of this ADR: 30 passed, 25 of which run in CI and 5 of which are `needs_data`.
- Coverage of src at that point: 98 percent, from the pytest-cov summary.

## Consequences

- A contributor without the CSVs can still run the full CI check set locally with
  `pytest -m "not needs_data"`.
- Strict mypy on src means every new data-path function needs annotations before it merges,
  which is a real cost on exploratory code. Exploratory code belongs in scripts/, which is
  linted but not type checked.
- Pinned exact versions mean dependency updates are deliberate and show up as their own commit,
  at the cost of not picking up patch releases automatically.
- CI cannot detect a stale reports/audit.json that is internally consistent but was generated
  from different data. The sha256 of both inputs is recorded in the artifact so that check can be
  made by hand, and `make reproduce` regenerates it from the raw files.
