"""The case study's numbers script reads the artifacts and prints every value it names.

The article itself lives on the author's site and cannot be held to the artifacts by a test the
way the README is. What can be held is the script the article's verification table points at:
it runs on a clean checkout without the data, every value it prints is non-empty, it carries
every value the README's script prints, and the values the article leans on hardest are the
artifacts' own.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from fraud_platform import config

SCRIPT = config.ROOT / "scripts" / "case_study_numbers.py"
REPORTS = config.ROOT / "reports"


@pytest.fixture(scope="module")
def numbers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("case_study_numbers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def values(numbers: ModuleType) -> dict[str, str]:
    return dict(numbers.case_study_numbers(REPORTS, config.ROOT / "loadtest"))


def test_every_value_is_a_non_empty_string(values: dict[str, str]) -> None:
    assert values
    for name, value in values.items():
        assert isinstance(value, str) and value.strip(), name


def test_the_readme_values_are_a_subset(numbers: ModuleType, values: dict[str, str]) -> None:
    readme = numbers.readme_module().inline_numbers(REPORTS, config.ROOT / "loadtest")
    for name, value in readme.items():
        assert values[name] == value, name


def test_the_thread_numbers_are_the_artifacts_own(values: dict[str, str]) -> None:
    with (REPORTS / "leakage_delta.json").open() as handle:
        leakage = json.load(handle)
    delta = leakage["deltas"]["per_variant"]["d_random_split"]["pr_auc_by_row"]["difference"]
    assert values["delta_d"] == (
        f"{delta['point']:+.4f} ({delta['low']:+.4f} to {delta['high']:+.4f})"
    )
    with (REPORTS / "metric_variance.json").open() as handle:
        shipped = json.load(handle)["by_row"]["per_model"]["shipped"]
    assert values["test_pr_auc"] == f"{shipped['point']:.4f}"
    assert values["test_pr_auc_interval"] == f"{shipped['low']:.4f} to {shipped['high']:.4f}"
    with (REPORTS / "parity.json").open() as handle:
        parity = json.load(handle)
    assert values["parity_max_difference"] == str(
        max(b["max_abs_difference_over_all_features"] for b in parity["backends"].values())
    )


def test_the_script_prints_its_named_values(
    numbers: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    import sys

    argv = sys.argv
    sys.argv = [str(SCRIPT)]
    try:
        numbers.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "## named numbers" in out
    assert "delta_d: " in out and "test_pr_auc: " in out


def test_the_script_has_no_em_dash() -> None:
    assert "\u2014" not in Path(SCRIPT).read_text(encoding="utf-8")
