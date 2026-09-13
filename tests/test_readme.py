"""The README quotes the artifacts and nothing else, on a clean checkout without the data.

Every table in the README's Results section is the output of `scripts/readme_numbers.py` pasted
verbatim, every inline number is one of the named values that script prints, every block in the
architecture diagram is named in the README's prose, and every relative link resolves. The
script reads only committed JSON, so these tests run in CI.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from types import ModuleType

import pytest

from fraud_platform import config

README = config.ROOT / "README.md"
DIAGRAM = config.ROOT / "docs" / "architecture.svg"
SCRIPT = config.ROOT / "scripts" / "readme_numbers.py"
EM_DASH = "\u2014"
# Prose the standard this README follows refuses, and the section headings it refuses.
BANNED_WORDS = (
    "comprehensive",
    "robust",
    "seamless",
    "leverage",
    "state-of-the-art",
    "enterprise",
    "production-grade",
    "learning project",
)
BANNED_HEADINGS = ("contributing", "maintainers", "code of conduct")


@pytest.fixture(scope="module")
def numbers() -> ModuleType:
    spec = importlib.util.spec_from_file_location("readme_numbers", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def unwrapped(text: str) -> str:
    """Prose is wrapped at 100 columns; a quoted value may straddle a line break."""
    return re.sub(r"\s+", " ", text)


def test_results_tables_are_the_script_output(readme: str, numbers: ModuleType) -> None:
    reports = config.REPORTS_DIR
    for table in (
        numbers.leakage_table(reports),
        numbers.latency_table(reports),
        numbers.comparison_table(reports),
    ):
        assert table in readme, f"table not in README verbatim:\n{table.splitlines()[0]}"


def test_every_inline_number_is_in_the_readme(readme: str, numbers: ModuleType) -> None:
    values = numbers.inline_numbers(config.REPORTS_DIR, config.ROOT / "loadtest")
    text = unwrapped(readme)
    missing = [f"{name}: {value}" for name, value in values.items() if value not in text]
    assert not missing, "README lacks:\n" + "\n".join(missing)


def test_no_em_dash_and_no_banned_prose(readme: str) -> None:
    assert EM_DASH not in readme
    lowered = readme.lower()
    hits = [word for word in BANNED_WORDS if word in lowered]
    assert not hits, f"banned prose in README: {hits}"
    headings = [line.lower() for line in readme.splitlines() if line.startswith("#")]
    for banned in BANNED_HEADINGS:
        assert not any(banned in h for h in headings), banned


def test_every_diagram_block_is_named_in_the_prose(readme: str) -> None:
    blocks = re.findall(r'data-node="([^"]+)"', DIAGRAM.read_text(encoding="utf-8"))
    assert len(blocks) >= 10
    approach = unwrapped(readme[readme.index("## Approach") : readme.index("## Data")]).lower()
    missing = [b for b in blocks if b.lower() not in approach]
    assert not missing, f"diagram blocks absent from the Approach prose: {missing}"


def test_exactly_one_emphasised_block(readme: str) -> None:
    svg = DIAGRAM.read_text(encoding="utf-8")
    emphasised = re.findall(r'class="node emphasis" data-node="([^"]+)"', svg)
    assert emphasised == ["Model registry"]
    assert "source of truth" in readme


def test_relative_links_resolve(readme: str) -> None:
    targets = re.findall(r"\]\(((?!https?://)[^)#]+)\)", readme)
    assert targets, "no relative links found"
    missing = [t for t in targets if not (config.ROOT / t).exists()]
    assert not missing, f"README links to missing paths: {missing}"


def test_license_file_and_line() -> None:
    text = unwrapped((config.ROOT / "LICENSE").read_text(encoding="utf-8"))
    assert text.startswith("MIT License")
    assert "not redistributable" in text
    assert "## License" in README.read_text(encoding="utf-8")


def test_length_is_front_loaded(readme: str) -> None:
    """The results table is reachable inside the first sixty lines, and the file stays short."""
    lines = readme.splitlines()
    first_table = next(i for i, line in enumerate(lines) if line.startswith("| Variant"))
    assert first_table < 60
    assert len(lines) <= 260


def test_script_runs_from_the_command_line(
    numbers: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = sys.argv
    sys.argv = ["readme_numbers.py"]
    try:
        numbers.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "## leakage delta" in out and "test_pr_auc:" in out
