"""The demo page is the stage 9 artifacts and nothing else, on a clean checkout without the data.

`docs/demo/index.html` is a self-contained page built by `scripts/demo_page.py` from the
committed artifacts under reports/. Every assertion here ties the committed page to a fresh
build or to the artifact it quotes: the page reproduces byte for byte, the JSON it carries is
what the extractor reads from the artifacts, and every cycle, gate verdict and headline number
the page can render is present in that JSON with the artifact's own value.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from fraud_platform import config, lifecycle

REPORTS = config.REPORTS_DIR
PAGE = config.ROOT / "docs" / "demo" / "index.html"
SCRIPT = config.ROOT / "scripts" / "demo_page.py"
EM_DASH = "\u2014"


def load(path: Path) -> dict[str, Any]:
    assert path.exists(), f"{path.relative_to(config.ROOT)} is missing"
    return dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("demo_page", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def page() -> str:
    assert PAGE.exists(), "docs/demo/index.html is missing; run `make demo-page`"
    return PAGE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def embedded(page: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="demo-data" type="application/json">(.*?)</script>', page, re.DOTALL
    )
    assert match is not None, "the page carries no embedded data block"
    return dict(json.loads(match.group(1).replace("<\\/", "</")))


@pytest.fixture(scope="module")
def lifecycle_demo() -> dict[str, Any]:
    return load(REPORTS / "lifecycle_demo.json")


# --- the page is a build of the artifacts --------------------------------------------------------


def test_committed_page_matches_a_fresh_build(
    builder: ModuleType, page: str, tmp_path: Path
) -> None:
    out = tmp_path / "index.html"
    out.write_text(builder.render(builder.extract(REPORTS)), encoding="utf-8")
    assert out.read_bytes() == PAGE.read_bytes(), (
        "docs/demo/index.html is stale against the artifacts; run `make demo-page`"
    )


def test_build_is_deterministic(builder: ModuleType) -> None:
    assert builder.render(builder.extract(REPORTS)) == builder.render(builder.extract(REPORTS))


def test_embedded_data_is_the_extraction(builder: ModuleType, embedded: dict[str, Any]) -> None:
    assert embedded == builder.extract(REPORTS)


def test_page_is_self_contained(page: str) -> None:
    # One stylesheet host for the typefaces, with a declared fallback stack; no other external
    # script, style or image, so a copy of the file is the whole deployment.
    external = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
    assert all(
        u.startswith(("https://fonts.googleapis.com", "https://fonts.gstatic.com"))
        for u in external
    ), external
    assert "<script src=" not in page
    assert 'system-ui, -apple-system, "Segoe UI", sans-serif' in page
    assert "ui-monospace" in page


def test_page_has_a_title_and_no_em_dash(page: str) -> None:
    assert "<title>Two Clocks and a Gate</title>" in page
    assert EM_DASH not in page
    assert EM_DASH not in SCRIPT.read_text(encoding="utf-8")


def test_builder_reads_no_data(page: str) -> None:
    # The builder opens JSON under reports/ and nothing else; the raw files and the cached
    # matrices never reach the page except by name inside a recorded regenerate command.
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert imported <= {"__future__", "argparse", "json", "pathlib", "typing"}, imported
    assert ".parquet" not in page


# --- what the page carries is what the artifacts hold --------------------------------------------


def test_every_cycle_is_carried(embedded: dict[str, Any], lifecycle_demo: dict[str, Any]) -> None:
    cycles = embedded["cycles"]
    assert len(cycles) == lifecycle_demo["n_cycles"] == len(lifecycle_demo["cycles"])
    for carried, recorded in zip(cycles, lifecycle_demo["cycles"], strict=True):
        assert carried["cycle"] == recorded["cycle"]
        assert carried["as_of_day"] == recorded["as_of_day"]
        assert carried["champion"] == recorded["champion"]
        assert carried["drift"] == recorded["drift"]
        assert carried["decision"] == recorded["decision"]
        assert carried["immature"] == recorded["immature"]
        assert len(carried["decay"]) == len(recorded["decay"])
        for a, b in zip(carried["decay"], recorded["decay"], strict=True):
            assert all(a[k] == b[k] for k in a), (recorded["cycle"], a)


def test_every_gate_verdict_is_carried_whole(
    embedded: dict[str, Any], lifecycle_demo: dict[str, Any]
) -> None:
    carried = [c["challenger"] for c in embedded["cycles"] if "challenger" in c]
    recorded = [c["challenger"] for c in lifecycle_demo["cycles"] if "challenger" in c]
    assert len(carried) == len(recorded)
    fitted = [c for c in carried if c["trained"]]
    assert len(fitted) == lifecycle_demo["n_challengers"]
    verdicts = {c["gate"]["promote"] for c in fitted}
    assert verdicts == {True, False}, "the page must show the gate in both directions"
    for a, b in zip(carried, recorded, strict=True):
        assert a["trained"] == b["trained"]
        if a["trained"]:
            assert a["gate"] == b["gate"]
            assert a["windows"] == b["windows"]
            assert a["registry"] == b["registry"]
            assert a.get("promotion") == b.get("promotion")
            assert a["gate"]["margin"] == lifecycle.PROMOTION_MARGIN
        else:
            assert a["reason"] == b["reason"]


def test_rollback_and_promotion_are_carried(
    embedded: dict[str, Any], lifecycle_demo: dict[str, Any]
) -> None:
    assert embedded["rollback"] == lifecycle_demo["rollback"]
    assert embedded["rollback"]["performed"] is True
    assert embedded["rollback"]["smoke"]["max_abs_score_difference_serving_vs_batch"] == 0.0
    promotion = lifecycle_demo["promotion"]
    for key in ("cycle", "version", "previous_version", "new_reference_pr_auc"):
        assert embedded["promotion"][key] == promotion[key]
    assert embedded["inputs"] == lifecycle_demo["inputs"]
    assert embedded["stream"] == lifecycle_demo["stream"]
    assert embedded["champion"] == lifecycle_demo["champion"]


def test_headline_numbers_come_from_their_artifacts(embedded: dict[str, Any]) -> None:
    variance = load(REPORTS / "metric_variance.json")
    operating = load(REPORTS / "operating_points.json")
    latency = load(REPORTS / "latency.json")
    parity = load(REPORTS / "parity.json")
    split = load(REPORTS / "split_summary.json")
    h = embedded["headline"]
    shipped = variance["by_row"]["per_model"]["shipped"]
    assert h["pr_auc"]["point"] == shipped["point"]
    assert h["pr_auc"]["low"] == shipped["low"]
    assert h["pr_auc"]["high"] == shipped["high"]
    assert h["roc_auc_test"] == operating["primary_metric"]["test"]["roc_auc"]
    assert (
        h["latency"]["p50_ms_median"]
        == latency["summary"]["pinned_p50_by_family"]["xgboost"]["median"]
    )
    assert h["parity"]["n_rows"] == parity["stream"]["n_rows"]
    assert h["parity"]["bit_identical"] == parity["backends"]["memory"]["bit_identical"]
    assert h["n_rows"] == split["totals"]["n_rows"]
    # the same reference the decay job carries, so the page's two quotes of it cannot diverge
    assert embedded["decay_job"]["reference"]["pr_auc"]["point"] == h["pr_auc"]["point"]


def test_constants_match_the_code(embedded: dict[str, Any]) -> None:
    assert embedded["inputs"]["promotion_margin"] == lifecycle.PROMOTION_MARGIN
    assert embedded["inputs"]["decay_tolerance"] == lifecycle.DECAY_TOLERANCE
    assert embedded["inputs"]["sustained_cycles"] == lifecycle.SUSTAINED_CYCLES
    assert embedded["inputs"]["maturity_days"] == lifecycle.MATURITY_DAYS
    band = embedded["noise_band"]["by_card"]["half_width"]
    assert band < lifecycle.PROMOTION_MARGIN <= band + 0.01


def test_every_source_names_its_command(embedded: dict[str, Any]) -> None:
    sources = embedded["sources"]
    assert sources[0]["artifact"] == "reports/lifecycle_demo.json"
    for s in sources:
        assert (config.ROOT / s["artifact"]).exists(), s["artifact"]
        assert s["regenerate_with"], s["artifact"]
