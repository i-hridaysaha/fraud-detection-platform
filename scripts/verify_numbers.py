"""Match every numeric token in the prose against a numeric leaf of a committed artifact.

Usage:
    make verify-numbers
    .venv/bin/python scripts/verify_numbers.py [--docs README.md METHODOLOGY.md docs/...]

Reads every JSON artifact under reports/, models/shipped_model.json and loadtest/results.json,
indexes every numeric leaf under the spellings the prose uses (rounded to one to six decimals,
signed, thousands-separated), then scans the documents for numeric tokens worth checking: any
token with a decimal point, a thousands separator, or four or more digits. A token that matches
no leaf is printed with its file and line for a reader to source by hand; the derived quotients
and the dates are the expected residue, and docs/verification.md accounts for each.

Exit status is 0 whatever the residue: the script is a reader's aid, and the judgement about an
unmatched token is made in docs/verification.md, not here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOTS = (ROOT / "reports", ROOT / "models", ROOT / "loadtest")
# docs/verification.md is the accounting for this script's residue and is not scanned: every
# unmatched token appears in it by construction.
DEFAULT_DOCS = (
    "README.md",
    "METHODOLOGY.md",
    "CHANGELOG.md",
    *sorted(
        str(p.relative_to(ROOT))
        for p in (ROOT / "docs").glob("*.md")
        if p.name != "verification.md"
    ),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs" / "adr").glob("*.md")),
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs" / "notes").glob("*.md")),
)
TOKEN = re.compile(
    r"(?<![\w.])[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[+-]?\d+\.\d+|(?<![\w.])\d{4,}"
)
# Tokens that are never measurements: ADR numbers, years, and the day indices of the split.
IGNORE = re.compile(r"^(?:00\d\d|20(?:19|2\d)|\d{4}-\d{2}-\d{2})$")


def leaves(value: Any) -> list[float]:
    """Every int or float anywhere inside a JSON value."""
    if isinstance(value, bool):
        return []
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in leaves(v)]
    if isinstance(value, list):
        return [x for v in value for x in leaves(v)]
    if isinstance(value, str):
        # timestamps and commands inside the artifacts carry numbers too; they count as sources
        return [float(m.replace(",", "")) for m in re.findall(r"-?\d+(?:\.\d+)?", value)]
    return []


def spellings(value: float) -> set[str]:
    out: set[str] = set()
    if value == int(value) and abs(value) < 1e15:
        n = int(value)
        out.update({str(n), f"{n:,}", f"{abs(n)}", f"{abs(n):,}"})
    # a float quoted to the nearest whole number, with or without the thousands separator
    out.update({f"{round(value):,}", str(round(value)), f"{abs(round(value)):,}"})
    for places in range(1, 9):
        rounded = f"{value:.{places}f}"
        out.add(rounded)
        out.add(f"{value:+.{places}f}")
        out.add(f"{value:,.{places}f}")
        out.add(rounded.lstrip("-"))
    # a rate quoted as a percentage or a ratio quoted as its reciprocal are derived, not spelled
    return out


def build_index() -> set[str]:
    index: set[str] = set()
    for root in ARTIFACT_ROOTS:
        for path in root.rglob("*.json"):
            with path.open() as handle:
                for leaf in leaves(json.load(handle)):
                    index.update(spellings(leaf))
    return index


def tokens(text: str) -> list[tuple[int, str]]:
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in TOKEN.finditer(line):
            token = match.group(0)
            if IGNORE.match(token.lstrip("+-")):
                continue
            found.append((number, token))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", nargs="*", default=list(DEFAULT_DOCS))
    args = parser.parse_args()
    index = build_index()
    scanned = unmatched = 0
    for doc in args.docs:
        path = ROOT / doc
        if not path.exists():
            continue
        misses = []
        for line, token in tokens(path.read_text(encoding="utf-8")):
            scanned += 1
            bare = token.lstrip("+")
            if bare in index or bare.lstrip("-") in index:
                continue
            unmatched += 1
            misses.append(f"  {doc}:{line}: {token}")
        if misses:
            print(f"{doc}: {len(misses)} unmatched")
            print("\n".join(misses))
    print(f"scanned {scanned} tokens, {unmatched} unmatched")


if __name__ == "__main__":
    main()
