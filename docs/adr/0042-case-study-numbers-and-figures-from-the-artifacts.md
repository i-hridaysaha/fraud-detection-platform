# 0042: The case study quotes one numbers script and draws from verbatim artifact copies

- **Status:** accepted
- **Date:** 2026-09-13
- **Stage:** 11

## Context

ADR 0041 gave the case study on the author's site the narrative and left it to be brought into
line with the artifacts here. The article lives outside this repository, so nothing in CI can
compare it with `reports/` the way `tests/test_readme.py` compares the README. The previous
version of the article, written against an earlier build, quoted numbers this repository does
not produce, and its figures had at one point carried 63 hand-typed values that went on
rendering after every one of them became false.

The question is how an article that cannot be tested here is nonetheless held to the rule that
every number comes from a committed artifact plus one command.

## Options considered

**Write the article from the stage documents and ADRs, by hand.** The documents are already
verified against the artifacts. Rejected: a number retyped from a verified document is still a
retyped number, and stage 10 found two last-place errors of exactly that kind in the README's
first draft and in `docs/eda.md`.

**Reuse the README's numbers script as it stands.** It prints the three tables and about sixty
named values. Rejected as sufficient: the article needs values the README does not quote (the
lag sweep gap, the entity-encoding decomposition, the band volumes, the gate verdicts by cycle,
the store latencies), and adding them to the README's script would make the README test
assert values the README never shows.

**One numbers script for the article, importing the README's, and a figure kit that reads
verbatim copies of the artifacts.** Chosen. `scripts/case_study_numbers.py` reuses
`readme_numbers.inline_numbers` and the three table functions and adds the values only the
article needs, each named, so the article's verification table can point at a key. On the site
side, a `sync_artifacts.py` copies the artifacts byte for byte (`shutil.copyfile`, provable with
`cmp`), an `artifacts.py` is the only module that opens a file, and `figures.py` types no
number. `tests/test_case_study_numbers.py` holds the script to the artifacts on a clean
checkout.

## Decision

The case study quotes numbers only in the spelling `scripts/case_study_numbers.py` prints
(`make case-study-numbers`), and its figures are rendered from copies of `reports/*.json` made
by the site's `sync_artifacts.py`. The article's verification table, kept in
`docs/notes/stage-11.md`, maps every number to a key of that script or names it as derived,
assumed or session-measured. `scripts/verify_numbers.py --docs <article>` is the mechanical
check, run at the stage close.

## Evidence

Judgement, not measurement, for the choice of mechanism. The measured inputs: the residue of
`scripts/verify_numbers.py` on the article at the stage close (recorded in the stage note), and
`tests/test_case_study_numbers.py`, which asserts the script runs without the data, prints every
README value unchanged, and reproduces the leakage delta, the shipped PR-AUC and the parity
difference from the artifacts.

## Consequences

- A number that changes in an artifact changes the article only through a rerun of the script
  and a re-sync of the figures; both are one command each, and the `cmp` check shows a stale copy.
- The article can still drift after the script is run, because nothing in CI reads it. The
  verification table and the numeric scan are the discipline, and they run at the stage close,
  not on every push.
- The site-side figure kit lives outside this repository and is not tested here. Its README
  records the checks it makes by hand: every PNG opened after rendering, the hero rendered with
  the safe-box guide on.
