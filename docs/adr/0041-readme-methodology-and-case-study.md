# 0041: The README is the proof, METHODOLOGY.md the detail, the case study the narrative

- **Status:** accepted
- **Date:** 2026-09-13
- **Stage:** 10

## Context

By the end of stage 9 the repository carried nine stage documents under `docs/` (3,506 lines),
39 decision records, ten stage notes, a demo page and a four-line placeholder README. A reader
arriving from the case study, or from GitHub directly, gives the repository between thirty
seconds and five minutes. Nothing in it was written for that reader.

The README standard the author's public repositories follow fixes the order (title, one-liner,
case-study link, results, why it is non-trivial, approach, data, model, evaluation, reproduce,
repo map, limitations, licence), the length (120 to 200 lines, front-loaded), and the rule that
results appear as a markdown table with the public-data caveat inside the section. Its one
structural instruction is that the README complements the case study rather than duplicating it:
the case study owns the narrative and the deep methodology, the README owns the proof.

The stage documents are too long for the README and too stage-shaped for a reader who wants
one place to find how a feature is defined or why a column was dropped.

## Options considered

**One long README holding everything.** Rejected. A 500-line README drowns the leakage table,
which is the repository's actual result, under the column plan. The standard names 300 lines as
the ceiling and this repository's material runs ten times that.

**A README that links the stage documents and nothing else.** Rejected. The stage documents are
organised by when a thing was measured, not by what a reader wants to know; the feature
definitions are in stage 4, their serving state in stage 8, their drift edges in stage 9. A
reader looking for "what does `vel_count_24h_entity` mean and where is it served from" should
not have to know the build order.

**Three documents with three jobs.** Chosen.

- The README carries the proof: the scope and provenance statement, the two stage 7 tables
  (the leakage delta and the label latency) ahead of the model comparison because they are the
  contribution and the comparison is context, every number with its interval and its
  configuration, the traps handled, the architecture diagram, the data, the model, the
  evaluation protocol, the reproduce commands, the repo map, the limitations, the prior work,
  and the licence line. Every table in it is the output of `scripts/readme_numbers.py`, and
  `tests/test_readme.py` fails if the README and the artifacts disagree.
- `METHODOLOGY.md` carries the detail the README cannot hold, organised by subject rather than
  by stage: the EDA summary, the column plan and its rules, the transformation and encoding
  decisions, the feature definitions read from the catalogue artifact, the graph finding, the
  modelling protocol, the experiments, the serving design, the monitoring jobs and the gate, a
  consolidated list of what is not established, and the ADR index.
- The case study, on the author's site, carries the narrative: why the system has the shape it
  has, the decisions in the order they were made, the lessons. The README links it once, near
  the top, and does not repeat it.

## Decision

The README is `README.md` as written in stage 10, held under 260 lines by
`tests/test_readme.py` with the first results table inside the first sixty lines; the length
exceeds the standard's 200-line target by the three results tables, which is the trade the
standard allows when the top of the file carries everything. `METHODOLOGY.md` sits at the
repository root beside it. The stage documents under `docs/` stay as they are and are linked
from METHODOLOGY.md by section, and the decision records are indexed in `docs/adr/README.md`.

No number appears in the README or in METHODOLOGY.md that does not appear in a committed
artifact; `scripts/verify_numbers.py` scans both, and `docs/verification.md` accounts for the
residue.

## Evidence

Judgement, not measurement. The measured inputs to the judgement: `wc -l README.md` (the line
count the test bounds), `wc -l docs/*.md` (3,506 lines at the start of the stage), and the
output of `scripts/verify_numbers.py` on the two documents, recorded in `docs/verification.md`.

## Consequences

- A number that changes in an artifact changes the README only through
  `scripts/readme_numbers.py`; editing a number by hand fails the test.
- METHODOLOGY.md duplicates numbers that the stage documents also carry. Both read from the
  same artifacts, and the verification pass checks both; the duplication is the price of a
  document organised by subject.
- The case study on the site was written against an earlier build of this platform and quotes
  numbers this repository does not produce. Bringing it into line with the artifacts here is
  outside the repository and is recorded in `docs/notes/stage-10.md` as work the release does
  not include.
- The README's line count is an explicit deviation from the standard's target, recorded here
  so it is not mistaken for drift.
