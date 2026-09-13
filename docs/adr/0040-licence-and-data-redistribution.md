# 0040: MIT for the code, the dataset excluded and not redistributed

- **Status:** accepted
- **Date:** 2026-09-13
- **Stage:** 10

## Context

The repository goes public at the end of stage 10. Until now it had no licence file: a
repository without one grants nobody the right to use, copy or modify the code, and reads as
unfinished. The code and the data are two different things under two different sets of terms,
and the licence has to say which is which.

The code is the author's own work. The data is not: `train_transaction.csv` and
`train_identity.csv` are provided by Vesta Corporation through the Kaggle competition
`ieee-fraud-detection`, obtained by the author under that competition's rules through the
author's own Kaggle account (ADR 0003). The Kaggle rules page renders client side and returned
only its title when fetched for this record (2026-09-13), as it did in stage 0 and stage 4, so
the terms are not restated here as fact; what is stated is what the author accepted and what
this repository does about it.

## Options considered

**No licence, or "all rights reserved".** Rejected. It defeats the purpose of publishing the
code and it reads as an oversight.

**A copyleft licence (GPL, AGPL).** Rejected. The repository is a reference implementation of
a set of patterns; the intent is that a reader can lift the causal window primitive, the parity
test or the gate into their own work without a licensing conversation. Copyleft would make that
harder for no benefit to the author.

**A permissive licence covering the code, with the dataset stated to be excluded.** Chosen.
MIT, because it is the shortest permissive licence, is understood without reading it, and is
the licence the author's README standard names for every repository linked from the site. The dataset exclusion is written into the same
file, below the licence text, so that a reader who opens LICENSE and nothing else sees both.

**A committed sample of the data for convenience.** Rejected in ADR 0003 and not reopened: the
data is not the author's to redistribute, in whole or in part.

## Decision

`LICENSE` carries the MIT text verbatim (from https://spdx.org/licenses/MIT.txt, fetched
2026-09-13), with the copyright line `Copyright (c) 2026 Hriday Saha`, followed by a separated
paragraph stating that the dataset is not covered, is not included, is not redistributable, was
obtained by the author under the competition rules, and must be obtained the same way by anyone
reproducing the artifacts. `.gitignore` continues to exclude `data/`, `*.csv` and `*.csv.zip`,
and the README's licence line repeats the two facts in one sentence.

The committed artifacts under `reports/` are aggregate measurements (counts, rates, scores,
intervals), not rows of the data, and are covered by the code licence. `reports/audit.json`
records the sha256 of the two files so a reader can confirm which bytes they were computed
from.

## Evidence

Judgement, not measurement. The facts the decision rests on are: the licence text as fetched
(the URL above), the file provenance recorded in `reports/audit.json` (`inputs`, with sizes and
sha256), and `git ls-files | grep -c '\.csv'` returning 0.

## Consequences

- Anyone may use the code under MIT terms. Nobody gets the data from this repository; `make
  data` fetches it under the reader's own Kaggle acceptance, and the CI job runs without it.
- The dataset paragraph in LICENSE is prose the author wrote about the author's own acceptance;
  it does not quote or paraphrase the competition rules, because they could not be fetched and
  read for this record.
- A future change of licence for the code is a new ADR. The data exclusion does not change
  with it.
