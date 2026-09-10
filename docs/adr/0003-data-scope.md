# 0003: Data scope

- **Status:** accepted
- **Date:** 2026-09-10
- **Stage:** 0

## Context

The Kaggle competition "ieee-fraud-detection" distributes five files. Listed with
`.venv/bin/kaggle competitions files ieee-fraud-detection`:

| File | Bytes |
| --- | --- |
| sample_submission.csv | 6,080,314 |
| test_identity.csv | 25,797,161 |
| test_transaction.csv | 613,194,934 |
| train_identity.csv | 26,529,680 |
| train_transaction.csv | 683,351,067 |

The repository has to decide which of those it ingests, and whether any of them are committed.
The data is provided by Vesta under the competition rules and is not the author's to
redistribute.

## Options considered

**Download all five and treat the test files as a holdout.** Rejected. The isFraud column is
present in train_transaction.csv and there is no file in the distribution carrying labels for the
test rows; sample_submission.csv is a prediction template, not a label file. Unlabelled rows
cannot be used for training, cannot be used for evaluation, and cannot answer any question in the
stage 0 audit. They would cost 639 MB and add a second, differently shaped dataset to keep
consistent for no measurable benefit. The competition leaderboard is not a goal of this repo, so
the one thing the test files are good for does not apply.

**Download the two training files only.** Chosen. 590,540 labelled rows over 182 relative days
is enough to build the chronological split, the entity analysis, the feature layer and the drift
work, and every one of those needs labels.

**Commit the CSVs, or a sample of them, for reproducibility.** Rejected on licensing before
convenience. The data is Vesta-provided under the competition rules and redistributing it, in
whole or as a sample, is not the author's call to make. A committed sample would also be a second
source of truth for numbers that the audit already measures on the full file, which breaks the
rule of exactly one source per number.

## Decision

Ingest `train_transaction.csv` and `train_identity.csv` only. Ship `scripts/download_data.sh`,
which fetches those two files by name and nothing else, and gitignore `data/` entirely. Record
the sha256 of both files inside `reports/audit.json` so a reader can confirm they audited the
same bytes.

Identity is kept as a separate table and joined on TransactionID rather than merged into one
wide file, because coverage is partial and label-dependent (below) and a merge would hide that.

## Evidence

- **Artifact:** reports/audit.json, sections `inputs`, `shape`, `identity_join`.
- **Command:** `make data && make audit`
- Downloaded sizes match the listing exactly: train_transaction.csv 683,351,067 bytes,
  sha256 3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642; train_identity.csv
  26,529,680 bytes, sha256 b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c.
- train_transaction.csv holds 590,540 rows and 394 columns, with 20,663 fraud rows, a rate of
  0.03499. train_identity.csv holds 144,233 rows and 41 columns.
- Identity join coverage is 0.2442 overall, 0.5477 on fraud rows and 0.2332 on non-fraud rows.
  Every TransactionID in train_identity.csv is present in train_transaction.csv (0 unmatched).

## Consequences

- Identity presence is more than twice as common on fraud rows as on non-fraud rows. Treating a
  missing identity join as "no information" would throw away signal, and any imputation of the
  identity columns has to keep the missingness itself visible to the model. This is measured
  here and picked up properly in stage 2.
- Three quarters of rows have no identity attributes, so any feature built from them is absent
  for most traffic and the serving path must handle that as a normal case.
- Nothing in this repository can be reproduced without Kaggle credentials and acceptance of the
  competition rules. That is stated in the download script and belongs in the README.
- Because the data is not committed, tests that read the CSVs are marked `needs_data` and are
  deselected in CI. The committed artifact reports/audit.json is what CI checks instead, through
  tests/test_audit_report.py.
