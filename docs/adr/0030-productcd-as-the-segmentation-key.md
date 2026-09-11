# 0030: ProductCD is the segmentation key, because there is one market to calibrate

- **Status:** accepted
- **Date:** 2026-09-11
- **Stage:** 6

## Context

The stage brief asked for per-segment thresholds, with `addr2` as the intended key: a billing
country, and a fraud operation calibrates per market because the base rate, the mix and the
tolerance for a false decline differ by market. Whether that is observable in this file is a
measurement, and stage 2 made it.

## Options considered

**`addr2`, per market.** Stage 2 measured the column on train at a top-value share of 0.9901
(`reports/eda/univariate.json`, `numeric.addr2`), 67 distinct values, and flagged it
`effectively_constant`. One value carries 0.9901 of the rows and the other 66 share the remaining
0.0099. A per-market calibration on that partition would be one market with a threshold and
sixty-six cells too small to tune anything in, and ADR 0010 already said what a rate quoted on a
cell of a few hundred rows means. Rejected: multi-market calibration is not observable here, and
saying so is the finding.

**No segmentation.** Rejected because the mechanic is the deliverable: a threshold and a
calibrator per partition, fitted on that partition's validation rows, is what a deployment does,
and the platform should carry it whichever key a deployment uses.

**`ProductCD`.** Chosen. Five levels, none in the rare tail, no nulls on train
(`reports/eda/univariate.json`, `categorical.ProductCD`: W 0.7207, C 0.1210, R 0.0729, H 0.0680,
S 0.0174 of train rows). It is not a market. It is the partition this file offers on which base
rates differ by an order of magnitude, and the same mechanic runs on it unchanged.

## Decision

The segmentation key is `config.SEGMENT_COLUMN = "ProductCD"`. For each level, the largest-F1
vertex and an isotonic calibrator are fitted on the level's validation rows, the cost-matrix edges
of ADR 0029 are applied to the level's calibrated probability, and the level's test rows are scored
with everything fixed. `addr2` is recorded as the key that could not be used and why.

## Evidence

`reports/operating_points.json`, regenerated with `make train` or
`.venv/bin/python scripts/train.py --sections thresholds`, block `segments`. Base rates carry
Wilson 95 percent intervals per `eda.wilson_interval`.

| Segment | Val rows | Val base rate | Test rows | Test base rate | Val PR-AUC | Test PR-AUC | Tuned threshold | Test precision | Test recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| W | 72,291 | 0.0205, 0.0195 to 0.0216 | 69,468 | 0.0186, 0.0176 to 0.0196 | 0.4027 | 0.2504 | 0.1344 | 0.2772 | 0.3248 |
| C | 9,125 | 0.1260, 0.1194 to 0.1330 | 9,388 | 0.1353, 0.1285 to 0.1423 | 0.8019 | 0.7664 | 0.2177 | 0.6980 | 0.7244 |
| R | 3,365 | 0.0475, 0.0409 to 0.0553 | 4,190 | 0.0506, 0.0444 to 0.0577 | 0.8487 | 0.8009 | 0.3833 | 0.7895 | 0.7075 |
| H | 2,350 | 0.0596, 0.0507 to 0.0699 | 2,562 | 0.0640, 0.0552 to 0.0742 | 0.7370 | 0.4982 | 0.4339 | 0.5772 | 0.4329 |
| S | 1,450 | 0.0738, 0.0614 to 0.0884 | 2,973 | 0.0484, 0.0413 to 0.0568 | 0.7312 | 0.5021 | 0.1750 | 0.5448 | 0.5069 |

**The segments differ enough to need their own thresholds.** The base rate runs from 0.0186 (W,
test) to 0.1353 (C, test), a factor of about seven, and the tuned thresholds from 0.1344 to
0.4339. W is 0.7207 of train rows and the segment the model ranks worst, test PR-AUC 0.2504; C and R
are above 0.76. A single file-wide threshold is a threshold for W.

**Where the base rate moved, the segment's own calibrator is the difference between a queue and
a flood.** S runs at 0.0738 on validation and 0.0484 on test with disjoint intervals. The global
calibrator, applied per segment, puts 1,297 of S's 2,973 test rows into the review band at a
realised cost of 0.4035 per row; S's own calibrator puts 207 there at 0.2619. On W, where the base
rate barely moved, the two calibrators realise 0.1207 and 0.1212. The per-segment and global band
reports for every level and both splits are in the artifact.

**What the smallest cells can support.** S has 1,450 validation rows and 107 fraud rows to fit a
threshold and a calibrator on, and its validation-to-test PR-AUC drop, 0.7312 to 0.5021, is the
second largest in the table after H's. ADR 0010's warning about sliced metrics applies to both.

## Consequences

- `config.SEGMENT_COLUMN` is the one place the key lives; a deployment with a real market column
  changes it there and reruns the thresholds section.
- The per-segment thresholds are reported, not shipped as the decisioning policy. The shipped
  policy is the global band pair of ADR 0029; a deployment choosing per-segment edges has the
  numbers to do so and the S row as the reason to.
- Nothing here says what `ProductCD` means. Its levels are labels in a file; the finding is that
  they partition the base rate, not what they are.
- Multi-market calibration stays not established on this dataset. A file with a billing-country
  column that is not 0.9901 one value is the prerequisite, and this repo does not have one.
