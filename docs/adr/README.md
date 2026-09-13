# Decision records

One record per non-obvious decision, immutable once accepted; a changed mind is a new record
that supersedes the old. Each carries Context, Options considered (at least two, each with why),
Decision, Evidence (an artifact and the command that regenerates it, or the words "judgement,
not measurement") and Consequences. This index is append-only.

| ADR | Stage | Decision |
| --- | --- | --- |
| [0001](0001-entity-key.md) | 0 | Entity key |
| [0002](0002-split-strategy.md) | 0 | Split strategy and boundaries |
| [0003](0003-data-scope.md) | 0 | Data scope |
| [0004](0004-tooling-baseline.md) | 0 | Tooling baseline |
| [0005](0005-memory-and-typing.md) | 1 | Memory and typing strategy for the joined frame |
| [0006](0006-train-only-enforcement.md) | 1 | Train-only enforcement as a runtime guard |
| [0007](0007-eda-on-train-only.md) | 2 | The exploratory analysis is computed on the train split |
| [0008](0008-spearman-over-pearson.md) | 2 | Rank correlation, and a screen-then-recompute implementation |
| [0009](0009-time-consistency-screen.md) | 2 | Time consistency as a feature screen |
| [0010](0010-metric-implications-of-class-balance.md) | 2 | What the measured class balance allows and forbids |
| [0011](0011-column-plan.md) | 3 | The column plan and the evidence each drop category requires |
| [0012](0012-missing-values-left-missing.md) | 3 | Missing values are left missing for the tree path |
| [0013](0013-d-column-normalisation.md) | 3 | D-column normalisation is measured and not applied |
| [0014](0014-target-encoding-label-lag.md) | 3 | The target encoding label lag is 14 days |
| [0015](0015-entity-target-encoding-excluded.md) | 3 | The entity key is excluded from target encoding |
| [0016](0016-v-column-reduction.md) | 3 | V-column reduction by block mean, chosen on a tie-break |
| [0017](0017-schema-contract.md) | 3 | The schema contract, and what it deliberately does not catch |
| [0018](0018-causal-window-implementation.md) | 4 | Causal windows are searchsorted over the sorted stream, and ties are excluded |
| [0019](0019-merchant-risk-not-reproducible.md) | 4 | Merchant risk is not reproducible on this dataset |
| [0020](0020-uid-aggregation-not-shipped.md) | 4 | The published UID aggregation is not shipped |
| [0021](0021-own-velocity-alongside-native-c.md) | 4 | Own velocity features are kept alongside the native C columns |
| [0022](0022-duplicate-content-features-shipped.md) | 4 | The duplicate-content features are shipped, in their causal form |
| [0023](0023-geographic-deviation-keyed-on-the-card.md) | 4 | Geographic deviation is keyed on the card, not on the entity |
| [0024](0024-incremental-graph-construction.md) | 5 | The graph is built incrementally over the stream, and a full-dataset component computation is refused |
| [0025](0025-component-fraud-rate-dropped.md) | 5 | The component fraud rate is built with a lag, measured, and dropped |
| [0026](0026-pr-auc-as-the-primary-metric.md) | 6 | PR-AUC is the primary metric, and ROC-AUC is reported beside it |
| [0027](0027-imbalance-strategy.md) | 6 | No reweighting ships; class weighting and SMOTE were measured on the same split and lost |
| [0028](0028-model-family-shipped.md) | 6 | XGBoost ships, on the stage 3 frame with the C, D and V blocks and nothing from stages 4 or 5 |
| [0029](0029-cost-matrix-and-band-edges.md) | 6 | Two band edges derived from an assumed cost matrix |
| [0030](0030-productcd-as-the-segmentation-key.md) | 6 | ProductCD is the segmentation key, because there is one market to calibrate |
| [0031](0031-leakage-variant-scope.md) | 7 | Four leakage variants, one per rule the pipeline enforces |
| [0032](0032-simulated-label-latency.md) | 7 | Label latency is simulated as a deterministic N day cutoff, swept over five values |
| [0033](0033-two-phase-feature-store-contract.md) | 8 | The online store reads and writes in two separate calls |
| [0034](0034-redis-sorted-sets-and-the-in-process-fallback.md) | 8 | Redis sorted sets for the shared store, an in-process dict as the fallback, one set of definitions |
| [0035](0035-serving-inference-pinned-to-one-thread.md) | 8 | Serving inference is pinned to one thread, for the measured reason |
| [0036](0036-drift-and-decay-as-separate-jobs.md) | 9 | Input drift and performance decay are two jobs on two clocks, never one |
| [0037](0037-psi-alert-edges-calibrated-on-the-accepted-windows.md) | 9 | PSI bands are reported as the convention says; the alert is calibrated on this file |
| [0038](0038-promotion-margin-and-retrain-triggers.md) | 9 | The promotion margin and the retrain triggers come from stage 6's noise band |
| [0039](0039-demo-page-as-a-static-build-of-the-artifacts.md) | 9, addendum | The demo is a static page built from the committed artifacts, not a live service |
| [0040](0040-licence-and-data-redistribution.md) | 10 | MIT for the code, the dataset excluded and not redistributed |
| [0041](0041-readme-methodology-and-case-study.md) | 10 | The README is the proof, METHODOLOGY.md the detail, the case study the narrative |
