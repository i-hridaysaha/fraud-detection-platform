# Verification

The stage 10 pass over the whole repository, run on 2026-09-13 on branch `stage/10-release`.
Four questions: does every number in the prose have an artifact and a command; was every
external citation fetched and read for this pass; does every decision record name its evidence
or say it has none; and does `make reproduce` on a clean clone give the numbers the prose quotes.
What failed is listed at the end, with what was done about it.

Every claim in this repository is one of four kinds: measured (a committed artifact plus the
command that regenerates it), derived (arithmetic on measured values, shown), assumed (an input
chosen here, labelled as one), or external (a source outside the repo, fetched and read in the
session that cites it). A claim that is none of these is not written.

## 1. Numbers

### 1.1 The README, one by one

Every table and inline number in `README.md` is produced by `scripts/readme_numbers.py` from the
artifacts below, and `tests/test_readme.py` asserts that the README carries exactly the script's
output. The test runs in CI on every push, on a clean checkout, without the data.

| README section | Numbers | Artifact | Regenerate with |
| --- | --- | --- | --- |
| Results, configuration | split rows, day ranges, test fraud rate 0.0348, 85,498 legitimate rows, accuracy 0.9652 | `reports/split_summary.json` (the last two derived from it: rows less fraud, one less the rate) | `make split-summary` |
| What each shortcut would have bought | the seven-row table, "nine tenths" (0.2593 over 0.2866, derived) | `reports/leakage_delta.json` (`variants`, `bootstrap.pr_auc_by_row`, `deltas.per_variant`) | `make leakage` |
| What late labels do | the five-row table, the implied rate 0.0004 at 60 days, the observed 0.0348 | `reports/label_latency.json` (`sweep`), `reports/split_summary.json` | `make latency` |
| Model comparison | the ten-row table | `reports/model_comparison.json` (`models`), `reports/metric_variance.json` (`by_row.per_model`), `reports/operating_points.json` (`primary_metric`) | `make train` |
| Model comparison, prose | +0.0328 (+0.0241 to +0.0418), -0.0039 (-0.0084 to +0.0008), 51 columns (237 less 186, derived), 0.5100 to 0.5805 | `reports/metric_variance.json` (`shipped_against_each`, negated to read shipped minus the other; `by_card.per_model.shipped`), `reports/model_comparison.json` (`n_columns`) | `make train` |
| Operating point | 0.2519, 0.6034, 0.4713, 0.5292, 78.2, band edges 0.05 and 0.5 | `reports/operating_points.json` (`tuned_vertex.test`, `band_edges`) | `make train` |
| Why this is non-trivial | 0.8480, 0.9664 | `reports/audit.json` (`entity_key_candidates.*.label_purity_multi_entities`) | `make audit` |
| Why this is non-trivial | 0.6666, 0.6598, 0.0440, 0.0170 | `reports/split_summary.json` (`entity_overlap.test`) | `make split-summary` |
| Why this is non-trivial | AUC 1.0000, identifiers and clocks | `reports/adversarial_validation.json` (`fits.full.auc.out_of_fold`, `entity_and_clock_share_of_top_10_gain` = 1.0) | `make adversarial` |
| Why this is non-trivial | 0.5477, 0.2332 | `reports/audit.json` (`identity_join`) | `make audit` |
| Why this is non-trivial | 14-day lag | `reports/encoding_spec.json` (`encoders.target.chosen_lag_days`) | `make prep` |
| Why this is non-trivial | 134,339 rows, 40 features, 0.0 | `reports/parity.json` (`stream.n_rows`, `backends.*`) | `make parity` |
| Data | 590,540, 394, 182 days, 144,233, 20,663, 0.0350 | `reports/split_summary.json` (`totals`), `reports/audit.json` (`shape`, `transaction_dt.span_days`, `identity_join.n_identity_rows`) | `make audit`, `make split-summary` |
| Data, excluded | 0.9999 | `reports/graph_summary.json` (`end_of_train_graph.giant_component.share_of_transactions`) | `make graph` |
| Data, excluded | -0.0052, +0.0029, -0.0076; -0.0049, -0.0015, -0.0079; 36 causal features | `reports/ablations.json` (`stacks[prepared+causal]`, `stacks[prepared+causal+graph]`, `against_reference.per_seed`), `reports/feature_summary.json` (`n_features_derived`) | `make train`, `make features` |
| Model | 186 columns, 0.6048 against 0.6184, -0.0663 to -0.0590, 0.2672 | `reports/leakage_delta.json` (`model.n_columns`), `reports/hyperparameter_search.json` (`final.tuned.val.pr_auc`, `final.default.val.pr_auc`), `reports/ablations.json` (`stacks[prepared+causal+graph|without=count]`), `reports/shap/shap_global.json` (`by_block.native_count.mean_abs_shap_share`) | `make train` |
| Evaluation | margin 0.06, half-width 0.0592 | `reports/lifecycle_demo.json` (`inputs.promotion_margin`), `reports/metric_variance.json` (`by_card.widest_paired_half_width`) | `make lifecycle-demo`, `make train` |
| Serving and monitoring | 37.6 ms, 0.081 ms | `reports/latency.json` (`summary.serving_path_whole_no_store_p50_ms.median`, `summary.serving_path_p50_ms.predict.median`) | `make serving-latency` |
| Serving and monitoring | 25.2, 62.1 | `loadtest/results.json` (`sweeps[].saturation.peak_throughput_rps`) | `make loadtest` (needs Redis) |
| Serving and monitoring | 10 weeks, 0.0, 18.4 | `reports/monitoring/drift.json` (`calibration.n_batches`, `in_control.leave_one_out.rate_any_alert`, `.mean_textbook_alert_band`) | `make monitor-drift` |
| Serving and monitoring | 0.0510, 0.06 | `reports/monitoring/decay.json` (`in_control.largest_drop`, `.decay_tolerance`) | `make monitor-decay` |
| Serving and monitoring | 14 cycles, 4 challengers, +0.0954 and +0.0621, +0.0424, 0.0 | `reports/lifecycle_demo.json` (`n_cycles`, `n_challengers`, `cycles[].challenger.gate`, `rollback`) | `make lifecycle-demo` |
| Limitations | 0.9901, 30 days | `reports/eda/univariate.json` (`numeric[addr2].top_value_share`), `reports/lifecycle_demo.json` (`inputs.maturity_days`) | `make eda`, `make lifecycle-demo` |
| Reproduce it | seed 42 | `config.SEED`; `tests/test_data_loader.py` refuses any other seed literal | `make test` |

Assumed inputs quoted in the README, labelled as such there: the cost matrix behind the bands,
the 30-day maturity, the 14-day lag grid, the latency sweep, the injected drift. External: the
NVIDIA Technical Blog post in section 2.

### 1.2 Every document, mechanically

`scripts/verify_numbers.py` (`make verify-numbers`) indexes every numeric leaf of every JSON
artifact under `reports/`, `models/` and `loadtest/` in the spellings the prose uses (one to
eight decimals, signed, thousands-separated, rounded to a whole number) and scans `README.md`,
`METHODOLOGY.md`, `CHANGELOG.md`, `docs/*.md`, `docs/adr/*.md` and `docs/notes/*.md`
for every token with a decimal point, a thousands separator or four or more digits. ADR numbers,
years and dates are skipped, and this file is not scanned.

Result on 2026-09-13: 5,280 tokens scanned, 66 unmatched after the one correction below. This
file is not scanned, since every unmatched token appears in it by construction. Every unmatched
token is one of the following, each sourced by hand:

| Tokens | Where | Kind | Source |
| --- | --- | --- | --- |
| 0.9652 | README | derived | 1 less the test fraud rate 0.03480 in `reports/split_summary.json` |
| 1.558, 1.084 | METHODOLOGY, docs/features.md | derived | quotients of the warm and cold rates in `reports/feature_summary.json` `coverage` |
| 244,197 | docs/eda.md, docs/prep.md, ADR 0008, ADR 0011 | derived | 291,163 rows with D4 present (`reports/eda/univariate.json`) less the 46,966 rows with both D4 and D12 present (`reports/eda/correlation.json`) |
| 0.2210 | docs/eda.md | derived | 1 less the DeviceInfo missing rate 0.7790 in `reports/eda/univariate.json` |
| 1.425 | docs/features.md, ADR 0022 | derived | 0.04157 over 0.02917, `reports/eda/quality.json` |
| 1,462 | docs/graph.md | derived | the sum of `hubs_by_column.n_hubs` at the 0.0001 share, `reports/graph_summary.json` |
| 0.9756 | docs/graph.md, ADR 0025, notes/stage-05 | derived | 1 less 0.0244, `reports/graph_summary.json` |
| 0.9459, 0.9677 | docs/experiments.md, ADR 0020 | external | the leaderboard scores on the NVIDIA post, section 2 |
| 0.3334 | ADR 0002, ADR 0006 | derived | 1 less 0.6666, `reports/split_summary.json` |
| 6,080,314; 25,797,161; 613,194,934 | ADR 0003 | measured, not from a committed artifact | `.venv/bin/kaggle competitions files ieee-fraud-detection`, run 2026-09-10, recorded in the ADR |
| 100,000; 50,000; 500,000 | ADR 0005, ADR 0008, notes/stage-01 | assumed | the chunk sizes in `scripts/memory_profile.py` and `fraud_platform.eda`, and the widening of the deliberate break |
| 42,577,454; 2,362,160; 87,274 | ADR 0005, notes/stage-01 | derived | differences and products of frame-byte leaves in `reports/memory_profile.json` |
| 2.2737, 5.879, 1.73 | ADR 0005, ADR 0008, notes/stage-01, notes/stage-02 | measured | mantissas of 2.2737e-13, 5.879e-08 and 1.73e-04, leaves in `reports/memory_profile.json` and `reports/eda/correlation.json` written in scientific notation |
| 662.85, 662.8500000000001 | ADR 0018, notes/stage-04 | measured, not from a committed artifact | a pytest failure output recorded verbatim in the stage 4 session |
| 68.96 | ADR 0022 | illustrative | a made-up amount in a sentence about float equality, labelled as such there |
| 21,750; 44,290; 44,291 | ADR 0026 | derived | 88,581 halved, times the artifact's recall and false-positive rate, labelled hypothetical |
| 27.43 | ADR 0027 | derived | 398,840 over 14,538, the negatives per positive on train; 398,840 is 413,378 less 14,538 in `reports/split_summary.json` |
| 0.2104 | ADR 0028 | derived | 0.6198 less 0.4094, `reports/model_comparison.json` |
| 3,506 | ADR 0041 | measured, not from a committed artifact | `wc -l docs/*.md` at the start of stage 10 |
| 156365; 124,020; 10423792; 9951526 | notes/stage-01 | measured, not from a committed artifact | the deliberate-break test outputs recorded in the stage 1 session |
| 3187; 4672; 8182; 11802; 15647; 16,129 | notes/stage-08 | measured, not from a committed artifact | the deliberate-break test outputs recorded in the stage 8 session |
| 0.2330; 244,197; 5,209; 5,271 | notes/stage-10 | this stage's own accounting | the corrected error, the derived difference above, and this scan's counts before the note was added (5,271 scanned, 62 unmatched; the note adds nine tokens and these four) |

The one correction: `docs/eda.md` quoted the `id_33` top-value share as 0.2330 where
`reports/eda/univariate.json` holds 0.23285, which rounds to 0.2329. Corrected. Nothing else in
the whole-repository scan was wrong; the earlier stages' own passes had caught the rest.

The check is necessary, not sufficient: a token that matches some leaf somewhere is not thereby
matched to the right leaf. The README is held to the right leaves by `tests/test_readme.py`;
METHODOLOGY.md was written from the stage documents and the build log, both of which were
matched leaf by leaf at their own stage close, and every one of its numbers appears in the table
above or in a stage document's verification table.

### 1.3 Figures

Every figure under `figures/` was re-rendered from the artifacts by `make figures` on 2026-09-13
and compared byte for byte with the committed file: all 24 identical before this stage's two
changes. The figure scripts carry no measured number as a literal; the only numeric literals are
layout constants, colours, the 0.10 and 0.25 PSI conventions (recorded as conventions in
`reports/eda/temporal.json`) and the window and day constants that `config.py` defines.

Two figures changed in this stage, then were re-rendered and inspected:

- `figures/ablations.png` gained a second panel drawing, per stack and per seed, the paired
  bootstrap interval of the validation PR-AUC difference against its reference, from
  `reports/ablations.json` `against_reference.per_seed`. Before this stage the figure drew the
  three seeds' points and the mean with no interval, and the ship rule reads the interval.
- `figures/duplicate_content.png`: the right panel's legend sat on the test-window marker; moved
  to the upper right.

Every model comparison drawn carries a bootstrap interval on PR-AUC: `model_comparison.png`,
`ablations.png`, `leakage_delta.png`, `label_latency.png`, `decay_in_control.png`,
`lifecycle_demo.png`. `v_reduction.png` and `time_consistency.png` draw single-feature or probe
ROC-AUC, not the primary metric, and their artifacts hold seed ranges rather than resamples; they
draw what the artifact holds.

All 24 figures were opened and read after rendering. `docs/architecture.svg` was drawn by hand,
rendered in a browser and read; `tests/test_readme.py` asserts that its fourteen blocks are all
named in the README's Approach section, that exactly one block (the model registry) carries the
emphasis colour, and that the file resolves.

## 2. External citations

Every URL cited anywhere in the repository, and whether it was fetched and read for this pass.

| Source | Cited in | Fetched 2026-09-13 | What was read |
| --- | --- | --- | --- |
| https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/ | README (prior work), METHODOLOGY, ADR 0020, ADR 0031, docs/experiments.md | yes, full text | authors and date (Carol McDonald and Chris Deotte, 26 January 2021); the UID line `X_train['UID'] = X_train.card1_addr1.astype(str)+'_'+np.floor(X_train.day-X_train.D1).astype(str)`; the time-consistency description (one feature, trained on the first month, tested on the last month of the training data); the UID aggregate list; public 0.9677 and private 0.9459, first place |
| https://spdx.org/licenses/MIT.txt | LICENSE, ADR 0040 | yes, full text | the MIT licence text, copied verbatim |
| https://keepachangelog.com/en/1.1.0/ | CHANGELOG.md | yes | the format: one section per version, newest first, ISO dates, Added / Changed / Deprecated / Removed / Fixed / Security, an Unreleased section, comparison links |
| https://semver.org/spec/v2.0.0.html | CHANGELOG.md | not fetched | linked by the Keep a Changelog convention; nothing is quoted from it |
| https://www.hridaysaha.com/projects/fraud-detection-platform | README (case study) | yes | resolves to the case study page, title "Fraud Detection Platform" |
| Kaggle: the competition page, its rules, the discussion post 111308 and the notebook `cdeotte/xgb-fraud-with-magic-0-9600` | not cited as sources | fetched, returned a title and no body (client-side rendering), as in stages 0 and 4 | nothing; no claim in the repository rests on them, and the competition rules are described in LICENSE and ADR 0040 only as what the author accepted |

The ADRs of earlier stages that cite the NVIDIA post (0020, 0031) did so from fetches made when
they were written and were not edited; the post was re-fetched for this pass and the quoted
line, the list and the scores are as those ADRs state.

## 3. Decision records

All 41 records under `docs/adr/` were read for their Evidence section. Each either names a
committed artifact (a JSON file under `reports/`, a test module, or a configuration file) with
the command that regenerates or runs it, or says in those words that the decision is judgement
rather than measurement. None has a missing or empty Evidence section.

| Evidence kind | Records |
| --- | --- |
| a JSON artifact under `reports/` (or `loadtest/`) and its command | 0001, 0002, 0003, 0005, 0007 to 0017, 0019, 0021 to 0038 |
| a test module or configuration file as the artifact, with its command | 0004 (pyproject, pre-commit, CI; `make lint && make test`), 0018 (`tests/test_causality.py`) |
| judgement, not measurement, with the measured inputs named | 0006, 0020, 0039, 0040, 0041 |

Records 0040 and 0041 are this stage's. The stage brief numbered them 0038 and 0039; both were
taken (0038 by stage 9, 0039 by the stage 9 addendum), the same shift every stage since 4 has
recorded.

## 4. Reproduce on a clean clone

Not run. `make reproduce` was not executed on a clean clone for this pass. The artifacts the
prose quotes are the committed outputs of the per-stage commands, run in this working tree.

## 5. Repository hygiene

Run on 2026-09-13 at the head of `stage/10-release`:

| Check | Result |
| --- | --- |
| `git log --format='%an <%ae>' \| sort -u` | one author, the repository owner |
| `grep -rn 'TODO(' src/ tests/ scripts/` | nothing |
| em dash grep over README.md, docs/, src/, tests/, scripts/, figures/ | one hit, the constant in `tests/test_readme.py` that the em dash test itself uses; `scripts/check_no_em_dash.sh` excludes nothing else and passes |
| the README word list in `tests/test_readme.py` over every markdown file, `src/`, `scripts/`, `tests/` | only the list itself |

## 6. What failed, and what was done

- `docs/eda.md`: one last-place rounding error (0.2330 for 0.2329), corrected.
- `figures/ablations.png` drew a model comparison with no interval on the primary metric; the
  interval panel was added.
- The case study on the author's site describes an earlier build and quotes numbers this
  repository does not produce (a PR-AUC of 0.5274 on 308 features under class weighting, against
  0.5451 on 186 columns with no reweighting here). It is linked from the README as the narrative
  and not quoted; bringing it into line is outside this repository and is recorded in
  `docs/notes/stage-10.md`.
- Nothing was deleted for want of a source. Two claims were rewritten before they were committed
  because their first drafts stated something the artifacts do not: that the immature-label
  regime is carried by the target encodings and the D columns (the stage 7 artifact says which
  columns carry it is not established), and that the C block's construction is "not published"
  (what is established is only that nothing fetched states it).
