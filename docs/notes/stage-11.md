# Stage 11: the case study

Nine stages produced the artifacts and stage 10 wrote the repository's front door. This stage
writes the page on the author's site, against the portfolio's own writing system and template,
and it is the one stage whose deliverable this repository cannot test. What the stage built here
is the machinery that makes the page checkable anyway: a numbers script the article's
verification table points at, a test that holds the script to the artifacts, and this note.

## What changed my mind

**The credential-strength check, before a word was written.** The writing system ranks pages by
hiring weight and forbids a personal project from outranking employer work in depth. Measured on
the live site with one `innerText` count per page: the employer-and-award page runs 2,122 words
and is an essay with one image; the four personal pages run 3,381 to 6,148 with the full
template; the live version of this page, written against the earlier build, runs 4,556. The
ranking is inverted, and the system's own remedy is to raise the strong page first and only then
trim the long ones. That remedy needs material this repository does not hold. So the page was
written at the bottom of the template's budget and held under the page it replaces, and the
inversion is reported rather than fixed.

**How much the template's scaffolding weighs.** The first complete draft ran 3,628 words of
visible body prose against a budget of 1,800 to 2,500, with six decisions each carrying the
template's four-part shape. Five trimming passes took it to 2,489: filler first, then depth
moved behind four toggles, then one decision (the entity key, whose numbers already lived in
Hidden Constraints) and one lesson (monitoring's two clocks, off the golden thread) removed. The
five remaining decisions still carry about 60 words each of labels, principle and takeaway
before a word of argument, and the whole page carries 4,563 visible words including five
alternatives tables and nine captions. The template's floor and the credential rule's ceiling
cannot both be met on this portfolio as it stands.

**The golden thread held, with a qualification the article had to carry.** The candidate thread
was that the techniques scoring highest on this data all read the future, evidenced by stage 7's
leakage delta. The random split is +0.2593 (+0.2378 to +0.2809) of test PR-AUC on its own and
nine tenths of the whole gap, so the thread stands. But two of the four shortcuts did not
inflate: the published post-processing deflates by 0.0554 on this window and the whole-file
encoder fit clears zero by row and not by card. The article says so in Results and in the
takeaway of Figure 6 rather than rounding the story up to "every shortcut leaks".

## Dead ends

- The card is not a crop of the banner, whatever the figure system's catalogue says. The six
  live cards share an HTML shell in `Wix/Homepage/covers-src/gen.py`, so the new card is a new
  `art_fraud()` in that file, exported with headless Chrome at 2400 by 1800, and the previous
  PNG is kept beside it. A kit-rendered card would have been the one card with a different
  treatment.
- The first architecture render put the monitoring band below the canvas and ran the feedback
  loop through two band labels; the first gate render ran its fourth row off the bottom; the
  arrow-of-time concept figure stacked its pill on its second chip. None of the three raised. The
  figure system's rule that every PNG is opened after rendering is the only thing that caught
  them.
- `gen.py` regenerates all seven card HTML files at once. The other six are byte-identical
  outputs of unchanged code, but the folder is not under version control, so only the mtimes say
  so.
- The Short Description came out at 41 words on the first count and 40 after one word went;
  the Key Insight at 43 and then 40. Both were counted by hand first and by `wc` second, and the
  hand count was wrong both times.

## Not established

- What the page reads like to the reader it is written for. It has not been read by one.
- Whether the list card's tag and figures, which present the project as employer work with
  production numbers, will be reconciled with the page's framing as an independent build on
  public data. That is a portfolio decision and is flagged in the article and the stage report.
- Whether the GitHub and demo links resolve on the day the page ships. The repository is private
  until stage 10 merges and the demo has no hosted URL yet.

## Verification

Every number in the article, by claim group, with its tag and its source. Keys are the names
`scripts/case_study_numbers.py` prints (`make case-study-numbers`); the artifact and the make
target that regenerates it follow. `scripts/verify_numbers.py --docs <article>` scanned 347
numeric tokens on the final draft and matched 339 against an artifact leaf; the eight it did not
match are the derived and session-measured rows below.

| Claim group | Tag | Source |
| --- | --- | --- |
| Rows 590,540; 20,663 fraud; rate 0.0350; 182 days; 434 joined columns | measured | `n_rows`, `n_fraud`, `fraud_rate`, `span_days`, `n_joined_columns`: `reports/split_summary.json`, `reports/audit.json`, `reports/column_plan.json`; `make split-summary`, `make audit`, `make prep` |
| Split: 413,378 train rows over days 1 to 120; 88,581 validation and test rows; test days 152 to 182; 3,083 test fraud; test rate 0.0348 | measured | `train_rows`, `train_days`, `val_rows`, `test_rows`, `test_days`, `n_test_fraud`, `test_fraud_rate`: `reports/split_summary.json`; `make split-summary` |
| Approve-everything accuracy 0.9652 | derived | `approve_all_accuracy`: one minus the test fraud rate |
| Entity key: 217,850 entities; purity 0.9664 against 0.8480; 0.1539 of `card1` values single-valued on start day | measured | `entities_key`, `purity_key`, `purity_card1`, `card1_single_start_day_share`: `reports/audit.json`; `make audit` |
| Entity overlap: 0.6598 of test rows unseen, fraud 0.0440 against 0.0170; 0.3402 seen by row | measured | `test_rows_unseen_share`, `test_fraud_rate_unseen`, `test_fraud_rate_seen`, `chrono_seen_entity_row_share`: `reports/split_summary.json`, `reports/leakage_delta.json`; `make split-summary`, `make leakage` |
| Random split population: 0.7410 seen | measured | `random_seen_entity_row_share`: `reports/leakage_delta.json` (`population.random.test`); `make leakage` |
| Leakage table, seven rows, all points, intervals and deltas; 186, 262 columns; 76 aggregate columns | measured | the `## leakage delta` table, `delta_a_by_row`, `delta_a_by_card`, `delta_b`, `delta_c`, `delta_d`, `delta_abc`, `delta_abcd`, `random_split_pr_auc`, `all_shortcuts_pr_auc`, `n_aggregate_columns`: `reports/leakage_delta.json`; `make leakage` |
| "Nine tenths of the gap"; "a third of the fraud" on 457 mixed entities | derived | 0.2593 over 0.2866 (`random_split_share_of_gap`); 1,090 of 3,083 from `reports/leakage_delta.json` (`postprocessing.diagnostic`); the 457 is measured there |
| Label latency: the five-row table; immature and excluded at 15; implied rate 0.0004 at 60 against observed 0.0348; recall 0.0032; 0.3 against 78.2 alerts a day | measured | the `## label latency` table, `latency_immature_15`, `latency_excluded_15`, `latency_immature_60_implied_rate`, `latency_observed_rate`, `latency_immature_60_alerts_per_day`, `latency_reference_alerts_per_day`: `reports/label_latency.json`; `make latency` |
| Latencies 15, 30, 45, 60, 90 days | assumed | the sweep's chosen values, `reports/label_latency.json` (`simulation.latency_days`) |
| Target encoding: gap +0.0812 at lag 0, +0.0077 at 14, below zero at 28; lag 14 | measured | `te_gap_lag_0`, `te_gap_lag_14`, `te_gap_lag_28`, `label_lag_days`: `reports/encoding_spec.json`; `make prep` |
| Entity encoding: 0.7835 validation; 0.9877 seen; 0.4915 unseen | measured | `entity_te_val_auc`, `entity_te_seen_auc`, `entity_te_unseen_auc`: `reports/encoding_spec.json`; `make prep` |
| 36 causal features; ablation moves -0.0052, +0.0029, -0.0076; 51 columns removed from 237 to 186 | measured, derived | `n_causal_features`, `causal_ablation_per_seed`, `n_default_columns`, `n_model_columns`, `n_columns_removed` (a subtraction): `reports/feature_summary.json`, `reports/ablations.json`, `reports/model_comparison.json`, `reports/leakage_delta.json`; `make features`, `make train`, `make leakage` |
| Parity: 134,339 rows, 40 features, difference 0.0 | measured | `parity_rows`, `parity_features`, `parity_max_difference`: `reports/parity.json`; `make parity` |
| Store: commit 0.90 ms on Redis against 0.039 in process | measured | `store_commit_ms_redis`, `store_commit_ms_memory`: `reports/latency.json`; `make serving-latency` |
| Request: median 37.6 ms; booster 0.081 ms; pin ratio 1.49 to 1.59; 25.2 requests per second | measured | `path_p50_ms`, `booster_p50_ms`, `peak_rps_one_worker`: `reports/latency.json` (`pinning` for the ratio), `loadtest/results.json`; `make serving-latency`, `make loadtest` |
| Cost bands: costs 10, 1, 0.5; edges 0.05 and 0.5; 2,500.1, 331.0, 46.9 rows a day; fraud shares 0.2530, 0.3831, 0.3639; rates 0.0101, 0.1159, 0.7770; cost per row 0.1492, 0.3480, 0.1729 | measured (the costs assumed) | `cost_missed_fraud`, `cost_false_decline`, `cost_review`, `band_low`, `band_high`, `band_*_per_day`, `band_*_fraud_share`, `band_review_fraud_rate`, `band_block_fraud_rate`, `cost_per_row_bands`, `cost_per_row_approve_all`, `cost_per_row_single_threshold`: `reports/operating_points.json`; `make train` |
| Operating point: threshold 0.2519, precision 0.6034, recall 0.4713, 78.2 alerts a day | measured | `threshold`, `test_precision`, `test_recall`, `alerts_per_day`: `reports/operating_points.json`; `make train` |
| Model comparison: the ten-row table; 9 models; validation 0.6198, 0.5745, 0.5373; shipped -0.0039 (-0.0084 to +0.0008) and +0.0328 (+0.0241 to +0.0418); card interval 0.5100 to 0.5805; 0.6048 against 0.6184 | measured | the `## model comparison` table, `n_models_compared`, `xgb_*_val_pr_auc`, `shipped_minus_default_xgboost`, `shipped_minus_best_forest`, `test_pr_auc_card_interval`, `search_*_val_pr_auc`: `reports/model_comparison.json`, `reports/metric_variance.json`, `reports/hyperparameter_search.json`; `make train` |
| Twelve stacks, three seeds | measured | `n_ablation_stacks`, `n_ablation_seeds`: `reports/ablations.json`; `make train` |
| Noise band 0.0592 by card; margin 0.06; tolerance 0.06; largest in-control drop 0.0510; headroom 0.0090; 30-day maturity | measured, derived (headroom), assumed (maturity) | `noise_half_width_by_card`, `promotion_margin`, `decay_tolerance`, `decay_largest_drop`, `decay_tolerance_headroom`, `maturity_days`: `reports/metric_variance.json`, `reports/monitoring/decay.json`, `reports/lifecycle_demo.json`; `make train`, `make monitor-decay`, `make lifecycle-demo` |
| Gate: drift flag at cycle 3, decay flag at cycle 8; four challengers; +0.0954 and +0.0621 promoted; +0.0424 refused under the margin; -0.0438 refused as not better | measured | `drift_flag_first_cycle`, `decay_flag_first_cycle`, `n_challengers`, `gate_promoted`, `gate_refused_under_margin`, `gate_refused_not_better`: `reports/lifecycle_demo.json`; `make lifecycle-demo` |
| Drift edges: 18.4 columns a week under the textbook band; false alert rate 0.0 on 10 weeks; 16 target encodings never alert | measured | `textbook_alerts_per_week`, `drift_false_alert_rate`, `drift_batches`: `reports/monitoring/drift.json` (`calibration.edges_raised` for the 16); `make monitor-drift` |
| Adversarial validation 1.0000, 0.9031, 0.8600 | measured | `adversarial_full_auc`, `adversarial_without_te_auc`, `adversarial_without_entity_and_clock_auc`: `reports/adversarial_validation.json`; `make adversarial` |
| Frame: 251 of 372 columns label-linked; 434 to 288 columns, 14 constant and 132 redundant; 15 D columns all worse normalised; 130 of 393 numeric columns move between windows; 59 flagged columns kept | measured | `n_label_linked_columns`, `n_columns_with_nulls`, `n_columns_kept`, `n_dropped_constant`, `n_dropped_redundant`, `d_columns`, `d_columns_psi_worsening`: `reports/eda/missingness.json`, `reports/column_plan.json`, `reports/prep/transforms.json`; the 130 of 393 is the count of per-column moves above 10 percent in `reports/eda/train_only.json` (docs/eda.md section 0) and the 59 is `time_consistency.n_flagged` in `reports/eda/temporal.json`; `make eda`, `make prep` |
| Repeat purchases: 1.333 on train; 0.814 and 0.883 on validation and test; hour window 2.865, 2.410, 2.019; "about 3.4 times"; six of ten deciles | measured, derived (3.4) | `dup_1h_ratio_by_split` and `reports/features/duplicate_content.json` (`causal_grouping_by_split`, `per_feature`); 0.10689 over 0.03161; `make features` |
| Segments: 207 against 1,297 review rows for the segment whose base rate moved | measured | `reports/operating_points.json` (`segments`); `make train` |
| "Six tenths" of the accepted level | derived | 0.3413 over 0.5451 from `reports/lifecycle_demo.json` and `reports/metric_variance.json` |
| No merchant column; billing country one value on 0.9901 of rows | measured | `n_merchant_name_matches`, `addr2_top_value_share`: `reports/features/merchant_proxies.json`, `reports/eda/univariate.json`; `make features`, `make eda` |
| 1,000 resamples; 95 percent intervals; seed 42 | measured, assumed | `n_boot`: `reports/metric_variance.json`; the level and the seed are choices in `config.py` |
| Hero geometry 2400 by 1000, safe box x 650 to 1750 and top 72 percent; card 2400 by 1800, 4 to 3 | external | `05 FIGURE SYSTEM.md`, measured there from the live Wix template |
| Page word counts 2,122; 3,381 to 6,148; 4,556; this page 2,489, 4,563, 1,247 | session-measured | live pages counted with one `innerText` script on 2026-09-13; this page by `count_words.py` |
| The first-place construction and its blog post | external | fetched and read in stage 4, quoted in ADR 0020 and ADR 0031 |
