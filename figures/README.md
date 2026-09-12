Figures for the README and the case study. Each is drawn by `make eda-figures`,
`make prep-figures`, `make features-figures`, `make graph-figures`, `make train-figures`,
`make experiment-figures` or `make monitoring-figures` from one artifact under
reports/ and from nothing else, at 200 DPI, so a figure and the prose beside it quote the same
numbers.

| File | Source | Shows |
| --- | --- | --- |
| fraud_rate_over_time.png | reports/eda/target.json | daily and weekly fraud rate over the train span |
| amount_distribution.png | reports/eda/univariate.json | TransactionAmt before and after log1p |
| missingness_blocks.png | reports/eda/missingness.json | which column blocks go missing together |
| fraud_rate_by_decile.png | reports/eda/bivariate.json | shape of the relationship for eight columns |
| psi_train_vs_test.png | reports/eda/temporal.json | distribution drift per feature |
| time_consistency.png | reports/eda/temporal.json | early window AUC against late window AUC |
| d_column_psi.png | reports/prep/transforms.json | D column PSI and validation AUC, raw against normalised |
| target_encoding_lag.png | reports/encoding_spec.json | what the label lag costs and what it removes |
| v_reduction.png | reports/prep/v_reduction.json | column count against validation AUC for four strategies |
| velocity_vs_native_c.png | reports/features/velocity_vs_c.json | this repo's point-in-time velocity against the native C columns |
| feature_coverage.png | reports/feature_summary.json | what a point-in-time entity feature covers against what a train-fitted one does |
| duplicate_content.png | reports/features/duplicate_content.json | fraud rate by time since the last identical purchase |
| graph_components.png | reports/graph_summary.json | the hub sweep of the end-of-train graph, and fraud rate by decile of the point-in-time component size |
| model_comparison.png | reports/metric_variance.json | test PR-AUC per model with its paired-bootstrap interval, and the shipped model's paired difference against each |
| ablations.png | reports/ablations.json | validation and test PR-AUC per feature stack, three seeds each |
| threshold_curve.png | reports/threshold_curve.json | precision, recall, F1 and alerts per day against the calibrated threshold on validation, band edges marked |
| calibration.png | reports/operating_points.json | observed fraud rate against predicted probability on test, raw score and calibrated |
| shap_global.png | reports/shap/shap_global.json | the twenty largest mean absolute SHAP values and the share per block |
| leakage_delta.png | reports/leakage_delta.json | test PR-AUC per leakage variant with its interval, and each variant's difference against the causal baseline |
| label_latency.png | reports/label_latency.json | test PR-AUC and the implied fraud rate against the simulated label latency, immature and excluded treatments |
| drift_calibration.png | reports/monitoring/drift.json | columns the textbook 0.20 band puts in alert on each in-control week against the calibrated rule with the week held out, and the score PSI per week against its edge |
| decay_in_control.png | reports/monitoring/decay.json | the shipped model's PR-AUC on each in-control weekly batch with its interval, the reference it was accepted at and the decay tolerance |
| adversarial_validation.png | reports/adversarial_validation.json | the twenty columns that best separate train from test once the target encodings are set aside, who-or-when against what |
| lifecycle_demo.png | reports/lifecycle_demo.json | the drift job per cycle, the decay job per matured batch, and the gate's verdicts on the injected drift |
