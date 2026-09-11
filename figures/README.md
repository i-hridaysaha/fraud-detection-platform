Figures for the README and the case study. Each is drawn by `make eda-figures`,
`make prep-figures` or `make features-figures` from one artifact under reports/ and from nothing
else, at 200 DPI, so a figure and the prose beside it quote the same numbers.

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
