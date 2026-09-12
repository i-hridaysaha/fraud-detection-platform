Stage 9 writes three JSON artifacts here, plus `reports/adversarial_validation.json` and
`reports/lifecycle_demo.json` one level up. Regenerate with `make monitor-drift`, `make
monitor-decay`, `make adversarial` and `make lifecycle-demo`; the last needs the raw files.

| File | Holds |
| --- | --- |
| drift_reference.json | the shipped model's 186 input columns and its scores as decile histograms of the training window, the in-control calibration on the weekly batches of val and test, and the alert edges |
| drift.json | the drift job over every in-control weekly batch, the textbook band against the calibrated rule, the leave-one-out false alert rate, and the cross-check against stage 2 |
| decay.json | the decay job over every in-control weekly batch once its labels matured: precision, recall, PR-AUC with its interval, and the drop against the stage 6 reference |
| ../adversarial_validation.json | train against test on the shipped stack, three fits, and the reading against stage 2 |
| ../lifecycle_demo.json | the whole loop on the val and test windows with a drift injected at day 141: every cycle's drift job, decay job, retrain decision, challenger, gate verdict and registry move, and the rollback drill |

The two jobs are separate on purpose. `drift.json` is written from the rows alone and
`monitoring.drift_report` raises if handed a frame that carries the label; `decay.json` is
written from labels the clock says are mature and `monitoring.performance_report` refuses a
batch before that. Nothing in either file is read from the other. `lifecycle.should_retrain`
is where they meet, and the demo artifact is the record of that meeting.

`tests/test_monitoring_artifacts.py` checks the files against each other and against stage 6
on every push: the edges against the calibration block, the decay reference against
`reports/metric_variance.json`, the promotion margin against the noise band, the gate verdicts
in the demo against their own numbers, and the rollback against the alias states it recorded.
