Stage 3 writes four JSON artifacts here, plus `reports/column_plan.json` and
`reports/encoding_spec.json` one level up. Regenerate the lot with `make prep`; regenerate one
with `.venv/bin/python scripts/prepare.py --sections <name>`.

| File | Holds |
| --- | --- |
| ../column_plan.json | keep or drop for each of the 434 raw columns, the reason, and the stage 2 artifact behind it |
| ../encoding_spec.json | the four encoders, the label-lag and smoothing sweeps, the entity target encoding measurement |
| transforms.json | D-column PSI before and after normalisation, the amount evidence, the clipping cost table |
| missing_policy.json | the per-group missing value policy, and what each library does with a null, measured |
| v_reduction.json | three V-column reduction strategies and the unreduced baseline, with seed variance |
| schema.json | the declared contract, the validation result for all three splits, the determinism check |

Every file carries the command that regenerates it, the git commit it was written at, and the
train row count it saw. Every fitted object behind these numbers was fitted on the train split
through `assert_train_only`. `tests/test_prep_artifacts.py` checks the internal consistency of
all six on every push, without needing the data.

The sections have an order and it is the order in the table. `column_plan` runs first because
every other section reads the plan rather than re-deciding it, and `schema` runs last because it
reads the three measured choices the other sections made: which D columns are normalised, which
V-reduction strategy ships, and the label lag and smoothing the encoder is fitted at.
