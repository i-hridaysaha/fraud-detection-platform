Stage 2 writes nine JSON artifacts here, one per section of docs/eda.md. Regenerate the lot with
`make eda`; regenerate one with `.venv/bin/python scripts/eda.py --sections <name>`.

| File | Holds |
| --- | --- |
| target.json | fraud rate overall, per split, per period, per category, and the class balance table |
| univariate.json | per column distribution for all 435 columns, and TransactionAmt in detail |
| missingness.json | missing rate per column, the blocks that share a mask, missingness against the label |
| correlation.json | Spearman screen and exact pairs, the V block reduction, near-duplicate pairs |
| bivariate.json | information value per feature with its missing-bin decomposition, fraud rate by decile |
| temporal.json | PSI against val and test, the time consistency screen, cold entities inside train |
| entities.json | transactions per entity, label purity, time between an entity's transactions |
| quality.json | duplicates, value range checks, email domains, DeviceInfo |
| train_only.json | what an EDA over the whole file would have said differently (ADR 0007) |

Every file carries the command that regenerates it, the git commit it was written at, and the
train row count it saw. Everything is computed on the train split; the three exceptions say so in
their own metadata. tests/test_eda_artifacts.py checks the internal consistency of all nine on
every push, without needing the data.
