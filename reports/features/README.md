Stage 4 writes three JSON artifacts here, plus `reports/feature_summary.json` one level up.
Regenerate the lot with `make features`; regenerate one with
`.venv/bin/python scripts/features.py --sections <name>`.

| File | Holds |
| --- | --- |
| ../feature_summary.json | every feature with its family, grain, window, serving state and null rate per split, plus drift, coverage and the state footprint |
| velocity_vs_c.json | this repo's point-in-time velocity against the native C columns, at the same grain |
| merchant_proxies.json | the column inventory screened for a merchant identifier, and the proxies used instead |
| duplicate_content.json | the causal duplicate-content features against stage 2's own grouping |

Every file carries the command that regenerates it, the git commit it was written at, and the
train row count it saw.

One thing about these artifacts differs from stage 3's and is worth stating rather than leaving to
be noticed. The features are computed over the **whole** time-sorted stream, train and later
windows together, and that is correct rather than tolerated: every feature reads only rows
strictly before its own timestamp, which is exactly what an authorisation endpoint has. A
validation row reads earlier training rows, as it would in production, and a training row cannot
read a validation row because validation rows are later in time. The only fitted object in the
stage is the dist1 bucket edges, and that goes through `assert_train_only` like everything in
stage 3.

`tests/test_causality.py` is what makes that claim checkable rather than asserted: it recomputes
every feature by brute force over `TransactionDT < t` on a sample, and it asserts that truncating
the stream at any time leaves every feature bit for bit unchanged on the rows before the cut.
`tests/test_feature_artifacts.py` checks the internal consistency of all four files on every push,
without needing the data, including that the numbers quoted from stage 2 and stage 3 are the
numbers those stages measured.

`reports/feature_summary.json`, block `running_state`, is the list stage 8 reads to know what the
online store must hold. There are three keyspaces, not one: the ADR 0001 entity, the card, and the
content group behind the duplicate features.
