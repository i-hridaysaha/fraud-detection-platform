"""What happens to a null, per column group, and the verification that the plan is possible.

Stage 2 measured that missingness here carries the label: 251 of the 372 columns with nulls have
a fraud rate that differs by at least 0.010 between their present and missing rows, at p below
1e-4. The largest gap in the frame belongs to a null and not to a value: D7 present runs at
0.1478 against 0.0274 when missing. Silent imputation deletes that.

So there is no global rule. There are three, and which one a column gets depends on what the
model reading it can do:

1. **A model that reads nulls gets nulls.** Nothing is filled. The split learns its own default
   direction and the information stays where stage 2 found it.
2. **A model that cannot gets an imputed value and an indicator.** The indicator is per
   missingness block, not per column, because stage 2 measured 24 blocks covering 392 of the 435
   columns and columns inside a block are null on identical rows: 251 indicators would carry the
   same information as 24 and cost seventeen times the width.
3. **Where the null has a meaning, it gets a sentinel that says so.** Never zero. Zero is a
   value these columns take: stage 2 measured 229 zero-inflated columns and D1 exactly zero on
   0.5013 of its non-null rows, so filling a null with zero merges two populations the analysis
   spent a section separating.

Rule 1 is a claim about libraries, and this module verifies it rather than asserting it.
`verify_native_nan_support` fits every candidate estimator on a frame with nulls and reports what
happened, twice: once to see whether the fit raises, and once on data where the null is the only
thing that predicts the label, to see whether the estimator learns a direction for it or merely
tolerates it. Those are different properties and only the second one matters here. The probe runs
on generated data, so it runs in CI and it re-runs whenever a pinned version changes. ADR 0012.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, transforms
from fraud_platform.data_loader import train_only

# --- choices ------------------------------------------------------------------------------

# Prefix for a block-level was-missing indicator. The suffix is the block's representative
# column, which is the alphabetically first member, so the name is stable across runs.
INDICATOR_PREFIX = "missing_block_"

# The identity join gets its own indicator regardless of the block structure. Stage 2 measured a
# fraud rate of 0.07312 on rows with an identity record against 0.02142 without, a factor of
# 3.41, and called it the single largest plain fact in the analysis. Reading that through 40
# columns going quiet is not the same as stating it.
IDENTITY_INDICATOR = "has_identity_record"

# id_01 and id_12 are the two columns measured to be null on precisely the rows where the join
# fails (missingness.json, identity_join.columns_whose_mask_is_exactly_the_join). The other 38
# carry additional nulls inside joined rows, so they cannot stand in for the join.
IDENTITY_JOIN_WITNESS = "id_01"

# A sentinel is placed this far below the smallest value the training window holds. The distance
# is arbitrary and the value is recorded per column in the spec; what matters is that it cannot
# collide with a real observation, and that the indicator beside it is what actually carries the
# meaning. See `SENTINEL_MEANING`.
SENTINEL_MARGIN = 1.0

SENTINEL_MEANING = (
    "the column measures days since an earlier event of a given kind. A null means the "
    "observation window holds no such earlier event for this card, so there is no origin to "
    "compute. The sentinel is placed below every value the training window contains, so it "
    "orders correctly for a tree and cannot be mistaken for an observation. It is not zero: "
    "stage 2 measured D1 exactly zero on 0.5013 of its non-null rows, which is a first observed "
    "transaction and a different thing entirely."
)

# The policy, per column group. `trees` and `complete_data` name what each kind of model gets,
# and `evidence` names the stage 2 measurement the row rests on.
GROUP_POLICY: dict[str, dict[str, str]] = {
    "count": {
        "trees": "no action: the block carries no nulls on train",
        "complete_data": "no action",
        "evidence": (
            "missingness.json blocks.all_blocks, the 63 column block at missing rate 0.0, which "
            "holds C1 to C14"
        ),
    },
    "timedelta": {
        "trees": "left missing",
        "complete_data": "sentinel below the train minimum, plus the block indicator",
        "evidence": (
            "missingness.json against_target: D7 runs 0.1478 present against 0.0274 missing, the "
            "largest gap measured in the frame, and D6, D8, D12, D13 and D14 all clear 0.07"
        ),
    },
    "vesta": {
        "trees": "left missing",
        "complete_data": "train median, plus the block indicator",
        "evidence": (
            "missingness.json blocks: 14 multi-column V blocks, so a per-block indicator carries "
            "what a per-column one would"
        ),
    },
    "identity": {
        "trees": "left missing",
        "complete_data": "train median for numeric, an explicit level for categorical, plus the "
        "block indicator and has_identity_record",
        "evidence": (
            "missingness.json identity_join: 0.07312 against 0.02142, a factor of 3.41, on a "
            "join that fails for 0.7341 of train rows"
        ),
    },
    "match": {
        "trees": "left missing",
        "complete_data": "an explicit category level, never a mode",
        "evidence": (
            "missingness.json against_target: M1 to M9 are label-linked, and univariate.json "
            "records M1 at 0.5380 missing with a top value share of 0.9999 on what remains"
        ),
    },
    "email": {
        "trees": "left missing",
        "complete_data": "an explicit category level, never a mode",
        "evidence": (
            "quality.json email_domains: R_emaildomain is missing on 0.7507 of rows and that "
            "state carries a measured rate of 0.02011"
        ),
    },
    "address": {
        "trees": "left missing",
        "complete_data": "train median, plus the block indicator",
        "evidence": (
            "bivariate.json: addr1 carries an information value of 0.4257, of which 0.3260 is the "
            "missing bin. Present rows alone give 0.0118. The absence is the feature"
        ),
    },
    "card": {
        "trees": "left missing",
        "complete_data": "train median for numeric, an explicit level for categorical",
        "evidence": "missingness.json missing_rate_by_column: card2 to card6 below 0.02 missing",
    },
    "distance": {
        "trees": "left missing",
        "complete_data": "train median, plus the block indicator",
        "evidence": (
            "bivariate.json deciles_reference.dist1: 255,775 rows missing at a fraud rate of "
            "0.0444 against a present-row trough of 0.0128"
        ),
    },
    "product": {
        "trees": "no action: ProductCD carries no nulls on train",
        "complete_data": "no action",
        "evidence": "univariate.json categorical, ProductCD at missing rate 0.0",
    },
    "amount": {
        "trees": "no action: TransactionAmt carries no nulls and the schema forbids one",
        "complete_data": "no action",
        "evidence": (
            "quality.json impossible_values: 0 negative amounts, 0 zero amounts, 0 non-finite "
            "amounts, minimum 0.251"
        ),
    },
    "meta": {
        "trees": "no action: the label, the identifier and the clock carry no nulls",
        "complete_data": "no action",
        "evidence": "quality.json impossible_values: n_rows_with_null_target and n_rows_with_null_time are 0",
    },
}


# --- library verification -------------------------------------------------------------------


def verify_native_nan_support(seed: int = config.SEED, n_rows: int = 4_000) -> dict[str, Any]:
    """Fit every candidate estimator on data with nulls and report what each one actually does.

    Two probes, because tolerating a null and learning from one are different behaviours and only
    the second justifies leaving 251 label-linked columns unimputed.

    `accepts_nan` fits on three columns with nulls scattered at random and records whether the
    fit raised. `learns_direction` fits on a single column that is pure noise wherever it is
    present and null on exactly the positive rows: an estimator that sends nulls to a learned
    side of the split scores an AUC of 1.0, and one that merely tolerates them scores 0.5. The
    label is not otherwise recoverable from the column, so the number is not ambiguous.

    Both probes run on generated data with a fixed seed, so this needs no CSV and runs wherever
    the tests run.
    """
    import lightgbm as lgb
    import sklearn
    import xgboost as xgb
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(seed)

    scattered_x = rng.normal(size=(n_rows, 3))
    scattered_y = (scattered_x[:, 0] + rng.normal(scale=0.5, size=n_rows) > 0).astype(int)
    scattered_x[rng.random((n_rows, 3)) < 0.3] = np.nan

    informative_y = (rng.random(n_rows) < 0.3).astype(int)
    informative_x = rng.normal(size=n_rows)
    informative_x[informative_y == 1] = np.nan
    informative_x = informative_x.reshape(-1, 1)

    builders: dict[str, Any] = {
        "lightgbm.LGBMClassifier": lambda: lgb.LGBMClassifier(
            n_estimators=20, random_state=seed, n_jobs=1, verbose=-1
        ),
        "xgboost.XGBClassifier": lambda: xgb.XGBClassifier(
            n_estimators=20, random_state=seed, n_jobs=1, verbosity=0
        ),
        "sklearn.HistGradientBoostingClassifier": lambda: HistGradientBoostingClassifier(
            max_iter=20, random_state=seed
        ),
        "sklearn.RandomForestClassifier": lambda: RandomForestClassifier(
            n_estimators=20, random_state=seed, n_jobs=1
        ),
        "sklearn.ExtraTreesClassifier": lambda: ExtraTreesClassifier(
            n_estimators=20, random_state=seed, n_jobs=1
        ),
        "sklearn.GradientBoostingClassifier": lambda: GradientBoostingClassifier(
            n_estimators=20, random_state=seed
        ),
        "sklearn.DecisionTreeClassifier": lambda: DecisionTreeClassifier(
            random_state=seed, max_depth=3
        ),
        "sklearn.LogisticRegression": lambda: LogisticRegression(max_iter=200),
    }

    results: list[dict[str, Any]] = []
    for name, builder in sorted(builders.items()):
        row: dict[str, Any] = {"estimator": name}
        try:
            builder().fit(scattered_x, scattered_y)
            row["accepts_nan"] = True
            row["error"] = None
        except Exception as error:
            row["accepts_nan"] = False
            row["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        if row["accepts_nan"]:
            model = builder().fit(informative_x, informative_y)
            scores = np.asarray(model.predict_proba(informative_x))[:, 1]
            auc = float(roc_auc_score(informative_y, scores))
            row["informative_null_auc"] = auc
            row["learns_direction"] = auc > 0.99
        else:
            row["informative_null_auc"] = None
            row["learns_direction"] = False
        results.append(row)

    return {
        "method": (
            "two probes on generated data. accepts_nan fits three columns with nulls at 0.3 and "
            "records whether the fit raises. informative_null_auc fits one column that is noise "
            "where present and null on exactly the positive rows, so an estimator that learns a "
            "direction for the null scores 1.0 and one that only tolerates it scores 0.5."
        ),
        "seed": seed,
        "n_rows": n_rows,
        "versions": {
            "lightgbm": lgb.__version__,
            "xgboost": xgb.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "estimators": results,
        "n_accepting_nan": sum(1 for r in results if r["accepts_nan"]),
        "n_learning_direction": sum(1 for r in results if r["learns_direction"]),
    }


# --- indicators --------------------------------------------------------------------------------


def indicator_name(representative: str) -> str:
    return f"{INDICATOR_PREFIX}{representative}"


def missing_indicator_spec(
    blocks: Sequence[Mapping[str, Any]], keep: Sequence[str]
) -> dict[str, Any]:
    """One indicator per missingness block that still has a column in the frame and any nulls.

    Blocks come from missingness.json, where they are defined by an identical null mask row for
    row. That definition is what makes one indicator per block correct rather than merely
    cheaper: inside a block the per-column indicators would be the same vector.

    A block whose columns were all dropped by the column plan gets no indicator. A block with no
    nulls gets none either, because a constant indicator is a wasted column.
    """
    kept = set(keep)
    entries: list[dict[str, Any]] = []
    for block in blocks:
        columns = sorted(c for c in block["columns"] if c in kept)
        if not columns or float(block["missing_rate"]) == 0.0:
            continue
        representative = columns[0]
        entries.append(
            {
                "indicator": indicator_name(representative),
                "representative": representative,
                "n_columns_covered": len(columns),
                "columns": columns,
                "missing_rate_train": float(block["missing_rate"]),
            }
        )
    entries.sort(key=lambda e: e["indicator"])
    return {
        "definition": (
            "one indicator per missingness block, computed from the block's representative "
            "column. Blocks are groups of columns whose null mask is identical row for row "
            "(missingness.json blocks.definition), so every member gives the same indicator."
        ),
        "n_indicators": len(entries),
        "n_columns_covered": sum(e["n_columns_covered"] for e in entries),
        "identity_indicator": {
            "name": IDENTITY_INDICATOR,
            "witness_column": IDENTITY_JOIN_WITNESS,
            "why": (
                "missingness.json identity_join measured the join at a fraud rate of 0.07312 "
                "against 0.02142, and named id_01 and id_12 as the two columns null on precisely "
                "the rows where the join fails"
            ),
        },
        "indicators": entries,
    }


def add_missing_indicators(frame: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    """Attach the block indicators and the identity-join indicator. Idempotent."""
    new: dict[str, Any] = {}
    for entry in spec["indicators"]:
        representative = entry["representative"]
        if representative not in frame.columns:
            continue
        new[entry["indicator"]] = frame[representative].isna().astype("int8")
    witness = spec["identity_indicator"]["witness_column"]
    if witness in frame.columns:
        new[spec["identity_indicator"]["name"]] = frame[witness].notna().astype("int8")
    return transforms.attach(frame, new)


# --- imputation for the models that need it -------------------------------------------------


@train_only
def fit_imputer(
    frame: pd.DataFrame,
    median_columns: Sequence[str],
    sentinel_columns: Sequence[str],
) -> dict[str, Any]:
    """Fill values for a complete-data pipeline: a train median, or a documented sentinel.

    Two kinds of column and they are not interchangeable. A median stands in for a value that
    exists and was not recorded. A sentinel stands in for a value that does not exist, which is
    what a delta with no earlier event to measure from is, and putting a median there would
    invent an event.
    """
    medians: dict[str, float] = {}
    for column in sorted(set(median_columns)):
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        medians[column] = float(np.median(finite))

    sentinels: dict[str, dict[str, Any]] = {}
    for column in sorted(set(sentinel_columns)):
        if column not in frame.columns:
            continue
        observed = frame[column].to_numpy(dtype="float64")
        finite = observed[np.isfinite(observed)]
        if finite.size == 0:
            continue
        minimum = float(finite.min())
        sentinels[column] = {
            "sentinel": minimum - SENTINEL_MARGIN,
            "train_min": minimum,
            "margin": SENTINEL_MARGIN,
        }

    return {
        "n_train_rows": len(frame),
        "sentinel_meaning": SENTINEL_MEANING,
        "n_median_columns": len(medians),
        "n_sentinel_columns": len(sentinels),
        "medians": medians,
        "sentinels": sentinels,
    }


def apply_imputer(frame: pd.DataFrame, fitted: Mapping[str, Any]) -> pd.DataFrame:
    """Fill with the fitted values. Idempotent, and it never touches a column it was not fitted on.

    Run `add_missing_indicators` before this, not after: once the fill has happened the null mask
    is gone and the indicator would be all zeros.
    """
    out = frame.copy()
    for column, value in fitted["medians"].items():
        if column in out.columns:
            out[column] = out[column].fillna(value)
    for column, entry in fitted["sentinels"].items():
        if column in out.columns:
            out[column] = out[column].fillna(entry["sentinel"])
    return out
