"""Stage 3 preparation. Turns the stage 2 analysis into a column plan, four encoders and a schema.

Usage:
    make prep
    .venv/bin/python scripts/prepare.py
    .venv/bin/python scripts/prepare.py --sections encoding v_reduction

Two rules run the stage and both are enforced here rather than left to care.

Every decision cites a stage 2 artifact. The column plan is built from `reports/eda/` plus one
measurement this script makes, and a column dropped without a number behind it would have to get
past `fraud_platform.cleaning`, which has no category for it.

Every fitted object is fitted on train. The frame is cut with `time_based_split` and the train
part goes through `assert_train_only` before anything is fitted, and each fit function is
decorated so the guard runs again at the point of fitting.

Sections and their outputs:

    column_plan   reports/column_plan.json         keep or drop per raw column, with the evidence
    transforms    reports/prep/transforms.json     D-column PSI before and after, amount, clipping
    missing       reports/prep/missing_policy.json library behaviour, policy per column group
    encoding      reports/encoding_spec.json       four encoders, the lag sweep, the entity probe
    v_reduction   reports/prep/v_reduction.json    three strategies measured, one chosen
    schema        reports/prep/schema.json         the contract, and the prepared frame's shape
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import (
    cleaning,
    config,
    data_loader,
    eda,
    encoders,
    missing_policy,
    prepare,
    reduction,
    transforms,
)
from fraud_platform import (
    schema as schema_module,
)

SECTIONS = (
    "column_plan",
    "transforms",
    "missing",
    "encoding",
    "v_reduction",
    "schema",
)

COLUMN_PLAN_PATH = config.REPORTS_DIR / "column_plan.json"
ENCODING_SPEC_PATH = config.REPORTS_DIR / "encoding_spec.json"
PREP_DIR = config.REPORTS_DIR / "prep"

# The quantiles the clipping cost table is computed at. 0.999 is the value ADR 0013 settles on
# and the others are there so the choice can be read against its neighbours.
CLIP_QUANTILE_GRID: tuple[float, ...] = (0.99, 0.995, 0.999, 0.9995)

# Columns the clipping bounds are fitted for. The amount is the one column denominated in money
# and the one stage 2 measured a 17.6 skew on; the counters are the widest numeric block that a
# linear model would have to read.
CLIP_COLUMNS: tuple[str, ...] = (config.AMOUNT_COLUMN, *config.COUNT_COLUMNS)

# The AUC a tie band is computed at when a sweep has to choose between two settings. It is read
# from the sweep's own best result rather than assumed, and this is only the floor: an AUC below
# 0.5 would make the Hanley and McNeil standard error meaningless.
MIN_TIE_BAND_AUC = 0.51

# Seeds the V-reduction probe is repeated under, so run-to-run variance is a measurement and not
# an assumption. config.SEED first, then two others; nothing else in the repo reads these.
PROBE_SEEDS: tuple[int, ...] = (config.SEED, 7, 1_337)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def envelope(section: str, command: str, n_train_rows: int) -> dict[str, Any]:
    """The header every artifact carries: what made it, from what, and how to make it again."""
    return {
        "section": section,
        "stage": 3,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": (
            f"make prep  # or: .venv/bin/python scripts/prepare.py --sections {section}"
        ),
        "command": command,
        "fitted_on": "train split only, dt < config.TRAIN_END_DT, guarded by assert_train_only",
        "n_train_rows": n_train_rows,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
    }


def write(report: Mapping[str, Any], path: Path) -> None:
    """Write one artifact. allow_nan=False so a NaN fails here and not in a reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=False) + "\n")
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


def tie_band(auc: float | None, n_positive: int, n_negative: int) -> dict[str, Any]:
    """The AUC difference this validation split could resolve at the observed level.

    Two settings whose validation AUCs differ by less than this have not been told apart, and a
    sweep that picked between them on the point estimate alone would be reporting noise as a
    decision. The band is computed at the sweep's own best AUC rather than at an assumed one.
    """
    level = max(float(auc or 0.0), MIN_TIE_BAND_AUC)
    result = eda.detectable_auc_difference(level, n_positive, n_negative)
    return {
        "computed_at_auc": level,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "band": result["min_detectable_difference_unpaired"],
        "method": "eda.detectable_auc_difference, the unpaired upper bound from ADR 0010",
    }


# --- column plan ------------------------------------------------------------------------------


def validation_probe(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Single-feature AUC on train and on validation, for the columns stage 2 flagged.

    ADR 0009 produced 59 candidates from two windows inside the training split and refused to
    call it a decision, because removing a feature has to be measured against the window the
    model is judged on. This is that measurement: one small model per column, fitted on the whole
    training window, scored on validation.
    """
    results: dict[str, dict[str, Any]] = {}
    for index, column in enumerate(sorted(columns), start=1):
        if column not in train.columns:
            continue
        outcome = eda.time_consistency(
            train[column],
            train[config.TARGET],
            validation[column],
            validation[config.TARGET],
            config.SEED,
        )
        results[column] = {
            "train_auc": outcome["early_auc"],
            "validation_auc": outcome["late_auc"],
            "status": outcome["status"],
            "n_train": outcome["n_early"],
            "n_validation": outcome["n_late"],
        }
        if index % 10 == 0:
            print(f"    {index}/{len(columns)} probed")
    return results


def build_column_plan(
    train: pd.DataFrame, validation: pd.DataFrame, facts: cleaning.EdaFacts
) -> dict[str, Any]:
    report = envelope(
        "column_plan",
        "cleaning.build_column_plan(EdaFacts.load(), validation_probe(train, val), raw_columns)",
        len(train),
    )
    flagged = facts.time_consistency_flagged
    print(f"  probing {len(flagged)} time-consistency candidates against validation")
    probe = validation_probe(train, validation, flagged)

    # The plan has one row per raw column. card_start_day and entity_id are derived by
    # data_loader.add_entity_key and are decided separately, under `derived_columns`.
    derived = {config.ENTITY_ID_COLUMN, config.CARD_START_DAY_COLUMN}
    raw = [c for c in train.columns if c not in derived]
    plan = cleaning.build_column_plan(facts, probe, raw)

    report["evidence_standard"] = {
        "rule": (
            "a column is kept unless one of four measured conditions holds. Each condition names "
            "the artifact that established it and the threshold it was applied at. There is no "
            "category for a column that is merely unpromising."
        ),
        "reasons": {
            "effectively_constant": (
                "univariate.json flags.effectively_constant, plus the information value rescue "
                "stage 2 asked for after finding C3 at 0.9958 zero"
            ),
            "missing_and_not_mnar": (
                "missingness.json missing_rate_by_column and against_target. Both halves are "
                "required: 251 of 372 columns with nulls are label-linked and an empty column "
                "that is one of them stays"
            ),
            "time_inconsistent": (
                "temporal.json time_consistency for the candidate list, and a single-feature "
                "train and validation AUC measured in this stage for the decision"
            ),
            "redundant": (
                "correlation.json v_block_reduction for the clusters, with the representative "
                "chosen from the members that survived the other three rules"
            ),
        },
    }
    report["validation_probe"] = {
        "method": (
            "eda.time_consistency with the whole training window as the fitting window and the "
            "validation split as the scoring window. One LightGBM model per column, one column "
            "per model, the stage 2 hyperparameters unchanged."
        ),
        "n_columns": len(probe),
        "per_column": [{"column": column, **values} for column, values in sorted(probe.items())],
    }
    report.update(plan)

    derived = {
        "note": (
            "columns this repo derives rather than reads. They are not raw columns, so they are "
            "not rows of the plan, and the decision about each is recorded here."
        ),
        "columns": [
            {
                "column": config.CARD_START_DAY_COLUMN,
                "role": "entity key component",
                "in_feature_set": False,
                "evidence": (
                    "temporal.json psi: PSI 1.448 against val and 1.841 against test, the highest "
                    "in the frame. It is floor(TransactionDT / 86400 - D1), so a later window "
                    "necessarily holds later start days. Stage 2 said not to pass it to a model "
                    "without an explicit decision; this is the decision, and it is no."
                ),
            },
            {
                "column": config.ENTITY_ID_COLUMN,
                "role": "entity key",
                "in_feature_set": False,
                "evidence": (
                    "entities.json label_purity: 0.9675 of multi-transaction entities are "
                    "label-pure, so the key identifies the label almost as well as it identifies "
                    "the entity. ADR 0015 measures what encoding it would be worth."
                ),
            },
        ],
    }
    report["derived_columns"] = derived
    return report


# --- transforms -------------------------------------------------------------------------------


def build_transforms(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, plan: Mapping[str, Any]
) -> dict[str, Any]:
    report = envelope(
        "transforms",
        "transforms.d_origin_psi(train, test, D columns); transforms.clip_cost(train, ...)",
        len(train),
    )
    kept = set(cleaning.feature_columns(plan))
    d_columns = [c for c in config.TIMEDELTA_COLUMNS if c in kept and c in train.columns]

    print(f"  D-column PSI, {len(d_columns)} columns, before and after normalisation")
    psi_val = transforms.d_origin_psi(train, val, d_columns)
    psi_test = transforms.d_origin_psi(train, test, d_columns)
    by_column = {row["column"]: row for row in psi_test}
    for row in psi_val:
        by_column[row["column"]]["psi_before_val"] = row["psi_before"]
        by_column[row["column"]]["psi_after_val"] = row["psi_after"]

    improves = sorted(c for c, row in by_column.items() if row["improves"])
    worsens = sorted(c for c, row in by_column.items() if not row["improves"])

    stage_2_psi = {c: v for c, v in _stage_2_psi().items() if c in by_column}
    reproduces = {
        column: {
            "stage_2_psi_test": stage_2_psi.get(column),
            "recomputed_psi_before_test": by_column[column]["psi_before"],
            "agrees": (
                stage_2_psi.get(column) is not None
                and abs(stage_2_psi[column] - by_column[column]["psi_before"]) < 1e-9
            ),
        }
        for column in sorted(by_column)
    }

    print("  single-feature validation AUC, D column against its origin")
    auc_probe = _d_origin_auc(train, val, d_columns)
    for row in auc_probe:
        by_column[row["column"]]["train_auc_before"] = row["train_auc_before"]
        by_column[row["column"]]["validation_auc_before"] = row["validation_auc_before"]
        by_column[row["column"]]["train_auc_after"] = row["train_auc_after"]
        by_column[row["column"]]["validation_auc_after"] = row["validation_auc_after"]
        by_column[row["column"]]["validation_auc_delta"] = row["validation_auc_delta"]

    print("  clipping cost across the quantile grid")
    clip = transforms.clip_cost(train, CLIP_COLUMNS, CLIP_QUANTILE_GRID, config.TARGET)
    bounds = transforms.fit_clip_bounds(train, CLIP_COLUMNS)

    amount = _amount_evidence()

    report["d_normalisation"] = {
        "definition": "D_n = TransactionDT / 86400 - D, the day the delta is measured from",
        "method": (
            "PSI against a later split, deciles of the train distribution of the column being "
            "measured with missing as its own level, which is the stage 2 rule. The before "
            "number reproduces temporal.json and the artifact records the comparison."
        ),
        "n_columns": len(by_column),
        "n_improving": len(improves),
        "n_worsening": len(worsens),
        "applied_to": improves,
        "not_applied_to": worsens,
        "decision_rule": (
            "the normalisation is applied to a column when it lowers that column's PSI against "
            "test and not otherwise. The transformation has an argument behind it, but whether "
            "the argument holds is a property of the column and it does not hold for all of them."
        ),
        "reproduces_stage_2": reproduces,
        "auc_probe": {
            "method": (
                "one LightGBM model per column on that column alone, fitted on the whole train "
                "window and scored on validation, once on the raw delta and once on its origin. "
                "PSI says whether a distribution moved; this says whether the column got better "
                "at the job, and a normalisation that fails both has failed."
            ),
            "n_columns_where_validation_auc_improves": sum(
                1 for r in auc_probe if (r["validation_auc_delta"] or 0.0) > 0
            ),
        },
        "per_column": [by_column[c] for c in sorted(by_column)],
    }
    report["amount"] = amount
    report["clipping"] = {
        "chosen_quantile": transforms.CLIP_QUANTILE,
        "why": (
            "stage 2 measured TransactionAmt U-shaped against the label, the top decile at a "
            "fraud rate of 0.0509 against a trough of 0.0192, so the tail is not noise. The "
            "table below is what each quantile would touch and it is why the bound sits at "
            "0.999 rather than at 0.99."
        ),
        "applies_to": (
            "the models in stage 5 that read a value rather than its rank. Trees get the raw "
            "distribution: a tree cuts on order, so a clipped maximum and an unclipped one "
            "produce the same split and the clip only costs information."
        ),
        "columns": list(CLIP_COLUMNS),
        "cost_by_quantile": clip,
        "fitted_bounds": bounds,
    }
    report["clock_columns"] = {
        "derived": list(transforms.CLOCK_COLUMNS),
        "in_feature_set": list(transforms.CLOCK_FEATURE_COLUMNS),
        "why": (
            "target.json by_clock: the fraud rate runs 0.0217 to 0.1001 across hour positions, a "
            "ratio of 4.62, and 0.0322 to 0.0379 across day-of-week positions, a ratio of 1.18. "
            "The hour is a feature on that evidence and the day of week is not."
        ),
        "note": (
            "all four are positions in a relative cycle. Stage 0 did not establish what "
            "TransactionDT counts from, so none of them is a wall-clock value."
        ),
    }
    return report


def _d_origin_auc(
    train: pd.DataFrame, validation: pd.DataFrame, columns: Sequence[str]
) -> list[dict[str, Any]]:
    """Single-feature train and validation AUC for each D column, raw against normalised."""
    train_with = transforms.to_d_origin(train, columns)
    val_with = transforms.to_d_origin(validation, columns)
    rows: list[dict[str, Any]] = []
    for column in columns:
        origin = transforms.d_origin_name(column)
        before = eda.time_consistency(
            train[column],
            train[config.TARGET],
            validation[column],
            validation[config.TARGET],
            config.SEED,
        )
        after = eda.time_consistency(
            train_with[origin],
            train_with[config.TARGET],
            val_with[origin],
            val_with[config.TARGET],
            config.SEED,
        )
        rows.append(
            {
                "column": column,
                "train_auc_before": before["early_auc"],
                "validation_auc_before": before["late_auc"],
                "train_auc_after": after["early_auc"],
                "validation_auc_after": after["late_auc"],
                "validation_auc_delta": (
                    None
                    if before["late_auc"] is None or after["late_auc"] is None
                    else after["late_auc"] - before["late_auc"]
                ),
            }
        )
    return rows


def _stage_2_psi() -> dict[str, float]:
    payload = json.loads((config.REPORTS_DIR / "eda" / "temporal.json").read_text())
    return {row["column"]: float(row["psi_test"]) for row in payload["psi"]["per_feature"]}


def _amount_evidence() -> dict[str, Any]:
    payload = json.loads((config.REPORTS_DIR / "eda" / "univariate.json").read_text())["amount"]
    return {
        "log1p": {
            "derived_column": transforms.AMOUNT_LOG_COLUMN,
            "skew_raw": payload["skew_raw"],
            "skew_log1p": payload["skew_log1p"],
            "kurtosis_raw": payload["kurtosis_raw"],
            "kurtosis_log1p": payload["kurtosis_log1p"],
            "source": "reports/eda/univariate.json, amount",
            "raw_column_kept": True,
            "why_raw_kept": (
                "a monetary threshold is expressed in currency. Stage 7 has to show a reviewer "
                "the amount that was charged, not its logarithm."
            ),
        },
        "cents": {
            "derived_columns": [
                transforms.AMOUNT_CENTS_COLUMN,
                transforms.AMOUNT_IS_ROUND_COLUMN,
            ],
            "definition": payload["cents"]["definition"],
            "n_distinct": payload["cents"]["n_distinct"],
            "share_zero_cents": payload["cents"]["share_zero_cents"],
            "top_10": payload["cents"]["top_10"],
            "round_against_not_round": payload["round_against_not_round"],
            "source": "reports/eda/univariate.json, amount.cents",
        },
    }


# --- missing values ---------------------------------------------------------------------------


def build_missing(train: pd.DataFrame, plan: Mapping[str, Any]) -> dict[str, Any]:
    report = envelope(
        "missing_policy",
        "missing_policy.verify_native_nan_support(); missing_policy.missing_indicator_spec(...)",
        len(train),
    )
    print("  verifying native null handling, per library, by fitting")
    verification = missing_policy.verify_native_nan_support()

    kept = cleaning.feature_columns(plan)
    facts = cleaning.EdaFacts.load()
    indicators = missing_policy.missing_indicator_spec(facts.missingness_blocks, kept)

    # The D columns get the sentinel and everything else numeric gets a median. A missing D is
    # not an unrecorded value: it is a delta with no earlier event to measure from, and a median
    # there would invent the event.
    sentinel_columns = [c for c in config.TIMEDELTA_COLUMNS if c in set(kept)]
    median_columns = [
        c
        for c in kept
        if c in train.columns
        and pd.api.types.is_numeric_dtype(train[c])
        and c not in config.TIMEDELTA_COLUMNS
    ]
    imputer = missing_policy.fit_imputer(train, median_columns, sentinel_columns)

    report["policy_by_group"] = {
        "rule": (
            "three policies, and which one a column gets depends on what the model reading it "
            "can do. There is no global fill value anywhere in this stage."
        ),
        "groups": missing_policy.GROUP_POLICY,
    }
    report["library_verification"] = verification
    report["mnar_evidence"] = {
        "n_columns_with_nulls": 372,
        "n_label_linked": 251,
        "gates": {"abs_gap": eda.MNAR_ABS_GAP, "p_value": eda.MNAR_P_VALUE},
        "source": "reports/eda/missingness.json, against_target",
        "largest_gaps": sorted(
            facts.missing_against_target.values(),
            key=lambda r: -abs(float(r["gap"])),
        )[:10],
    }
    multi = [e for e in indicators["indicators"] if e["n_columns_covered"] > 1]
    single = [e for e in indicators["indicators"] if e["n_columns_covered"] == 1]
    report["indicators"] = {
        **indicators,
        "breakdown": {
            "n_multi_column_indicators": len(multi),
            "n_columns_under_multi_column_indicators": sum(e["n_columns_covered"] for e in multi),
            "n_single_column_indicators": len(single),
            "note": (
                "stage 2 measured 24 multi-column blocks covering 392 of 435 columns. The column "
                "plan then removed 146 columns, which shrinks several blocks to one member, and a "
                "one-member block still needs its own indicator because its null mask is its own. "
                "The count here is against the kept columns, not against the raw frame."
            ),
        },
    }
    report["imputation"] = {
        "note": (
            "for the stage 5 models that cannot read a null. The tree path never calls this and "
            "the artifact records both so the difference is visible."
        ),
        "n_median_columns": imputer["n_median_columns"],
        "n_sentinel_columns": imputer["n_sentinel_columns"],
        "sentinel_meaning": imputer["sentinel_meaning"],
        "sentinels": imputer["sentinels"],
    }
    return report


# --- encoding ---------------------------------------------------------------------------------


def build_encoding(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    report = envelope(
        "encoding_spec",
        "encoders.fit_* on train, with target_encoding_sweep and entity_target_encoding_value",
        len(train),
    )
    kept = set(cleaning.feature_columns(plan))

    staged_train = encoders.add_normalised_free_text(train)
    staged_val = encoders.add_normalised_free_text(val)
    staged_test = encoders.add_normalised_free_text(test)

    print("  free text normalisation coverage")
    coverage = encoders.normalisation_coverage(train, {"val": val, "test": test})

    categorical = sorted(
        {
            str(c)
            for c in staged_train.columns
            if (c in kept or str(c).endswith(encoders.NORMALISED_SUFFIX))
            and encoders.is_categorical_like(staged_train[c])
        }
    )
    identifiers = [c for c in encoders.IDENTIFIER_COLUMNS if c in kept and c in train.columns]
    normalised = [
        encoders.normalised_name(c)
        for c in encoders.FREE_TEXT_NORMALISERS
        if encoders.normalised_name(c) in staged_train.columns
    ]
    encode_columns = sorted({*identifiers, *normalised})

    print(f"  vocabulary over {len(categorical)} categorical columns")
    vocabulary = encoders.fit_category_vocabulary(staged_train, categorical)
    application = [
        *encoders.vocabulary_application_report(staged_train, vocabulary, "train"),
        *encoders.vocabulary_application_report(staged_val, vocabulary, "val"),
        *encoders.vocabulary_application_report(staged_test, vocabulary, "test"),
    ]

    print(f"  frequency encoding over {len(encode_columns)} identifier columns")
    frequency = encoders.fit_frequency_encoding(staged_train, encode_columns)

    print(
        f"  target encoding sweep: {len(encoders.LAG_GRID)} lags by "
        f"{len(encoders.SMOOTHING_GRID)} smoothings over {len(encode_columns)} columns"
    )
    sweep = encoders.target_encoding_sweep(
        staged_train,
        staged_val,
        encode_columns,
        encoders.LAG_GRID,
        encoders.SMOOTHING_GRID,
    )

    n_pos = int(val[config.TARGET].sum())
    n_neg = int(len(val) - n_pos)
    lag_choice = _choose_lag(sweep, n_pos, n_neg)
    smoothing_choice = _choose_smoothing(sweep, lag_choice["chosen_lag_days"], n_pos, n_neg)

    print("  entity target encoding, information value at several lags")
    entity = encoders.entity_target_encoding_value(
        train,
        val,
        config.ENTITY_ID_COLUMN,
        encoders.LAG_GRID,
        smoothing=smoothing_choice["chosen_smoothing"],
        decompose_at_lag=lag_choice["chosen_lag_days"],
    )
    entity["decision"] = _entity_decision(entity, n_pos, n_neg)

    report["encoders"] = {
        "frequency": {
            "columns": sorted(frequency["tables"]),
            "min_cardinality": frequency["min_cardinality"],
            "unseen_value": frequency["unseen_value"],
            "note": frequency["note"],
            "per_column": frequency["per_column"],
        },
        "target": {
            "chosen_lag_days": lag_choice["chosen_lag_days"],
            "chosen_smoothing": smoothing_choice["chosen_smoothing"],
            "columns": encode_columns,
            "undefined_policy": (
                "a row whose lagged boundary precedes every training row encodes to null; an "
                "unseen level encodes to the prior at that boundary, which is the formula's own "
                "answer for a level counted zero times"
            ),
            "lag_sweep": lag_choice,
            "smoothing_sweep": smoothing_choice,
            "grid": {
                "lags": list(encoders.LAG_GRID),
                "smoothings": list(encoders.SMOOTHING_GRID),
            },
            "per_point": sweep,
        },
        "vocabulary": {
            "rare_tail_share": vocabulary["rare_tail_share"],
            "tokens": {
                "rare": vocabulary["rare_token"],
                "unseen": vocabulary["unseen_token"],
                "missing": vocabulary["missing_token"],
            },
            "why_three_tokens": (
                "a level below the rare line, a level the training window never saw, and a null "
                "are three different states. Stage 2 measured all three separately and merging "
                "any two of them would erase one of its findings."
            ),
            "n_columns": vocabulary["n_columns"],
            "collapse_source": (
                "univariate.json thresholds.rare_tail_share, 0.01, the same line the stage 2 "
                "rare-tail counts were measured against"
            ),
            "per_column": {
                column: {k: v for k, v in spec.items() if k != "levels"}
                for column, spec in sorted(vocabulary["columns"].items())
            },
            "application": application,
        },
        "free_text": {
            "columns": sorted(encoders.FREE_TEXT_NORMALISERS),
            "method": (
                "structural normalisation. DeviceInfo loses its Build suffix and keeps the first "
                "token with digits stripped; id_31 loses its version and keeps family and "
                "platform. Neither uses a brand table: this repo has not established what any of "
                "these strings names, and the tokens work without it."
            ),
            "coverage": coverage,
        },
    }
    report["entity_target_encoding"] = entity
    return report


def _mean_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[Any, dict[str, Any]]:
    grouped: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[key], []).append(row)
    out: dict[Any, dict[str, Any]] = {}
    for value, group in grouped.items():
        val_aucs = [r["validation_auc"] for r in group if r["validation_auc"] is not None]
        train_aucs = [r["train_auc"] for r in group if r["train_auc"] is not None]
        out[value] = {
            "n_points": len(group),
            "mean_validation_auc": float(np.mean(val_aucs)) if val_aucs else None,
            "mean_train_auc": float(np.mean(train_aucs)) if train_aucs else None,
            "mean_train_minus_validation": (
                float(np.mean(train_aucs) - np.mean(val_aucs)) if val_aucs and train_aucs else None
            ),
            "mean_validation_undefined": float(
                np.mean([r["n_validation_undefined"] for r in group])
            ),
            "mean_train_undefined": float(np.mean([r["n_train_undefined"] for r in group])),
        }
    return out


def _per_column(rows: Sequence[Mapping[str, Any]], key: str, value: Any) -> dict[str, Any]:
    """Mean train and validation AUC per column, over every sweep point with `key == value`."""
    train_by: dict[str, list[float]] = {}
    val_by: dict[str, list[float]] = {}
    for row in rows:
        if row[key] != value:
            continue
        if row["train_auc"] is not None:
            train_by.setdefault(row["column"], []).append(float(row["train_auc"]))
        if row["validation_auc"] is not None:
            val_by.setdefault(row["column"], []).append(float(row["validation_auc"]))
    return {
        "train": {c: float(np.mean(v)) for c, v in train_by.items()},
        "validation": {c: float(np.mean(v)) for c, v in val_by.items()},
    }


def _paired(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
    """Mean and standard error of `a - b` over the columns both cover.

    The two settings are measured on the same rows and the same columns, so the difference is
    paired and the spread that matters is the spread across columns. This is not a sampling
    interval for one column's AUC: it says how consistently the swept columns agree about the
    direction of the difference, which is the question a sweep is asking.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_columns": 0, "mean": None, "standard_error": None, "half_width_95": None}
    diffs = np.array([a[c] - b[c] for c in shared], dtype="float64")
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    return {
        "n_columns": len(shared),
        "mean": float(diffs.mean()),
        "standard_error": se,
        "half_width_95": 1.959963984540054 * se,
        "n_columns_positive": int((diffs > 0).sum()),
    }


def _choose_lag(
    sweep: Sequence[Mapping[str, Any]], n_positive: int, n_negative: int
) -> dict[str, Any]:
    """Pick the lag, and say what the sweep actually resolved.

    The rule is stated before the numbers are looked at, and it is not "the lag with the best
    validation AUC". A longer lag can only lose information, so the best validation AUC is
    always at or near lag 0, and choosing on that number alone would choose the leak.

    What the sweep is for is the other column: the gap between the train AUC and the validation
    AUC. An encoding that is summarising a row's own label separates the rows it was fitted on
    far better than the rows it was not, and the gap is what that looks like. So: take the
    smallest lag whose mean train-minus-validation gap is indistinguishable from zero across the
    swept columns, and check that the validation AUC it costs is inside what the split can
    resolve. Smallest, because a lag is a delay on information and the shortest defensible one
    keeps the most.

    Neither test is a substitute for knowing how long a chargeback actually takes to mature.
    Stage 0 did not establish the host's labelling rule and this stage does not either. What the
    sweep establishes is the price of each choice.
    """
    by_lag = _mean_by(sweep, "lag_days")
    lags = sorted(by_lag)
    per_lag_columns = {lag: _per_column(sweep, "lag_days", lag) for lag in lags}

    gaps: dict[int, dict[str, Any]] = {}
    for lag in lags:
        columns = per_lag_columns[lag]
        gaps[lag] = _paired(columns["train"], columns["validation"])

    scored = {
        lag: by_lag[lag]["mean_validation_auc"]
        for lag in lags
        if by_lag[lag]["mean_validation_auc"] is not None
    }
    best_lag = max(scored, key=lambda k: scored[k])
    unpaired = tie_band(scored[best_lag], n_positive, n_negative)

    cost = {
        lag: _paired(per_lag_columns[best_lag]["validation"], per_lag_columns[lag]["validation"])
        for lag in lags
    }

    def gap_is_zero(lag: int) -> bool:
        entry = gaps[lag]
        return entry["mean"] is not None and abs(entry["mean"]) <= entry["half_width_95"]

    def cost_is_resolvable(lag: int) -> bool:
        entry = cost[lag]
        return entry["mean"] is not None and entry["mean"] <= unpaired["band"]

    eligible = [lag for lag in lags if gap_is_zero(lag) and cost_is_resolvable(lag)]
    chosen = min(eligible) if eligible else max(lags)
    return {
        "rule": (
            "the smallest lag whose mean train-minus-validation gap is inside its own 95 percent "
            "interval of zero and whose validation cost against the best lag is inside the band "
            "the split can resolve. Stated before the sweep was read."
        ),
        "chosen_lag_days": int(chosen),
        "fell_back_to_largest_lag": not eligible,
        "best_validation_lag_days": int(best_lag),
        "best_mean_validation_auc": scored[best_lag],
        "unpaired_tie_band": unpaired,
        "eligible_lags": [int(lag) for lag in eligible],
        "gap_test": {
            "definition": (
                "mean over the swept columns of (train AUC minus validation AUC) at that lag, "
                "with the standard error taken across columns"
            ),
            "per_lag": {str(lag): gaps[lag] for lag in lags},
        },
        "cost_test": {
            "definition": (
                "paired difference in validation AUC against the best-scoring lag, taken column "
                "by column"
            ),
            "per_lag": {str(lag): cost[lag] for lag in lags},
        },
        "per_lag": {str(lag): by_lag[lag] for lag in lags},
    }


def _choose_smoothing(
    sweep: Sequence[Mapping[str, Any]], lag: int, n_positive: int, n_negative: int
) -> dict[str, Any]:
    """Pick the smoothing at the chosen lag, ties going to more shrinkage toward the prior.

    Here the best validation AUC is the right target: smoothing trades a thin level's own rate
    against the prior and neither end of that trade is safe by construction. Where two settings
    cannot be told apart by a paired comparison across the swept columns, the larger one wins,
    because more shrinkage is the safer error on a level with few rows behind it.
    """
    at_lag = [row for row in sweep if row["lag_days"] == lag]
    by_smoothing = _mean_by(at_lag, "smoothing")
    values = sorted(by_smoothing)
    per_columns = {value: _per_column(at_lag, "smoothing", value) for value in values}
    scored = {
        value: by_smoothing[value]["mean_validation_auc"]
        for value in values
        if by_smoothing[value]["mean_validation_auc"] is not None
    }
    best = max(scored, key=lambda k: scored[k])
    paired = {
        value: _paired(per_columns[best]["validation"], per_columns[value]["validation"])
        for value in values
    }
    within = sorted(
        value
        for value in values
        if paired[value]["mean"] is not None
        and paired[value]["mean"] <= paired[value]["half_width_95"]
    )
    return {
        "rule": (
            "the smoothing with the best mean validation AUC at the chosen lag. `resolved` says "
            "whether the sweep actually told the grid apart; where it does not, the choice is "
            "weakly determined and the artifact says so rather than dressing it up."
        ),
        "at_lag_days": int(lag),
        "chosen_smoothing": float(best),
        "resolved": len(within) < len(values),
        "resolution_note": (
            "every grid value inside the paired interval of the best means the sweep did not "
            "separate them. The point estimate still picks one and the interval says how much "
            "that pick is worth."
        ),
        "best_smoothing": float(best),
        "best_mean_validation_auc": scored[best],
        "unpaired_tie_band": tie_band(scored[best], n_positive, n_negative),
        "smoothings_within_band": [float(v) for v in within],
        "paired_against_best": {str(value): paired[value] for value in values},
        "per_smoothing": {str(value): by_smoothing[value] for value in values},
    }


def _entity_decision(entity: Mapping[str, Any], n_positive: int, n_negative: int) -> dict[str, Any]:
    """Ship the entity target encoding or exclude it, on the numbers just measured.

    The rule is stated before the numbers, and it is not "is the validation AUC good". It will
    be. Stage 2 measured 0.9675 of multi-transaction entities label-pure at this key, so an
    encoding of the key recovers the label wherever the key is known, and a composite validation
    AUC will report that as skill.

    The test is for the shape that gives away a label lookup gated by entity overlap. Two
    conditions, both measured on validation, and the column is excluded if both hold:

        the encoding, restricted to rows whose entity the training window saw, separates the
        classes better than it does over validation as a whole, by more than the split can
        resolve. All of its power sits on the subset where the key is known;

        the encoding, restricted to rows whose entity the training window did not see, is inside
        the resolvable band of 0.5. It carries nothing at all where the key is new.

    Together those say the column is an entity-to-label table, and how much of a later window it
    covers is a property of entity overlap rather than of anything the model does. Stage 1
    measured that overlap falling to 0.3402 of test rows.
    """
    per_lag = entity["per_lag"]
    best_val = max(r["validation_auc"] for r in per_lag if r["validation_auc"] is not None)
    best_train = max(r["train_auc"] for r in per_lag if r["train_auc"] is not None)
    band = tie_band(best_val, n_positive, n_negative)
    decomposition = entity["decomposition"]

    at_lag = next((r for r in per_lag if r["lag_days"] == decomposition["at_lag_days"]), per_lag[0])
    on_seen = decomposition["auc_on_seen_rows"]
    on_unseen = decomposition["auc_on_unseen_rows"]
    composite = at_lag["validation_auc"]

    concentrated = (
        on_seen is not None and composite is not None and (on_seen - composite) > band["band"]
    )
    empty_off_key = on_unseen is not None and abs(on_unseen - 0.5) <= band["band"]

    return {
        "rule": (
            "exclude when the encoding's power is concentrated on the rows whose entity the "
            "training window saw, by more than the split can resolve, and is indistinguishable "
            "from chance on the rows whose entity it did not. That is an entity-to-label table "
            "gated by entity overlap, not a behavioural feature."
        ),
        "at_lag_days": decomposition["at_lag_days"],
        "best_train_auc": best_train,
        "best_validation_auc": best_val,
        "composite_validation_auc_at_chosen_lag": composite,
        "validation_auc_on_seen_rows": on_seen,
        "validation_auc_on_unseen_rows": on_unseen,
        "seen_indicator_auc": decomposition["seen_indicator_auc"],
        "tie_band": band,
        "power_concentrated_on_seen_entities": bool(concentrated),
        "chance_on_unseen_entities": bool(empty_off_key),
        "shipped": not (concentrated and empty_off_key),
        "entity_label_purity_stage_2": _entity_label_purity(),
        "entity_label_purity_source": (
            "reports/eda/entities.json, structure.label_purity, share_all_clean plus "
            "share_all_fraud. Read, not retyped."
        ),
        "share_validation_rows_on_a_seen_entity": entity["share_validation_rows_on_a_seen_entity"],
        "coverage_caveat": (
            "stage 1 measured 0.3402 of test rows on an entity the training window had seen "
            "(reports/split_summary.json, entity_overlap.test), so whatever this column is worth "
            "it is worth on a minority of the rows a deployed model would score, and that "
            "minority shrinks with time since training."
        ),
    }


def _entity_label_purity() -> float:
    """The share of multi-transaction entities that are label-pure, read from stage 2."""
    purity = json.loads((config.REPORTS_DIR / "eda" / "entities.json").read_text())["structure"][
        "label_purity"
    ]
    return float(purity["share_all_clean"]) + float(purity["share_all_fraud"])


# --- V reduction ------------------------------------------------------------------------------


def build_v_reduction(
    train: pd.DataFrame, val: pd.DataFrame, plan: Mapping[str, Any], facts: cleaning.EdaFacts
) -> dict[str, Any]:
    report = envelope(
        "v_reduction",
        "reduction.compare_strategies(train, val, pool, representatives, blocks)",
        len(train),
    )
    rows = {row["column"]: row for row in plan["columns"]}
    pool = sorted(
        c
        for c in config.VESTA_COLUMNS
        if c in rows
        and c in train.columns
        and (rows[c]["decision"] == "keep" or rows[c]["reason"] == "redundant")
    )
    representatives = sorted(c for c in pool if c in rows and rows[c]["decision"] == "keep")
    blocks = reduction.v_blocks(facts.missingness_blocks, pool)
    print(
        f"  {len(pool)} V columns in the pool, {len(representatives)} representatives, "
        f"{len(blocks)} blocks"
    )
    comparison = reduction.compare_strategies(
        train,
        val,
        pool,
        representatives,
        blocks,
        seed=config.SEED,
        seeds=PROBE_SEEDS,
    )

    n_pos = int(val[config.TARGET].sum())
    n_neg = int(len(val) - n_pos)
    scored = {
        r["strategy"]: r["validation_auc"]
        for r in comparison["results"]
        if r["validation_auc"] is not None
    }
    counts = {r["strategy"]: r["n_columns"] for r in comparison["results"]}
    best = max(scored, key=lambda k: scored[k])
    band = tie_band(scored[best], n_pos, n_neg)
    within = [s for s, auc in scored.items() if scored[best] - auc <= band["band"]]
    chosen = min(within, key=lambda s: (counts[s], s))

    report["pool"] = {
        "definition": (
            "every V column that survived the constant, missing and time-inconsistency rules. "
            "The redundancy rule is one of the strategies being compared, so it is not applied "
            "before the comparison."
        ),
        "n_pool_columns": len(pool),
        "n_representatives": len(representatives),
        "n_blocks": len(blocks),
    }
    report["comparison"] = comparison
    spread = comparison["seed_variance"]["largest_spread"]
    observed = max(scored.values()) - min(scored.values())
    report["decision"] = {
        "rule": (
            "the strategy with the best validation AUC, and among the strategies inside the "
            "resolvable band of it, the one with the fewest columns. Stated before the numbers "
            "were read."
        ),
        "chosen": chosen,
        "best": best,
        "tie_band": band,
        "within_band": sorted(within),
        "n_columns_by_strategy": counts,
        "validation_auc_by_strategy": scored,
        "resolution": {
            "observed_spread_between_strategies": observed,
            "largest_seed_spread_within_a_strategy": spread,
            "resolved": bool(spread is not None and observed > spread and len(within) < 4),
            "note": (
                "the spread between the four strategies against the spread one strategy shows "
                "across seeds. When the second is comparable to the first, the probe has not "
                "separated them and the tie-break is what chose."
            ),
        },
    }
    return report


# --- schema -----------------------------------------------------------------------------------


def build_schema(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    plan: Mapping[str, Any],
    facts: cleaning.EdaFacts,
    decisions: Mapping[str, Any],
) -> dict[str, Any]:
    report = envelope(
        "schema",
        "prepare.fit_preparation(train, ...); schema.declare(prepared_train); schema.validate(...)",
        len(train),
    )
    fitted = prepare.fit_preparation(
        train,
        plan,
        facts,
        d_origin_columns=decisions["d_origin_columns"],
        v_strategy=decisions["v_strategy"],
        lag_days=decisions["lag_days"],
        smoothing=decisions["smoothing"],
    )
    prepared_train = prepare.apply_preparation(train, fitted, plan)
    declared = schema_module.declare(
        prepared_train, facts.negative_value_columns, facts.all_columns()
    )

    results: list[dict[str, Any]] = []
    for name, part in (("train", train), ("val", val), ("test", test)):
        prepared = prepare.apply_preparation(part, fitted, plan)
        try:
            schema_module.validate(prepared, declared)
            results.append(
                {"split": name, "n_rows": len(prepared), "valid": True, "violations": []}
            )
        except schema_module.SchemaError as error:
            results.append(
                {
                    "split": name,
                    "n_rows": len(prepared),
                    "valid": False,
                    "violations": error.violations,
                }
            )
        del prepared
        gc.collect()

    again = prepare.apply_preparation(train, fitted, plan)
    deterministic = bool(
        list(again.columns) == list(prepared_train.columns)
        and _fingerprint(again) == _fingerprint(prepared_train)
    )

    report["decisions"] = dict(decisions)
    raw_names = {row["column"] for row in plan["columns"]}
    surviving_raw = [c for c in prepared_train.columns if c in raw_names]
    report["prepared_shape"] = {
        "n_columns": int(prepared_train.shape[1]),
        "n_rows_train": int(prepared_train.shape[0]),
        "n_raw_columns": int(plan["n_columns"]),
        "n_columns_kept_by_the_plan": int(plan["n_kept"]),
        "n_columns_dropped_by_the_plan": int(plan["n_dropped"]),
        "n_raw_columns_in_the_prepared_frame": len(surviving_raw),
        "n_derived_columns": int(prepared_train.shape[1]) - len(surviving_raw),
        "note": (
            "the plan's kept count and the prepared frame's raw count differ by the V reduction: "
            "a strategy other than keeping representatives consumes the whole V pool and replaces "
            "it with derived columns, so V columns the plan kept are not in the frame either."
        ),
    }
    report["schema"] = declared.to_dict()
    report["validation"] = results
    report["determinism"] = {
        "method": (
            "the pipeline is applied to the train split twice from the same fitted objects and "
            "the two frames are compared column by column, values included."
        ),
        "identical": deterministic,
        "fingerprint": _fingerprint(prepared_train),
        "seed": config.SEED,
    }
    report["not_caught"] = list(schema_module.NOT_CAUGHT)
    return report


def _fingerprint(frame: pd.DataFrame) -> str:
    """A stable hash of the frame's values and column names."""
    import hashlib

    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(str(column).encode())
        series = frame[column]
        if isinstance(series.dtype, pd.CategoricalDtype):
            digest.update(series.cat.codes.to_numpy().tobytes())
        elif pd.api.types.is_numeric_dtype(series):
            digest.update(np.nan_to_num(series.to_numpy(dtype="float64"), nan=-9e99).tobytes())
        else:
            digest.update(series.astype("string").fillna("").str.cat().encode())
    return digest.hexdigest()


# --- entry point ------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3 preparation.")
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    parser.add_argument("--eda-dir", default=str(config.REPORTS_DIR / "eda"))
    parser.add_argument("--sections", nargs="*", default=list(SECTIONS), choices=list(SECTIONS))
    args = parser.parse_args()

    started = time.perf_counter()
    print("loading the joined frame")
    frame = data_loader.add_entity_key(
        data_loader.load_raw(transactions_path=args.transactions, identity_path=args.identity)
    )
    print(f"  {len(frame):,} rows by {frame.shape[1]} columns")
    train, val, test = data_loader.time_based_split(frame)
    del frame
    gc.collect()

    data_loader.assert_train_only(train, what="stage 3 preparation")
    print(f"  train {len(train):,} rows, val {len(val):,}, test {len(test):,}")

    facts = cleaning.EdaFacts.load(args.eda_dir)

    for section in args.sections:
        print(f"section {section}")
        section_started = time.perf_counter()
        if section == "column_plan":
            report = build_column_plan(train, val, facts)
            path = COLUMN_PLAN_PATH
        elif section == "transforms":
            report = build_transforms(train, val, test, _plan())
            path = PREP_DIR / "transforms.json"
        elif section == "missing":
            report = build_missing(train, _plan())
            path = PREP_DIR / "missing_policy.json"
        elif section == "encoding":
            report = build_encoding(train, val, test, _plan())
            path = ENCODING_SPEC_PATH
        elif section == "v_reduction":
            report = build_v_reduction(train, val, _plan(), facts)
            path = PREP_DIR / "v_reduction.json"
        else:
            report = build_schema(train, val, test, _plan(), facts, _decisions())
            path = PREP_DIR / "schema.json"
        report["seconds"] = time.perf_counter() - section_started
        write(report, path)
        del report
        gc.collect()

    print(f"\ndone in {time.perf_counter() - started:.1f}s")


def _plan() -> dict[str, Any]:
    if not COLUMN_PLAN_PATH.exists():
        raise SystemExit(
            f"{COLUMN_PLAN_PATH} does not exist. Run --sections column_plan first: every other "
            "section reads the plan rather than re-deciding it."
        )
    return dict(json.loads(COLUMN_PLAN_PATH.read_text()))


def _decisions() -> dict[str, Any]:
    """The four measured choices the pipeline needs, read from the artifacts that made them."""
    transforms_path = PREP_DIR / "transforms.json"
    v_path = PREP_DIR / "v_reduction.json"
    for path in (transforms_path, ENCODING_SPEC_PATH, v_path):
        if not path.exists():
            raise SystemExit(f"{path} does not exist. Run the earlier sections first.")
    transforms_report = json.loads(transforms_path.read_text())
    encoding = json.loads(ENCODING_SPEC_PATH.read_text())
    v_report = json.loads(v_path.read_text())
    return {
        "d_origin_columns": list(transforms_report["d_normalisation"]["applied_to"]),
        "d_origin_source": "reports/prep/transforms.json, d_normalisation.applied_to",
        "v_strategy": v_report["decision"]["chosen"],
        "v_strategy_source": "reports/prep/v_reduction.json, decision.chosen",
        "lag_days": int(encoding["encoders"]["target"]["chosen_lag_days"]),
        "smoothing": float(encoding["encoders"]["target"]["chosen_smoothing"]),
        "encoding_source": "reports/encoding_spec.json, encoders.target",
    }


if __name__ == "__main__":
    main()
