"""The preparation pipeline: one fit on train, one transform that every split goes through.

Everything the earlier modules in this stage build is a piece. This is the assembly, and it
exists so there is exactly one order of operations rather than one per caller. The order is not
arbitrary and two steps in it would be wrong the other way round:

- indicators are computed before imputation, because imputation destroys the null mask the
  indicator reads;
- the target encoding runs on the raw column and not on the vocabulary-collapsed one, because
  collapsing the rare tail first would pool levels whose rates the encoder is there to separate.

`fit_preparation` takes the train split and returns every fitted object in one dictionary.
`apply_preparation` takes that dictionary and any split. The fitted objects are built through
`assert_train_only` inside their own modules, so a caller cannot assemble this pipeline on a
frame that reaches past the training boundary even by accident.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_platform import cleaning, config, encoders, missing_policy, reduction, transforms
from fraud_platform.data_loader import assert_train_only

# Column groups the free-text normalisation replaces rather than supplements. The raw column
# stays in the frame: stage 7 will want to show a reviewer the string that was actually sent.
PIPELINE_STEPS: tuple[str, ...] = (
    "apply_column_plan",
    "add_clock_columns",
    "add_amount_features",
    "to_d_origin",
    "add_normalised_free_text",
    "add_missing_indicators",
    "apply_category_vocabulary",
    "apply_frequency_encoding",
    "apply_target_encoding",
    "reduce_v_columns",
)


def fit_preparation(
    train: pd.DataFrame,
    plan: Mapping[str, Any],
    facts: cleaning.EdaFacts,
    d_origin_columns: Sequence[str],
    v_strategy: str,
    lag_days: int,
    smoothing: float,
    clip_columns: Sequence[str] = (),
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Fit every stateful piece of the pipeline on the train split.

    `d_origin_columns`, `v_strategy`, `lag_days` and `smoothing` are decisions, not defaults.
    Each of them is the output of a measurement the stage 3 script runs and records, and they
    arrive as arguments so that the pipeline cannot be assembled without them having been made.
    """
    assert_train_only(train, what="stage 3 preparation pipeline")

    kept = cleaning.feature_columns(plan)
    v_pool = cleaning.v_reduction_pool(plan)
    spare = v_pool if v_strategy in ("pca", "block_mean") else []
    cleaned = cleaning.apply_column_plan(train, plan, spare=spare)
    staged = transforms.add_amount_features(transforms.add_clock_columns(cleaned))
    staged = transforms.to_d_origin(staged, d_origin_columns)
    staged = encoders.add_normalised_free_text(staged)

    indicator_spec = missing_policy.missing_indicator_spec(facts.missingness_blocks, kept)

    categorical = sorted(
        {
            column
            for column in staged.columns
            if column in kept or column.endswith(encoders.NORMALISED_SUFFIX)
        }
        & _categorical_columns(staged)
    )
    identifiers = [c for c in encoders.IDENTIFIER_COLUMNS if c in staged.columns]
    normalised = [
        encoders.normalised_name(c)
        for c in encoders.FREE_TEXT_NORMALISERS
        if encoders.normalised_name(c) in staged.columns
    ]

    vocabulary = encoders.fit_category_vocabulary(staged, [*categorical, *normalised])
    frequency = encoders.fit_frequency_encoding(staged, [*identifiers, *normalised])
    target = encoders.fit_target_encoding(
        staged, [*identifiers, *normalised], lag_days=lag_days, smoothing=smoothing
    )

    blocks = reduction.v_blocks(facts.missingness_blocks, v_pool)
    v_fitted: dict[str, Any] = {
        "strategy": v_strategy,
        "pool": v_pool,
        "spared_from_the_plan": sorted(spare),
        "blocks": blocks,
    }
    if v_strategy == "pca":
        v_fitted["pca"] = reduction.fit_block_pca(staged, blocks, seed=seed)
    elif v_strategy == "block_mean":
        v_fitted["block_mean"] = reduction.fit_block_mean(staged, blocks)

    clip = transforms.fit_clip_bounds(staged, clip_columns) if clip_columns else None

    return {
        "seed": seed,
        "steps": list(PIPELINE_STEPS),
        "d_origin_columns": list(d_origin_columns),
        "indicators": indicator_spec,
        "vocabulary": vocabulary,
        "frequency": frequency,
        "target_encoding": target,
        "v_reduction": v_fitted,
        "clip": clip,
        "kept_columns": kept,
    }


def _categorical_columns(frame: pd.DataFrame) -> set[str]:
    return {str(column) for column in frame.columns if encoders.is_categorical_like(frame[column])}


def apply_preparation(
    frame: pd.DataFrame, fitted: Mapping[str, Any], plan: Mapping[str, Any]
) -> pd.DataFrame:
    """Run the pipeline over any split. Deterministic: same input, same output, every time.

    The output is sorted by column name at the end. Nothing downstream should depend on column
    order, and sorting means nothing can start to by accident.
    """
    v = fitted["v_reduction"]
    out = cleaning.apply_column_plan(frame, plan, spare=v["spared_from_the_plan"])
    out = transforms.add_clock_columns(out)
    out = transforms.add_amount_features(out)
    out = transforms.to_d_origin(out, fitted["d_origin_columns"])
    out = encoders.add_normalised_free_text(out)
    out = missing_policy.add_missing_indicators(out, fitted["indicators"])
    out = encoders.apply_category_vocabulary(out, fitted["vocabulary"])
    out = encoders.apply_frequency_encoding(out, fitted["frequency"])
    out = encoders.apply_target_encoding(out, fitted["target_encoding"])

    if v["strategy"] in ("pca", "block_mean"):
        derived = (
            reduction.apply_block_pca(out, v["pca"])
            if v["strategy"] == "pca"
            else reduction.apply_block_mean(out, v["block_mean"])
        )
        out = out.drop(columns=[c for c in v["pool"] if c in out.columns])
        out = pd.concat([out, derived], axis=1)
    # The `representative` strategy needs no step here: it is what the column plan's redundancy
    # rule already did, and nothing was spared from that drop.

    if fitted["clip"] is not None:
        out = transforms.apply_clip(out, fitted["clip"])

    return out[sorted(out.columns)]


# --- the recorded decisions ------------------------------------------------------------------


def read_decisions(reports_dir: Path = config.REPORTS_DIR) -> dict[str, Any]:
    """The four measured choices the pipeline needs, read from the artifacts that made them.

    Stage 3's driver writes the artifacts and then reads them back through this function to
    assemble the schema section; stage 6 reads the same four values the same way to rebuild the
    prepared frame, so there is one copy of each decision and it is in the artifact.
    """
    transforms_path = reports_dir / "prep" / "transforms.json"
    encoding_path = reports_dir / "encoding_spec.json"
    v_path = reports_dir / "prep" / "v_reduction.json"
    for path in (transforms_path, encoding_path, v_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Run the earlier stage 3 sections first."
            )
    transforms_report = json.loads(transforms_path.read_text())
    encoding = json.loads(encoding_path.read_text())
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


def read_column_plan(reports_dir: Path = config.REPORTS_DIR) -> dict[str, Any]:
    """The stage 3 column plan, which every later section reads rather than re-deciding."""
    path = reports_dir / "column_plan.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run scripts/prepare.py --sections column_plan first."
        )
    return dict(json.loads(path.read_text()))
