"""Stage 7, experiment 1: the leakage delta.

Four shortcuts that are standard in a competition solution and a defect at an authorisation
endpoint, switched on one at a time and then together, with the model family, the
hyperparameters, the seed and the column stack held at the shipped model's. The delta between
the shipped configuration and each variant, with a bootstrap interval, is what the train-only,
strictly causal discipline of stages 1 to 5 is worth on this data. A delta inside the noise is
reported as inside the noise.

  (a) every encoder fitted on all 590,540 rows instead of the 413,378 train rows
  (b) entity aggregates over every row, in the style of the published UID construction
      that ADR 0020 kept out of `fraud_platform.features`
  (c) every prediction replaced by the entity's mean prediction over the scored window
  (d) a random 70/15/15 split in place of the chronological one

Nothing here is importable from `fraud_platform`, and nothing in `fraud_platform` can build
(a) or (d): both need rows past the training boundary in a fit, which the train-only guard
refuses. The guard is switched off here, by name, inside a context manager, so that the file
that does it is this one and a diff that did it anywhere else would be visible for what it is.

One artifact, reports/leakage_delta.json, from `make leakage`.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn
import xgboost

from fraud_platform import cleaning, config, data_loader, evaluation, modelling, prepare, transforms

OUT_PATH = config.REPORTS_DIR / "leakage_delta.json"
BUNDLE_PATH = config.ROOT / "models" / "shipped_model.json"
OPERATING_POINTS_PATH = config.REPORTS_DIR / "operating_points.json"
VARIANCE_PATH = config.REPORTS_DIR / "metric_variance.json"

N_BOOTSTRAP = 1_000
FAMILY = "xgboost"
IMBALANCE = "none"

# The random split reproduces the chronological row fractions of ADR 0002 and nothing else.
RANDOM_SPLIT_FRACTIONS: tuple[float, float] = (0.70, 0.85)

# The columns the published construction aggregates per entity: the counters, the match flags
# and the timedeltas normalised to an origin day. Every column in each group, from the raw
# file, whether or not stage 3's column plan kept it, because the construction being priced is
# the published one and not this repo's edit of it.
AGGREGATE_COLUMNS: dict[str, tuple[str, ...]] = {
    "count": config.COUNT_COLUMNS,
    "match": config.MATCH_COLUMNS,
    "timedelta": config.TIMEDELTA_COLUMNS,
}
AGGREGATE_PREFIX = "uid_"
AGGREGATE_STATISTICS: tuple[str, ...] = ("mean", "std")

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "baseline",
        "letters": "",
        "description": "the shipped configuration: encoders fitted on train through the guard, "
        "no entity aggregate, no post-processing, the chronological split",
        "encoder_fit_rows": "train",
        "entity_aggregates": False,
        "postprocessing": False,
        "split": "chronological",
    },
    {
        "id": "a_full_fit_encoders",
        "letters": "a",
        "description": "every encoder fitted on all rows, including the validation and test "
        "windows and their labels, at the same lag and smoothing",
        "encoder_fit_rows": "all",
        "entity_aggregates": False,
        "postprocessing": False,
        "split": "chronological",
    },
    {
        "id": "b_full_data_entity_aggregates",
        "letters": "b",
        "description": "mean and standard deviation of the C, M and normalised D columns per "
        "entity over every row of the dataset, attached to every row of the entity",
        "encoder_fit_rows": "train",
        "entity_aggregates": True,
        "postprocessing": False,
        "split": "chronological",
    },
    {
        "id": "c_entity_mean_postprocessing",
        "letters": "c",
        "description": "the baseline's predictions, each replaced by the mean prediction over "
        "the entity's rows in the same scored window",
        "encoder_fit_rows": "train",
        "entity_aggregates": False,
        "postprocessing": True,
        "split": "chronological",
    },
    {
        "id": "d_random_split",
        "letters": "d",
        "description": "a seeded random permutation cut at the chronological row fractions, "
        "encoders fitted on the random train rows, scored on the random test rows",
        "encoder_fit_rows": "train",
        "entity_aggregates": False,
        "postprocessing": False,
        "split": "random",
    },
    {
        "id": "abc_chronological",
        "letters": "abc",
        "description": "(a), (b) and (c) together on the chronological split, so the delta "
        "stays paired with the baseline",
        "encoder_fit_rows": "all",
        "entity_aggregates": True,
        "postprocessing": True,
        "split": "chronological",
    },
    {
        "id": "abcd_random",
        "letters": "abcd",
        "description": "all four together: the competition-style pipeline end to end",
        "encoder_fit_rows": "all",
        "entity_aggregates": True,
        "postprocessing": True,
        "split": "random",
    },
)


# --- scaffolding -----------------------------------------------------------------------------


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def envelope() -> dict[str, Any]:
    return {
        "section": "leakage_delta",
        "stage": 7,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": "make leakage  # or: .venv/bin/python scripts/leakage_delta.py",
        "command": ".venv/bin/python scripts/leakage_delta.py",
        "split": {
            "train_end_dt": config.TRAIN_END_DT,
            "val_end_dt": config.VAL_END_DT,
            "rule": "dt < boundary, ADR 0002",
        },
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "platform": platform.platform(),
        },
        "seed": config.SEED,
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def write(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(report), indent=2, allow_nan=False) + "\n")
    print(f"  wrote {path.relative_to(config.ROOT)}")


def read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"{path.relative_to(config.ROOT)} does not exist; run `make train` first")
    return dict(json.loads(path.read_text()))


# --- the guard, switched off in the open ------------------------------------------------------


@contextlib.contextmanager
def guard_switched_off() -> Iterator[None]:
    """The train-only guard replaced by a pass-through for the duration of one leaky fit.

    `data_loader.assert_train_only` is what the `train_only` decorator calls, and
    `fraud_platform.prepare` holds its own bound name; both are replaced and both are restored
    on exit, whatever happens inside. This is the only place in the repository that does this,
    and the reason it is a context manager rather than a flag is so that it cannot be left on.
    """

    def pass_through(frame: pd.DataFrame, *, what: str) -> pd.DataFrame:
        return frame

    original = data_loader.assert_train_only
    data_loader.assert_train_only = pass_through
    prepare.assert_train_only = pass_through
    try:
        yield
    finally:
        data_loader.assert_train_only = original
        prepare.assert_train_only = original


# --- splits -----------------------------------------------------------------------------------


def random_split_masks(frame: pd.DataFrame, seed: int) -> dict[str, pd.Series[bool]]:
    """A seeded permutation cut at the chronological row fractions. What a competition does."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    position = np.empty(len(frame), dtype="int64")
    position[order] = np.arange(len(frame))
    first = int(RANDOM_SPLIT_FRACTIONS[0] * len(frame))
    second = int(RANDOM_SPLIT_FRACTIONS[1] * len(frame))
    return {
        "train": pd.Series(position < first, index=frame.index),
        "val": pd.Series((position >= first) & (position < second), index=frame.index),
        "test": pd.Series(position >= second, index=frame.index),
    }


def prepare_splits(
    frame: pd.DataFrame,
    fit_mask: pd.Series[bool],
    masks: Mapping[str, pd.Series[bool]],
    plan: Mapping[str, Any],
    facts: cleaning.EdaFacts,
    decisions: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Stage 3's pipeline fitted on the rows `fit_mask` selects, applied to every split."""
    fitted = prepare.fit_preparation(
        frame[fit_mask],
        plan,
        facts,
        d_origin_columns=decisions["d_origin_columns"],
        v_strategy=decisions["v_strategy"],
        lag_days=decisions["lag_days"],
        smoothing=decisions["smoothing"],
    )
    out = {}
    for name, mask in masks.items():
        part = frame[mask].sort_values(config.TIME_COLUMN, kind="stable")
        out[name] = prepare.apply_preparation(part, fitted, plan)
        print(f"    prepared {name}: {out[name].shape[0]:,} rows by {out[name].shape[1]}")
    return out


# --- the published aggregation, over every row ------------------------------------------------


def full_data_entity_aggregates(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-entity mean and standard deviation of the aggregate columns over every row.

    The construction of ADR 0020: groupby the entity, take the group's statistic, attach it to
    every row of the group. A match flag is its level code (F, T; M0, M1, M2), a timedelta is
    its origin day (`transforms.to_d_origin`), a counter is itself. The standard deviation is
    pandas' with one degree of freedom, so an entity with one row gets a null, as the published
    construction does. Nothing here is restricted to rows before the row being scored, which is
    the whole point.
    """
    values: dict[str, pd.Series] = {}
    for column in AGGREGATE_COLUMNS["count"]:
        values[column] = frame[column].astype("float64")
    for column in AGGREGATE_COLUMNS["match"]:
        codes = frame[column].cat.codes.astype("float64")
        values[column] = codes.where(codes >= 0)
    with_origin = transforms.to_d_origin(
        frame[[config.TIME_COLUMN, *AGGREGATE_COLUMNS["timedelta"]]],
        AGGREGATE_COLUMNS["timedelta"],
    )
    for column in AGGREGATE_COLUMNS["timedelta"]:
        values[transforms.d_origin_name(column)] = with_origin[transforms.d_origin_name(column)]

    table = pd.DataFrame(values, index=frame.index)
    keys = frame[config.ENTITY_ID_COLUMN]
    grouped = table.groupby(keys, sort=False).agg(list(AGGREGATE_STATISTICS))
    grouped.columns = [f"{AGGREGATE_PREFIX}{column}_{stat}" for column, stat in grouped.columns]
    out = grouped.reindex(keys.to_numpy())
    out.index = frame.index
    return out.astype("float32")


def aggregate_column_names() -> list[str]:
    names = []
    for group, columns in AGGREGATE_COLUMNS.items():
        for column in columns:
            base = transforms.d_origin_name(column) if group == "timedelta" else column
            names.extend(f"{AGGREGATE_PREFIX}{base}_{stat}" for stat in AGGREGATE_STATISTICS)
    return names


# --- the post-processing ----------------------------------------------------------------------


def entity_mean_postprocessing(
    scores: npt.NDArray[np.float64], keys: npt.NDArray[Any]
) -> npt.NDArray[np.float64]:
    """Every row's score replaced by the mean score over its entity's rows in the same window.

    For the first transaction of an entity in the window, every other row in the mean is a
    later one. That is what makes it a competition step and not a serving step.
    """
    series = pd.Series(np.asarray(scores, dtype="float64"))
    return series.groupby(np.asarray(keys), sort=False).transform("mean").to_numpy(dtype="float64")


def postprocessing_diagnostic(
    frame: pd.DataFrame, scores: npt.NDArray[np.float64]
) -> dict[str, Any]:
    """Where the entity mean acts, on one scored window.

    The rows are partitioned by the kind of entity they sit on: a single row in the window, a
    multi-row entity whose rows are all legitimate, all fraud, or mixed. The mean changes nothing
    on a singleton and pulls every row of a mixed entity toward one number, so the share of the
    window's fraud rows inside mixed entities is the share the step can hurt. An entity whose
    key has a null component (ADR 0001 keeps it as its own bucket) pools rows that share a
    `card1` and nothing else, and the step is also scored with those entities left alone.
    """
    y = frame[config.TARGET].to_numpy(dtype="int64")
    keys = frame[config.ENTITY_ID_COLUMN]
    per_entity = (
        pd.DataFrame({"y": y}).groupby(keys.to_numpy(), sort=False)["y"].agg(["sum", "count"])
    )
    kind_of_entity = pd.Series(
        np.select(
            [
                per_entity["count"] == 1,
                per_entity["sum"] == 0,
                per_entity["sum"] == per_entity["count"],
            ],
            ["singleton", "multi_row_all_legitimate", "multi_row_all_fraud"],
            default="multi_row_mixed",
        ),
        index=per_entity.index,
    )
    kind = keys.map(kind_of_entity).to_numpy()
    partition = {}
    for name in ("singleton", "multi_row_all_legitimate", "multi_row_all_fraud", "multi_row_mixed"):
        rows = kind == name
        partition[name] = {
            "n_entities": int((kind_of_entity == name).sum()),
            "n_rows": int(rows.sum()),
            "n_fraud_rows": int(y[rows].sum()),
            "share_of_rows": float(rows.mean()),
            "share_of_fraud_rows": float(y[rows].sum() / y.sum()),
        }
    null_component = keys.str.contains(config.ENTITY_NULL_TOKEN, regex=False).to_numpy()
    averaged = entity_mean_postprocessing(scores, keys.to_numpy())
    return {
        "on": "the baseline's scores over the chronological test window",
        "partition_by_entity_kind": partition,
        "null_key_component": {
            "rule": f"an entity key with {config.ENTITY_NULL_TOKEN!r} in it, ADR 0001",
            "row_share": float(null_component.mean()),
            "fraud_rate": float(y[null_component].mean()),
            "n_entities": int(keys[null_component].nunique()),
        },
        "pr_auc": {
            "baseline": evaluation.pr_auc(y, scores),
            "entity_mean": evaluation.pr_auc(y, averaged),
            "entity_mean_with_null_key_entities_left_alone": evaluation.pr_auc(
                y, np.where(null_component, scores, averaged)
            ),
        },
    }


# --- one fit ----------------------------------------------------------------------------------


def fit_and_score(
    prepared: Mapping[str, pd.DataFrame],
    columns: Sequence[str],
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    """One xgboost fit on the train frame, scored on val and test. The same call for every variant."""
    matrices = {split: modelling.tree_matrix(frame, columns) for split, frame in prepared.items()}
    y_fit = prepared["train"][config.TARGET].to_numpy(dtype="int64")
    model = modelling.make_model(FAMILY, IMBALANCE, y_fit, config.SEED, hyperparameters)
    started = time.perf_counter()
    model.fit(matrices["train"].to_numpy(dtype="float32"), y_fit)
    fit_seconds = time.perf_counter() - started
    scores = {
        split: modelling.score(model, matrices[split].to_numpy(dtype="float32"))
        for split in ("val", "test")
    }
    del model, matrices
    gc.collect()
    return {
        "n_columns": len(columns),
        "n_fit_rows": int(y_fit.size),
        "n_fit_positive": int(y_fit.sum()),
        "fit_seconds": fit_seconds,
        "scores": scores,
    }


def metrics(prepared: Mapping[str, pd.DataFrame], scores: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in ("val", "test"):
        y = prepared[split][config.TARGET].to_numpy(dtype="int64")
        keys = prepared[split][config.ENTITY_ID_COLUMN].to_numpy()
        y_card, s_card = evaluation.card_level(y, scores[split], keys)
        out[split] = {
            "pr_auc": evaluation.pr_auc(y, scores[split]),
            "roc_auc": evaluation.roc_auc(y, scores[split]),
            "mean_score": evaluation.mean_score(y, scores[split]),
            "card_level": {
                "pr_auc": evaluation.pr_auc(y_card, s_card),
                "roc_auc": evaluation.roc_auc(y_card, s_card),
            },
        }
    return out


def population(prepared: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """What the scored windows hold: rows, positives, and how much of each is reachable by an
    entity-level shortcut. Seen means the entity has a row in the fit split; multi-row means it
    has more than one row inside the scored window itself."""
    train_entities = set(prepared["train"][config.ENTITY_ID_COLUMN].to_numpy())
    out: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        frame = prepared[split]
        keys = frame[config.ENTITY_ID_COLUMN]
        y = frame[config.TARGET].to_numpy(dtype="int64")
        rows_per_entity = keys.map(keys.value_counts())
        entry: dict[str, Any] = {
            "n_rows": len(frame),
            "n_positive": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "n_entities": int(keys.nunique()),
            "min_dt": int(frame[config.TIME_COLUMN].min()),
            "max_dt": int(frame[config.TIME_COLUMN].max()),
            "multi_row_entity_row_share": float((rows_per_entity > 1).mean()),
        }
        if split != "train":
            seen = keys.isin(train_entities)
            entry["seen_entity_row_share"] = float(seen.mean())
            entry["seen_entity_fraud_rate"] = float(y[seen.to_numpy()].mean())
            entry["unseen_entity_fraud_rate"] = float(y[~seen.to_numpy()].mean())
        out[split] = entry
    return out


# --- the experiment ---------------------------------------------------------------------------


def run(transactions: str, identity: str, n_boot: int, sample_fraction: float | None) -> None:
    started = time.perf_counter()
    bundle = read(BUNDLE_PATH)
    columns: list[str] = list(bundle["columns"])
    hyperparameters: dict[str, Any] = dict(bundle["hyperparameters"])
    if bundle["family"] != FAMILY or bundle["imbalance"] != IMBALANCE:
        raise SystemExit("this script is written for the shipped xgboost model without reweighting")

    print("loading the joined frame")
    frame = data_loader.add_entity_key(
        data_loader.load_raw(transactions_path=transactions, identity_path=identity)
    )
    if sample_fraction is not None:
        keep = np.random.default_rng(config.SEED).random(len(frame)) < sample_fraction
        frame = frame[keep]
        print(f"  smoke run on a {sample_fraction:.2f} sample: {len(frame):,} rows")
    print(f"  {len(frame):,} rows by {frame.shape[1]} columns")

    plan = prepare.read_column_plan()
    facts = cleaning.EdaFacts.load()
    decisions = prepare.read_decisions()
    chronological = data_loader.split_masks(frame)
    random_masks = random_split_masks(frame, config.SEED)
    all_rows = pd.Series(True, index=frame.index)

    print("baseline preparation: fitted on train, through the guard")
    data_loader.assert_train_only(frame[chronological["train"]], what="stage 7 baseline")
    prepared = {
        "baseline": prepare_splits(
            frame, chronological["train"], chronological, plan, facts, decisions
        )
    }
    print("variant (a) preparation: fitted on every row, guard switched off")
    with guard_switched_off():
        prepared["a"] = prepare_splits(frame, all_rows, chronological, plan, facts, decisions)
        print("variant (d) preparation: fitted on the random train rows, guard switched off")
        prepared["d"] = prepare_splits(
            frame, random_masks["train"], random_masks, plan, facts, decisions
        )
        print("variant (abcd) preparation: fitted on every row, applied to the random splits")
        prepared["abcd"] = prepare_splits(frame, all_rows, random_masks, plan, facts, decisions)

    if sample_fraction is not None:
        # A sample can drop a column below the frequency encoder's cardinality floor, so the
        # smoke run reads whatever survived rather than failing on a column the full run has.
        present = set(prepared["baseline"]["train"].columns)
        columns = [c for c in columns if c in present]
        print(f"  smoke run reads {len(columns)} of the shipped columns")

    print("the published aggregation over every row")
    aggregates = full_data_entity_aggregates(frame)
    aggregate_names = aggregate_column_names()
    assert list(aggregates.columns) == aggregate_names
    del frame
    gc.collect()

    def with_aggregates(parts: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {
            split: pd.concat([part, aggregates.loc[part.index]], axis=1)
            for split, part in parts.items()
        }

    fits: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, npt.NDArray[np.float64]]] = {}
    keys = {
        "chronological": {
            s: prepared["baseline"][s][config.ENTITY_ID_COLUMN].to_numpy() for s in ("val", "test")
        },
        "random": {
            s: prepared["d"][s][config.ENTITY_ID_COLUMN].to_numpy() for s in ("val", "test")
        },
    }

    def record(variant_id: str, fitted: dict[str, Any], split_kind: str, postprocess: bool) -> None:
        s = fitted.pop("scores")
        if postprocess:
            s = {
                split: entity_mean_postprocessing(s[split], keys[split_kind][split]) for split in s
            }
        scores[variant_id] = s
        fits[variant_id] = fitted

    print("fitting: baseline")
    record(
        "baseline",
        fit_and_score(prepared["baseline"], columns, hyperparameters),
        "chronological",
        False,
    )
    print("fitting: (a)")
    record(
        "a_full_fit_encoders",
        fit_and_score(prepared["a"], columns, hyperparameters),
        "chronological",
        False,
    )
    print("fitting: (b)")
    record(
        "b_full_data_entity_aggregates",
        fit_and_score(
            with_aggregates(prepared["baseline"]), [*columns, *aggregate_names], hyperparameters
        ),
        "chronological",
        False,
    )
    print("post-processing: (c)")
    fits["c_entity_mean_postprocessing"] = dict(fits["baseline"])
    scores["c_entity_mean_postprocessing"] = {
        split: entity_mean_postprocessing(scores["baseline"][split], keys["chronological"][split])
        for split in ("val", "test")
    }
    print("fitting: (d)")
    record(
        "d_random_split", fit_and_score(prepared["d"], columns, hyperparameters), "random", False
    )
    print("fitting: (abc)")
    record(
        "abc_chronological",
        fit_and_score(
            with_aggregates(prepared["a"]), [*columns, *aggregate_names], hyperparameters
        ),
        "chronological",
        True,
    )
    print("fitting: (abcd)")
    record(
        "abcd_random",
        fit_and_score(
            with_aggregates(prepared["abcd"]), [*columns, *aggregate_names], hyperparameters
        ),
        "random",
        True,
    )

    report = envelope()
    report["model"] = {
        "family": FAMILY,
        "imbalance": IMBALANCE,
        "stack": bundle["stack"],
        "n_columns": len(columns),
        "columns": columns,
        "hyperparameters": hyperparameters,
        "hyperparameters_are_module_defaults": hyperparameters
        == modelling.DEFAULT_HYPERPARAMETERS[FAMILY],
        "seed": config.SEED,
        "held_fixed": "family, hyperparameters, seed and the shipped column stack are the same in "
        "every variant; (b) and the combined variants read the aggregate columns in addition",
    }
    report["shipped_model_test_pr_auc"] = read(OPERATING_POINTS_PATH)["primary_metric"]["test"][
        "pr_auc"
    ]
    report["stage_6_noise_band"] = read(VARIANCE_PATH)["noise_band"]
    report["guard"] = {
        "baseline": "on: the fit rows went through data_loader.assert_train_only",
        "leaky_variants": "switched off inside scripts/leakage_delta.py:guard_switched_off for "
        "the preparation fits of (a), (d) and the combined variants, and restored after",
    }
    report["aggregation"] = {
        "adr": "0020",
        "key": config.ENTITY_ID_COLUMN,
        "groups": {k: list(v) for k, v in AGGREGATE_COLUMNS.items()},
        "statistics": list(AGGREGATE_STATISTICS),
        "n_columns": len(aggregate_names),
        "columns": aggregate_names,
        "over": "every row of the dataset, all three windows, no restriction to earlier rows",
    }
    report["postprocessing"] = {
        "rule": "each score replaced by the mean score over the entity's rows in the scored "
        "window (validation rows with validation rows, test rows with test rows)",
        "key": config.ENTITY_ID_COLUMN,
        "diagnostic": postprocessing_diagnostic(
            prepared["baseline"]["test"], scores["baseline"]["test"]
        ),
    }
    report["random_split"] = {
        "seed": config.SEED,
        "fractions": list(RANDOM_SPLIT_FRACTIONS),
        "rule": "a seeded permutation of the rows cut at the chronological row fractions",
    }
    report["population"] = {
        "chronological": population(prepared["baseline"]),
        "random": population(prepared["d"]),
    }
    report["variants"] = [
        {
            **variant,
            **fits[variant["id"]],
            **metrics(
                prepared["baseline" if variant["split"] == "chronological" else "d"],
                scores[variant["id"]],
            ),
        }
        for variant in VARIANTS
    ]

    print("bootstrapping")
    y_test = {
        "chronological": prepared["baseline"]["test"][config.TARGET].to_numpy(dtype="int64"),
        "random": prepared["d"]["test"][config.TARGET].to_numpy(dtype="int64"),
    }
    groups = {
        "chronological": [v["id"] for v in VARIANTS if v["split"] == "chronological"],
        "random": [v["id"] for v in VARIANTS if v["split"] == "random"],
    }
    bootstrap: dict[str, Any] = {}
    drawn: dict[tuple[str, str, str], dict[str, Any]] = {}
    for metric, resample in (("pr_auc", "rows"), ("roc_auc", "rows"), ("pr_auc", "groups")):
        label = f"{metric}_by_{'row' if resample == 'rows' else 'card'}"
        bootstrap[label] = {}
        for kind, names in groups.items():
            print(f"  {label} on the {kind} test rows ({len(names)} models)")
            drawn[(metric, resample, kind)] = evaluation.bootstrap_draws(
                y_test[kind],
                {name: scores[name]["test"] for name in names},
                n_boot,
                config.SEED if kind == "chronological" else config.SEED + 1,
                groups=keys[kind]["test"] if resample == "groups" else None,
                metric=metric,
            )
            bootstrap[label][kind] = evaluation.paired_summary(drawn[(metric, resample, kind)])

    deltas: dict[str, Any] = {}
    for variant in VARIANTS:
        if variant["id"] == "baseline":
            continue
        entry: dict[str, Any] = {"against": "baseline", "split": variant["split"]}
        for metric, resample in (("pr_auc", "rows"), ("roc_auc", "rows"), ("pr_auc", "groups")):
            label = f"{metric}_by_{'row' if resample == 'rows' else 'card'}"
            if variant["split"] == "chronological":
                pair = next(
                    p
                    for p in bootstrap[label]["chronological"]["pairs"]
                    if {p["first"], p["second"]} == {variant["id"], "baseline"}
                )
                sign = 1.0 if pair["first"] == variant["id"] else -1.0
                difference = dict(pair["difference"])
                if sign < 0:
                    difference = {
                        "point": -difference["point"],
                        "low": -difference["high"],
                        "high": -difference["low"],
                        "half_width": difference["half_width"],
                        "bootstrap_mean": -difference["bootstrap_mean"],
                    }
                entry[label] = {
                    "resample": "paired",
                    "variant": bootstrap[label]["chronological"]["per_model"][variant["id"]],
                    "baseline": bootstrap[label]["chronological"]["per_model"]["baseline"],
                    "difference": difference,
                    "sign_agreement": pair["sign_agreement"],
                    "excludes_zero": pair["excludes_zero"],
                }
            else:
                independent = evaluation.independent_difference(
                    drawn[(metric, resample, "random")],
                    variant["id"],
                    drawn[(metric, resample, "chronological")],
                    "baseline",
                )
                entry[label] = {
                    "resample": "independent: the variant is scored on the random test rows, "
                    "the baseline on the chronological ones",
                    "variant": bootstrap[label]["random"]["per_model"][variant["id"]],
                    "baseline": bootstrap[label]["chronological"]["per_model"]["baseline"],
                    "difference": independent["difference"],
                    "sign_agreement": independent["sign_agreement"],
                    "excludes_zero": independent["excludes_zero"],
                }
        primary = entry["pr_auc_by_row"]
        entry["inflates"] = bool(primary["difference"]["point"] > 0 and primary["excludes_zero"])
        deltas[variant["id"]] = entry
    report["bootstrap"] = bootstrap
    report["deltas"] = {
        "rule": "a variant inflates the metric when the 95 percent interval of its test PR-AUC "
        "difference against the baseline, on the row resample, excludes zero and the point is "
        "positive; otherwise it is recorded as a null. Chronological variants are paired on "
        "identical resamples; the random-split variants are scored on different rows and their "
        "interval is the difference of independent draws",
        "n_boot": n_boot,
        "per_variant": deltas,
    }
    report["seconds"] = time.perf_counter() - started
    write(report, OUT_PATH)
    for variant in VARIANTS:
        m = next(v for v in report["variants"] if v["id"] == variant["id"])
        line = f"  {variant['id']:32s} test PR-AUC {m['test']['pr_auc']:.4f}  ROC-AUC {m['test']['roc_auc']:.4f}"
        if variant["id"] in deltas:
            d = deltas[variant["id"]]["pr_auc_by_row"]["difference"]
            line += f"  delta {d['point']:+.4f} ({d['low']:+.4f} to {d['high']:+.4f})"
        print(line)
    print(f"done in {report['seconds']:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7 experiment 1: the leakage delta.")
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    parser.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    parser.add_argument(
        "--sample-fraction",
        type=float,
        default=None,
        help="smoke run on a seeded row sample; the artifact it writes is not the committed one",
    )
    args = parser.parse_args()
    run(args.transactions, args.identity, args.n_boot, args.sample_fraction)


if __name__ == "__main__":
    main()
