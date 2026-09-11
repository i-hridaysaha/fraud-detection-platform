"""Stage 4 features. Builds the behavioural features and writes what stages 6, 8 and 9 read.

Usage:
    make features
    .venv/bin/python scripts/features.py
    .venv/bin/python scripts/features.py --sections velocity_c

One rule runs the stage and it is not enforced here, because it cannot be. The features are
point-in-time by construction, not by a guard: `fraud_platform.features` reads only the rows of
the stream that fall strictly before the row it is computing, so the whole frame is the correct
input and a window check would pass whatever it was handed. `tests/test_causality.py` is the
enforcement: brute-force recomputation over `TransactionDT < t`, plus truncation invariance, plus
the tie exclusion. This script reports the result of that suite rather than repeating it.

The one fitted object in the stage, the dist1 bucket edges, is fitted on the train split through
`assert_train_only` like everything in stage 3.

Sections and their outputs:

    catalogue    reports/feature_summary.json            every feature, its family, its window,
                                                         its serving state, its null rate per
                                                         split, its drift, and what the online
                                                         store has to hold
    velocity_c   reports/features/velocity_vs_c.json     this repo's point-in-time velocity
                                                         against the native C columns, at the
                                                         same grain
    merchant     reports/features/merchant_proxies.json  what merchant risk would need, what the
                                                         file holds instead
    duplicates   reports/features/duplicate_content.json the causal duplicate-content features
                                                         against stage 2's grouping
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config, data_loader, eda, encoders, features

SECTIONS = ("catalogue", "velocity_c", "merchant", "duplicates")

FEATURE_SUMMARY_PATH = config.REPORTS_DIR / "feature_summary.json"
FEATURES_DIR = config.REPORTS_DIR / "features"
EDA_DIR = config.REPORTS_DIR / "eda"

# The columns the features read, plus the C block the cross-check needs and the label. Reading a
# subset rather than the whole 434 column frame is what keeps this script inside a few GB.
INPUT_COLUMNS: tuple[str, ...] = (
    config.ID_COLUMN,
    config.TARGET,
    config.TIME_COLUMN,
    config.AMOUNT_COLUMN,
    "card1",
    "card2",
    "addr1",
    "addr2",
    "dist1",
    "D1",
    "ProductCD",
    "P_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
    *config.COUNT_COLUMNS,
)

# What a merchant risk feature would be built on, and what this file offers in its place. The
# regex is the screen: it runs over the real column names and the artifact records that it found
# nothing, so "there is no merchant column" is a measurement rather than a recollection.
MERCHANT_NAME_PATTERN = re.compile(
    r"merch|mcc|categor|\bsic\b|acquir|terminal|store|retail|seller|vendor", re.IGNORECASE
)

# The columns that stand in for a merchant, each with what it is a proxy for. Nothing here claims
# any of them identifies a merchant. ADR 0019.
MERCHANT_PROXIES: tuple[tuple[str, str], ...] = (
    (
        "ProductCD",
        "the product class of the purchase, five levels, the closest thing to a category",
    ),
    ("P_emaildomain", "the purchaser's email domain, which correlates with where they buy"),
    ("R_emaildomain", "the recipient's email domain, and its agreement with the purchaser's"),
    ("dist1", "a distance, unit not established, between two points the file does not name"),
    ("dist2", "a second distance on a different missingness block"),
    ("addr1", "the billing region, which is a property of the account and not of the seller"),
    ("addr2", "a coarser address field"),
    ("card4", "the card network"),
    ("card6", "the card funding type, credit against debit"),
)


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
        "stage": 4,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": (
            f"make features  # or: .venv/bin/python scripts/features.py --sections {section}"
        ),
        "command": command,
        "computed_on": (
            "the whole time-sorted stream. Every feature reads only rows strictly before its own "
            "timestamp, so the whole frame is the correct input; the one fitted object, the dist1 "
            "bucket edges, is fitted on the train split through assert_train_only"
        ),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False, sort_keys=False) + "\n")
    print(f"  wrote {path.relative_to(config.ROOT)}")


def load_inputs(transactions: str, identity: str) -> pd.DataFrame:
    frame = data_loader.add_entity_key(
        data_loader.load_raw(
            transactions_path=transactions, identity_path=identity, columns=list(INPUT_COLUMNS)
        )
    )
    return encoders.add_normalised_free_text(frame)


# --- catalogue --------------------------------------------------------------------------------


def build_catalogue(
    frame: pd.DataFrame,
    built: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    buckets: Mapping[str, Any],
    cfg: features.FeatureConfig,
) -> dict[str, Any]:
    """Every feature with its declared shape and its measured null rate per split.

    The declared half comes from `features.catalogue`, so the artifact cannot list a feature the
    module does not build or omit one it does. The measured half is the null rate per split and
    the PSI against the two later splits, which is the number stage 9 will want a baseline for.
    """
    report = envelope(
        "catalogue",
        ".venv/bin/python scripts/features.py --sections catalogue",
        int(masks["train"].sum()),
    )
    report["config"] = cfg.to_dict()
    report["n_features_derived"] = len(features.feature_names(cfg))
    report["n_features_listed"] = len(features.catalogue(cfg))
    report["families"] = {
        family: sorted(spec.name for spec in features.catalogue(cfg) if spec.family == family)
        for family in features.FAMILIES
    }
    report["running_state"] = {
        "definition": (
            "a feature requires running state when its value depends on earlier rows of a key, "
            "so a serving path has to hold something per key rather than computing from the "
            "request alone. state_key names which keyspace and state_fields names what is held"
        ),
        "n_requiring_state": sum(
            1 for spec in features.catalogue(cfg) if spec.requires_running_state
        ),
        "by_state_key": {
            key: sorted(spec.name for spec in features.catalogue(cfg) if spec.state_key == key)
            for key in (*features.GRAINS, "none")
        },
    }

    joined = pd.concat([frame[[config.TARGET, "D1"]], built], axis=1)
    per_feature: list[dict[str, Any]] = []
    for spec in features.catalogue(cfg):
        row = spec.to_dict()
        row["null_rate_by_split"] = {
            name: features.null_rates(joined[mask], [spec.name])[spec.name]
            for name, mask in masks.items()
        }
        if spec.name in joined.columns:
            train_values = joined.loc[masks["train"], spec.name]
            row["drift"] = {
                name: eda.population_stability_index(train_values, joined.loc[mask, spec.name])[
                    "psi"
                ]
                for name, mask in masks.items()
                if name != "train"
            }
        else:
            row["drift"] = None
        per_feature.append(row)
    report["per_feature"] = per_feature

    report["dist1_buckets"] = dict(buckets)
    report["state_footprint"] = {
        "whole_stream": features.state_footprint(frame, cfg),
        "train": features.state_footprint(frame[masks["train"]], cfg),
    }
    report["coverage"] = _coverage(frame, built, masks)
    report["entity_key_pins_addr1"] = _addr_degeneracy(frame, masks)
    report["tenure_origin"] = _tenure_origin(frame, masks)
    report["not_built"] = list(features.NOT_BUILT)
    return report


def _coverage(
    frame: pd.DataFrame, built: pd.DataFrame, masks: Mapping[str, pd.Series[bool]]
) -> dict[str, Any]:
    """How many rows have any history for the features to read.

    Stage 2 measured 0.5850 of train entities as singletons carrying 0.2331 of rows, and stage 1
    measured 0.6598 of test rows on an entity the training window never saw. This is the same
    fact stated as feature coverage: the share of rows where the entity has no earlier
    transaction at all, per split, and the fraud rate on each side of that line.
    """
    labels = frame[config.TARGET].to_numpy(dtype="int64")
    out: dict[str, Any] = {
        "definition": (
            "a row is cold for a grain when the stream holds no strictly earlier transaction on "
            "that key. Cold rows take the null branch of every comparison feature and zero on "
            "every count feature"
        )
    }
    for name, mask in masks.items():
        rows = mask.to_numpy(dtype=bool)
        split: dict[str, Any] = {"n_rows": int(rows.sum())}
        for label, column in (
            ("entity", "rec_prior_count"),
            ("card", features.velocity_name("count", "24h", "card")),
            ("content", "dup_prior_count"),
        ):
            if column not in built.columns:
                continue
            has_history = built[column].to_numpy() > 0
            cold = rows & ~has_history
            warm = rows & has_history
            split[label] = {
                "share_cold": float(cold.sum() / rows.sum()) if rows.sum() else None,
                "n_cold": int(cold.sum()),
                "fraud_rate_cold": float(labels[cold].mean()) if cold.sum() else None,
                "fraud_rate_warm": float(labels[warm].mean()) if warm.sum() else None,
                "interval_cold": eda.wilson_interval(int(labels[cold].sum()), int(cold.sum())),
                "interval_warm": eda.wilson_interval(int(labels[warm].sum()), int(warm.sum())),
            }
        if name != "train":
            seen = set(frame.loc[masks["train"], config.ENTITY_ID_COLUMN].unique())
            in_train = frame.loc[mask, config.ENTITY_ID_COLUMN].isin(seen).to_numpy(dtype=bool)
            any_prior = built.loc[mask, "rec_prior_count"].to_numpy() > 0
            split["entity_history_against_training_overlap"] = {
                "share_entity_seen_in_train": float(in_train.mean()),
                "share_with_any_earlier_row_in_the_stream": float(any_prior.mean()),
                "why_they_differ": (
                    "an entity the training window never saw can still have transacted earlier "
                    "inside this split, and at authorisation the online store holds that history "
                    "whatever window the model was fitted on. The first number is what a "
                    "train-fitted entity statistic covers, which is what ADR 0015 excluded. The "
                    "second is what a point-in-time behavioural feature covers"
                ),
            }
        out[name] = split
    out["note"] = (
        "the card and content lines use a window rather than all history, so 'cold' there means "
        "no earlier transaction inside that window and not no earlier transaction at all"
    )
    return out


def _addr_degeneracy(frame: pd.DataFrame, masks: Mapping[str, pd.Series[bool]]) -> dict[str, Any]:
    """The measurement behind ADR 0023: addr1 cannot vary inside an entity, and does inside a card.

    addr1 is a component of the ADR 0001 key, so this is a fact about the key rather than about
    the data, and measuring it is how the repo distinguishes the two.
    """
    train = frame[masks["train"]]
    out: dict[str, Any] = {
        "why": (
            "addr1 is a component of the ADR 0001 entity key, so a deviation from the entity's "
            "own modal addr1 is identically zero and carries nothing. The geography family is "
            "keyed on card1 instead"
        )
    }
    for grain, key in (("entity", config.ENTITY_ID_COLUMN), ("card", "card1")):
        grain_out: dict[str, Any] = {"n_keys": int(train[key].nunique())}
        for column in ("addr1", "addr2"):
            distinct = train.groupby(key, observed=True)[column].nunique(dropna=True)
            grain_out[column] = {
                "max_distinct_per_key": int(distinct.max()),
                "share_keys_with_more_than_one": float((distinct > 1).mean()),
                "mean_distinct_per_key": float(distinct.mean()),
            }
        out[grain] = grain_out
    return out


def _tenure_origin(frame: pd.DataFrame, masks: Mapping[str, pd.Series[bool]]) -> dict[str, Any]:
    """D1's normalised origin, measured and not shipped.

    The stage asked for D1 and its normalised origin. ADR 0013 already decided that the D-column
    normalisation is applied to no column, on a measurement of all 15. This re-measures the one
    column stage 4 was asked about, because a decision inherited without a number is not evidence,
    and because D1's origin is `card_start_day` without the floor and that is worth showing
    rather than asserting.
    """
    from fraud_platform import transforms

    with_origin = transforms.to_d_origin(frame, ["D1"])
    origin = transforms.d_origin_name("D1")
    train_values = with_origin.loc[masks["train"], origin]
    out: dict[str, Any] = {
        "column": origin,
        "shipped": False,
        "why_not": (
            "ADR 0013 measured the D-column normalisation raising PSI on all 15 columns and "
            "lowering single-feature validation AUC on 12. This column is card_start_day without "
            "the floor, which stage 2 measured at the highest PSI in the frame"
        ),
        "psi_against": {
            name: eda.population_stability_index(train_values, with_origin.loc[mask, origin])["psi"]
            for name, mask in masks.items()
            if name != "train"
        },
        "psi_of_d1_raw": {
            name: eda.population_stability_index(
                frame.loc[masks["train"], "D1"], frame.loc[mask, "D1"]
            )["psi"]
            for name, mask in masks.items()
            if name != "train"
        },
    }
    difference = (with_origin[origin] - frame[config.CARD_START_DAY_COLUMN]).dropna()
    out["against_card_start_day"] = {
        "relationship": "D1_origin = card_start_day + the fractional part of the transaction day",
        "min_difference": float(difference.min()),
        "max_difference": float(difference.max()),
        "n_compared": int(difference.size),
    }
    return out


# --- velocity against the native C columns -----------------------------------------------------


def build_velocity_c(
    frame: pd.DataFrame,
    built: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    cfg: features.FeatureConfig,
) -> dict[str, Any]:
    """This repo's point-in-time velocity against the native C columns, at the same grain.

    The comparison is only like for like at the card grain. Whatever the C columns are counting,
    they are described as card-associated counts, and this repo's entity key is finer than a card
    by construction, so an entity-grain velocity and a card-associated counter are two different
    quantities. Both grains are reported and the card grain is the comparison.

    What is measured here is correlation and single-feature information value. Model importances
    are a stage 6 measurement and are not attempted here: a model has not been fitted in this
    repo yet, and an importance without one would be an invention.
    """
    report = envelope(
        "velocity_c",
        ".venv/bin/python scripts/features.py --sections velocity_c",
        int(masks["train"].sum()),
    )
    report["what_the_c_columns_are"] = {
        "claim": "counts associated with the payment card",
        "status": "not established in this repo",
        "why": (
            "the competition's own data description is rendered client side and returned no text "
            "when fetched in stage 0 and again in stage 4, so the meaning of the C block is not "
            "restated anywhere here as fact. What is measured is their structure: stage 1 found "
            "C1 to C14 integer valued with no nulls, stage 2 found C8 with C10 at Spearman 0.9710 "
            "over all 413,378 train rows and named the block running counters, and the "
            "correlations below are against this repo's own counts"
        ),
    }
    train = masks["train"].to_numpy(dtype=bool)
    labels = frame[config.TARGET].to_numpy(dtype="int64")
    target = frame.loc[masks["train"], config.TARGET]
    c_columns = [c for c in config.COUNT_COLUMNS if c in frame.columns]
    velocity_counts = [
        features.velocity_name("count", suffix, grain)
        for grain in cfg.velocity_grains
        for suffix, _ in cfg.windows
    ]
    velocity_counts = [name for name in velocity_counts if name in built.columns]

    joined = pd.concat(
        [frame.loc[masks["train"], c_columns], built.loc[masks["train"], velocity_counts]], axis=1
    )
    pairs = [(own, native) for own in velocity_counts for native in c_columns]
    spearman = eda.exact_spearman(joined, pairs)

    per_feature: list[dict[str, Any]] = []
    for own in velocity_counts:
        row_values = {native: spearman.get(f"{own}|{native}") for native in c_columns}
        ranked = sorted(
            ((native, value) for native, value in row_values.items() if value is not None),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        spec = next(spec for spec in features.catalogue(cfg) if spec.name == own)
        iv = eda.information_value(built.loc[masks["train"], own], target)
        per_feature.append(
            {
                "feature": own,
                "grain": spec.grain,
                "window_seconds": spec.window_seconds,
                "information_value": iv["iv"],
                "n_bins": iv["n_bins"],
                "n_bins_requested": iv["n_bins_requested"],
                "zero_against_positive": eda.zero_against_positive(built[own], labels, train),
                "by_split": eda.zero_against_positive_by_split(built[own], labels, masks),
                "spearman_against_c": row_values,
                "closest_c_column": ranked[0][0] if ranked else None,
                "closest_abs_rho": abs(ranked[0][1]) if ranked else None,
                "max_abs_rho": max((abs(value) for _, value in ranked), default=None),
            }
        )
    report["per_feature"] = per_feature

    report["native_c_columns"] = [
        {
            "column": native,
            "information_value": eda.information_value(frame.loc[masks["train"], native], target)[
                "iv"
            ],
            "zero_against_positive": eda.zero_against_positive(frame[native], labels, train),
            "by_split": eda.zero_against_positive_by_split(frame[native], labels, masks),
            "max_abs_rho_against_own_velocity": max(
                (
                    abs(value)
                    for own in velocity_counts
                    if (value := spearman.get(f"{own}|{native}")) is not None
                ),
                default=None,
            ),
        }
        for native in c_columns
    ]

    finite = [abs(value) for value in spearman.values() if value is not None and np.isfinite(value)]
    report["summary"] = {
        "n_pairs": len(pairs),
        "n_pairs_measured": len(finite),
        "max_abs_rho": float(max(finite)) if finite else None,
        "median_abs_rho": float(np.median(finite)) if finite else None,
        "n_pairs_above_0.5": int(sum(1 for value in finite if value >= 0.5)),
        "n_pairs_above_0.8": int(sum(1 for value in finite if value >= 0.8)),
        "reading": (
            "a high correlation would mean this repo has reconstructed the native counters and "
            "one of the two is redundant. A low one means they count different things, and stage "
            "6 has two candidate blocks to measure rather than one"
        ),
        "on_the_information_values": (
            "the velocity counts are zero-inflated, so ten equal-frequency bins collapse and the "
            "information value understates them. n_bins records the collapse and "
            "zero_against_positive is the number to read instead. The C columns are counters over "
            "a longer history and do not have the problem, which is part of the comparison in "
            "itself: the two blocks are not zero on the same rows"
        ),
    }
    report["stage_6_owes"] = (
        "separate model importances for the two blocks, and a like-for-like model comparison with "
        "each block switched off through features.FeatureConfig.native_groups. ADR 0021"
    )
    del joined
    gc.collect()
    assert train.any()
    return report


# --- merchant risk ------------------------------------------------------------------------------


def build_merchant(frame: pd.DataFrame, masks: Mapping[str, pd.Series[bool]]) -> dict[str, Any]:
    """What merchant risk would need, and what the file holds instead.

    Two measurements and one statement. The measurements: the full column inventory screened for
    a merchant or category name, which finds nothing, and the stage 2 information value of each
    proxy. The statement: a proxy's information value is not a merchant risk feature's
    information value, and no number here should be read as one. ADR 0019.
    """
    report = envelope(
        "merchant",
        ".venv/bin/python scripts/features.py --sections merchant",
        int(masks["train"].sum()),
    )
    transaction_columns = list(pd.read_csv(config.TRANSACTIONS_PATH, nrows=0).columns)
    identity_columns = [
        c for c in pd.read_csv(config.IDENTITY_PATH, nrows=0).columns if c != config.ID_COLUMN
    ]
    all_columns = transaction_columns + identity_columns
    hits = [c for c in all_columns if MERCHANT_NAME_PATTERN.search(c)]
    numbered = re.compile(r"^(V|C|D|M|id_)\d+$")
    report["column_inventory"] = {
        "n_columns": len(all_columns),
        "n_transaction_columns": len(transaction_columns),
        "n_identity_columns": len(identity_columns),
        "screen_pattern": MERCHANT_NAME_PATTERN.pattern,
        "n_name_matches": len(hits),
        "name_matches": hits,
        "unnumbered_columns": [c for c in all_columns if not numbered.match(c)],
        "reading": (
            "every column is either a numbered member of the V, C, D, M or id_ blocks or one of "
            "the named columns listed. None of the named columns is a merchant identifier or a "
            "merchant category code, and the numbered blocks are unlabelled, so nothing in the "
            "file can be asserted to identify a seller"
        ),
    }

    bivariate = json.loads((EDA_DIR / "bivariate.json").read_text())
    iv_by_column = {str(row["column"]): row for row in bivariate["per_feature"]}
    quality = json.loads((EDA_DIR / "quality.json").read_text())
    proxies: list[dict[str, Any]] = []
    for column, what in MERCHANT_PROXIES:
        row = iv_by_column.get(column)
        proxies.append(
            {
                "column": column,
                "proxy_for": what,
                "present_in_file": column in all_columns,
                "information_value": None if row is None else float(row["iv"]),
                "information_value_present_bins": (
                    None if row is None else float(row["iv_present_bins"])
                ),
                "source": "reports/eda/bivariate.json per_feature, read not recomputed",
            }
        )
    report["proxies"] = proxies
    report["email_agreement"] = {
        "why_here": (
            "the strongest of the proxies is not a column but a comparison between two: stage 2 "
            "measured purchaser and recipient domain agreement carrying more rate than either "
            "domain alone"
        ),
        "measured": quality["email_domains"].get("agreement"),
        "source": "reports/eda/quality.json email_domains",
    }
    report["not_comparable"] = (
        "None of these is a merchant risk feature and no number above should be read as one. A "
        "merchant risk feature is a statistic over a seller's own transaction history: its "
        "chargeback rate, its dispute rate, the share of its volume that is card-not-present. "
        "Those are computed over a merchant identifier and this file has none, so the feature "
        "cannot be built here at any strength. Published fraud results that include merchant risk "
        "are therefore not comparable with anything in this repo, in either direction, and the "
        "gap is a property of the dataset rather than of the pipeline. ADR 0019"
    )
    return report


# --- duplicate content --------------------------------------------------------------------------


def build_duplicates(
    frame: pd.DataFrame,
    built: pd.DataFrame,
    masks: Mapping[str, pd.Series[bool]],
    cfg: features.FeatureConfig,
) -> dict[str, Any]:
    """The causal duplicate-content features against stage 2's own grouping.

    Stage 2 asked the question with a groupby, which marks every member of a group including the
    first, and it is not a feature: at scoring time the later members of a group do not exist
    yet. The causal version marks a row only when an identical row already happened. The two
    counts should differ by the first row of each group and by the tied rows, and the artifact
    reports both so the difference is a number rather than a claim. ADR 0022.
    """
    report = envelope(
        "duplicates",
        ".venv/bin/python scripts/features.py --sections duplicates",
        int(masks["train"].sum()),
    )
    quality = json.loads((EDA_DIR / "quality.json").read_text())
    stage_2 = quality["duplicates"]["repeated_core_fields"]
    report["stage_2_grouping"] = {
        "columns": list(stage_2["columns"]),
        "n_groups": int(stage_2["n_groups"]),
        "n_rows_in_groups": int(stage_2["n_rows_in_groups"]),
        "n_extra_rows": int(stage_2["n_extra_rows"]),
        "fraud_rate_in_groups": float(stage_2["fraud_rate_in_groups"]),
        "fraud_rate_outside_groups": float(stage_2["fraud_rate_outside_groups"]),
        "source": "reports/eda/quality.json duplicates.repeated_core_fields, read not recomputed",
        "why_it_is_not_a_feature": (
            "a groupby marks every member of a group, including the first, from statistics that "
            "include the group's later rows. At authorisation those rows do not exist"
        ),
    }
    report["strict_content_grouping"] = {
        "definition": quality["duplicates"]["same_content"]["definition"],
        "n_groups": int(quality["duplicates"]["same_content"]["n_groups"]),
        "n_rows_in_groups": int(quality["duplicates"]["same_content"]["n_rows_in_groups"]),
        "fraud_rate_in_groups": quality["duplicates"]["same_content"]["fraud_rate_in_groups"],
        "why_not_used": (
            "10 rows in 413,378 and no fraud among them, because the C and V blocks are running "
            "counters so two rows that are the same purchase are never the same row. The core "
            "field definition is the one with something in it"
        ),
    }

    train = masks["train"].to_numpy(dtype=bool)
    labels = frame[config.TARGET].to_numpy(dtype="int64")
    prior = built["dup_prior_count"].to_numpy()
    marked = train & (prior > 0)
    unmarked = train & (prior == 0)

    ts = frame[config.TIME_COLUMN].to_numpy(dtype="int64")
    content = features.content_codes(frame)
    tied = pd.DataFrame({"c": content, "t": ts}).duplicated(keep=False).to_numpy() & train

    report["causal_grouping"] = {
        "definition": (
            "a row is marked when at least one strictly earlier transaction agrees with it on "
            f"{list(features.CONTENT_COLUMNS)}. Same-timestamp ties are excluded, as everywhere"
        ),
        "n_rows_marked": int(marked.sum()),
        "share_rows_marked": float(marked.sum() / train.sum()),
        "fraud_rate_marked": float(labels[marked].mean()),
        "fraud_rate_unmarked": float(labels[unmarked].mean()),
        "interval_marked": eda.wilson_interval(int(labels[marked].sum()), int(marked.sum())),
        "interval_unmarked": eda.wilson_interval(int(labels[unmarked].sum()), int(unmarked.sum())),
        "lift": float(labels[marked].mean() / labels[unmarked].mean()),
    }
    report["causal_grouping_by_split"] = {
        "why": (
            "stage 2 measured this grouping on train and so did the block above. A separation on "
            "the training window is a candidate and not a result, which is the precedent ADR 0009 "
            "set, so the same comparison is made on val and test whether or not it survives"
        ),
        **eda.zero_against_positive_by_split(built["dup_prior_count"], labels, masks),
    }
    report["against_stage_2"] = {
        "stage_2_n_extra_rows": int(stage_2["n_extra_rows"]),
        "causal_n_rows_marked": int(marked.sum()),
        "difference": int(stage_2["n_extra_rows"]) - int(marked.sum()),
        "n_tied_rows_in_content_groups": int(tied.sum()),
        "why_they_can_differ": (
            "stage 2's n_extra_rows is every group member beyond the first in file order. The "
            "causal count is every row with a strictly earlier identical one, which drops a row "
            "whose only identical companion shares its timestamp"
        ),
    }

    per_feature: list[dict[str, Any]] = []
    target = frame.loc[masks["train"], config.TARGET]
    for name in ("dup_prior_count", "dup_seconds_since_prev", "dup_prior_count_1h"):
        values = built.loc[masks["train"], name]
        iv = eda.information_value(values, target)
        entry: dict[str, Any] = {
            "feature": name,
            "information_value": iv["iv"],
            "n_bins": iv["n_bins"],
            "n_bins_requested": iv["n_bins_requested"],
            "null_rate_train": float(values.isna().mean()),
            "deciles": eda.decile_table(values, target),
        }
        if name != "dup_seconds_since_prev":
            entry["zero_against_positive"] = eda.zero_against_positive(built[name], labels, train)
            entry["by_split"] = eda.zero_against_positive_by_split(built[name], labels, masks)
        per_feature.append(entry)
    report["per_feature"] = per_feature
    report["window_seconds"] = cfg.duplicate_window_seconds
    report["decision"] = (
        "shipped. The rule was written before the numbers: ship when the causal version separates "
        "the label on the train split by more than the split resolves, and drop it otherwise. "
        "ADR 0022 records the rule, the numbers and the outcome"
    )
    return report


# --- entry point --------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4 features.")
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    parser.add_argument("--sections", nargs="*", default=list(SECTIONS), choices=list(SECTIONS))
    args = parser.parse_args()

    started = time.perf_counter()
    print("loading the columns the features read")
    frame = load_inputs(args.transactions, args.identity)
    print(f"  {len(frame):,} rows by {frame.shape[1]} columns")

    masks = data_loader.split_masks(frame)
    train = frame[masks["train"]]
    data_loader.assert_train_only(train, what="stage 4 dist1 bucket edges")
    print(
        f"  train {int(masks['train'].sum()):,} rows, val {int(masks['val'].sum()):,}, "
        f"test {int(masks['test'].sum()):,}"
    )

    cfg = features.DEFAULT_CONFIG
    buckets = features.fit_dist_buckets(train)
    del train
    gc.collect()

    print("building the features over the whole stream")
    build_started = time.perf_counter()
    built = features.build_features(frame, buckets, cfg)
    build_seconds = time.perf_counter() - build_started
    print(f"  {built.shape[1]} features on {len(built):,} rows in {build_seconds:.1f}s")

    for section in args.sections:
        print(f"section {section}")
        section_started = time.perf_counter()
        if section == "catalogue":
            report = build_catalogue(frame, built, masks, buckets, cfg)
            report["build_seconds"] = build_seconds
            path = FEATURE_SUMMARY_PATH
        elif section == "velocity_c":
            report = build_velocity_c(frame, built, masks, cfg)
            path = FEATURES_DIR / "velocity_vs_c.json"
        elif section == "merchant":
            report = build_merchant(frame, masks)
            path = FEATURES_DIR / "merchant_proxies.json"
        else:
            report = build_duplicates(frame, built, masks, cfg)
            path = FEATURES_DIR / "duplicate_content.json"
        report["seconds"] = time.perf_counter() - section_started
        write(report, path)
        del report
        gc.collect()

    print(f"\ndone in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
