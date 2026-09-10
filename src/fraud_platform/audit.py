"""First-pass audit of the IEEE-CIS training data.

Narrow on purpose. This answers only what is needed to fix the entity key and the split:
shape, the meaning of TransactionDT, label granularity, entity key candidates, identity join
coverage, and the chronological split boundaries. Distribution, missingness and correlation
work belongs to stage 2.

Every public function returns a dict carrying its numbers and the expression that produced
them, so reports/audit.json can be read without also reading this file.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86_400

# Loading all 394 transaction columns costs several GB and none of the audit questions need
# them. The column count is read separately from the header.
AUDIT_COLUMNS = ["TransactionID", "isFraud", "TransactionDT", "card1", "addr1", "D1"]

ENTITY_KEYS: dict[str, list[str]] = {
    "card1": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_addr1_cardday": ["card1", "addr1", "card_start_day"],
}


def _float(value: Any) -> float:
    """Cast a numpy scalar to a plain float so json.dump does not choke on it."""
    return float(value)


def _int(value: Any) -> int:
    return int(value)


def _mean_or_none(values: pd.Series) -> float | None:
    """Mean of a possibly empty selection. Empty gives None, not NaN, so the report stays
    valid JSON."""
    if len(values) == 0:
        return None
    return float(values.mean())


def _ratio(numerator: float, denominator: float) -> float | None:
    """A rate, or None when the denominator is empty.

    None reaches the report as JSON null, which reads as "not defined on this input" rather
    than as a zero that a reader could mistake for a measurement.
    """
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def load_transactions(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the audit subset of train_transaction.csv."""
    return pd.read_csv(path, usecols=columns if columns is not None else AUDIT_COLUMNS)


def count_columns(path: str) -> int:
    """Column count without reading a single row of data."""
    return len(pd.read_csv(path, nrows=0).columns)


def add_card_start_day(df: pd.DataFrame) -> pd.DataFrame:
    """Attach floor(TransactionDT / 86400 - D1), the candidate card-start-day component.

    D1 is documented as days since the card began, so this difference should be near constant
    for a given card. Whether it actually is gets measured in `verify_card_start_day`, not
    assumed here.
    """
    out = df.copy()
    out["card_start_day"] = np.floor(out["TransactionDT"] / SECONDS_PER_DAY - out["D1"])
    return out


def basic_shape(df: pd.DataFrame, n_columns: int) -> dict[str, Any]:
    n_rows = len(df)
    n_fraud = _int(df["isFraud"].sum())
    return {
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_fraud": n_fraud,
        "n_non_fraud": n_rows - n_fraud,
        "fraud_rate": _float(n_fraud / n_rows),
        "command": (
            "len(df); len(pd.read_csv(path, nrows=0).columns); "
            "df['isFraud'].sum(); df['isFraud'].mean()"
        ),
    }


def transaction_dt_profile(df: pd.DataFrame) -> dict[str, Any]:
    """What TransactionDT is, measured rather than assumed.

    The field is a relative offset, not wall-clock time. Two things are checked here. First,
    the greatest common divisor of the gaps between consecutive distinct values, which gives
    the granularity of the clock. Second, the count of transactions per hour position within a
    86400-unit cycle: if 86400 units really is a day, that profile shows the diurnal shape of
    card activity, with a deep overnight trough. A unit other than a second would smear it flat.
    """
    dt = df["TransactionDT"]
    unique_sorted = np.sort(dt.unique())
    gaps = np.diff(unique_sorted)
    granularity = _int(np.gcd.reduce(gaps.astype(np.int64)))

    hour_position = (dt // 3600) % 24
    per_hour = hour_position.value_counts().sort_index()
    trough_hour = _int(per_hour.idxmin())
    peak_hour = _int(per_hour.idxmax())

    span_seconds = _int(dt.max() - dt.min())
    return {
        "min": _int(dt.min()),
        "max": _int(dt.max()),
        "span_seconds": span_seconds,
        "span_days": _float(span_seconds / SECONDS_PER_DAY),
        "is_integer_valued": bool(dt.dtype.kind in "iu"),
        "granularity_units": granularity,
        "is_monotonic_increasing_in_file_order": bool(dt.is_monotonic_increasing),
        "n_distinct": _int(dt.nunique()),
        "n_rows": len(df),
        "diurnal_check": {
            "counts_by_hour_position": {str(k): _int(v) for k, v in per_hour.items()},
            "trough_hour_position": trough_hour,
            "peak_hour_position": peak_hour,
            "peak_over_trough_ratio": _float(per_hour.max() / per_hour.min()),
        },
        "command": (
            "dt = df['TransactionDT']; np.gcd.reduce(np.diff(np.sort(dt.unique()))); "
            "dt.is_monotonic_increasing; ((dt // 3600) % 24).value_counts().sort_index()"
        ),
    }


def label_granularity(df: pd.DataFrame, key: list[str]) -> dict[str, Any]:
    """Are labels transaction-level or entity-level?

    Reported over entities with two or more transactions. An entity is pure when every one of
    its transactions carries the same label.
    """
    grouped = df.groupby(key, dropna=False)["isFraud"].agg(["size", "sum"])
    multi = grouped[grouped["size"] >= 2]
    n_multi = len(multi)

    all_fraud = _int((multi["sum"] == multi["size"]).sum())
    all_non_fraud = _int((multi["sum"] == 0).sum())
    mixed = n_multi - all_fraud - all_non_fraud

    return {
        "key": key,
        "n_entities_with_2_or_more_transactions": n_multi,
        "n_all_fraud": all_fraud,
        "n_all_non_fraud": all_non_fraud,
        "n_mixed": mixed,
        "fraction_all_fraud": _ratio(all_fraud, n_multi),
        "fraction_all_non_fraud": _ratio(all_non_fraud, n_multi),
        "fraction_mixed": _ratio(mixed, n_multi),
        "command": (
            f"g = df.groupby({key}, dropna=False)['isFraud'].agg(['size', 'sum']); "
            "m = g[g['size'] >= 2]; (m['sum'] == m['size']).mean(), (m['sum'] == 0).mean()"
        ),
    }


def card1_profile(df: pd.DataFrame) -> dict[str, Any]:
    counts = df["card1"].value_counts()
    return {
        "cardinality": _int(df["card1"].nunique(dropna=False)),
        "n_missing": _int(df["card1"].isna().sum()),
        "transactions_per_card1": {
            "median": _float(counts.median()),
            "p90": _float(counts.quantile(0.90)),
            "max": _int(counts.max()),
            "mean": _float(counts.mean()),
        },
        "n_appearing_exactly_once": _int((counts == 1).sum()),
        "share_of_card1_values_appearing_exactly_once": _float((counts == 1).mean()),
        "share_of_rows_on_single_transaction_card1": _float(_int((counts == 1).sum()) / len(df)),
        "command": (
            "c = df['card1'].value_counts(); df['card1'].nunique(); "
            "c.median(), c.quantile(0.90), c.max(), (c == 1).mean()"
        ),
    }


def verify_card_start_day(df: pd.DataFrame, key: list[str]) -> dict[str, Any]:
    """Check the claim that TransactionDT/86400 - D1 is near constant within a card.

    Reported over groups with two or more non-null values. Constant means one distinct value.
    """
    subset = df.dropna(subset=["card_start_day"])
    grouped = subset.groupby(key, dropna=False)["card_start_day"]
    sizes = grouped.size()
    n_distinct = grouped.nunique()
    spread = grouped.max() - grouped.min()

    multi = sizes >= 2
    n_multi = _int(multi.sum())
    single_valued = _int((n_distinct[multi] == 1).sum())
    within_one_day = _int((spread[multi] <= 1).sum())

    return {
        "key": key,
        "n_groups_with_2_or_more_non_null": n_multi,
        "n_groups_single_valued": single_valued,
        "fraction_single_valued": _ratio(single_valued, n_multi),
        "fraction_spread_within_1_day": _ratio(within_one_day, n_multi),
        "median_spread_days": _float(spread[multi].median()),
        "p90_spread_days": _float(spread[multi].quantile(0.90)),
        "distinct_values_per_group": {
            "median": _float(n_distinct[multi].median()),
            "p90": _float(n_distinct[multi].quantile(0.90)),
            "max": _int(n_distinct[multi].max()),
        },
        "n_rows_with_null_card_start_day": _int(df["card_start_day"].isna().sum()),
        "command": (
            f"g = df.dropna(subset=['card_start_day']).groupby({key}, dropna=False)"
            "['card_start_day']; s = g.size(); (g.nunique()[s >= 2] == 1).mean()"
        ),
    }


def entity_key_stats(df: pd.DataFrame, name: str, key: list[str]) -> dict[str, Any]:
    """Entity count, transactions per entity, and label purity for one candidate key."""
    grouped = df.groupby(key, dropna=False)["isFraud"].agg(["size", "sum"])
    sizes = grouped["size"]
    multi = grouped[sizes >= 2]
    n_multi = len(multi)
    pure = _int(((multi["sum"] == 0) | (multi["sum"] == multi["size"])).sum())

    n_rows_with_null_component = _int(df[key].isna().any(axis=1).sum())

    return {
        "name": name,
        "key": key,
        "n_entities": len(grouped),
        "n_rows_with_null_key_component": n_rows_with_null_component,
        "transactions_per_entity": {
            "mean": _float(sizes.mean()),
            "median": _float(sizes.median()),
            "p90": _float(sizes.quantile(0.90)),
            "max": _int(sizes.max()),
        },
        "n_singleton_entities": _int((sizes == 1).sum()),
        "share_singleton_entities": _float((sizes == 1).mean()),
        "share_rows_in_singleton_entities": _float(_int((sizes == 1).sum()) / len(df)),
        "n_entities_with_2_or_more_transactions": n_multi,
        "n_label_pure_multi_entities": pure,
        "label_purity_multi_entities": _ratio(pure, n_multi),
        "command": (
            f"g = df.groupby({key}, dropna=False)['isFraud'].agg(['size', 'sum']); "
            "m = g[g['size'] >= 2]; "
            "((m['sum'] == 0) | (m['sum'] == m['size'])).mean(); g['size'].describe()"
        ),
    }


def identity_join_coverage(df: pd.DataFrame, identity_ids: pd.Series) -> dict[str, Any]:
    """How many transactions have a matching row in train_identity, overall and by label."""
    has_identity = df["TransactionID"].isin(set(identity_ids))
    fraud_mask = df["isFraud"] == 1

    return {
        "n_identity_rows": _int(len(identity_ids)),
        "n_identity_ids_not_in_transactions": _int(
            (~identity_ids.isin(set(df["TransactionID"]))).sum()
        ),
        "coverage_overall": _mean_or_none(has_identity),
        "coverage_fraud": _mean_or_none(has_identity[fraud_mask]),
        "coverage_non_fraud": _mean_or_none(has_identity[~fraud_mask]),
        "n_matched": _int(has_identity.sum()),
        "command": (
            "ids = pd.read_csv(identity_path, usecols=['TransactionID'])['TransactionID']; "
            "h = df['TransactionID'].isin(set(ids)); "
            "h.mean(), h[df['isFraud'] == 1].mean(), h[df['isFraud'] == 0].mean()"
        ),
    }


def chronological_split(
    df: pd.DataFrame, fractions: tuple[float, float, float] = (0.70, 0.15, 0.15)
) -> dict[str, Any]:
    """Cut the data by TransactionDT at row-count quantiles.

    Boundaries are TransactionDT values, and assignment is `dt < boundary`, so a timestamp
    shared by several transactions never straddles two splits. That makes the realised row
    fractions land close to but not exactly on the requested ones, which is the honest
    trade: a split that leaks a timestamp across the boundary is worse than a split that is
    0.1 percent off.
    """
    train_frac, val_frac, _ = fractions
    dt = df["TransactionDT"]
    train_boundary = _int(np.quantile(dt, train_frac))
    val_boundary = _int(np.quantile(dt, train_frac + val_frac))

    masks = {
        "train": dt < train_boundary,
        "val": (dt >= train_boundary) & (dt < val_boundary),
        "test": dt >= val_boundary,
    }

    splits: dict[str, Any] = {}
    for name, mask in masks.items():
        part = df[mask]
        span = _int(part["TransactionDT"].max() - part["TransactionDT"].min())
        splits[name] = {
            "n_rows": len(part),
            "row_fraction": _float(len(part) / len(df)),
            "dt_min": _int(part["TransactionDT"].min()),
            "dt_max": _int(part["TransactionDT"].max()),
            "span_days": _float(span / SECONDS_PER_DAY),
            "day_index_min": math.floor(_int(part["TransactionDT"].min()) / SECONDS_PER_DAY),
            "day_index_max": math.floor(_int(part["TransactionDT"].max()) / SECONDS_PER_DAY),
            "n_fraud": _int(part["isFraud"].sum()),
            "fraud_rate": _float(part["isFraud"].mean()),
        }

    return {
        "requested_fractions": list(fractions),
        "boundaries": {
            "train_end_exclusive_dt": train_boundary,
            "val_end_exclusive_dt": val_boundary,
            "train_end_day_index": train_boundary / SECONDS_PER_DAY,
            "val_end_day_index": val_boundary / SECONDS_PER_DAY,
        },
        "splits": splits,
        "command": (
            "b1, b2 = np.quantile(df['TransactionDT'], 0.70), "
            "np.quantile(df['TransactionDT'], 0.85); "
            "df[df['TransactionDT'] < b1], "
            "df[(df['TransactionDT'] >= b1) & (df['TransactionDT'] < b2)], "
            "df[df['TransactionDT'] >= b2]"
        ),
    }


def entity_overlap(df: pd.DataFrame, split: dict[str, Any], key: list[str]) -> dict[str, Any]:
    """How much of val and test is made of entities the training window never saw."""
    dt = df["TransactionDT"]
    b1 = split["boundaries"]["train_end_exclusive_dt"]
    b2 = split["boundaries"]["val_end_exclusive_dt"]

    def entity_series(part: pd.DataFrame) -> pd.Series:
        # One string per row so a multi-column key can go into a set. A null component becomes
        # the literal "NA", which puts all rows missing that component into one bucket. That
        # matches groupby(dropna=False) elsewhere in this module, so the counts agree.
        parts = [part[column].astype("string").fillna("NA") for column in key]
        joined = parts[0]
        for column_values in parts[1:]:
            joined = joined + "|" + column_values
        return joined

    train = df[dt < b1]
    val = df[(dt >= b1) & (dt < b2)]
    test = df[dt >= b2]

    train_entities = set(entity_series(train))

    out: dict[str, Any] = {"key": key, "n_train_entities": len(train_entities)}
    for name, part in (("val", val), ("test", test)):
        ents = entity_series(part)
        unseen_rows = ~ents.isin(train_entities)
        unique_ents = set(ents)
        unseen_entities = unique_ents - train_entities
        out[name] = {
            "n_entities": len(unique_ents),
            "n_entities_unseen_in_train": len(unseen_entities),
            "share_entities_unseen_in_train": _ratio(len(unseen_entities), len(unique_ents)),
            "n_rows_on_unseen_entities": _int(unseen_rows.sum()),
            "share_rows_on_unseen_entities": _mean_or_none(unseen_rows),
            "fraud_rate_on_unseen_entity_rows": _mean_or_none(part.loc[unseen_rows, "isFraud"]),
            "fraud_rate_on_seen_entity_rows": _mean_or_none(part.loc[~unseen_rows, "isFraud"]),
        }

    out["command"] = (
        f"e = lambda p: reduce(lambda a, b: a + '|' + b, "
        f"[p[c].astype('string').fillna('NA') for c in {key}]); "
        "set(e(test)) - set(e(train))"
    )
    return out


def d1_profile(df: pd.DataFrame) -> dict[str, Any]:
    """What D1 looks like, as the ground for treating it as a day counter.

    Two things matter for the entity key. D1 is non-negative and bounded well above the 182 day
    observation window, which is consistent with a counter that started before the data did.
    And the implied card start day never falls after the transaction itself, which a counter
    running the other way would violate.
    """
    d1 = df["D1"]
    day_index = np.floor(df["TransactionDT"] / SECONDS_PER_DAY)
    start_after_transaction = _int((df["card_start_day"] > day_index).sum())

    return {
        "min": _float(d1.min()),
        "max": _float(d1.max()),
        "n_null": _int(d1.isna().sum()),
        "share_null": _float(d1.isna().mean()),
        "share_zero": _float((d1 == 0).mean()),
        "n_negative": _int((d1 < 0).sum()),
        "n_rows_with_card_start_day_after_transaction_day": start_after_transaction,
        "observation_window_days": _float(
            (df["TransactionDT"].max() - df["TransactionDT"].min()) / SECONDS_PER_DAY
        ),
        "command": (
            "d1 = df['D1']; d1.min(), d1.max(), (d1 == 0).mean(), (d1 < 0).sum(); "
            "(df['card_start_day'] > np.floor(df['TransactionDT'] / 86400)).sum()"
        ),
    }
