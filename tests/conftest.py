"""Fixtures shared by the stage 4 test modules.

Only one thing lives here and it is here because two modules need the same frame: a generated
transaction stream containing every case the features have a branch for. `tests/test_causality.py`
checks the causal rule against it and `tests/test_features_units.py` checks everything else, and a
second copy of the builder would be a second definition of what "a frame with ties in it" means.

Earlier stages build their frames inside their own test modules, which stays true. This is the
first case of two modules sharing one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, encoders, features

DAY = config.SECONDS_PER_DAY


def build_feature_frame(n_days: int = 30, per_day: int = 30) -> pd.DataFrame:
    """A frame built to contain every case the features branch on.

    Deliberately present: same-timestamp ties on one entity and on one card, entities with a
    single transaction, cards using several addresses, repeated amounts inside a day, rows with no
    identity record at all and rows with part of one, and repeated content on the six core fields.
    """
    rng = np.random.default_rng(config.SEED)
    n = n_days * per_day
    days = np.repeat(np.arange(1, n_days + 1), per_day)
    ts = (days * DAY + rng.integers(0, DAY, n)).astype("int64")

    frame = pd.DataFrame(
        {
            config.ID_COLUMN: np.arange(2_987_000, 2_987_000 + n, dtype="int64"),
            config.TARGET: (rng.random(n) < 0.06).astype("int8"),
            config.TIME_COLUMN: ts,
            # A small amount set, so a repeat inside a window and a repeated content key happen.
            config.AMOUNT_COLUMN: rng.choice([25.0, 68.95, 117.0, 250.5, 499.99], n),
            "card1": rng.integers(1_000, 1_012, n).astype("float64"),
            "card2": rng.choice([100.0, 200.0, np.nan], n),
            "addr1": rng.choice([100.0, 200.0, 300.0, np.nan], n, p=[0.4, 0.3, 0.2, 0.1]),
            "addr2": rng.choice([87.0, 60.0, np.nan], n, p=[0.6, 0.3, 0.1]),
            "dist1": rng.choice([1.0, 3.0, 12.0, 60.0, 400.0, np.nan], n),
            "D1": rng.integers(0, 20, n).astype("float64"),
            "ProductCD": rng.choice(["W", "C", "H"], n),
            "P_emaildomain": rng.choice(["a.com", "b.com", None], n),
            "DeviceType": rng.choice(["mobile", "desktop", None], n, p=[0.3, 0.3, 0.4]),
            "id_30": rng.choice(["ios 11.1.2", "windows 10", None], n, p=[0.3, 0.3, 0.4]),
            "DeviceInfo": rng.choice(
                ["Windows", "SM-J700M Build/MMB29K", "iOS Device", None],
                n,
                p=[0.2, 0.2, 0.2, 0.4],
            ),
            "id_31": rng.choice(
                ["chrome 62.0", "safari generic", "mobile safari 11.0", None],
                n,
                p=[0.25, 0.15, 0.2, 0.4],
            ),
        }
    )
    frame = frame.sort_values(config.TIME_COLUMN, kind="stable").reset_index(drop=True)

    # Forced ties. Three rows share a timestamp on one entity and two more on a different card.
    # Without these the tie branch is never reached and the exclusion is untested.
    frame.loc[10:12, config.TIME_COLUMN] = int(frame.loc[10, config.TIME_COLUMN])
    frame.loc[10:12, "card1"] = 1_000.0
    frame.loc[10:12, "addr1"] = 100.0
    frame.loc[10:12, "D1"] = 5.0
    frame.loc[40:41, config.TIME_COLUMN] = int(frame.loc[40, config.TIME_COLUMN])
    frame.loc[40:41, "card1"] = 1_001.0

    frame = frame.sort_values(config.TIME_COLUMN, kind="stable").reset_index(drop=True)
    frame[config.CARD_START_DAY_COLUMN] = np.floor(frame[config.TIME_COLUMN] / DAY - frame["D1"])
    parts = [
        frame[column].astype("string").fillna(config.ENTITY_NULL_TOKEN)
        for column in config.ENTITY_KEY_COLUMNS
    ]
    entity = parts[0]
    for part in parts[1:]:
        entity = entity + "|" + part
    frame[config.ENTITY_ID_COLUMN] = entity
    return encoders.add_normalised_free_text(frame)


@pytest.fixture(scope="session")
def feature_frame() -> pd.DataFrame:
    return build_feature_frame()


@pytest.fixture(scope="session")
def dist_buckets(feature_frame: pd.DataFrame) -> dict[str, Any]:
    boundary = int(feature_frame[config.TIME_COLUMN].quantile(0.7))
    train = feature_frame[feature_frame[config.TIME_COLUMN] < boundary]
    return features.fit_dist_buckets(train)


@pytest.fixture(scope="session")
def built_features(feature_frame: pd.DataFrame, dist_buckets: dict[str, Any]) -> pd.DataFrame:
    return features.build_features(feature_frame, dist_buckets)
