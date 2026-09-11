"""Constants every later stage reads instead of restating.

Three kinds of thing live here and they are not interchangeable:

- Paths and names, which are facts about the files on disk.
- Split boundaries and the entity key, which are stage 0 decisions recorded in ADR 0001 and
  ADR 0002. Changing one of these numbers changes what every later result means, so they are
  written once, here, and a test asserts they still match reports/audit.json.
- The typing spec, which is stage 1's own decision (ADR 0005). It is measured, not guessed:
  the integer ranges below come from reports/split_summary.json and the audit before it.

SEED is the single source of randomness for the whole repo. Anything stochastic reads it from
here rather than carrying a literal.
"""

from __future__ import annotations

from pathlib import Path

# The repo root, resolved from this file so a script run from any directory finds the data.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "figures"

TRANSACTIONS_PATH = DATA_DIR / "train_transaction.csv"
IDENTITY_PATH = DATA_DIR / "train_identity.csv"

SPLIT_SUMMARY_PATH = REPORTS_DIR / "split_summary.json"
MEMORY_PROFILE_PATH = REPORTS_DIR / "memory_profile.json"

# Every stochastic component reads this. One constant, no per-module literals.
SEED = 42

ID_COLUMN = "TransactionID"
TARGET = "isFraud"
TIME_COLUMN = "TransactionDT"
AMOUNT_COLUMN = "TransactionAmt"

# TransactionDT is a relative offset in seconds and 86,400 units is a day. Measured in stage 0:
# folding the column into a position within an 86,400 unit cycle gives the diurnal shape of card
# activity, peak to trough 16.99. What the offset is measured from is not established, so day
# indices in this repo are relative.
SECONDS_PER_DAY = 86_400

# ADR 0002. Boundaries are TransactionDT values and assignment is `dt < boundary`, so a timestamp
# shared by several transactions never straddles two splits. They are the 0.70 and 0.85 quantiles
# of TransactionDT, frozen here as integers so no later stage has to re-run a quantile on a
# filtered frame and get a different answer.
TRAIN_END_DT = 10_437_998
VAL_END_DT = 13_151_845


SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")

# ADR 0001. card_start_day is floor(TransactionDT / 86400 - D1) and is derived at load time by
# fraud_platform.data_loader.add_entity_key, not present in the CSV.
CARD_START_DAY_COLUMN = "card_start_day"
ENTITY_ID_COLUMN = "entity_id"
ENTITY_KEY_COLUMNS: tuple[str, ...] = ("card1", "addr1", CARD_START_DAY_COLUMN)
ENTITY_KEY_SOURCE_COLUMNS: tuple[str, ...] = ("card1", "addr1", TIME_COLUMN, "D1")

# Placeholder for a null component of the entity key. ADR 0001 keeps null components as their own
# bucket rather than dropping the row, so every row belongs to exactly one entity and the counts
# reconcile. The audit used the same literal, so entity counts here and there agree.
ENTITY_NULL_TOKEN = "NA"

# --- column groups -------------------------------------------------------------------------
# Named by the prefix the dataset uses. Nothing here asserts what the columns mean; the groups
# exist so later stages can address a block by name. tests/test_temporal_contract.py checks that
# these groups exactly partition the 434 columns of the joined frame.

CARD_COLUMNS: tuple[str, ...] = tuple(f"card{i}" for i in range(1, 7))
ADDRESS_COLUMNS: tuple[str, ...] = ("addr1", "addr2")
DISTANCE_COLUMNS: tuple[str, ...] = ("dist1", "dist2")
EMAIL_COLUMNS: tuple[str, ...] = ("P_emaildomain", "R_emaildomain")
PRODUCT_COLUMNS: tuple[str, ...] = ("ProductCD",)
COUNT_COLUMNS: tuple[str, ...] = tuple(f"C{i}" for i in range(1, 15))
TIMEDELTA_COLUMNS: tuple[str, ...] = tuple(f"D{i}" for i in range(1, 16))
MATCH_COLUMNS: tuple[str, ...] = tuple(f"M{i}" for i in range(1, 10))
VESTA_COLUMNS: tuple[str, ...] = tuple(f"V{i}" for i in range(1, 340))

# From train_identity.csv, present only for the 0.2442 of rows that join (stage 0 audit).
IDENTITY_NUMBERED_COLUMNS: tuple[str, ...] = tuple(f"id_{i:02d}" for i in range(1, 39))
DEVICE_COLUMNS: tuple[str, ...] = ("DeviceType", "DeviceInfo")
IDENTITY_COLUMNS: tuple[str, ...] = IDENTITY_NUMBERED_COLUMNS + DEVICE_COLUMNS

# Columns that arrive from the CSV as strings. Measured: 31 of the 434, none with cardinality
# above 1,786 against 590,540 rows, so all of them pay as categoricals (ADR 0005).
CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    *MATCH_COLUMNS,
    "id_12",
    "id_15",
    "id_16",
    "id_23",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
    *DEVICE_COLUMNS,
)

# The 18 columns measured to hold integers with no nulls, with the narrowest dtype their measured
# range fits. A null appearing here in future data makes read_csv raise, which is the intent: a
# silent promotion to float would hide a change in the file.
INTEGER_COLUMN_DTYPES: dict[str, str] = {
    ID_COLUMN: "int32",  # 2,987,000 to 3,577,539
    TARGET: "int8",  # 0 to 1
    TIME_COLUMN: "int32",  # 86,400 to 15,811,131
    "card1": "int16",  # 1,000 to 18,396
    **dict.fromkeys(COUNT_COLUMNS, "int16"),  # 0 to 5,691 across C1 to C14
}

# Columns held at float64 because float32 would round them measurably. Only the amount qualifies:
# a float32 roundtrip moves it by up to 0.000375 on a maximum of 31,937.391, and it is the one
# column denominated in money. Across every other numeric column the same roundtrip moves nothing
# by more than 2.3e-13 (ADR 0005).
EXACT_FLOAT_COLUMNS: tuple[str, ...] = (AMOUNT_COLUMN,)

# Everything numeric that is not named above.
DEFAULT_NUMERIC_DTYPE = "float32"

# --- stage 6: the decision cost matrix --------------------------------------------------------
# ASSUMPTION. The relative cost of each outcome a decision can produce, in units of one false
# decline. Nothing in this repo measures these: the file carries no chargeback amount, no margin
# and no review cost. They are inputs chosen so that the band edges can be derived from something
# explicit, and ADR 0028 labels them as inputs. The derivation from them is not an assumption and
# lives in fraud_platform.evaluation.cost_bands. Change a number here and the bands move; the
# artifact records which numbers produced which edges.
COST_FALSE_DECLINE = 1.0  # a legitimate transaction blocked
COST_MISSED_FRAUD = 10.0  # a fraudulent transaction approved
COST_REVIEW = 0.5  # one analyst look, paid whatever the transaction turns out to be

# The segmentation key for per-segment thresholds. Stage 2 measured addr2 at a top-value share of
# 0.9901 on train (reports/eda/univariate.json), so a per-market calibration has one market to
# calibrate. ProductCD is the partition with five levels none of which is rare. ADR 0029.
SEGMENT_COLUMN = "ProductCD"
