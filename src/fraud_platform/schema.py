"""The declared shape of the prepared frame, and the validator that refuses anything else.

Stage 8 puts a serving path in front of this pipeline, and the failure that path has is not a
crash. It is a row that arrives with a column renamed, a count arriving as a string, or a
negative amount, and the model scoring it anyway and returning a number nobody can tell is
wrong. This module exists so that row raises instead.

**What the schema constrains.** Structure, and only structure: the set of columns, the dtype
family of each, whether a null is allowed, and the sign and finiteness bounds stage 2 measured.
Every constraint here traces to `reports/eda/quality.json` under `impossible_values`, which is
the section that went looking for values the data should not be able to take and found none.

**What it deliberately does not constrain, and why.** No upper bounds and no distributional
limits. Stage 2's train-only section measured 130 of the 393 numeric columns moving a median or
a 99th percentile by more than 10 percent when the later splits are folded in, with V332 going
from 1,650 to 60,335. A cap fitted on the training window would reject legitimate later rows for
the offence of being later, and it would do it silently at exactly the moment the drift work in
stage 9 wants to see them. Drift is a monitoring question, and putting it in a validator turns
every distribution shift into an outage. No category allow-lists either, for the same reason
inverted: stage 2 measured 240 DeviceInfo levels and 22 id_31 levels that appear only after the
training window. Rejecting an unseen level would reject 0.0456 of val and test on id_31 alone.
The encoders have an explicit unseen token; that is where a new level is handled, not here.

ADR 0017 is the argument, with the list of what falls through.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud_platform import config

# --- choices ------------------------------------------------------------------------------

# The four columns where a null is a structural impossibility rather than a distributional
# accident. Everything downstream indexes, splits or scores on these, so a null in one of them
# is a broken row and not a sparse one. Stage 2 measured all four null-free on train
# (quality.json: n_rows_with_null_target 0, n_rows_with_null_time 0, n_non_finite_amounts 0),
# and that measurement is why the line is drawn here and nowhere else: a column that merely
# happened to be complete on train is declared nullable, because completeness on one window is
# not a property of the column.
NON_NULLABLE: tuple[str, ...] = (
    config.ID_COLUMN,
    config.TARGET,
    config.TIME_COLUMN,
    config.AMOUNT_COLUMN,
)

# Dtype families. The schema constrains the family and not the exact width, because stage 1's
# dtype map narrows an integer column to the smallest type its measured range fits and a later
# frame with a wider range would legitimately arrive as int32 where train was int16.
FAMILIES: tuple[str, ...] = ("integer", "float", "categorical", "boolean")


class SchemaError(ValueError):
    """Raised when a frame does not match the declared schema. Carries every violation found."""

    def __init__(self, violations: Sequence[Mapping[str, Any]]) -> None:
        self.violations = [dict(v) for v in violations]
        lines = [f"{v['column']}: {v['problem']}" for v in self.violations]
        super().__init__(f"{len(self.violations)} schema violation(s):\n  " + "\n  ".join(lines))


@dataclass(frozen=True)
class ColumnSpec:
    """One column's contract."""

    name: str
    family: str
    nullable: bool = True
    minimum: float | None = None
    maximum: float | None = None
    unique: bool = False
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.name,
            "family": self.family,
            "nullable": self.nullable,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unique": self.unique,
            "note": self.note,
        }


@dataclass(frozen=True)
class Schema:
    """The whole contract: the columns, and whether an unlisted column is allowed."""

    columns: tuple[ColumnSpec, ...]
    allow_extra_columns: bool = False
    name: str = "prepared"

    def spec(self, column: str) -> ColumnSpec | None:
        for candidate in self.columns:
            if candidate.name == column:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "allow_extra_columns": self.allow_extra_columns,
            "n_columns": len(self.columns),
            "columns": [spec.to_dict() for spec in self.columns],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Schema:
        return cls(
            columns=tuple(
                ColumnSpec(
                    name=str(row["column"]),
                    family=str(row["family"]),
                    nullable=bool(row["nullable"]),
                    minimum=row["minimum"],
                    maximum=row["maximum"],
                    unique=bool(row.get("unique", False)),
                    note=row.get("note"),
                )
                for row in payload["columns"]
            ),
            allow_extra_columns=bool(payload["allow_extra_columns"]),
            name=str(payload["name"]),
        )


def family_of(dtype: Any) -> str:
    """The declared family a pandas dtype belongs to."""
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_string_dtype(dtype) or dtype is object or dtype == np.dtype("O"):
        return "categorical"
    return str(dtype)


# Bounds for the columns this stage derives. A derived column has no stage 2 measurement behind
# it and never will, so its bounds come from the arithmetic that produces it: a count cannot be
# negative, a smoothed rate of a zero-or-one label lies in the unit interval, an indicator is an
# indicator. A derivation whose range is genuinely unbounded, such as a standardised block mean,
# is declared with no bounds rather than with a guess. Each entry is (suffix or exact name,
# minimum, maximum, why).
DERIVED_BOUNDS: tuple[tuple[str, float | None, float | None, str], ...] = (
    ("_freq", 0.0, None, "a count of training rows, so never negative and with no ceiling"),
    ("_te", 0.0, 1.0, "a smoothed mean of a zero-or-one label, so inside the unit interval"),
    ("_log1p", 0.0, None, "log1p of an amount stage 2 measured strictly positive"),
    ("_cents", 0.0, 99.0, "round(amount * 100) mod 100"),
    ("_is_round", 0.0, 1.0, "an indicator"),
    ("day_index", 0.0, None, "floor(TransactionDT / 86400), and TransactionDT is non-negative"),
    ("week_index", 0.0, None, "day_index // 7"),
    ("hour_position", 0.0, 23.0, "position inside an 86,400 unit cycle, in hours"),
    ("dow_position", 0.0, 6.0, "position inside a seven day cycle"),
    ("missing_block_", 0.0, 1.0, "an indicator"),
    ("has_identity_record", 0.0, 1.0, "an indicator"),
    ("vblockmean_", None, None, "a mean of standardised columns, signed and unbounded"),
    ("vpca_", None, None, "a principal component score, signed and unbounded"),
)


def _derived_bounds(column: str) -> tuple[float | None, float | None, str] | None:
    """Bounds for a derived column, or None when the column is not one this stage derives."""
    for key, minimum, maximum, note in DERIVED_BOUNDS:
        if column == key or column.endswith(key) or column.startswith(key):
            return minimum, maximum, note
    return None


def declare(
    frame: pd.DataFrame,
    negative_value_columns: Iterable[str],
    measured_columns: Iterable[str],
    name: str = "prepared",
) -> Schema:
    """Build the schema from a prepared train frame and the stage 2 sign measurements.

    The dtype and the column set come from the frame, which is the honest source: the contract
    should describe what the pipeline actually produces, not what someone believes it produces.
    The bounds come from `quality.json`, which listed the 17 columns that hold negative values
    and explained every one of them. A column on that list gets no lower bound. A numeric column
    off it gets a lower bound of zero, because stage 2 looked for a negative there and found
    none, and a negative arriving later is a change in the data and not a wide tail.

    `TransactionAmt` is the one raw column with a strict bound: stage 2 measured 0 negative
    amounts and 0 zero amounts across the split, with a minimum of 0.251.

    `measured_columns` is the set stage 2 actually profiled. A column outside it is one this
    stage derived, and its bounds come from `DERIVED_BOUNDS` instead: "no negative was observed"
    is a statement about a measurement, and there is no measurement of a column that did not
    exist when the measuring was done.
    """
    negatives = set(negative_value_columns)
    measured = set(measured_columns)
    specs: list[ColumnSpec] = []
    for column in frame.columns:
        family = family_of(frame[column].dtype)
        nullable = column not in NON_NULLABLE
        minimum: float | None = None
        maximum: float | None = None
        note: str | None = None
        if family in ("integer", "float"):
            derived = _derived_bounds(str(column))
            if column == config.AMOUNT_COLUMN:
                minimum = 0.0
                note = (
                    "strictly positive: quality.json measured 0 negative and 0 zero amounts, "
                    "minimum 0.251"
                )
            elif column in negatives:
                note = (
                    "no lower bound: quality.json lists this column among the 17 that hold "
                    "negative values, all of them explicable"
                )
            elif column in measured:
                minimum = 0.0
                note = "quality.json found no negative value in this column on train"
            elif derived is not None:
                minimum, maximum, note = derived
                note = f"derived column: {note}"
            else:
                note = "no bound: neither measured in stage 2 nor a derivation with a known range"
        specs.append(
            ColumnSpec(
                name=str(column),
                family=family,
                nullable=nullable,
                minimum=minimum,
                maximum=maximum,
                unique=column == config.ID_COLUMN,
                note=note,
            )
        )
    return Schema(columns=tuple(specs), allow_extra_columns=False, name=name)


def validate(frame: pd.DataFrame, schema: Schema) -> None:
    """Check a frame against the schema and raise with every violation, not just the first.

    Raising on the first would turn a broken producer into a queue of one-line fixes. The caller
    gets the whole list.
    """
    violations: list[dict[str, Any]] = []
    declared = {spec.name for spec in schema.columns}
    present = set(frame.columns)

    for column in sorted(declared - present):
        violations.append({"column": column, "problem": "declared column is missing"})
    if not schema.allow_extra_columns:
        for column in sorted(present - declared):
            violations.append({"column": column, "problem": "column is not in the schema"})

    for spec in schema.columns:
        if spec.name not in present:
            continue
        series = frame[spec.name]
        actual = family_of(series.dtype)
        if actual != spec.family:
            violations.append(
                {
                    "column": spec.name,
                    "problem": f"dtype family is {actual}, schema declares {spec.family}",
                }
            )
            continue
        # One array per column and numpy from here: the pandas per-column overhead was most of
        # the serving path's time on a one-row frame (stage 8), and the checks are the same.
        array = series.to_numpy()
        n_null = int(np.isnan(array).sum()) if spec.family == "float" else int(pd.isna(array).sum())
        if not spec.nullable and n_null:
            violations.append(
                {"column": spec.name, "problem": f"{n_null} null value(s) in a non-nullable column"}
            )
        if spec.unique and series.duplicated().any():
            violations.append(
                {
                    "column": spec.name,
                    "problem": f"{int(series.duplicated().sum())} duplicate value(s) in a unique column",
                }
            )
        if spec.family in ("integer", "float"):
            values = array.astype("float64")
            if spec.family == "float" and np.isinf(values).any():
                violations.append({"column": spec.name, "problem": "non-finite value"})
            if spec.minimum is not None:
                below = int((values < spec.minimum).sum())
                if below:
                    violations.append(
                        {
                            "column": spec.name,
                            "problem": (
                                f"{below} value(s) below the declared minimum {spec.minimum}, "
                                f"lowest {float(np.nanmin(values))}"
                            ),
                        }
                    )
            if spec.maximum is not None:
                above = int((values > spec.maximum).sum())
                if above:
                    violations.append(
                        {
                            "column": spec.name,
                            "problem": (
                                f"{above} value(s) above the declared maximum {spec.maximum}, "
                                f"highest {float(np.nanmax(values))}"
                            ),
                        }
                    )

    if violations:
        raise SchemaError(violations)


NOT_CAUGHT: tuple[str, ...] = (
    "a value that is the right type, in range, and wrong. The schema knows nothing about what a "
    "correct amount is for a given card",
    "distribution drift. There are no upper bounds and no distributional limits, deliberately: "
    "stage 2 measured 130 of 393 numeric columns moving a percentile by more than 10 percent "
    "when the later splits are included, so a train-fitted cap would reject later data for "
    "being later. Stage 9 monitors drift; the validator does not",
    "a category level the training window never saw. Stage 2 measured 240 such levels on "
    "DeviceInfo and 22 on id_31 covering 0.0456 of val and test. The encoders have an explicit "
    "unseen token and that is where the case is handled",
    "a row that is internally inconsistent across columns, such as a D column implying an origin "
    "later than its own timestamp. Nothing here compares two columns to each other",
    "a label that is wrong. Stage 0 did not establish the host's labelling rule and the "
    "validator cannot check what has not been established",
    "a duplicated row. Stage 2 measured 10 near-duplicate rows in 413,378 and found the "
    "core-field repeat grouping to be signal rather than dirt, so duplication is not an error "
    "condition here",
    "a column arriving in a different order. Column order is not part of the contract",
)
