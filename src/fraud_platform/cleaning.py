"""The column plan: which raw columns reach a model, and the stage 2 number behind each answer.

A column is kept unless one of four measured conditions holds, and each condition names the
artifact that established it. There is no fifth category and there is no "not useful": a column
whose removal cannot be argued from `reports/eda/` stays in the frame and takes its chances with
the model. ADR 0011 is the argument for the four and for the evidence each one requires.

    effectively_constant     one value covers CONSTANT_SHARE of the non-null rows, and the
                             column's information value does not clear CONSTANT_RESCUE_IV.
                             Both halves are needed. Stage 2 found C3 at 0.9958 zero and asked
                             for exactly this rescue, because a column that is almost constant
                             can still separate the classes on the rows where it is not.

    missing_and_not_mnar     the missing rate is at least MISSING_DROP_RATE and the column is
                             not on the label-linked list in missingness.json. A column missing
                             on almost every row is worth nothing; a column missing on almost
                             every row whose missingness tracks the label is worth keeping, and
                             stage 2 measured 251 of those.

    time_inconsistent       the single-feature model fitted on the whole training window ranks
                             the validation split below chance. Stage 2's screen compared two
                             windows inside train and produced a candidate list of 59; ADR 0009
                             said explicitly that turning that list into a decision needs a
                             measurement against a held-out window, and this is it.

    redundant                the column sits in a correlation group at REDUNDANT_RHO and is not
                             the representative the group kept. Representatives are chosen from
                             the members that survived the other three tests, so a group never
                             trades a column that holds up over time for one that does not.

Order matters and is fixed: constant, then missing, then time inconsistency, then redundancy.
Redundancy runs last because it is the only rule whose answer depends on which columns are still
standing. A column matching more than one condition is recorded under the first and carries the
rest in `also_matched`, so the plan never double-counts a drop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from fraud_platform import config, eda

# --- choices ------------------------------------------------------------------------------
# Each of these is a line drawn somewhere. The plan artifact records every one of them next to
# the numbers they were applied to, so a reader can move a line and re-read the consequences.

# Reuses the stage 2 constant, so the flag list in univariate.json and the drop list here cannot
# drift apart.
CONSTANT_SHARE = eda.CONSTANT_SHARE

# An effectively constant column is kept anyway when its information value reaches this. 0.02 is
# the bottom band boundary in bivariate.json `iv_bands`, so the rescue line is a line stage 2
# already drew rather than a new one.
CONSTANT_RESCUE_IV = 0.02

# A column this empty is dropped unless its missingness is label-linked. Stage 2 measured 12
# columns above 0.90 missing and the plan records how many this catches.
MISSING_DROP_RATE = 0.95

# A single-feature model that ranks validation below this has inverted on the split the model
# will be judged on. 0.50 is chance, not a choice about strictness.
TIME_INCONSISTENT_VAL_AUC = 0.50

# Reuses the stage 2 grouping threshold for the same reason CONSTANT_SHARE is reused.
REDUNDANT_RHO = eda.REDUNDANT_RHO

DROP_REASONS: tuple[str, ...] = (
    "effectively_constant",
    "missing_and_not_mnar",
    "time_inconsistent",
    "redundant",
)

# Column groups, by the prefix the dataset uses. Order matters: the identity block is checked
# before the general prefixes so id_01 is not read as a card column.
_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("identity", config.IDENTITY_COLUMNS),
    ("card", config.CARD_COLUMNS),
    ("address", config.ADDRESS_COLUMNS),
    ("distance", config.DISTANCE_COLUMNS),
    ("email", config.EMAIL_COLUMNS),
    ("product", config.PRODUCT_COLUMNS),
    ("amount", (config.AMOUNT_COLUMN,)),
    ("count", config.COUNT_COLUMNS),
    ("timedelta", config.TIMEDELTA_COLUMNS),
    ("match", config.MATCH_COLUMNS),
    ("vesta", config.VESTA_COLUMNS),
)


def column_group(column: str) -> str:
    """The named block a column belongs to, or `meta` for the label, the clock and the key."""
    if column in (config.ID_COLUMN, config.TARGET, config.TIME_COLUMN):
        return "meta"
    for name, members in _GROUPS:
        if column in members:
            return name
    return "other"


def column_role(column: str) -> str:
    """What a column is for, which is a different question from whether it is kept."""
    if column == config.TARGET:
        return "target"
    if column == config.ID_COLUMN:
        return "identifier"
    if column == config.TIME_COLUMN:
        return "time"
    return "feature"


class EdaFacts:
    """Every stage 2 number the plan reads, loaded once and addressed by column name.

    Nothing is computed here. This class exists so the plan builder cannot quietly invent a
    number: every lookup it offers returns a value that came out of a committed artifact, and a
    column the artifacts do not mention returns None rather than a default.
    """

    def __init__(self, sections: Mapping[str, Mapping[str, Any]]) -> None:
        self._sections = dict(sections)

        univariate = self._sections["univariate"]
        self.profiles: dict[str, dict[str, Any]] = {
            str(row["column"]): dict(row)
            for row in [*univariate["numeric"], *univariate["categorical"]]
        }
        self.constant_flags: set[str] = set(univariate["flags"]["effectively_constant"])

        missingness = self._sections["missingness"]
        self.missing_rate: dict[str, float] = {
            str(row["column"]): float(row["missing_rate"])
            for row in missingness["missing_rate_by_column"]
        }
        self.label_linked: set[str] = set(missingness["against_target"]["label_linked_columns"])
        self.missing_against_target: dict[str, dict[str, Any]] = {
            str(row["column"]): dict(row) for row in missingness["against_target"]["per_column"]
        }
        self.identity_join: dict[str, Any] = dict(missingness["identity_join"])
        self.missingness_blocks: list[dict[str, Any]] = [
            dict(block) for block in missingness["blocks"]["all_blocks"]
        ]

        bivariate = self._sections["bivariate"]
        self.information_value: dict[str, dict[str, Any]] = {
            str(row["column"]): dict(row) for row in bivariate["per_feature"]
        }

        temporal = self._sections["temporal"]
        self.time_consistency: dict[str, dict[str, Any]] = {
            str(row["column"]): dict(row) for row in temporal["time_consistency"]["per_feature"]
        }
        self.time_consistency_flagged: list[str] = list(
            temporal["time_consistency"]["flagged_columns"]
        )
        self.psi: dict[str, dict[str, Any]] = {
            str(row["column"]): dict(row) for row in temporal["psi"]["per_feature"]
        }

        correlation = self._sections["correlation"]
        self.correlation_blocks: list[dict[str, Any]] = [
            dict(block) for block in correlation["v_block_reduction"]["blocks"]
        ]
        self.pairs_outside_v: list[dict[str, Any]] = [
            dict(pair) for pair in correlation["strongest_pairs_outside_v"]["pairs"]
        ]

        quality = self._sections["quality"]
        self.negative_value_columns: list[str] = list(
            quality["impossible_values"]["columns_with_negative_values"]
        )
        self.device_info: dict[str, Any] = dict(quality["device_info"])
        self.email_domains: dict[str, Any] = dict(quality["email_domains"])

        self.n_train_rows: int = int(univariate["n_train_rows"])

    @classmethod
    def load(cls, eda_dir: Path | str | None = None) -> EdaFacts:
        """Read the six artifacts the plan depends on from reports/eda/."""
        directory = Path(eda_dir or config.REPORTS_DIR / "eda")
        wanted = ("univariate", "missingness", "bivariate", "temporal", "correlation", "quality")
        sections = {name: json.loads((directory / f"{name}.json").read_text()) for name in wanted}
        return cls(sections)

    def iv(self, column: str) -> float | None:
        row = self.information_value.get(column)
        return None if row is None else float(row["iv"])

    def iv_present_bins(self, column: str) -> float | None:
        row = self.information_value.get(column)
        return None if row is None else float(row["iv_present_bins"])

    def all_columns(self) -> list[str]:
        """Every column stage 2 profiled, in a fixed order."""
        return sorted(self.profiles)


def build_column_plan(
    facts: EdaFacts,
    validation_probe: Mapping[str, Mapping[str, Any]],
    raw_columns: Sequence[str],
) -> dict[str, Any]:
    """One entry per raw column: keep or drop, why, and the artifact that says so.

    `validation_probe` carries the train and validation AUC of a single-feature model for the
    columns stage 2 flagged as time inconsistent. It is a measurement and not a lookup, so it
    arrives as an argument: the plan cannot be built from the artifacts alone, which is the
    point ADR 0009 made when it refused to turn its candidate list into a decision.
    """
    columns = sorted(raw_columns)
    entries: dict[str, dict[str, Any]] = {}
    for column in columns:
        profile = facts.profiles.get(column, {})
        entries[column] = {
            "column": column,
            "group": column_group(column),
            "role": column_role(column),
            "decision": "keep",
            "reason": None,
            "also_matched": [],
            "evidence": {},
            "measured": {
                "missing_rate": facts.missing_rate.get(column),
                "top_value_share": profile.get("top_value_share"),
                "iv": facts.iv(column),
                "iv_present_bins": facts.iv_present_bins(column),
                "label_linked_missingness": column in facts.label_linked,
            },
        }

    def mark(column: str, reason: str, evidence: dict[str, Any]) -> None:
        entry = entries.get(column)
        if entry is None:
            return
        if entry["decision"] == "drop":
            entry["also_matched"].append(reason)
            return
        entry["decision"] = "drop"
        entry["reason"] = reason
        entry["evidence"] = evidence

    protected = {config.ID_COLUMN, config.TARGET, config.TIME_COLUMN}

    # --- 1. effectively constant, with the rescue stage 2 asked for -------------------------
    rescued: list[dict[str, Any]] = []
    for column in sorted(facts.constant_flags):
        if column in protected or column not in entries:
            continue
        share = float(facts.profiles[column]["top_value_share"])
        value = facts.iv(column)
        if value is not None and value >= CONSTANT_RESCUE_IV:
            rescued.append(
                {
                    "column": column,
                    "top_value_share": share,
                    "iv": value,
                    "iv_present_bins": facts.iv_present_bins(column),
                    "kept_because": (
                        f"information value {value:.4f} clears the {CONSTANT_RESCUE_IV} rescue "
                        "line, so the column separates the classes on the rows where it is not "
                        "its dominant value"
                    ),
                }
            )
            entries[column]["measured"]["rescued_from_effectively_constant"] = True
            continue
        mark(
            column,
            "effectively_constant",
            {
                "artifact": "reports/eda/univariate.json",
                "field": "flags.effectively_constant",
                "top_value": facts.profiles[column].get("top_value"),
                "top_value_share": share,
                "threshold": CONSTANT_SHARE,
                "iv": value,
                "rescue_threshold": CONSTANT_RESCUE_IV,
            },
        )

    # --- 2. missing above the threshold and not label-linked --------------------------------
    for column in columns:
        if column in protected:
            continue
        rate = facts.missing_rate.get(column)
        if rate is None or rate < MISSING_DROP_RATE:
            continue
        linked = column in facts.label_linked
        against = facts.missing_against_target.get(column, {})
        evidence = {
            "artifact": "reports/eda/missingness.json",
            "field": "missing_rate_by_column and against_target",
            "missing_rate": rate,
            "threshold": MISSING_DROP_RATE,
            "label_linked": linked,
            "gap": against.get("gap"),
            "p_value": against.get("p_value"),
            "mnar_gates": {"abs_gap": eda.MNAR_ABS_GAP, "p_value": eda.MNAR_P_VALUE},
        }
        if linked:
            entries[column]["measured"]["kept_despite_missing_rate"] = True
            entries[column]["measured"]["mnar_evidence"] = evidence
            continue
        mark(column, "missing_and_not_mnar", evidence)

    # --- 3. inverted on validation ----------------------------------------------------------
    for column in sorted(validation_probe):
        if column in protected or column not in entries:
            continue
        probe = validation_probe[column]
        val_auc = probe.get("validation_auc")
        screen = facts.time_consistency.get(column, {})
        evidence = {
            "artifact": "reports/eda/temporal.json plus reports/column_plan.json",
            "field": "time_consistency.flagged, re-measured against the validation split",
            "stage_2_early_auc": screen.get("early_auc"),
            "stage_2_late_auc": screen.get("late_auc"),
            "train_auc": probe.get("train_auc"),
            "validation_auc": val_auc,
            "threshold": TIME_INCONSISTENT_VAL_AUC,
        }
        entries[column]["measured"]["validation_probe"] = {
            "train_auc": probe.get("train_auc"),
            "validation_auc": val_auc,
        }
        if val_auc is None or float(val_auc) >= TIME_INCONSISTENT_VAL_AUC:
            entries[column]["measured"]["survived_time_consistency_on_validation"] = True
            continue
        mark(column, "time_inconsistent", evidence)

    # --- 4. redundancy, from the survivors --------------------------------------------------
    groups_recorded: list[dict[str, Any]] = []
    for block in facts.correlation_blocks:
        for group in block["groups"]:
            members = [m for m in group["members"] if m in entries]
            if len(members) < 2:
                continue
            survivors = [m for m in members if entries[m]["decision"] == "keep"]
            record = {
                "block_missing_rate": block["block_missing_rate"],
                "members": sorted(members),
                "stage_2_representative": group["representative"],
                "min_abs_rho_to_representative": group["min_abs_rho_to_representative"],
                "n_members_already_dropped": len(members) - len(survivors),
            }
            if not survivors:
                record["representative"] = None
                record["note"] = "every member was dropped by an earlier rule"
                groups_recorded.append(record)
                continue
            representative = _choose_representative(survivors, facts)
            record["representative"] = representative
            record["dropped"] = sorted(m for m in survivors if m != representative)
            groups_recorded.append(record)
            for member in record["dropped"]:
                mark(
                    member,
                    "redundant",
                    {
                        "artifact": "reports/eda/correlation.json",
                        "field": "v_block_reduction.blocks[].groups[]",
                        "cluster": sorted(members),
                        "representative_kept": representative,
                        "min_abs_rho_to_representative": group["min_abs_rho_to_representative"],
                        "threshold": REDUNDANT_RHO,
                    },
                )

    # Pairs outside the V block are only actionable when the correlation was measured on the
    # whole split. Stage 2 said so about D4 and D12: they rank identically on their 46,966
    # complete cases and differ in missingness on 244,197 rows, which is most of what either
    # column carries. The rule is the coverage, not the pair.
    outside_v_recorded: list[dict[str, Any]] = []
    for pair in facts.pairs_outside_v:
        if pair["a"] not in entries or pair["b"] not in entries:
            continue
        complete = int(pair["n_complete_pairs"])
        full_coverage = complete == facts.n_train_rows
        record = {
            "pair": [pair["a"], pair["b"]],
            "rho": pair["rho"],
            "n_complete_pairs": complete,
            "n_train_rows": facts.n_train_rows,
            "actionable": full_coverage,
        }
        if not full_coverage:
            record["note"] = (
                "the correlation was measured on a subset of rows, so it says nothing about the "
                "rows where one of the pair is missing and the other is not"
            )
            outside_v_recorded.append(record)
            continue
        survivors = [c for c in (pair["a"], pair["b"]) if entries[c]["decision"] == "keep"]
        if len(survivors) < 2:
            record["note"] = "one of the pair was already dropped by an earlier rule"
            outside_v_recorded.append(record)
            continue
        representative = _choose_representative(survivors, facts)
        dropped = next(c for c in survivors if c != representative)
        record["representative_kept"] = representative
        record["dropped"] = dropped
        outside_v_recorded.append(record)
        mark(
            dropped,
            "redundant",
            {
                "artifact": "reports/eda/correlation.json",
                "field": "strongest_pairs_outside_v.pairs",
                "cluster": [pair["a"], pair["b"]],
                "representative_kept": representative,
                "rho": pair["rho"],
                "n_complete_pairs": complete,
                "threshold": REDUNDANT_RHO,
            },
        )

    rows = [entries[column] for column in columns]
    dropped_by_reason = {
        reason: sorted(r["column"] for r in rows if r["reason"] == reason)
        for reason in DROP_REASONS
    }
    kept = [r["column"] for r in rows if r["decision"] == "keep"]
    return {
        "thresholds": {
            "effectively_constant_share": CONSTANT_SHARE,
            "effectively_constant_rescue_iv": CONSTANT_RESCUE_IV,
            "missing_drop_rate": MISSING_DROP_RATE,
            "time_inconsistent_validation_auc": TIME_INCONSISTENT_VAL_AUC,
            "redundant_rho": REDUNDANT_RHO,
        },
        "rule_order": list(DROP_REASONS),
        "n_columns": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(rows) - len(kept),
        "n_dropped_by_reason": {k: len(v) for k, v in dropped_by_reason.items()},
        "dropped_by_reason": dropped_by_reason,
        "kept_columns": sorted(kept),
        "rescued_from_effectively_constant": rescued,
        "correlation_groups": groups_recorded,
        "pairs_outside_v": outside_v_recorded,
        "columns": rows,
    }


def _choose_representative(candidates: Sequence[str], facts: EdaFacts) -> str:
    """Pick the column a correlation group keeps.

    Two rules, in order. Prefer a column stage 2's time consistency screen did not flag, because
    a group that trades a stable column for an inverted one has made itself worse while looking
    like it made itself smaller. Among those, prefer the highest information value from the
    present bins, which is the ranking stage 2 asked for: the missing-bin term belongs to the
    missingness indicator, not to the column. Ties break on the name so the plan is the same on
    every run.
    """
    flagged = set(facts.time_consistency_flagged)

    def key(column: str) -> tuple[int, float, str]:
        value = facts.iv_present_bins(column)
        return (
            1 if column in flagged else 0,
            -(value if value is not None else -1.0),
            column,
        )

    return sorted(candidates, key=key)[0]


def feature_columns(plan: Mapping[str, Any]) -> list[str]:
    """The kept columns that are features, in sorted order. Excludes the label, id and clock."""
    return sorted(
        row["column"]
        for row in plan["columns"]
        if row["decision"] == "keep" and row["role"] == "feature"
    )


def apply_column_plan(
    frame: pd.DataFrame, plan: Mapping[str, Any], spare: Sequence[str] = ()
) -> pd.DataFrame:
    """Drop the columns the plan drops, leaving everything else untouched.

    A column the plan does not mention is left alone rather than dropped. The plan is a list of
    decisions about the columns stage 2 saw; a column that arrived since then is a schema
    question, and `fraud_platform.schema` is where that is answered loudly.

    `spare` holds a column back from the drop. It exists for one case and should not grow past
    it: the redundancy rule and the V-column reduction are the same decision measured two ways,
    so when the reduction chooses a strategy other than keeping representatives, the columns the
    redundancy rule dropped are the inputs to it and have to survive this step. The V reduction
    then removes them itself. See `fraud_platform.prepare`.
    """
    kept_back = set(spare)
    dropped = {
        row["column"]
        for row in plan["columns"]
        if row["decision"] == "drop" and row["column"] not in kept_back
    }
    present = [column for column in frame.columns if column in dropped]
    return frame.drop(columns=present)


def v_reduction_pool(plan: Mapping[str, Any]) -> list[str]:
    """Every V column the reduction strategies were compared over.

    That is the V columns surviving the constant, missing and time-inconsistency rules, whether
    or not the redundancy rule then removed them, because the redundancy rule is one of the
    strategies being compared and cannot be applied before the comparison.
    """
    return sorted(
        row["column"]
        for row in plan["columns"]
        if row["column"] in config.VESTA_COLUMNS
        and (row["decision"] == "keep" or row["reason"] == "redundant")
    )
