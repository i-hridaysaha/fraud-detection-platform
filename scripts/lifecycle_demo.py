"""Stage 9: the lifecycle end to end on a pinned seed, with a drift injected on a known day.

The live stream is the val and test windows replayed in weekly batches through the shipped
model, one batch per cycle. From a chosen batch on, every row carries an upstream change the
model never saw: amounts arrive in cents (times 100) and the C counters are rescaled (3c + 7).
Each cycle runs the drift job on the batch that closed, the decay job on every batch whose
labels matured under the clock, the retrain decision, and, when it says so, a challenger fitted
on the mature window through the same pipeline and put to the gate against the champion. A
promotion moves the registry alias; the script then exercises the rollback and checks that the
alias points back at the original version and that the version scores as it did.

    .venv/bin/python scripts/lifecycle_demo.py \\
        --transactions data/train_transaction.csv --identity data/train_identity.csv

Writes reports/lifecycle_demo.json. The registry it writes to is mlruns/lifecycle_demo/,
gitignored and rebuilt on every run, so nothing here touches the stage 8 production alias.
What the run does and does not establish is stated in docs/monitoring.md, not here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np
import pandas as pd
import sklearn
import xgboost

from fraud_platform import (
    cleaning,
    config,
    data_loader,
    encoders,
    evaluation,
    features,
    lifecycle,
    modelling,
    monitoring,
    prepare,
    serving,
)

MODELS_DIR = config.ROOT / "models"
REPORT_PATH = config.REPORTS_DIR / "lifecycle_demo.json"
OPERATING_POINTS_PATH = config.REPORTS_DIR / "operating_points.json"
SCHEMA_PATH = config.REPORTS_DIR / "prep" / "schema.json"
DEMO_MLRUNS = config.ROOT / "mlruns" / "lifecycle_demo"

# The demo's inputs. Every one is a choice and the artifact labels them so.
DRIFT_FROM_BATCH = 3
AMOUNT_FACTOR = 100.0
COUNT_SCALE = 3.0
COUNT_SHIFT = 7.0
WINDOWS = lifecycle.ChallengerWindows(fit_days=60, calibration_days=14, gate_days=14)
N_BOOTSTRAP = 1000
N_SMOKE_ROWS = 200
# Cycles keep ticking after the stream ends until every batch's labels have matured and the
# decision for the last one has been taken.
MAX_CYCLES = 40


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
        "section": "lifecycle_demo",
        "stage": 9,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "regenerate_with": (
            "make lifecycle-demo  # or: .venv/bin/python scripts/lifecycle_demo.py "
            "--transactions data/train_transaction.csv --identity data/train_identity.csv"
        ),
        "command": ".venv/bin/python scripts/lifecycle_demo.py",
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
            "mlflow": __import__("mlflow").__version__,
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


def day(dt: int) -> float:
    return round(monitoring.day_index(dt), 2)


# --- the champion -------------------------------------------------------------------------------


class Champion:
    """A model in production: its booster, its pipeline, and the two references the jobs read."""

    def __init__(
        self,
        name: str,
        version: int | None,
        model: Any,
        fitted: dict[str, Any],
        plan: dict[str, Any],
        columns: Sequence[str],
        threshold: float,
        train_end_dt: int,
        drift_reference: monitoring.DriftReference,
        decay_reference: dict[str, Any],
        accepted: dict[str, Any],
    ) -> None:
        self.name = name
        self.version = version
        self.model = model
        self.fitted = fitted
        self.plan = plan
        self.columns = list(columns)
        self.threshold = threshold
        self.train_end_dt = train_end_dt
        self.drift_reference = drift_reference
        self.decay_reference = decay_reference
        self.accepted = accepted

    def prepared(self, frame: pd.DataFrame) -> pd.DataFrame:
        return prepare.apply_preparation(frame, self.fitted, self.plan)

    def scores(self, prepared: pd.DataFrame) -> np.ndarray:
        x = modelling.tree_matrix(prepared, self.columns).to_numpy(dtype="float32")
        return modelling.score(self.model, x)


def batches_of(frame: pd.DataFrame) -> list[dict[str, int]]:
    return monitoring.batch_bounds(frame[config.TIME_COLUMN].to_numpy(dtype="int64"))


def references_for(
    prepared_fit: pd.DataFrame,
    fit_scores: np.ndarray,
    columns: Sequence[str],
    accepted_windows: Mapping[str, tuple[pd.DataFrame, np.ndarray]],
    gate_y: np.ndarray,
    gate_scores: np.ndarray,
    built_from: Mapping[str, Any],
) -> tuple[monitoring.DriftReference, dict[str, Any]]:
    """The drift reference on a fit window calibrated on the windows the model was accepted on,
    and the decay reference draws on the window its acceptance number came from."""
    drift = monitoring.build_reference(
        prepared_fit, fit_scores, columns, accepted_windows, built_from=built_from
    )
    decay = monitoring.reference_draws(gate_y, gate_scores, N_BOOTSTRAP, config.SEED)
    return drift, decay


def build_champion(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[Champion, dict[str, Any]]:
    """The shipped model with the stage 3 pipeline refitted on train, as `make register` does."""
    description = read(MODELS_DIR / "shipped_model.json")
    model = xgboost.XGBClassifier()
    model.load_model(MODELS_DIR / description["model_file"])
    columns = list(description["columns"])
    plan = prepare.read_column_plan()
    facts = cleaning.EdaFacts.load()
    decisions = prepare.read_decisions()
    started = time.perf_counter()
    fitted = prepare.fit_preparation(
        train,
        plan,
        facts,
        d_origin_columns=decisions["d_origin_columns"],
        v_strategy=decisions["v_strategy"],
        lag_days=int(decisions["lag_days"]),
        smoothing=float(decisions["smoothing"]),
    )
    buckets = features.fit_dist_buckets(encoders.add_normalised_free_text(train))
    prepared = {
        name: prepare.apply_preparation(part, fitted, plan)
        for name, part in (("train", train), ("val", val), ("test", test))
    }
    scores = {
        name: modelling.score(model, modelling.tree_matrix(part, columns).to_numpy(dtype="float32"))
        for name, part in prepared.items()
    }
    points = read(OPERATING_POINTS_PATH)
    threshold = float(points["tuned_vertex"]["val"]["threshold"])
    y_test = test[config.TARGET].to_numpy(dtype="int64")
    drift_reference, decay_reference = references_for(
        prepared["train"],
        scores["train"],
        columns,
        {"val": (prepared["val"], scores["val"]), "test": (prepared["test"], scores["test"])},
        y_test,
        scores["test"],
        {"model": "models/shipped_model.ubj", "fit": "stage 3 on train, this run"},
    )
    accepted = {
        "window": "test",
        "pr_auc": float(decay_reference["points"]["reference"]),
        "stage_6_recorded": float(points["primary_metric"]["test"]["pr_auc"]),
        "source": "reports/operating_points.json, primary_metric.test",
    }
    champion = Champion(
        "shipped",
        None,
        model,
        fitted,
        plan,
        columns,
        threshold,
        config.TRAIN_END_DT,
        drift_reference,
        decay_reference,
        accepted,
    )
    y_val = val[config.TARGET].to_numpy(dtype="int64")
    calibrator = lifecycle.calibrator_breakpoints(scores["val"], y_val, "val")
    bands = evaluation.cost_bands()
    serving_json = {
        "model_name": serving.MODEL_NAME,
        "alias": serving.MODEL_ALIAS,
        "family": description["family"],
        "stack": description["stack"],
        "hyperparameters": description["hyperparameters"],
        "columns": columns,
        "n_columns": len(columns),
        "raw_columns": lifecycle.raw_columns(train),
        "dist_buckets": buckets,
        "decisions": decisions,
        "calibrator": calibrator,
        "bands": {
            "low": bands["low"],
            "high": bands["high"],
            "on": "calibrated probability",
            "costs": bands["costs"],
            "source": "evaluation.cost_bands() from config, as reports/operating_points.json",
        },
        "tuned_threshold": {
            "threshold": threshold,
            "on": "raw score",
            "rule": points["tuned_vertex"]["val"]["rule"],
            "source": "reports/operating_points.json, tuned_vertex.val",
        },
        "schema": read(SCHEMA_PATH)["schema"],
        "inference_threads": serving.INFERENCE_THREADS,
        "stage_6_git_commit": description["git_commit"],
    }
    about = {
        "fit_seconds": time.perf_counter() - started,
        "description": description,
        "serving": serving_json,
        "hyperparameters": description["hyperparameters"],
        "decisions": decisions,
    }
    return champion, about


# --- the injected drift ---------------------------------------------------------------------------


def inject_drift(frame: pd.DataFrame, from_dt: int) -> pd.DataFrame:
    """Rows at or after `from_dt` arrive with amounts in cents and the C counters rescaled."""
    out = frame.copy()
    hit = out[config.TIME_COLUMN].to_numpy(dtype="int64") >= from_dt
    out.loc[hit, config.AMOUNT_COLUMN] = out.loc[hit, config.AMOUNT_COLUMN] * AMOUNT_FACTOR
    for column in config.COUNT_COLUMNS:
        if column in out.columns:
            values = out.loc[hit, column].astype("float64") * COUNT_SCALE + COUNT_SHIFT
            out.loc[hit, column] = values.astype(out[column].dtype)
    return out


# --- the registry --------------------------------------------------------------------------------


def fresh_registry() -> lifecycle.Registry:
    if DEMO_MLRUNS.exists():
        shutil.rmtree(DEMO_MLRUNS)
    DEMO_MLRUNS.mkdir(parents=True)
    uri = f"sqlite:///{DEMO_MLRUNS / 'mlflow.db'}"
    os.environ[serving.ENV_TRACKING_URI] = uri
    return lifecycle.Registry(uri, DEMO_MLRUNS / "artifacts", experiment="lifecycle-demo")


def register(
    registry: lifecycle.Registry,
    model: Any,
    fitted: Mapping[str, Any],
    plan: Mapping[str, Any],
    serving_json: Mapping[str, Any],
    params: Mapping[str, Any],
    alias: str,
) -> dict[str, Any]:
    bundle_dir = Path(tempfile.mkdtemp(prefix="lifecycle-bundle-"))
    paths = serving.write_bundle(bundle_dir, model, fitted, plan, _clean(serving_json))
    out = registry.register(paths, params, alias)
    shutil.rmtree(bundle_dir, ignore_errors=True)
    return out


def smoke(bundle: serving.ServingBundle, rows: pd.DataFrame, expected: np.ndarray) -> float:
    """Rows through the serving path of a loaded bundle against batch scores. The stage 8 check."""
    raw = set(bundle.raw_columns)
    largest = 0.0
    for position in range(len(rows)):
        record = rows.iloc[position]
        payload = {
            str(k): (v.item() if isinstance(v, np.generic) else v)
            for k, v in record.items()
            if str(k) in raw and str(k) != config.TARGET and not pd.isna(v)
        }
        out = serving.score_transaction(bundle, payload, None)
        largest = max(largest, abs(out["score"] - float(expected[position])))
    return largest


# --- main -------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transactions", default=str(config.TRANSACTIONS_PATH))
    parser.add_argument("--identity", default=str(config.IDENTITY_PATH))
    args = parser.parse_args()
    started = time.perf_counter()

    print("loading the joined frame")
    frame = data_loader.add_entity_key(
        data_loader.load_raw(transactions_path=args.transactions, identity_path=args.identity)
    )
    train, val, test = data_loader.time_based_split(frame)
    print("fitting the champion's pipeline and building its references")
    champion, champion_about = build_champion(train, val, test)
    hyperparameters = champion_about["hyperparameters"]
    schema_report = read(SCHEMA_PATH)

    stream = pd.concat([val, test]).sort_values(config.TIME_COLUMN, kind="stable")
    bounds = batches_of(stream)
    drift_from_dt = int(bounds[DRIFT_FROM_BATCH]["start_dt"])
    live = inject_drift(stream, drift_from_dt)
    history = pd.concat([train, live]).sort_values(config.TIME_COLUMN, kind="stable")
    ts = live[config.TIME_COLUMN].to_numpy(dtype="int64")
    labels = live[config.TARGET].to_numpy(dtype="int64")
    print(
        f"stream: {len(live)} rows in {len(bounds)} weekly batches from day "
        f"{day(bounds[0]['start_dt'])}; drift injected from batch {DRIFT_FROM_BATCH}, day "
        f"{day(drift_from_dt)}"
    )

    registry = fresh_registry()
    registered = register(
        registry,
        champion.model,
        champion.fitted,
        champion.plan,
        champion_about["serving"],
        {"role": "champion", "source": "models/shipped_model.ubj"},
        lifecycle.PRODUCTION_ALIAS,
    )
    champion.version = int(registered["version"])
    original = champion
    print(f"  champion registered as version {champion.version}")

    cycles: list[dict[str, Any]] = []
    decisions_log: list[dict[str, Any]] = []
    evaluated: set[int] = set()
    promotion: dict[str, Any] | None = None
    challengers = 0
    cycle = 0
    while cycle < MAX_CYCLES:
        as_of = (
            int(bounds[min(cycle, len(bounds) - 1)]["end_dt"])
            + max(0, cycle - len(bounds) + 1) * monitoring.BATCH_DAYS * config.SECONDS_PER_DAY
        )
        record: dict[str, Any] = {
            "cycle": cycle,
            "as_of_dt": as_of,
            "as_of_day": day(as_of),
            "champion": {"name": champion.name, "version": champion.version},
        }

        # The drift job, on the batch that closed this cycle. No labels.
        if cycle < len(bounds):
            b = bounds[cycle]
            mask = monitoring.batch_mask(ts, b)
            prepared = champion.prepared(live.loc[mask])
            scores = champion.scores(prepared)
            report = monitoring.drift_report(
                champion.drift_reference, prepared.drop(columns=[config.TARGET]), scores, b
            )
            top = sorted(report["per_feature"], key=lambda f: -f["psi"])[:5]
            record["drift"] = {
                "batch": b["batch"],
                "start_day": day(b["start_dt"]),
                "end_day": day(b["end_dt"]),
                "n_rows": int(mask.sum()),
                "drifted_rows": int((ts[mask] >= drift_from_dt).sum()),
                "score_psi": report["score"]["psi"],
                "score_alert": report["score"]["alert"],
                "n_alert": report["aggregate"]["n_alert"],
                "feature_alert": report["aggregate"]["feature_alert"],
                "n_alert_band": report["aggregate"]["n_alert_band"],
                "drift_alert": report["drift_alert"],
                "top_psi": [(f["column"], round(f["psi"], 4)) for f in top],
            }
            drift_alert: bool | None = report["drift_alert"]
        else:
            record["drift"] = None
            drift_alert = None

        # The decay job, on every batch whose labels matured under this cycle's clock.
        decay_flags: list[dict[str, Any]] = []
        for b in bounds:
            if b["batch"] in evaluated:
                continue
            if not monitoring.is_mature(b["end_dt"], as_of, lifecycle.MATURITY_DAYS):
                continue
            mask = monitoring.batch_mask(ts, b)
            scores = champion.scores(champion.prepared(live.loc[mask]))
            report = monitoring.performance_report(
                labels[mask],
                scores,
                champion.threshold,
                champion.decay_reference,
                b,
                as_of,
                lifecycle.MATURITY_DAYS,
                N_BOOTSTRAP,
                config.SEED,
            )
            evaluated.add(b["batch"])
            decay_flags.append(
                {
                    "batch": b["batch"],
                    "end_day": day(b["end_dt"]),
                    "matured_at_day": day(report["matured_at_dt"]),
                    "drifted_rows": int((ts[mask] >= drift_from_dt).sum()),
                    "n_rows": report["n_rows"],
                    "n_fraud": report["n_fraud"],
                    "pr_auc": report["pr_auc"]["point"],
                    "pr_auc_low": report["pr_auc"]["low"],
                    "pr_auc_high": report["pr_auc"]["high"],
                    "precision": report["precision"],
                    "recall": report["recall"],
                    "reference_pr_auc": report["reference_pr_auc"],
                    "drop": report["drop"]["point"],
                    "drop_low": report["drop"]["low"],
                    "drop_excludes_zero": report["drop"]["excludes_zero"],
                    "decay": lifecycle.decay_flag(report),
                }
            )
        record["decay"] = decay_flags
        record["immature"] = [
            b["batch"]
            for b in bounds
            if b["batch"] not in evaluated
            and not monitoring.is_mature(b["end_dt"], as_of, lifecycle.MATURITY_DAYS)
        ]

        decisions_log.append(
            {
                "cycle": cycle,
                "drift_alert": drift_alert,
                "decay": (bool(any(d["decay"] for d in decay_flags)) if decay_flags else None),
            }
        )
        decision = lifecycle.should_retrain(decisions_log)
        record["decision"] = decision
        shown = (
            f"drift {record['drift']['drift_alert']}" if record["drift"] else "no batch"
        ) + f", matured {[d['batch'] for d in decay_flags]}, retrain {decision['reason']}"
        print(f"cycle {cycle} (day {day(as_of)}): {shown}")

        if decision["retrain"]:
            mature_end = as_of - lifecycle.MATURITY_DAYS * config.SECONDS_PER_DAY
            windows = lifecycle.challenger_bounds(mature_end, WINDOWS)
            if windows["gate_start_dt"] < champion.train_end_dt:
                record["challenger"] = {
                    "trained": False,
                    "reason": (
                        "the gate window would overlap the champion's training window: too "
                        "little mature data since the champion was fitted"
                    ),
                    "mature_end_day": day(mature_end),
                    "gate_start_day": day(windows["gate_start_dt"]),
                    "champion_train_end_day": day(champion.train_end_dt),
                }
                print("  retrain asked, gate not possible yet")
            else:
                challengers += 1
                print(
                    f"  fitting challenger {challengers} on days {day(windows['fit_start_dt'])} to "
                    f"{day(windows['fit_end_dt'])}, gate {day(windows['gate_start_dt'])} to "
                    f"{day(windows['gate_end_dt'])}"
                )
                challenger = lifecycle.fit_challenger(
                    history, windows, champion.columns, hyperparameters, schema_report
                )
                gate_rows = history.loc[
                    (history[config.TIME_COLUMN] >= windows["gate_start_dt"])
                    & (history[config.TIME_COLUMN] < windows["gate_end_dt"])
                ]
                y_gate = gate_rows[config.TARGET].to_numpy(dtype="int64")
                s_champion = champion.scores(champion.prepared(gate_rows))
                s_challenger = challenger.scores(gate_rows)
                verdict = lifecycle.gate(
                    y_gate,
                    s_champion,
                    s_challenger,
                    gate_rows[config.ENTITY_ID_COLUMN].to_numpy(),
                    N_BOOTSTRAP,
                    config.SEED,
                )
                drifted_in_fit = int(
                    (
                        (history[config.TIME_COLUMN] >= max(windows["fit_start_dt"], drift_from_dt))
                        & (history[config.TIME_COLUMN] < windows["fit_end_dt"])
                    ).sum()
                )
                record["challenger"] = {
                    "trained": True,
                    "number": challengers,
                    "trigger": decision["reason"],
                    "windows": {k: day(v) for k, v in windows.items()},
                    "fit": challenger.fit,
                    "drifted_rows_in_fit_window": drifted_in_fit,
                    "gate_rows": int(y_gate.size),
                    "gate_fraud": int(y_gate.sum()),
                    "gate_drifted_rows": int(
                        (gate_rows[config.TIME_COLUMN].to_numpy() >= drift_from_dt).sum()
                    ),
                    "gate": verdict,
                }
                print(
                    f"  gate: champion {verdict['champion']['point']:.4f}, challenger "
                    f"{verdict['challenger']['point']:.4f}, difference "
                    f"{verdict['difference']['point']:+.4f} ({verdict['difference']['low']:+.4f} "
                    f"to {verdict['difference']['high']:+.4f}): {verdict['reason']}"
                )
                registered = register(
                    registry,
                    challenger.model,
                    challenger.fitted,
                    challenger.plan,
                    challenger.serving,
                    {
                        "role": "challenger",
                        "number": challengers,
                        "trigger": decision["reason"],
                        "gate_promote": verdict["promote"],
                    },
                    lifecycle.CHALLENGER_ALIAS,
                )
                record["challenger"]["registry"] = {
                    "version": registered["version"],
                    "alias": registered["alias"],
                }
                if verdict["promote"]:
                    moved = registry.promote(int(registered["version"]))
                    record["challenger"]["promotion"] = moved
                    print(f"  promoted: {moved}")
                    prepared_fit = prepare.apply_preparation(
                        history.loc[
                            (history[config.TIME_COLUMN] >= windows["fit_start_dt"])
                            & (history[config.TIME_COLUMN] < windows["fit_end_dt"])
                        ],
                        challenger.fitted,
                        challenger.plan,
                    )
                    calibration_rows = history.loc[
                        (history[config.TIME_COLUMN] >= windows["calibration_start_dt"])
                        & (history[config.TIME_COLUMN] < windows["calibration_end_dt"])
                    ]
                    prepared_calibration = prepare.apply_preparation(
                        calibration_rows, challenger.fitted, challenger.plan
                    )
                    prepared_gate = prepare.apply_preparation(
                        gate_rows, challenger.fitted, challenger.plan
                    )
                    new_columns = list(challenger.serving["columns"])
                    new_scores = {
                        name: modelling.score(
                            challenger.model,
                            modelling.tree_matrix(part, new_columns).to_numpy("float32"),
                        )
                        for name, part in (
                            ("fit", prepared_fit),
                            ("calibration", prepared_calibration),
                            ("gate", prepared_gate),
                        )
                    }
                    drift_reference, decay_reference = references_for(
                        prepared_fit,
                        new_scores["fit"],
                        new_columns,
                        {
                            "calibration": (prepared_calibration, new_scores["calibration"]),
                            "gate": (prepared_gate, new_scores["gate"]),
                        },
                        y_gate,
                        new_scores["gate"],
                        {"model": f"challenger {challengers}", "fit": "the fit window"},
                    )
                    promotion = {
                        "cycle": cycle,
                        "challenger": challengers,
                        "version": int(registered["version"]),
                        "previous_version": champion.version,
                        "gate": verdict,
                        "windows": {k: day(v) for k, v in windows.items()},
                        "new_reference_pr_auc": float(decay_reference["points"]["reference"]),
                        "new_drift_reference": {
                            "feature_count_edge": drift_reference.feature_count_edge,
                            "score_alert_edge": drift_reference.score_alert_edge,
                            "n_calibration_batches": drift_reference.calibration["n_batches"],
                        },
                    }
                    champion = Champion(
                        f"challenger {challengers}",
                        int(registered["version"]),
                        challenger.model,
                        challenger.fitted,
                        challenger.plan,
                        new_columns,
                        challenger.fit["tuned_threshold"],
                        int(windows["fit_end_dt"]),
                        drift_reference,
                        decay_reference,
                        {
                            "window": "gate",
                            "pr_auc": float(decay_reference["points"]["reference"]),
                        },
                    )
                    # The new champion's history starts here: the decay it is judged by is its
                    # own, and a batch matured under the old one is not re-read.
                    decisions_log = []
        cycles.append(record)
        cycle += 1
        if len(evaluated) == len(bounds) and cycle >= len(bounds):
            break

    # The rollback drill: one step back from whatever the alias points at, then check that the
    # service would load the version it pointed at before the last promotion and that the
    # version scores through the serving path as it does through the batch path.
    print("rollback drill")
    before = registry.aliases()
    rollback: dict[str, Any] = {"aliases_before": before, "what": "one step back, a drill"}
    if before[lifecycle.PREVIOUS_ALIAS] is None:
        rollback["performed"] = False
        rollback["reason"] = "no promotion happened, so there is nothing to roll back"
    else:
        moved = registry.rollback()
        rollback["performed"] = True
        rollback["moved"] = moved
        rollback["aliases_after"] = registry.aliases()
        loaded, about = serving.load_from_registry()
        rows = test.iloc[: N_SMOKE_ROWS * 40 : 40].head(N_SMOKE_ROWS)
        expected = modelling.score(
            loaded.model,
            modelling.tree_matrix(
                prepare.apply_preparation(rows, loaded.fitted, loaded.plan), loaded.columns
            ).to_numpy("float32"),
        )
        rollback["loaded"] = {**about, "n_columns": len(loaded.columns)}
        rollback["smoke"] = {
            "n_rows": len(rows),
            "max_abs_score_difference_serving_vs_batch": smoke(loaded, rows, expected),
        }
        rollback["production_is_the_previous_version"] = bool(
            about["version"] == before[lifecycle.PREVIOUS_ALIAS]
        )
        rollback["production_is_the_original_version"] = bool(about["version"] == original.version)
        print(f"  production back at version {about['version']}, smoke {rollback['smoke']}")

    report_out = {
        **envelope(),
        "inputs": {
            "what": "every value here is a choice of the demo, not a measurement",
            "batch_days": monitoring.BATCH_DAYS,
            "drift_from_batch": DRIFT_FROM_BATCH,
            "drift_from_day": day(drift_from_dt),
            "injection": {
                "amount": f"{config.AMOUNT_COLUMN} times {AMOUNT_FACTOR:g}",
                "counts": f"every C column to {COUNT_SCALE:g} c + {COUNT_SHIFT:g}",
                "applied_to": "raw rows at or after drift_from_day, before preparation",
            },
            "maturity_days": lifecycle.MATURITY_DAYS,
            "challenger_windows": {
                "fit_days": WINDOWS.fit_days,
                "calibration_days": WINDOWS.calibration_days,
                "gate_days": WINDOWS.gate_days,
            },
            "promotion_margin": lifecycle.PROMOTION_MARGIN,
            "decay_tolerance": lifecycle.DECAY_TOLERANCE,
            "sustained_cycles": lifecycle.SUSTAINED_CYCLES,
            "n_boot": N_BOOTSTRAP,
            "stream": "the val and test windows in order, real consecutive weeks, no row replayed",
        },
        "stream": {
            "n_rows": len(live),
            "n_batches": len(bounds),
            "first_day": day(bounds[0]["start_dt"]),
            "last_day": day(int(ts.max())),
            "n_drifted_rows": int((ts >= drift_from_dt).sum()),
        },
        "champion": {
            "version": original.version,
            "accepted": original.accepted,
            "threshold": original.threshold,
            "train_end_day": day(original.train_end_dt),
            "drift_reference": {
                "feature_count_edge": original.drift_reference.feature_count_edge,
                "score_alert_edge": original.drift_reference.score_alert_edge,
                "n_calibration_batches": original.drift_reference.calibration["n_batches"],
            },
            "pipeline_fit_seconds": champion_about["fit_seconds"],
        },
        "n_cycles": len(cycles),
        "n_challengers": challengers,
        "promotion": promotion,
        "rollback": rollback,
        "cycles": cycles,
        "seconds": time.perf_counter() - started,
    }
    write(report_out, REPORT_PATH)
    print(f"done in {report_out['seconds']:.0f} s")


if __name__ == "__main__":
    main()
