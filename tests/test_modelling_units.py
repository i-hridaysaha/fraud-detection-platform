"""The pieces of `fraud_platform.modelling`, on generated data so they run in CI.

What is checked: the column rules (what a model may read, what a native-group switch removes,
what a stack is), the two preparation paths (the tree path keeps nulls and codes, the
complete-data path fills, clips, one-hots and standardises, and refuses a frame past the training
boundary), the SMOTE plan and its materialisation, the model factory, the convergence report, and
the expanding-window folds that never fit on a row later than their validation rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_platform import config, features, graph_features, modelling
from fraud_platform.data_loader import TrainOnlyError

DAY = config.SECONDS_PER_DAY


def small_prepared(n: int = 240, seed: int = config.SEED) -> pd.DataFrame:
    """A frame shaped like a stage 3 prepared frame with stage 4 and 5 columns attached."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            config.ID_COLUMN: np.arange(n, dtype="int64"),
            config.TARGET: (rng.random(n) < 0.2).astype("int8"),
            config.TIME_COLUMN: (DAY + np.sort(rng.integers(0, 30 * DAY, n))).astype("int64"),
            "day_index": np.arange(n) // 8,
            "week_index": np.arange(n) // 56,
            "dow_position": np.arange(n) % 7,
            "hour_position": rng.integers(0, 24, n).astype("int8"),
            config.ENTITY_ID_COLUMN: rng.choice(["a|1|2", "b|1|3", "c|2|2"], n),
            config.CARD_START_DAY_COLUMN: rng.integers(0, 10, n).astype("float64"),
            config.AMOUNT_COLUMN: np.exp(rng.normal(4, 1.5, n)),
            "C1": rng.integers(0, 50, n).astype("int16"),
            "C2": rng.integers(0, 50, n).astype("int16"),
            "D1": np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 100, n)).astype("float32"),
            "M1": rng.choice(["T", "F", None], n),
            "M1_cat": pd.Categorical(rng.choice(["T", "F", "(missing)"], n)),
            "ProductCD": rng.choice(["W", "C"], n),
            "ProductCD_cat": pd.Categorical(
                rng.choice(["W", "C"], n), categories=["W", "C", "(rare)"]
            ),
            "card1_te": rng.random(n).astype("float32"),
            "DeviceInfo_cat": pd.Categorical(rng.choice(["a", "b"], n)),
            "DeviceInfo_te": rng.random(n).astype("float32"),
            "DeviceInfo_norm_cat": pd.Categorical(rng.choice(["a", "b"], n)),
            "DeviceInfo_norm_te": rng.random(n).astype("float32"),
            "vblockmean_V1": rng.random(n).astype("float32"),
            "missing_block_D1": rng.integers(0, 2, n).astype("int8"),
            "missing_block_C1": np.zeros(n, dtype="int8"),
            "has_identity_record": rng.random(n) < 0.3,
            "vel_count_1h_entity": rng.integers(0, 5, n).astype("float32"),
            "rec_seconds_since_prev": np.where(
                rng.random(n) < 0.4, np.nan, rng.random(n) * 1e5
            ).astype("float32"),
            "graph_degree_card1": rng.integers(0, 9, n).astype("float32"),
        }
    )
    return frame


CAUSAL = ["vel_count_1h_entity", "rec_seconds_since_prev"]
GRAPH = ["graph_degree_card1"]


@pytest.fixture(scope="module")
def prepared() -> pd.DataFrame:
    return small_prepared()


@pytest.fixture(scope="module")
def prepared_columns(prepared: pd.DataFrame) -> list[str]:
    # The driver reads the stage 3 frame before the later blocks are joined onto it.
    return modelling.prepared_feature_columns(prepared.drop(columns=[*CAUSAL, *GRAPH]))


# --- column rules ---------------------------------------------------------------------------


def test_excluded_columns_and_raw_categoricals_are_not_read(prepared_columns: list[str]) -> None:
    for column in modelling.EXCLUDED_COLUMNS:
        assert column not in prepared_columns
    assert "M1" not in prepared_columns
    assert "ProductCD" not in prepared_columns
    assert "M1_cat" in prepared_columns
    assert "ProductCD_cat" in prepared_columns
    assert "hour_position" in prepared_columns
    assert prepared_columns == sorted(prepared_columns)
    # The normalised free-text column replaces the raw one; the raw one's encodings are not read.
    assert "DeviceInfo_cat" not in prepared_columns and "DeviceInfo_te" not in prepared_columns
    assert "DeviceInfo_norm_cat" in prepared_columns and "DeviceInfo_norm_te" in prepared_columns


def test_native_group_members_cover_raw_derived_and_indicator_columns(
    prepared_columns: list[str],
) -> None:
    assert modelling.native_group_members(prepared_columns, "count") == [
        "C1",
        "C2",
        "missing_block_C1",
    ]
    assert modelling.native_group_members(prepared_columns, "timedelta") == [
        "D1",
        "missing_block_D1",
    ]
    assert modelling.native_group_members(prepared_columns, "match") == ["M1_cat"]
    assert modelling.native_group_members(prepared_columns, "vesta") == ["vblockmean_V1"]
    with pytest.raises(KeyError):
        modelling.native_group_members(prepared_columns, "identity")


def test_stack_columns_follow_the_switches(prepared_columns: list[str]) -> None:
    default = modelling.stack_columns(prepared_columns, CAUSAL, GRAPH, modelling.DEFAULT_STACK)
    assert default == [*prepared_columns, *CAUSAL]
    assert modelling.DEFAULT_STACK.graph is graph_features.DEFAULT_ENABLED
    everything = modelling.stack_columns(
        prepared_columns, CAUSAL, GRAPH, modelling.FeatureStack(causal=True, graph=True)
    )
    assert everything == [*prepared_columns, *CAUSAL, *GRAPH]
    bare = modelling.stack_columns(
        prepared_columns, CAUSAL, GRAPH, modelling.FeatureStack(causal=False, graph=False)
    )
    assert bare == prepared_columns
    no_count = modelling.stack_columns(
        prepared_columns, CAUSAL, GRAPH, modelling.DEFAULT_STACK.without_group("count")
    )
    assert "C1" not in no_count and "missing_block_C1" not in no_count
    assert "D1" in no_count
    none = modelling.FeatureStack(native_groups=())
    assert (
        modelling.stack_columns(prepared_columns, CAUSAL, GRAPH, none)
        == [
            c
            for c in prepared_columns
            if c
            not in {
                "C1",
                "C2",
                "D1",
                "M1_cat",
                "vblockmean_V1",
                "missing_block_D1",
                "missing_block_C1",
            }
        ]
        + CAUSAL
    )


def test_stack_names_are_distinct_and_describe_the_switches() -> None:
    names = {
        modelling.FeatureStack(causal=c, graph=g, native_groups=n).name
        for c in (True, False)
        for g in (True, False)
        for n in (features.NATIVE_GROUPS, (), ("count",))
    }
    assert len(names) == 12
    assert modelling.DEFAULT_STACK.name == "prepared+causal|native=count,timedelta,match,vesta"
    assert modelling.FeatureStack(native_groups=()).name.endswith("native=none")
    with pytest.raises(KeyError):
        modelling.DEFAULT_STACK.without_group("graph")
    assert modelling.DEFAULT_STACK.to_dict()["native_groups"] == list(features.NATIVE_GROUPS)


# --- the tree path --------------------------------------------------------------------------


def test_tree_matrix_keeps_nulls_and_codes_categoricals(
    prepared: pd.DataFrame, prepared_columns: list[str]
) -> None:
    matrix = modelling.tree_matrix(prepared, prepared_columns)
    assert list(matrix.columns) == prepared_columns
    assert matrix["D1"].isna().sum() == prepared["D1"].isna().sum()
    assert matrix["M1_cat"].dtype == np.dtype("int16")
    assert set(matrix["M1_cat"].unique()) <= {0, 1, 2}
    assert matrix["has_identity_record"].dtype == np.dtype("int8")
    assert matrix["C1"].dtype == np.dtype("int16")
    mask = modelling.interpolable_mask(matrix)
    assert mask[prepared_columns.index("D1")]
    assert not mask[prepared_columns.index("M1_cat")]
    assert not mask[prepared_columns.index("missing_block_D1")]


def test_tree_codes_are_the_vocabulary_position_on_every_frame(prepared: pd.DataFrame) -> None:
    later = prepared.copy()
    later["ProductCD_cat"] = pd.Categorical(["C"] * len(later), categories=["W", "C", "(rare)"])
    codes = modelling.tree_matrix(later, ["ProductCD_cat"])["ProductCD_cat"]
    assert (codes == 1).all()


# --- the complete-data path -----------------------------------------------------------------


@pytest.fixture(scope="module")
def complete(prepared: pd.DataFrame, prepared_columns: list[str]) -> dict[str, object]:
    columns = [*prepared_columns, *CAUSAL, *GRAPH]
    fitted = modelling.fit_complete_data_path(prepared, columns)
    matrix, names, mask = modelling.apply_complete_data_path(prepared, fitted)
    return {"fitted": fitted, "matrix": matrix, "names": names, "mask": mask, "columns": columns}


def test_complete_path_refuses_a_frame_past_the_boundary(prepared: pd.DataFrame) -> None:
    late = prepared.copy()
    late.loc[late.index[-1], config.TIME_COLUMN] = config.TRAIN_END_DT
    with pytest.raises(TrainOnlyError):
        modelling.fit_complete_data_path(late, ["C1"])


def test_complete_path_has_no_null_and_one_hot_rows_sum_to_one(complete: dict[str, object]) -> None:
    matrix = complete["matrix"]
    names = complete["names"]
    assert isinstance(matrix, np.ndarray) and isinstance(names, list)
    assert matrix.dtype == np.dtype("float32")
    assert not np.isnan(matrix).any()
    assert matrix.shape[1] == len(names)
    for column in ("M1_cat", "ProductCD_cat"):
        block = [i for i, n in enumerate(names) if n.startswith(f"{column}=")]
        assert np.allclose(matrix[:, block].sum(axis=1), 1.0)


def test_complete_path_standardises_and_clips_on_train(
    prepared: pd.DataFrame, complete: dict[str, object]
) -> None:
    fitted = complete["fitted"]
    matrix = complete["matrix"]
    names = complete["names"]
    mask = complete["mask"]
    assert isinstance(fitted, dict) and isinstance(matrix, np.ndarray)
    assert isinstance(names, list) and isinstance(mask, np.ndarray)
    numeric = fitted["numeric_columns"]
    assert mask[: len(numeric)].all() and not mask[len(numeric) :].any()
    means = matrix[:, : len(numeric)].mean(axis=0)
    assert np.allclose(means, 0.0, atol=1e-3)
    assert config.AMOUNT_COLUMN in fitted["clip"]["bounds"]
    assert "C1" in fitted["clip"]["bounds"]
    assert "D1" in fitted["imputer"]["sentinels"]
    assert fitted["imputer"]["sentinels"]["D1"]["sentinel"] < prepared["D1"].min()
    assert "rec_seconds_since_prev" in fitted["extra_indicator_columns"]
    assert "D1" not in fitted["extra_indicator_columns"]
    indicator = names.index("rec_seconds_since_prev_missing")
    assert matrix[:, indicator].sum() == prepared["rec_seconds_since_prev"].isna().sum()
    assert fitted["n_output_columns"] == matrix.shape[1]


def test_complete_path_applies_train_statistics_to_a_later_frame(
    prepared: pd.DataFrame, complete: dict[str, object]
) -> None:
    fitted = complete["fitted"]
    assert isinstance(fitted, dict)
    later = prepared.copy()
    later[config.AMOUNT_COLUMN] = later[config.AMOUNT_COLUMN] * 100
    matrix, names, _ = modelling.apply_complete_data_path(later, fitted)
    amount = names.index(config.AMOUNT_COLUMN)
    upper = fitted["clip"]["bounds"][config.AMOUNT_COLUMN]["upper"]
    mean = fitted["mean"][amount]
    std = fitted["std"][amount]
    assert np.allclose(matrix[:, amount].max(), (upper - mean) / std, atol=1e-4)


# --- smote ----------------------------------------------------------------------------------


def test_smote_plan_balances_the_classes_from_minority_neighbours() -> None:
    rng = np.random.default_rng(config.SEED)
    space = rng.normal(size=(200, 4)).astype("float32")
    y = (np.arange(200) < 30).astype("int64")
    plan = modelling.smote_plan(space, y, seed=1)
    assert plan["n_synthetic"] == 170 - 30
    assert plan["base"].size == plan["neighbour"].size == plan["position"].size == 140
    assert (y[plan["base"]] == 1).all() and (y[plan["neighbour"]] == 1).all()
    assert (plan["base"] != plan["neighbour"]).all()
    assert (plan["position"] >= 0).all() and (plan["position"] < 1).all()
    again = modelling.smote_plan(space, y, seed=1)
    assert np.array_equal(plan["base"], again["base"])
    assert not np.array_equal(plan["base"], modelling.smote_plan(space, y, seed=2)["base"])
    # The neighbour really is among the k nearest positives.
    minority = space[:30]
    for base, neighbour in zip(plan["base"][:20], plan["neighbour"][:20], strict=True):
        distances = np.linalg.norm(minority - space[base], axis=1)
        nearest = set(np.argsort(distances)[1 : modelling.SMOTE_NEIGHBOURS + 1])
        assert neighbour in nearest


def test_smote_plan_edge_cases() -> None:
    space = np.zeros((10, 2), dtype="float32")
    with pytest.raises(ValueError):
        modelling.smote_plan(space, np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0]), seed=0)
    balanced = modelling.smote_plan(
        np.random.default_rng(0).normal(size=(20, 2)).astype("float32"),
        np.array([1] * 10 + [0] * 10),
        seed=0,
    )
    assert balanced["n_synthetic"] == 0
    with pytest.raises(ValueError):
        modelling.smote_plan(space, np.ones(10, dtype="int64"), seed=0, k=0)


def test_materialise_smote_interpolates_floats_and_copies_codes() -> None:
    matrix = np.array(
        [[0.0, 10.0, 3], [1.0, np.nan, 4], [4.0, 20.0, 5], [9.0, 30.0, 6]], dtype="float32"
    )
    y = np.array([1, 1, 0, 0])
    plan = {
        "base": np.array([0, 1]),
        "neighbour": np.array([1, 0]),
        "position": np.array([0.25, 0.5], dtype="float32"),
        "n_synthetic": 2,
    }
    interpolable = np.array([True, True, False])
    out_x, out_y = modelling.materialise_smote(matrix, y, plan, interpolable)
    assert out_x.shape == (6, 3) and out_y.tolist() == [1, 1, 0, 0, 1, 1]
    assert out_x[4, 0] == pytest.approx(0.25)
    assert np.isnan(out_x[4, 1])
    assert out_x[4, 2] == 3
    assert out_x[5, 0] == pytest.approx(0.5)
    assert out_x[5, 2] == 4
    same_x, same_y = modelling.materialise_smote(
        matrix, y, {"base": np.zeros(0, dtype="int64")}, interpolable
    )
    assert same_x is matrix and same_y is y


# --- the families ---------------------------------------------------------------------------


def test_make_model_applies_the_strategy_the_way_each_library_takes_it() -> None:
    y = np.array([1] * 10 + [0] * 90)
    assert modelling.class_weight_ratio(y) == 9.0
    lr = modelling.make_model("logistic_regression", "class_weight", y)
    assert lr.class_weight == "balanced"
    assert modelling.make_model("logistic_regression", "smote", y).class_weight is None
    rf = modelling.make_model("random_forest", "class_weight", y)
    assert rf.class_weight == "balanced" and rf.n_estimators == 200
    xgb = modelling.make_model("xgboost", "class_weight", y)
    assert xgb.scale_pos_weight == 9.0
    assert modelling.make_model("xgboost", "none", y).scale_pos_weight == 1.0
    assert (
        modelling.make_model("xgboost", "none", y, hyperparameters={"max_depth": 3}).max_depth == 3
    )
    with pytest.raises(ValueError):
        modelling.make_model("catboost", "none", y)
    with pytest.raises(ValueError):
        modelling.make_model("xgboost", "undersample", y)
    with pytest.raises(ValueError):
        modelling.class_weight_ratio(np.zeros(5, dtype="int64"))


def test_convergence_reports_the_cap_honestly() -> None:
    rng = np.random.default_rng(config.SEED)
    x = rng.normal(size=(300, 5))
    y = (x[:, 0] + 0.3 * rng.normal(size=300) > 0).astype("int64")
    fitted = modelling.make_model("logistic_regression", "none", y).fit(x, y)
    report = modelling.convergence(fitted)
    assert report is not None and report["converged"] and report["n_iter"] < report["max_iter"]
    capped = modelling.make_model("logistic_regression", "none", y, hyperparameters={"max_iter": 1})
    with pytest.warns(Warning):
        capped.fit(x, y)
    report = modelling.convergence(capped)
    assert report is not None and not report["converged"] and report["n_iter"] == 1
    assert modelling.convergence(modelling.make_model("xgboost", "none", y)) is None
    assert modelling.score(fitted, x).shape == (300,)


# --- time-aware folds -----------------------------------------------------------------------


def test_expanding_window_folds_never_fit_on_a_later_row() -> None:
    rng = np.random.default_rng(config.SEED)
    ts = rng.integers(0, 1_000_000, 5_000).astype("int64")
    # Ties on purpose: the same timestamp must land in one block.
    ts[:50] = ts[0]
    folds = modelling.expanding_window_folds(ts, 3)
    assert [f["fold"] for f in folds] == [0, 1, 2]
    for fold in folds:
        assert ts[fold["fit"]].max() < ts[fold["validate"]].min()
        assert fold["fit_end_ts"] < fold["validate_start_ts"] <= fold["validate_end_ts"]
        assert np.intersect1d(fold["fit"], fold["validate"]).size == 0
    assert folds[0]["fit"].size < folds[1]["fit"].size < folds[2]["fit"].size
    assert set(folds[1]["fit"]) == set(folds[0]["fit"]) | set(folds[0]["validate"])
    tie_rows = set(np.flatnonzero(ts == ts[0]))
    for fold in folds:
        for side in ("fit", "validate"):
            overlap = tie_rows & set(fold[side])
            assert overlap in (set(), tie_rows)
    with pytest.raises(ValueError):
        modelling.expanding_window_folds(ts, 0)
    with pytest.raises(ValueError):
        modelling.expanding_window_folds(np.zeros(10, dtype="int64"), 2)
