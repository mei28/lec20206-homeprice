import numpy as np
import pandas as pd
import polars as pl
import pytest

from kgl import lesson
from kgl.env import Env
from kgl.io import read_metrics


def test_refit_rounds_scales_by_the_row_ratio():
    # S1: best iteration 1,276 on 180,632 training rows, refit on 202,323 rows.
    assert lesson.refit_rounds(1276, n_all=202_323, n_train=180_632) == 1429


def test_record_run_writes_metrics_config_and_oof(tmp_path):
    env = Env(kind="local", root=tmp_path, input_dir=tmp_path / "input", output_dir=tmp_path / "output")
    holdout = pl.DataFrame({"id": [1, 2], "y": [10.0, 20.0], "fold": [0, 0]})
    out = lesson.record_run(
        env,
        exp="s9_demo",
        base="s8_demo",
        run="base",
        score=0.25,
        metric="rmsle",
        config={"comp": "c", "id_col": "id"},
        holdout=holdout,
        pred=np.array([11.0, 19.0]),
        notes="demo",
        commit="abc123",
    )
    assert out == tmp_path / "output" / "s9_demo" / "base"
    metrics = read_metrics(out / "metrics.json")
    assert (metrics.exp, metrics.base, metrics.run, metrics.cv, metrics.cv_folds) == (
        "s9_demo",
        "s8_demo",
        "base",
        0.25,
        [0.25],
    )
    assert metrics.config == {"comp": "c", "id_col": "id"} and metrics.git_commit == "abc123"
    assert pl.read_parquet(out / "oof.parquet")["pred"].to_list() == [11.0, 19.0]
    assert (out / "config.json").exists()


def test_fit_lgbm_holdout_stops_early_and_predicts_the_holdout():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=2000)})
    y = 2 * x + rng.normal(scale=0.5, size=2000)
    tr, va = np.arange(1500), np.arange(1500, 2000)
    params = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 7, "verbose": -1, "num_threads": 1}
    booster, pred = lesson.fit_lgbm_holdout(X, y, tr, va, params, max_rounds=2000, early_stopping_rounds=20)
    assert 0 < booster.best_iteration < 2000
    assert pred.shape == (500,)
    assert np.sqrt(np.mean((pred - y[va]) ** 2)) < 0.7


def test_fit_lgbm_holdout_uses_row_weights():
    # Two groups with the same feature but different targets: the weighted fit leans to the heavy group.
    X = pd.DataFrame({"x": np.zeros(400)})
    y = np.r_[np.zeros(200), np.ones(200)]
    tr, va = np.arange(400), np.arange(400)
    params = {"objective": "regression", "learning_rate": 0.5, "num_leaves": 2, "verbose": -1, "num_threads": 1}
    weight = np.r_[np.full(200, 9.0), np.ones(200)]
    _, pred = lesson.fit_lgbm_holdout(X, y, tr, va, params, max_rounds=50, early_stopping_rounds=10, weight=weight)
    assert pred.mean() == pytest.approx(0.1, abs=0.01)


def test_catboost_holdout_and_refit_handle_categories_with_missing_values():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "ward": pd.Categorical(rng.choice(["港区", "青梅市", None], size=n), categories=["港区", "青梅市"]),
        }
    )
    y = 2 * X["x"].to_numpy() + np.where(X["ward"] == "港区", 1.0, 0.0) + rng.normal(scale=0.1, size=n)
    tr, va = np.arange(450), np.arange(450, n)
    params = {"learning_rate": 0.2, "depth": 3, "loss_function": "RMSE", "random_seed": 0, "thread_count": 1}
    rounds, pred = lesson.fit_catboost_holdout(X, y, tr, va, params, max_rounds=300, early_stopping_rounds=20)
    assert 0 < rounds <= 300
    assert np.sqrt(np.mean((pred - y[va]) ** 2)) < 0.5
    test_pred = lesson.refit_catboost(X, y, X.iloc[:5], params, rounds=rounds)
    assert test_pred.shape == (5,)
