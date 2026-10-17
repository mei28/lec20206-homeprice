"""Helpers shared by the stage notebooks: holdout training, the refit rule, and run records.

The stage notebooks validate on one time holdout (fold 0) and submit a model refit on every training row.
"""

import datetime as dt
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoostRegressor

from kgl.env import Env, run_dir
from kgl.io import Metrics, write_config, write_metrics, write_oof


def fit_lgbm_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    params: dict[str, Any],
    *,
    max_rounds: int,
    early_stopping_rounds: int,
    weight: np.ndarray | None = None,
) -> tuple[lgb.Booster, np.ndarray]:
    """Train on `tr_idx`, stop early on `va_idx`, and return the booster with its holdout predictions.

    `weight` (one value per row of X) weights the training rows; the holdout is always scored unweighted.
    """
    train_weight = None if weight is None else weight[tr_idx]
    train_set = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx], weight=train_weight)
    valid_set = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=train_set)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    return booster, np.asarray(booster.predict(X.iloc[va_idx], num_iteration=booster.best_iteration))


def refit_rounds(best_iteration: int, *, n_all: int, n_train: int) -> int:
    """Rounds for the refit on all rows: the holdout's best iteration scaled by how many more rows there are."""
    return int(round(best_iteration * n_all / n_train))


def record_run(
    env: Env,
    *,
    exp: str,
    base: str,
    run: str,
    score: float,
    metric: str,
    config: dict[str, Any],
    holdout: pl.DataFrame,
    pred: np.ndarray,
    notes: str,
    commit: str,
) -> Path:
    """Write config.json, oof.parquet (`holdout` plus a `pred` column), and metrics.json for one run."""
    finished = dt.datetime.now().isoformat(timespec="seconds")
    out = run_dir(env, exp, run)
    write_config(out, config, [])
    write_oof(out, holdout.with_columns(pl.Series("pred", pred)))
    write_metrics(
        out,
        Metrics(
            exp=exp,
            base=base,
            run=run,
            cv=score,
            cv_folds=[score],
            metric=metric,
            config=config,
            argv=[],
            git_commit=commit,
            started_at=finished,
            finished_at=finished,
            notes=notes,
        ),
    )
    return out


def _for_catboost(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """CatBoost takes categories as strings without missing values: missing becomes the string "NA"."""
    categorical = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    out = X.copy()
    for c in categorical:
        out[c] = out[c].cat.add_categories("NA").fillna("NA").astype(str)
    return out, categorical


def fit_catboost_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    params: dict[str, Any],
    *,
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[int, np.ndarray]:
    """Train CatBoost on `tr_idx`, stop early on `va_idx`; return the rounds used and the holdout predictions."""
    X_cat, categorical = _for_catboost(X)
    model = CatBoostRegressor(iterations=max_rounds, od_type="Iter", od_wait=early_stopping_rounds, verbose=0, **params)
    model.fit(
        X_cat.iloc[tr_idx],
        y[tr_idx],
        cat_features=categorical,
        eval_set=(X_cat.iloc[va_idx], y[va_idx]),
        use_best_model=True,
    )
    return model.get_best_iteration() + 1, np.asarray(model.predict(X_cat.iloc[va_idx]))


def refit_catboost(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, params: dict[str, Any], *, rounds: int
) -> np.ndarray:
    """Train CatBoost on every row of X for `rounds` rounds and predict X_test."""
    X_cat, categorical = _for_catboost(X)
    model = CatBoostRegressor(iterations=rounds, verbose=0, **params)
    model.fit(X_cat, y, cat_features=categorical)
    return np.asarray(model.predict(_for_catboost(X_test)[0]))


def fit_xgb_holdout(
    X: pd.DataFrame,
    y: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    params: dict[str, Any],
    *,
    max_rounds: int,
    early_stopping_rounds: int,
) -> tuple[int, np.ndarray]:
    """Train XGBoost on `tr_idx`, stop early on `va_idx`; return the rounds used and the holdout predictions."""
    train = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], enable_categorical=True)
    valid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], enable_categorical=True)
    booster = xgb.train(
        params,
        train,
        max_rounds,
        evals=[(valid, "holdout")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
    )
    pred = booster.predict(valid, iteration_range=(0, booster.best_iteration + 1))
    return booster.best_iteration + 1, np.asarray(pred)


def refit_xgb(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, params: dict[str, Any], *, rounds: int
) -> np.ndarray:
    """Train XGBoost on every row of X for `rounds` rounds and predict X_test."""
    booster = xgb.train(params, xgb.DMatrix(X, label=y, enable_categorical=True), rounds)
    return np.asarray(booster.predict(xgb.DMatrix(X_test, enable_categorical=True)))
