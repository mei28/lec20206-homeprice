"""Stackers that see the members' predictions and a few row features (S12 and later).

Every function takes one matrix Z so that `ensemble.cross_predict` can split it by rows. Its layout:
columns [0, n_members) hold the members' log predictions, column n_members holds the contract-price flag
(1.0 or 0.0), and the remaining columns hold other row features.

The nonlinear stackers (MLP, LightGBM) predict a correction to the members' mean, so a stacker that learns
nothing falls back to the plain average.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from kgl import ensemble


def correction_inputs(Z: np.ndarray, *, n_members: int, with_rows: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Inputs for the nonlinear stackers (deviations from the members' mean, the mean, row features) and the mean."""
    H = Z[:, :n_members]
    mean = H.mean(axis=1)
    rows = [Z[:, n_members:]] if with_rows else []
    return np.hstack([H - mean[:, None], mean[:, None], *rows]), mean


def fit_ridge(Z: np.ndarray, y: np.ndarray, *, n_members: int, alpha: float) -> Ridge:
    """The S10 Ridge stacking: the members' predictions only, the same weights for every row."""
    return ensemble.fit_stacking(Z[:, :n_members], y, alpha=alpha)


def predict_ridge(model: Ridge, Z: np.ndarray, *, n_members: int) -> np.ndarray:
    return model.predict(Z[:, :n_members])


def by_type(Z: np.ndarray, *, n_members: int) -> np.ndarray:
    """Member predictions split into a contract-price copy and a survey-price copy, plus the contract flag."""
    H, contract = Z[:, :n_members], Z[:, n_members : n_members + 1]
    return np.hstack([H * contract, H * (1 - contract), contract])


def fit_ridge_by_type(Z: np.ndarray, y: np.ndarray, *, n_members: int, alpha: float) -> Ridge:
    """Ridge with separate member weights for contract-price and survey-price rows."""
    return Ridge(alpha=alpha).fit(by_type(Z, n_members=n_members), y)


def predict_ridge_by_type(model: Ridge, Z: np.ndarray, *, n_members: int) -> np.ndarray:
    return model.predict(by_type(Z, n_members=n_members))


def fit_mlp(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    n_members: int,
    hidden: tuple[int, ...],
    alpha: float,
    seeds: tuple[int, ...],
    valid_share: float,
) -> list[Pipeline]:
    """One MLP per seed on the correction inputs; each stops early on its own random `valid_share` of the rows."""
    X, mean = correction_inputs(Z, n_members=n_members)
    return [
        make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=hidden,
                alpha=alpha,
                early_stopping=True,
                validation_fraction=valid_share,
                max_iter=500,
                random_state=seed,
            ),
        ).fit(X, y - mean)
        for seed in seeds
    ]


def predict_mlp(models: list[Pipeline], Z: np.ndarray, *, n_members: int) -> np.ndarray:
    """The members' mean plus the seeds' average correction."""
    X, mean = correction_inputs(Z, n_members=n_members)
    return mean + np.mean([m.predict(X) for m in models], axis=0)


def fit_lgbm(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    n_members: int,
    params: dict[str, Any],
    max_rounds: int,
    early_stopping_rounds: int,
    valid_share: float,
    seed: int,
    with_rows: bool = True,
) -> lgb.Booster:
    """LightGBM on the correction inputs, early-stopped on a random `valid_share` of the rows."""
    X, mean = correction_inputs(Z, n_members=n_members, with_rows=with_rows)
    inner = np.random.default_rng(seed).random(len(y)) < valid_share
    train_set = lgb.Dataset(X[~inner], label=(y - mean)[~inner])
    valid_set = lgb.Dataset(X[inner], label=(y - mean)[inner], reference=train_set)
    return lgb.train(
        params,
        train_set,
        num_boost_round=max_rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )


def predict_lgbm(booster: lgb.Booster, Z: np.ndarray, *, n_members: int, with_rows: bool = True) -> np.ndarray:
    """The members' mean plus the booster's correction."""
    X, mean = correction_inputs(Z, n_members=n_members, with_rows=with_rows)
    return mean + booster.predict(X, num_iteration=booster.best_iteration)
