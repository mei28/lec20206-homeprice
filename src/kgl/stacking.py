"""Stackers that see the members' predictions and a few row features (S12 and later).

Every function takes one matrix Z so that `ensemble.cross_predict` can split it by rows. Its layout:
columns [0, n_members) hold the members' log predictions, column n_members holds the contract-price flag
(1.0 or 0.0), and the remaining columns hold other row features.

The nonlinear stackers (MLP, LightGBM) predict a correction to the members' mean, so a stacker that learns
nothing falls back to the plain average.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from kgl import ensemble

ROW_FEATURES = ["contract", "log_area", "station_minutes", "te_district"]


def row_features(X: pd.DataFrame) -> np.ndarray:
    """The row features the stackers may condition on, in the order of ROW_FEATURES (from the S5 feature frame)."""
    return np.column_stack(
        [
            (X["価格情報区分"] == "成約価格情報").to_numpy(dtype=float),
            np.log(X["area"].to_numpy()),
            X["station_minutes"].to_numpy(),
            X["te_district"].to_numpy(),
        ]
    )


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


def stackers(
    n_members: int,
    *,
    alpha: float,
    mlp_hidden: tuple[int, ...],
    mlp_alpha: float,
    mlp_seeds: tuple[int, ...],
    lgbm_params: dict[str, Any],
    lgbm_max_rounds: int,
    lgbm_early_stopping_rounds: int,
    valid_share: float,
    seed: int,
) -> dict[str, tuple[Callable[..., Any], Callable[..., np.ndarray]]]:
    """The four S12 stackers as (fit(Z, y), predict(model, Z)) pairs."""
    return {
        "ridge": (
            partial(fit_ridge, n_members=n_members, alpha=alpha),
            partial(predict_ridge, n_members=n_members),
        ),
        "ridge_by_type": (
            partial(fit_ridge_by_type, n_members=n_members, alpha=alpha),
            partial(predict_ridge_by_type, n_members=n_members),
        ),
        "mlp": (
            partial(
                fit_mlp,
                n_members=n_members,
                hidden=mlp_hidden,
                alpha=mlp_alpha,
                seeds=mlp_seeds,
                valid_share=valid_share,
            ),
            partial(predict_mlp, n_members=n_members),
        ),
        "lgbm": (
            partial(
                fit_lgbm,
                n_members=n_members,
                params=lgbm_params,
                max_rounds=lgbm_max_rounds,
                early_stopping_rounds=lgbm_early_stopping_rounds,
                valid_share=valid_share,
                seed=seed,
            ),
            partial(predict_lgbm, n_members=n_members),
        ),
    }


def stacker_average(
    H: np.ndarray,
    T: np.ndarray,
    rows_holdout: np.ndarray,
    rows_test: np.ndarray,
    y: np.ndarray,
    first_part: np.ndarray,
    found: dict[str, tuple[Callable[..., Any], Callable[..., np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray]:
    """S12's recipe (S13 onward): the `found` stackers and the weighted average, then their plain average.

    H and T hold the members' holdout and test log predictions (one column per member), rows_* the row features.
    The holdout predictions come from `ensemble.cross_predict` on `first_part`; the test predictions from
    combiners fit on every holdout row.
    """
    Z_h, Z_t = np.hstack([H, rows_holdout]), np.hstack([T, rows_test])
    parts = [
        (ensemble.cross_predict(Z_h, y, first_part, fit, predict), predict(fit(Z_h, y), Z_t))
        for fit, predict in found.values()
    ]
    parts.append(
        (
            ensemble.cross_predict(H, y, first_part, ensemble.fit_weights, lambda w, Hp: Hp @ w),
            T @ ensemble.fit_weights(H, y),
        )
    )
    return np.mean([h for h, _ in parts], axis=0), np.mean([t for _, t in parts], axis=0)


def stacker_average_with_block(
    H: np.ndarray,
    T: np.ndarray,
    rows_holdout: np.ndarray,
    rows_test: np.ndarray,
    y: np.ndarray,
    first_part: np.ndarray,
    H_block: np.ndarray,
    rows_block: np.ndarray,
    y_block: np.ndarray,
    found: dict[str, tuple[Callable[..., Any], Callable[..., np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray]:
    """`stacker_average` with an earlier block of rows (S15: 2024Q3-Q4) added to every combiner's training rows.

    The holdout predictions come from `ensemble.cross_predict_with_block`; the test predictions from combiners fit
    on the block and every holdout row.
    """
    Z_h, Z_t, Z_b = np.hstack([H, rows_holdout]), np.hstack([T, rows_test]), np.hstack([H_block, rows_block])
    Z_all, H_all, y_all = np.vstack([Z_b, Z_h]), np.vstack([H_block, H]), np.concatenate([y_block, y])
    parts = [
        (
            ensemble.cross_predict_with_block(Z_h, y, first_part, Z_b, y_block, fit, predict),
            predict(fit(Z_all, y_all), Z_t),
        )
        for fit, predict in found.values()
    ]
    parts.append(
        (
            ensemble.cross_predict_with_block(
                H, y, first_part, H_block, y_block, ensemble.fit_weights, lambda w, Hp: Hp @ w
            ),
            T @ ensemble.fit_weights(H_all, y_all),
        )
    )
    return np.mean([h for h, _ in parts], axis=0), np.mean([t for _, t in parts], axis=0)


def stacker_average_forward(
    H: np.ndarray,
    rows: np.ndarray,
    H_block: np.ndarray,
    rows_block: np.ndarray,
    y_block: np.ndarray,
    found: dict[str, tuple[Callable[..., Any], Callable[..., np.ndarray]]],
) -> np.ndarray:
    """The stacker average fit on the earlier block only, predicting the rows of H (forward in time only)."""
    Z, Z_b = np.hstack([H, rows]), np.hstack([H_block, rows_block])
    parts = [predict(fit(Z_b, y_block), Z) for fit, predict in found.values()]
    parts.append(H @ ensemble.fit_weights(H_block, y_block))
    return np.mean(parts, axis=0)
