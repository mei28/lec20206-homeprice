"""TabNet as a member (S13 onward): its inputs, the model, and holdout and refit training.

Inputs: category codes first (0 = missing, embedded by TabNet), then numbers with missing values filled by the
median and standardized, all statistics from the fitting rows. The target is centered before training so the
output starts near zero, and the center is added back to the predictions.
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor


def inputs(X_fit: pd.DataFrame, X_apply: list[pd.DataFrame]) -> tuple[list[np.ndarray], list[int]]:
    """Category codes (0 = missing) and standardized numeric columns, with statistics from X_fit only."""
    cat_cols = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]
    num_cols = [c for c in X_fit.columns if c not in cat_cols]
    median = X_fit[num_cols].median()
    filled = X_fit[num_cols].fillna(median)
    mean, std = filled.mean(), filled.std()
    assert (std > 0).all(), std[std == 0]

    def matrix(X: pd.DataFrame) -> np.ndarray:
        num = ((X[num_cols].fillna(median) - mean) / std).to_numpy(dtype=np.float32)
        cat = np.column_stack([X[c].cat.codes.to_numpy() + 1 for c in cat_cols]).astype(np.float32)
        return np.hstack([cat, num])

    cat_dims = [len(X_fit[c].cat.categories) + 1 for c in cat_cols]
    return [matrix(X) for X in X_apply], cat_dims


def make(cat_dims: list[int], params: dict[str, Any], *, device: str) -> TabNetRegressor:
    return TabNetRegressor(
        n_d=params["n_d"],
        n_a=params["n_a"],
        n_steps=params["n_steps"],
        gamma=params["gamma"],
        lambda_sparse=params["lambda_sparse"],
        cat_idxs=list(range(len(cat_dims))),
        cat_dims=cat_dims,
        cat_emb_dim=params["cat_emb_dim"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": params["lr"]},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": params["lr_step_size"], "gamma": params["lr_gamma"]},
        seed=params["seed"],
        device_name=device,
        verbose=0,
    )


def fit_holdout(
    X: pd.DataFrame, y: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray, params: dict[str, Any], *, device: str
) -> tuple[int, np.ndarray]:
    """Train on `tr_idx`, stop early on `va_idx`; return the epochs used (best + 1) and the holdout predictions."""
    (X_tr, X_va), cat_dims = inputs(X.iloc[tr_idx], [X.iloc[tr_idx], X.iloc[va_idx]])
    center = float(y[tr_idx].mean())
    model = make(cat_dims, params, device=device)
    model.fit(
        X_tr,
        (y[tr_idx] - center).reshape(-1, 1),
        eval_set=[(X_va, (y[va_idx] - center).reshape(-1, 1))],
        eval_metric=["rmse"],
        max_epochs=params["max_epochs"],
        patience=params["patience"],
        batch_size=params["batch_size"],
        virtual_batch_size=params["virtual_batch_size"],
    )
    return int(model.best_epoch) + 1, model.predict(X_va).ravel().astype(np.float64) + center


def refit(
    X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, params: dict[str, Any], *, epochs: int, device: str
) -> np.ndarray:
    """Train on every row of X for `epochs` epochs and predict X_test."""
    (X_all, X_te), cat_dims = inputs(X, [X, X_test])
    center = float(y.mean())
    model = make(cat_dims, params, device=device)
    model.fit(
        X_all,
        (y - center).reshape(-1, 1),
        max_epochs=epochs,
        batch_size=params["batch_size"],
        virtual_batch_size=params["virtual_batch_size"],
    )
    return model.predict(X_te).ravel().astype(np.float64) + center
