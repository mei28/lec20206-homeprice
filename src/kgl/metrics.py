"""Evaluation metrics addressed by name. Add the competition metric here, with a test, during onboarding."""

from collections.abc import Callable

import numpy as np
from sklearn import metrics as skm

Metric = Callable[[np.ndarray, np.ndarray], float]


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


# name -> (function, greater_is_better)
_METRICS: dict[str, tuple[Metric, bool]] = {
    "accuracy": (lambda y, p: float(skm.accuracy_score(y, p)), True),
    "auc": (lambda y, p: float(skm.roc_auc_score(y, p)), True),
    "logloss": (lambda y, p: float(skm.log_loss(y, p)), False),
    "f1_macro": (lambda y, p: float(skm.f1_score(y, p, average="macro")), True),
    "rmse": (_rmse, False),
    "mae": (lambda y, p: float(skm.mean_absolute_error(y, p)), False),
}


def get_metric(name: str) -> Metric:
    if name not in _METRICS:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(_METRICS)}")
    return _METRICS[name][0]


def greater_is_better(name: str) -> bool:
    if name not in _METRICS:
        raise KeyError(f"unknown metric {name!r}; known: {sorted(_METRICS)}")
    return _METRICS[name][1]


def score_folds(y: np.ndarray, oof: np.ndarray, folds: np.ndarray, name: str) -> tuple[float, list[float]]:
    """Score each validation fold separately; the CV score is the mean of the per-fold scores."""
    metric = get_metric(name)
    y, oof, folds = np.asarray(y), np.asarray(oof), np.asarray(folds)
    per_fold = [metric(y[folds == k], oof[folds == k]) for k in sorted(set(folds.tolist())) if k >= 0]
    return float(np.mean(per_fold)), per_fold
