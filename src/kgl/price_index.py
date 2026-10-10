"""Price indices per trade period, built from training rows only, and their extrapolation to later periods.

An index value is a log price level relative to the first period. Training targets are deflated by it
(log price minus index), and predictions for later periods add the extrapolated index back.
"""

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import Ridge


def median_index(t: np.ndarray, log_value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median log value per period, relative to the first period. Moves with the mix of rows."""
    periods = np.unique(t)
    medians = np.array([np.median(log_value[t == p]) for p in periods])
    return periods, medians - medians[0]


def hedonic_index(t: np.ndarray, log_value: np.ndarray, controls: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
    """Period effects of a regression of `log_value` on period dummies plus `controls`.

    Holding the controls fixed, the period coefficients measure the price change of a comparable property,
    so a shift in the mix of rows (location, type, size, age) does not move the index.
    """
    periods, position = np.unique(t, return_inverse=True)
    rows = np.arange(len(t))
    dummies = sp.csr_matrix((np.ones(len(t)), (rows, position)), shape=(len(t), len(periods)))[:, 1:]
    # A tiny ridge penalty keeps the solve stable when controls are collinear (one-hot groups with an intercept).
    model = Ridge(alpha=1e-3).fit(sp.hstack([dummies, controls]).tocsr(), log_value)
    return periods, np.concatenate([[0.0], model.coef_[: len(periods) - 1]])


def extrapolate(periods: np.ndarray, index: np.ndarray, target: np.ndarray, *, how: str, k: int = 8) -> np.ndarray:
    """Index values for periods after the last one: `flat` keeps the last level, `linear` extends the
    straight line fitted to the last `k` periods."""
    if how == "flat":
        return np.full(len(target), index[-1], dtype=float)
    if how == "linear":
        slope, intercept = np.polyfit(periods[-k:], index[-k:], 1)
        return intercept + slope * np.asarray(target, dtype=float)
    raise ValueError(f"unknown how={how!r}; use 'flat' or 'linear'")


def lookup(periods: np.ndarray, index: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Index values for rows whose periods are among `periods`."""
    position = np.searchsorted(periods, t)
    known = (position < len(periods)) & (periods[np.minimum(position, len(periods) - 1)] == t)
    if not known.all():
        raise KeyError(f"periods without an index value: {sorted(set(np.asarray(t)[~known].tolist()))[:5]}")
    return index[position]


def offset(periods: np.ndarray, index: np.ndarray, t: np.ndarray, *, k: int) -> np.ndarray:
    """Index value per row: looked up for periods the index covers, linearly extrapolated (last `k`
    periods) for periods after the last one. Any other period raises."""
    t = np.asarray(t, dtype=float)
    out = np.empty(len(t))
    later = t > periods[-1]
    out[later] = extrapolate(periods, index, t[later], how="linear", k=k)
    out[~later] = lookup(periods, index, t[~later])
    return out
