"""Combining member predictions (log scale): non-negative weights, Ridge stacking, and fair holdout scoring.

With one time holdout, a combiner fit on the holdout and scored on it looks better than it is.
`cross_predict` fits on one part of the holdout and predicts the other, both ways.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import Ridge


def fit_weights(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-negative weights per member (columns of H) that minimize the squared error; no intercept."""
    weights, _ = nnls(H, y)
    return weights


def fit_stacking(H: np.ndarray, y: np.ndarray, *, alpha: float) -> Ridge:
    """A Ridge meta-model on the members' predictions; coefficients may be negative, with an intercept."""
    return Ridge(alpha=alpha).fit(H, y)


def cross_predict(
    H: np.ndarray,
    y: np.ndarray,
    first_part: np.ndarray,
    fit: Callable[[np.ndarray, np.ndarray], Any],
    predict: Callable[[Any, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Predict the rows outside `first_part` with a combiner fit on `first_part`, and the reverse."""
    out = np.empty(len(y))
    for fit_rows in (first_part, ~first_part):
        combiner = fit(H[fit_rows], y[fit_rows])
        out[~fit_rows] = predict(combiner, H[~fit_rows])
    return out


def cross_predict_with_block(
    H: np.ndarray,
    y: np.ndarray,
    first_part: np.ndarray,
    H_block: np.ndarray,
    y_block: np.ndarray,
    fit: Callable[[np.ndarray, np.ndarray], Any],
    predict: Callable[[Any, np.ndarray], np.ndarray],
) -> np.ndarray:
    """`cross_predict` with an extra block of rows (an earlier period) added to every fit; predicts H's rows only."""
    out = np.empty(len(y))
    for fit_rows in (first_part, ~first_part):
        combiner = fit(np.vstack([H_block, H[fit_rows]]), np.concatenate([y_block, y[fit_rows]]))
        out[~fit_rows] = predict(combiner, H[~fit_rows])
    return out
