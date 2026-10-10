"""Target encoding: the smoothed mean of a target per category value.

A group with n rows and sum s gets (s + m * prior) / (n + m), so groups with few rows lean on the prior.
A value never seen in the fitting rows gets the prior. Missing keys form their own group.
"""

import numpy as np
import polars as pl


def smoothed_mean(
    fit_keys: pl.Series, fit_values: np.ndarray, apply_keys: pl.Series, *, prior: float, m: float
) -> np.ndarray:
    """Smoothed mean of `fit_values` per key, looked up for `apply_keys`."""
    stats = (
        pl.DataFrame({"key": fit_keys, "value": fit_values})
        .group_by("key")
        .agg(pl.col("value").sum().alias("total"), pl.len().alias("n"))
    )
    looked_up = pl.DataFrame({"key": apply_keys}).join(
        stats, on="key", how="left", nulls_equal=True, maintain_order="left"
    )
    encoded = (looked_up["total"].fill_null(0.0) + m * prior) / (looked_up["n"].fill_null(0) + m)
    return encoded.to_numpy()


def oof_smoothed_mean(keys: pl.Series, values: np.ndarray, folds: np.ndarray, *, m: float) -> np.ndarray:
    """Out-of-fold encoding: each row is encoded from the rows of the other folds only, so a row's own
    target never enters its feature. The prior is the mean of the other folds' values."""
    out = np.empty(len(values))
    for k in np.unique(folds):
        apply_rows, fit_rows = np.flatnonzero(folds == k), np.flatnonzero(folds != k)
        out[apply_rows] = smoothed_mean(
            keys.gather(fit_rows), values[fit_rows], keys.gather(apply_rows), prior=float(values[fit_rows].mean()), m=m
        )
    return out
