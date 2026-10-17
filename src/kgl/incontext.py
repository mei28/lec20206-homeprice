"""Helpers for in-context models such as TabPFN (S13 onward): choosing the context rows and predicting in chunks.

An in-context model reads its training rows (the context) at prediction time, so the context size is bounded by
GPU memory, and long predictions are split into chunks.
"""

from typing import Any

import numpy as np
import pandas as pd


def most_recent(rows: np.ndarray, periods: np.ndarray, n: int) -> np.ndarray:
    """The n rows of `rows` with the latest trade periods (ties keep row order), in row order."""
    order = np.argsort(-periods[rows], kind="stable")
    return np.sort(rows[order[:n]])


def predict_in_chunks(model: Any, X: pd.DataFrame, step: int) -> np.ndarray:
    """`model.predict` on consecutive chunks of `step` rows, concatenated as float64."""
    return np.concatenate([model.predict(X.iloc[i : i + step]) for i in range(0, len(X), step)]).astype(np.float64)
