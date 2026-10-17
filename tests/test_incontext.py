import numpy as np
import pandas as pd

from kgl import incontext


def test_most_recent_keeps_the_latest_periods_in_row_order():
    periods = np.array([2020.0, 2024.5, 2023.0, 2024.5, 2024.75, 2019.0])
    rows = np.array([0, 1, 2, 3, 5])  # row 4 (the latest) is not a candidate
    assert incontext.most_recent(rows, periods, 3).tolist() == [1, 2, 3]


def test_most_recent_breaks_ties_by_row_order():
    periods = np.array([2024.0, 2024.0, 2024.0, 2023.0])
    assert incontext.most_recent(np.arange(4), periods, 2).tolist() == [0, 1]


class _RowSum:
    """A stand-in model that records the chunk sizes it was asked to predict."""

    def __init__(self):
        self.chunks = []

    def predict(self, X):
        self.chunks.append(len(X))
        return X.sum(axis=1).to_numpy().astype(np.float32)


def test_predict_in_chunks_matches_one_call_and_returns_float64():
    X = pd.DataFrame({"a": np.arange(7.0), "b": np.ones(7)})
    model = _RowSum()
    pred = incontext.predict_in_chunks(model, X, 3)
    assert model.chunks == [3, 3, 1]
    assert pred.dtype == np.float64
    assert pred.tolist() == (X["a"] + 1).tolist()
