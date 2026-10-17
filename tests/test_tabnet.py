import numpy as np
import pandas as pd
import pytest

from kgl import tabnet

PARAMS = {
    "n_d": 4,
    "n_a": 4,
    "n_steps": 2,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
    "cat_emb_dim": 2,
    "lr": 0.02,
    "lr_step_size": 10,
    "lr_gamma": 0.9,
    "batch_size": 256,
    "virtual_batch_size": 64,
    "max_epochs": 12,
    "patience": 5,
    "seed": 0,
}


@pytest.fixture
def rows():
    rng = np.random.default_rng(0)
    n = 1200
    city = rng.choice(["a", "b", "c"], size=n).astype(object)
    city[:10] = None
    X = pd.DataFrame({"city": pd.Categorical(city, categories=["a", "b", "c"]), "x": rng.normal(size=n)})
    X.loc[5:9, "x"] = np.nan
    effect = pd.Series(city).map({"a": -1.0, "b": 0.0, "c": 1.0}).fillna(0.0).to_numpy()
    y = 5.0 + 2 * np.nan_to_num(X["x"].to_numpy()) + effect + rng.normal(scale=0.1, size=n)
    return X, y


def test_inputs_put_category_codes_first_with_zero_for_missing_and_scale_with_the_fit_rows(rows):
    X, _ = rows
    fit, apply = X.iloc[:800], X.iloc[800:]
    (m_fit, m_apply), cat_dims = tabnet.inputs(fit, [fit, apply])
    assert cat_dims == [4]
    assert m_fit.dtype == np.float32 and m_fit.shape == (800, 2)
    assert m_fit[:10, 0].tolist() == [0.0] * 10
    assert m_fit[10:, 0].min() >= 1.0
    median = fit["x"].median()
    filled = fit["x"].fillna(median)
    expected = ((apply["x"].fillna(median) - filled.mean()) / filled.std()).to_numpy(dtype=np.float32)
    assert m_apply[:, 1] == pytest.approx(expected)


def test_fit_holdout_stops_early_and_predicts_the_holdout_on_the_target_scale(rows):
    X, y = rows
    tr, va = np.arange(900), np.arange(900, 1200)
    epochs, pred = tabnet.fit_holdout(X, y, tr, va, PARAMS, device="cpu")
    assert 1 <= epochs <= PARAMS["max_epochs"]
    assert pred.shape == (300,) and pred.dtype == np.float64
    assert np.sqrt(np.mean((pred - y[va]) ** 2)) < np.std(y[va])


def test_refit_trains_on_every_row_and_predicts_the_test_rows(rows):
    X, y = rows
    pred = tabnet.refit(X.iloc[:900], y[:900], X.iloc[900:], PARAMS, epochs=10, device="cpu")
    assert pred.shape == (300,) and pred.dtype == np.float64
    assert abs(pred.mean() - y[900:].mean()) < 1.0
