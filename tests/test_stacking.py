import numpy as np
import pytest

from kgl import ensemble, stacking

N_MEMBERS = 3
LGBM_PARAMS = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 7, "verbose": -1, "num_threads": 1}


@pytest.fixture
def stacked_rows():
    """Z: three members' predictions, then a contract flag and one more row feature; y depends on the row feature."""
    rng = np.random.default_rng(0)
    n = 2000
    truth = rng.normal(17.0, 0.5, size=n)
    members = np.column_stack([truth + rng.normal(scale=s, size=n) for s in (0.1, 0.2, 0.3)])
    contract = (rng.random(n) < 0.6).astype(float)
    feature = rng.normal(size=n)
    y = truth + 0.2 * feature
    return np.column_stack([members, contract, feature]), y


def test_correction_inputs_are_deviations_from_the_mean_then_the_mean_then_row_features(stacked_rows):
    Z, _ = stacked_rows
    X, mean = stacking.correction_inputs(Z, n_members=N_MEMBERS)
    assert X.shape == (len(Z), N_MEMBERS + 1 + 2)
    assert mean == pytest.approx(Z[:, :N_MEMBERS].mean(axis=1))
    assert X[:, :N_MEMBERS].sum(axis=1) == pytest.approx(np.zeros(len(Z)), abs=1e-9)
    assert X[:, N_MEMBERS] == pytest.approx(mean)
    assert X[:, N_MEMBERS + 1 :] == pytest.approx(Z[:, N_MEMBERS:])


def test_correction_inputs_without_rows_keep_only_the_members(stacked_rows):
    Z, _ = stacked_rows
    X, _ = stacking.correction_inputs(Z, n_members=N_MEMBERS, with_rows=False)
    assert X.shape == (len(Z), N_MEMBERS + 1)


def test_ridge_stacker_is_the_s10_stacking_on_the_members_only(stacked_rows):
    Z, y = stacked_rows
    model = stacking.fit_ridge(Z, y, n_members=N_MEMBERS, alpha=1.0)
    expected = ensemble.fit_stacking(Z[:, :N_MEMBERS], y, alpha=1.0).predict(Z[:, :N_MEMBERS])
    assert stacking.predict_ridge(model, Z, n_members=N_MEMBERS) == pytest.approx(expected)


def test_by_type_keeps_each_row_in_the_copy_of_its_own_type(stacked_rows):
    Z, _ = stacked_rows
    split = stacking.by_type(Z, n_members=N_MEMBERS)
    contract = Z[:, N_MEMBERS] == 1.0
    assert split.shape == (len(Z), 2 * N_MEMBERS + 1)
    assert (split[contract, N_MEMBERS : 2 * N_MEMBERS] == 0).all()
    assert (split[~contract, :N_MEMBERS] == 0).all()
    assert split[:, -1] == pytest.approx(Z[:, N_MEMBERS])


def test_ridge_by_type_predicts_close_to_the_truth(stacked_rows):
    Z, y = stacked_rows
    model = stacking.fit_ridge_by_type(Z, y, n_members=N_MEMBERS, alpha=1.0)
    pred = stacking.predict_ridge_by_type(model, Z, n_members=N_MEMBERS)
    assert np.sqrt(np.mean((pred - y) ** 2)) < np.sqrt(np.mean((Z[:, :N_MEMBERS].mean(axis=1) - y) ** 2))


def test_mlp_stacker_learns_a_correction_from_the_row_features(stacked_rows):
    Z, y = stacked_rows
    models = stacking.fit_mlp(Z, y, n_members=N_MEMBERS, hidden=(8,), alpha=1e-3, seeds=(0, 1), valid_share=0.2)
    pred = stacking.predict_mlp(models, Z, n_members=N_MEMBERS)
    mean_rmse = np.sqrt(np.mean((Z[:, :N_MEMBERS].mean(axis=1) - y) ** 2))
    assert len(models) == 2
    assert np.sqrt(np.mean((pred - y) ** 2)) < mean_rmse


def test_lgbm_stacker_learns_a_correction_from_the_row_features(stacked_rows):
    Z, y = stacked_rows
    booster = stacking.fit_lgbm(
        Z, y, n_members=N_MEMBERS, params=LGBM_PARAMS, max_rounds=200, early_stopping_rounds=20, valid_share=0.2, seed=0
    )
    pred = stacking.predict_lgbm(booster, Z, n_members=N_MEMBERS)
    mean_rmse = np.sqrt(np.mean((Z[:, :N_MEMBERS].mean(axis=1) - y) ** 2))
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.8 * mean_rmse


def test_lgbm_stacker_without_rows_cannot_use_the_row_feature(stacked_rows):
    Z, y = stacked_rows
    kwargs = {"n_members": N_MEMBERS, "params": LGBM_PARAMS, "max_rounds": 200, "early_stopping_rounds": 20}
    with_rows = stacking.fit_lgbm(Z, y, valid_share=0.2, seed=0, **kwargs)
    without = stacking.fit_lgbm(Z, y, valid_share=0.2, seed=0, with_rows=False, **kwargs)
    assert with_rows.num_feature() == N_MEMBERS + 1 + 2
    assert without.num_feature() == N_MEMBERS + 1
    pred = stacking.predict_lgbm(without, Z, n_members=N_MEMBERS, with_rows=False)
    assert pred.shape == (len(Z),)
