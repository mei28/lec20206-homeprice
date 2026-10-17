import numpy as np
import pytest

from kgl import ensemble


@pytest.fixture
def members():
    """Two members: the truth plus independent noise, and a member that is pure noise around the mean."""
    rng = np.random.default_rng(0)
    y = rng.normal(17.0, 0.5, size=400)
    good = y + rng.normal(scale=0.1, size=400)
    noisy = 17.0 + rng.normal(scale=0.5, size=400)
    return np.column_stack([good, noisy]), y


def test_nonnegative_weights_favor_the_informative_member(members):
    H, y = members
    weights = ensemble.fit_weights(H, y)
    assert (weights >= 0).all()
    assert weights[0] > 0.9 and weights[1] < 0.1


def test_stacking_predicts_close_to_the_truth(members):
    H, y = members
    model = ensemble.fit_stacking(H, y, alpha=1.0)
    assert np.sqrt(np.mean((model.predict(H) - y) ** 2)) < 0.11


def test_cross_predict_scores_each_half_with_a_combiner_fit_on_the_other(members):
    H, y = members
    first_half = np.arange(400) < 200
    seen = []

    def fit(Hf, yf):
        seen.append(len(yf))
        return float(yf.mean())

    out = ensemble.cross_predict(H, y, first_half, fit, lambda mean, Hp: np.full(len(Hp), mean))
    assert seen == [200, 200]
    assert out[:200] == pytest.approx(y[200:].mean())
    assert out[200:] == pytest.approx(y[:200].mean())


def test_cross_predict_with_an_earlier_block_adds_it_to_every_fit(members):
    H, y = members
    first_half = np.arange(400) < 200
    H_early, y_early = H[:100] + 0.0, y[:100] + 10.0  # a block whose mean differs, to see it in the fits
    seen = []

    def fit(Hf, yf):
        seen.append(len(yf))
        return float(yf.mean())

    out = ensemble.cross_predict_with_block(
        H, y, first_half, H_early, y_early, fit, lambda mean, Hp: np.full(len(Hp), mean)
    )
    assert seen == [300, 300]
    assert out[first_half] == pytest.approx(np.full(200, np.concatenate([y_early, y[~first_half]]).mean()))
    assert out[~first_half] == pytest.approx(np.full(200, np.concatenate([y_early, y[first_half]]).mean()))
