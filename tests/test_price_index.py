import numpy as np
import pytest
import scipy.sparse as sp

from kgl import price_index


def test_median_index_is_relative_to_the_first_period():
    t = np.array([2020.0, 2020.0, 2020.25, 2020.25, 2020.5])
    log_value = np.array([1.0, 3.0, 2.5, 2.5, 3.0])
    periods, index = price_index.median_index(t, log_value)
    assert periods.tolist() == [2020.0, 2020.25, 2020.5]
    assert index.tolist() == [0.0, 0.5, 1.0]


@pytest.fixture
def composition_shift():
    """Cheap segment's share grows from 20% to 80% while every price rises 0.1 per year."""
    rng = np.random.default_rng(0)
    periods = np.arange(2019.0, 2025.0, 0.25)
    t, cheap = [], []
    for i, p in enumerate(periods):
        share = 0.2 + 0.6 * i / (len(periods) - 1)
        t += [p] * 400
        cheap += list(rng.random(400) < share)
    t, cheap = np.array(t), np.array(cheap, dtype=float)
    log_value = 0.1 * (t - 2019.0) - 0.5 * cheap + rng.normal(scale=0.05, size=len(t))
    return t, cheap, log_value


def test_median_index_is_biased_by_a_composition_shift(composition_shift):
    t, _, log_value = composition_shift
    _, index = price_index.median_index(t, log_value)
    assert index[-1] < 0.3  # the true rise over 5.75 years is 0.575


def test_hedonic_index_recovers_the_time_effect_under_a_composition_shift(composition_shift):
    t, cheap, log_value = composition_shift
    periods, index = price_index.hedonic_index(t, log_value, sp.csr_matrix(cheap[:, None]))
    assert index[0] == 0.0
    assert index[-1] == pytest.approx(0.1 * (periods[-1] - periods[0]), abs=0.02)


def test_extrapolate_flat_keeps_the_last_level():
    periods, index = np.array([2024.0, 2024.25, 2024.5]), np.array([0.0, 0.1, 0.2])
    assert price_index.extrapolate(periods, index, np.array([2024.75, 2025.0]), how="flat").tolist() == [0.2, 0.2]


def test_extrapolate_linear_continues_the_trend_of_the_last_k_periods():
    periods, index = np.array([2024.0, 2024.25, 2024.5, 2024.75]), np.array([5.0, 0.0, 0.1, 0.2])
    out = price_index.extrapolate(periods, index, np.array([2025.0, 2025.25]), how="linear", k=3)
    assert out == pytest.approx([0.3, 0.4])


def test_extrapolate_rejects_unknown_methods():
    with pytest.raises(ValueError, match="how"):
        price_index.extrapolate(np.array([2024.0]), np.array([0.0]), np.array([2024.25]), how="cubic")


def test_lookup_maps_known_periods_and_rejects_unknown_ones():
    periods, index = np.array([2024.0, 2024.25]), np.array([0.0, 0.1])
    assert price_index.lookup(periods, index, np.array([2024.25, 2024.0])).tolist() == [0.1, 0.0]
    with pytest.raises(KeyError):
        price_index.lookup(periods, index, np.array([2024.5]))
