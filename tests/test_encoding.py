import numpy as np
import polars as pl
import pytest

from kgl import encoding


def test_smoothed_mean_pulls_small_groups_toward_the_prior():
    fit_keys = pl.Series(["a", "a", "b"])
    out = encoding.smoothed_mean(fit_keys, np.array([1.0, 3.0, 10.0]), pl.Series(["a", "b", "c"]), prior=0.0, m=2.0)
    # a: (1 + 3 + 2 * 0) / (2 + 2); b: 10 / (1 + 2); c is unseen, so it is the prior
    assert out == pytest.approx([1.0, 10 / 3, 0.0])


def test_smoothed_mean_treats_missing_keys_as_their_own_group():
    out = encoding.smoothed_mean(
        pl.Series([None, None, "a"]), np.array([2.0, 4.0, 9.0]), pl.Series([None, "a"]), prior=0.0, m=0.0
    )
    assert out.tolist() == [3.0, 9.0]


def test_oof_smoothed_mean_never_uses_a_rows_own_fold():
    keys = pl.Series(["a", "a", "a", "a"])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    out = encoding.oof_smoothed_mean(keys, values, np.array([0, 0, 1, 1]), m=0.0)
    assert out.tolist() == [3.5, 3.5, 1.5, 1.5]


def test_oof_smoothed_mean_uses_the_other_folds_mean_as_prior():
    keys = pl.Series(["a", "b", "a", "c"])
    values = np.array([1.0, 5.0, 3.0, 7.0])
    out = encoding.oof_smoothed_mean(keys, values, np.array([0, 0, 1, 1]), m=1.0)
    # fold 0 rows see fold 1 (a: 3, c: 7; prior 5): a -> (3 + 5) / 2, b unseen -> 5
    # fold 1 rows see fold 0 (a: 1, b: 5; prior 3): a -> (1 + 3) / 2, c unseen -> 3
    assert out.tolist() == [4.0, 5.0, 2.0, 3.0]
