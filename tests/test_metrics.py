import numpy as np
import pytest

from kgl import metrics


def test_accuracy_hand_computed():
    y = np.array([1, 0, 1, 1])
    pred = np.array([1, 0, 0, 1])
    assert metrics.get_metric("accuracy")(y, pred) == 0.75


def test_rmse_hand_computed():
    y = np.array([0.0, 0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0, 1.0])
    assert metrics.get_metric("rmse")(y, pred) == 1.0


def test_auc_hand_computed():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    assert metrics.get_metric("auc")(y, score) == 0.75


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        metrics.get_metric("no_such_metric")


def test_greater_is_better():
    assert metrics.greater_is_better("auc") is True
    assert metrics.greater_is_better("rmse") is False


def test_score_folds_returns_mean_and_per_fold():
    y = np.array([1, 0, 1, 0])
    oof = np.array([1, 0, 0, 0])
    folds = np.array([0, 0, 1, 1])
    mean, per_fold = metrics.score_folds(y, oof, folds, "accuracy")
    assert per_fold == [1.0, 0.5]
    assert mean == 0.75


def test_rmsle_hand_computed():
    # log1p(y) = [0, 1], log1p(pred) = [1, 1] -> squared errors [1, 0] -> sqrt(0.5)
    y = np.array([0.0, np.e - 1])
    pred = np.array([np.e - 1, np.e - 1])
    assert metrics.get_metric("rmsle")(y, pred) == pytest.approx(np.sqrt(0.5))


def test_rmsle_rejects_negative_prediction():
    # Kaggle rejects negative predictions for RMSLE; the local metric must not clip them silently.
    with pytest.raises(ValueError):
        metrics.get_metric("rmsle")(np.array([1.0]), np.array([-1.0]))
