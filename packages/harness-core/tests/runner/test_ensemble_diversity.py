import numpy as np
import pytest
from harnessml.core.runner.analysis.ensemble_diversity import compute_diversity


def test_identical_models_zero_disagreement():
    preds = np.array([[0.8, 0.8], [0.3, 0.3], [0.9, 0.9]])
    assert compute_diversity(preds, method="disagreement") == 0.0


def test_opposite_models_high_disagreement():
    preds = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    diversity = compute_diversity(preds, method="disagreement")
    assert diversity > 0.5


def test_q_statistic_range():
    labels = np.array([1, 0, 1, 0, 1])
    preds = np.array([[0.9, 0.8], [0.2, 0.3], [0.8, 0.7], [0.1, 0.2], [0.7, 0.6]])
    q = compute_diversity(preds, method="q_statistic", labels=labels)
    assert -1 <= q <= 1


def test_kappa_identical():
    labels = np.array([1, 0, 1, 0, 1])
    preds = np.array([[0.9, 0.9], [0.2, 0.2], [0.8, 0.8], [0.1, 0.1], [0.7, 0.7]])
    k = compute_diversity(preds, method="kappa", labels=labels)
    assert k == pytest.approx(1.0)


def test_correlation_identical():
    preds = np.array([[0.8, 0.8], [0.3, 0.3], [0.9, 0.9], [0.1, 0.1]])
    corr = compute_diversity(preds, method="correlation")
    assert corr == pytest.approx(1.0)


def test_correlation_diverse():
    np.random.seed(42)
    preds = np.random.rand(100, 5)
    corr = compute_diversity(preds, method="correlation")
    assert abs(corr) < 0.3  # Random predictions should have low correlation


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown"):
        compute_diversity(np.array([[0.5]]), method="bogus")


def test_q_statistic_requires_labels():
    with pytest.raises(ValueError, match="requires labels"):
        compute_diversity(np.array([[0.5, 0.5]]), method="q_statistic")


def test_single_model():
    preds = np.array([[0.8], [0.3], [0.9]])
    assert compute_diversity(preds, method="disagreement") == 0.0
    assert compute_diversity(preds, method="correlation") == 0.0
