"""Tests for LogisticRegression auto-scaling max_iter for large feature sets."""
import numpy as np
import pytest
from harnessml.core.models.wrappers.logistic import LogisticRegressionModel


def test_auto_scales_max_iter_for_wide_data():
    """LogisticRegression should auto-increase max_iter for >50 features."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 100
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    # No max_iter in params — should auto-scale
    model = LogisticRegressionModel(params={})
    model.fit(X, y)

    # max_iter should have been increased
    assert model._model.max_iter >= 1000


def test_no_auto_scale_when_max_iter_explicit():
    """When max_iter is explicitly set, auto-scaling should not override."""
    rng = np.random.default_rng(42)
    n_samples = 200
    n_features = 100
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    model = LogisticRegressionModel(params={"max_iter": 200})
    model.fit(X, y)

    # Should respect the user's explicit setting
    assert model._model.max_iter == 200


def test_no_auto_scale_for_small_feature_sets():
    """For feature sets <= 50, max_iter should not be modified."""
    rng = np.random.default_rng(42)
    n_samples = 100
    n_features = 5
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    model = LogisticRegressionModel(params={})
    model.fit(X, y)

    # Default sklearn max_iter is 100
    assert model._model.max_iter == 100


def test_auto_scale_proportional_to_features():
    """max_iter should scale with the number of features."""
    rng = np.random.default_rng(42)
    n_samples = 300
    n_features = 200
    X = rng.standard_normal((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    model = LogisticRegressionModel(params={})
    model.fit(X, y)

    # 200 features * 10 = 2000
    assert model._model.max_iter == 2000
