"""Tests for Logistic Regression model wrapper: fit, predict, save/load, feature importances."""
import numpy as np
import pytest
from harnessml.core.models.wrappers.logistic import LogisticRegressionModel


def _make_model():
    return LogisticRegressionModel(params={"max_iter": 500})


def test_fit_predict_binary(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_save_load_roundtrip(synthetic_binary_data, tmp_path):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    model.save(tmp_path / "logistic")
    loaded = LogisticRegressionModel.load(tmp_path / "logistic")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_feature_importance(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    coefs = model._model.coef_
    assert coefs.shape[-1] == X.shape[1]
    assert np.all(np.isfinite(coefs))
