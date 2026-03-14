"""Tests for SVM model wrapper: fit, predict, save/load."""
import numpy as np
from harnessml.core.models.wrappers.svm import SVMModel


def _make_model():
    return SVMModel(params={"C": 1.0, "kernel": "rbf"})


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
    model.save(tmp_path / "svm")
    loaded = SVMModel.load(tmp_path / "svm")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_feature_importance(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    # SVM does not natively provide feature importances
    assert model.feature_importances is None
