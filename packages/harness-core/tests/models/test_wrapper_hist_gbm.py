"""Tests for HistGradientBoosting model wrapper: fit, predict, save/load, feature importances."""
import numpy as np
import pytest
from harnessml.core.models.wrappers.hist_gbm import HistGradientBoostingModel


def _make_model():
    return HistGradientBoostingModel(params={"max_iter": 50, "max_depth": 3})


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
    model.save(tmp_path / "hist_gbm")
    loaded = HistGradientBoostingModel.load(tmp_path / "hist_gbm")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_feature_importance(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    # HistGradientBoosting does not expose feature_importances via our wrapper property
    assert model.feature_importances is None
