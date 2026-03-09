"""Tests for XGBoost model wrapper: fit, predict, save/load, feature importances."""
import numpy as np
import pytest

try:
    import xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

pytestmark = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")


def _make_model():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel
    return XGBoostModel(
        params={"max_depth": 3, "n_estimators": 10, "verbosity": 0},
        mode="classifier",
    )


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
    model.save(tmp_path / "xgb")
    from harnessml.core.models.wrappers.xgboost import XGBoostModel
    loaded = XGBoostModel.load(tmp_path / "xgb")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_feature_importance(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    importances = model._model.feature_importances_
    assert len(importances) == X.shape[1]
    assert np.all(np.isfinite(importances))
