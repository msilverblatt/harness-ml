"""Tests for XGBoost model wrapper (classifier + regressor modes)."""
import numpy as np
import pytest

try:
    import xgboost  # noqa: F401

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

pytestmark = pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")


def test_xgb_classifier():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"max_depth": 3, "n_estimators": 10, "verbosity": 0},
        mode="classifier",
    )
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False


def test_xgb_regressor():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"max_depth": 1, "n_estimators": 10, "verbosity": 0},
        mode="regressor",
        cdf_scale=5.0,
    )
    X = np.random.randn(100, 3)
    y = X[:, 0] * 5 + np.random.randn(100)  # margin target
    model.fit(X, y)
    margins = model.predict_margin(X)
    assert margins.shape == (100,)
    probs = model.predict_proba(X)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is True


def test_xgb_regressor_cdf_scale():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"n_estimators": 10, "verbosity": 0},
        mode="regressor",
        cdf_scale=5.0,
    )
    X = np.random.randn(50, 2)
    y = X[:, 0] * 3
    model.fit(X, y)
    probs = model.predict_proba(X)
    # With larger CDF scale, probabilities should be less extreme
    assert 0.1 < np.mean(probs) < 0.9


def test_xgb_regressor_no_cdf_scale_raises():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"n_estimators": 10, "verbosity": 0},
        mode="regressor",
    )
    X = np.random.randn(50, 2)
    y = X[:, 0] * 3
    model.fit(X, y)
    with pytest.raises(ValueError, match="cdf_scale"):
        model.predict_proba(X)


def test_xgb_classifier_predict_margin_raises():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"n_estimators": 10, "verbosity": 0},
        mode="classifier",
    )
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_margin(X)


def test_xgb_invalid_mode():
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    with pytest.raises(ValueError, match="mode must be"):
        XGBoostModel(params={}, mode="invalid")


def test_xgb_save_load(tmp_path):
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"max_depth": 2, "n_estimators": 10, "verbosity": 0},
        mode="classifier",
    )
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = XGBoostModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_xgb_regressor_save_load(tmp_path):
    from harnessml.core.models.wrappers.xgboost import XGBoostModel

    model = XGBoostModel(
        params={"max_depth": 1, "n_estimators": 10, "verbosity": 0},
        mode="regressor",
        cdf_scale=5.0,
    )
    X = np.random.randn(50, 2)
    y = X[:, 0] * 3
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = XGBoostModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    np.testing.assert_array_almost_equal(
        model.predict_margin(X), loaded.predict_margin(X)
    )
    assert loaded.is_regression is True
    assert loaded._cdf_scale == 5.0
