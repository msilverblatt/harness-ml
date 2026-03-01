"""Tests for CatBoost, LightGBM, and RandomForest model wrappers."""
import pytest
import numpy as np

try:
    import catboost

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    import lightgbm

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
def test_catboost_fit_predict():
    from easyml.models.wrappers.catboost import CatBoostModel

    model = CatBoostModel(params={"iterations": 10, "depth": 3, "verbose": 0})
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
def test_catboost_save_load(tmp_path):
    from easyml.models.wrappers.catboost import CatBoostModel

    model = CatBoostModel(params={"iterations": 10, "depth": 3, "verbose": 0})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = CatBoostModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
def test_catboost_predict_margin_raises():
    from easyml.models.wrappers.catboost import CatBoostModel

    model = CatBoostModel(params={"iterations": 10, "verbose": 0})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_margin(X)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
def test_lightgbm_fit_predict():
    from easyml.models.wrappers.lightgbm import LightGBMModel

    model = LightGBMModel(params={"n_estimators": 10, "max_depth": 3, "verbose": -1})
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
def test_lightgbm_save_load(tmp_path):
    from easyml.models.wrappers.lightgbm import LightGBMModel

    model = LightGBMModel(params={"n_estimators": 10, "max_depth": 3, "verbose": -1})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = LightGBMModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
def test_lightgbm_predict_margin_raises():
    from easyml.models.wrappers.lightgbm import LightGBMModel

    model = LightGBMModel(params={"n_estimators": 10, "verbose": -1})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_margin(X)


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------


def test_random_forest_fit_predict():
    from easyml.models.wrappers.random_forest import RandomForestModel

    model = RandomForestModel(params={"n_estimators": 10, "max_depth": 3})
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


def test_random_forest_save_load(tmp_path):
    from easyml.models.wrappers.random_forest import RandomForestModel

    model = RandomForestModel(params={"n_estimators": 10, "max_depth": 3})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = RandomForestModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True


def test_random_forest_predict_margin_raises():
    from easyml.models.wrappers.random_forest import RandomForestModel

    model = RandomForestModel(params={"n_estimators": 10})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_margin(X)
