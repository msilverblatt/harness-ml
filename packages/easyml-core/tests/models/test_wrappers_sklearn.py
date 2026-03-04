"""Tests for sklearn model wrappers (LogisticRegression, ElasticNet)."""
import numpy as np
import pytest


def test_logistic_regression_fit_predict():
    from easyml.core.models.wrappers.logistic import LogisticRegressionModel

    model = LogisticRegressionModel(params={"C": 1.0, "max_iter": 200})
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


def test_logistic_regression_save_load(tmp_path):
    from easyml.core.models.wrappers.logistic import LogisticRegressionModel

    model = LogisticRegressionModel(params={"C": 1.0, "max_iter": 200})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = LogisticRegressionModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True


def test_logistic_regression_predict_margin_raises():
    from easyml.core.models.wrappers.logistic import LogisticRegressionModel

    model = LogisticRegressionModel()
    with pytest.raises(NotImplementedError):
        model.predict_margin(np.array([[1, 2]]))


def test_elastic_net_fit_predict():
    from easyml.core.models.wrappers.elastic_net import ElasticNetModel

    model = ElasticNetModel(params={"C": 0.01, "l1_ratio": 1.0, "max_iter": 500})
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


def test_elastic_net_save_load(tmp_path):
    from easyml.core.models.wrappers.elastic_net import ElasticNetModel

    model = ElasticNetModel(params={"C": 0.1, "l1_ratio": 0.5, "max_iter": 500})
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = ElasticNetModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True


def test_elastic_net_defaults_to_saga_solver_and_l1_ratio():
    from easyml.core.models.wrappers.elastic_net import ElasticNetModel

    # Ensure that even with minimal params, solver and l1_ratio are set correctly
    model = ElasticNetModel(params={"max_iter": 200})
    assert model._model.solver == "saga"
    assert model._model.l1_ratio == 0.5

    # Explicit l1_ratio overrides the default
    model2 = ElasticNetModel(params={"l1_ratio": 0.8, "max_iter": 200})
    assert model2._model.l1_ratio == 0.8


def test_elastic_net_predict_margin_raises():
    from easyml.core.models.wrappers.elastic_net import ElasticNetModel

    model = ElasticNetModel(params={"l1_ratio": 0.5})
    with pytest.raises(NotImplementedError):
        model.predict_margin(np.array([[1, 2]]))
