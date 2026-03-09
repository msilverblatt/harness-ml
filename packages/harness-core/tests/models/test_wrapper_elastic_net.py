"""Tests for ElasticNet model wrapper: fit, predict, save/load, feature importances."""
import numpy as np
import pytest
from harnessml.core.models.wrappers.elastic_net import ElasticNetModel


def _make_model(**kwargs):
    return ElasticNetModel(
        params={"max_iter": 500},
        mode="classifier",
        **kwargs,
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
    model.save(tmp_path / "enet")
    loaded = ElasticNetModel.load(tmp_path / "enet")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_feature_importance(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    # ElasticNet in classifier mode uses LogisticRegression which has coef_
    coefs = model._model.coef_
    assert coefs.shape[-1] == X.shape[1]
    assert np.all(np.isfinite(coefs))
