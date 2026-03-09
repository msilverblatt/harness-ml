"""Tests for MLP model wrapper: fit, predict, save/load, multi-seed, normalize."""
import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def _make_model(**kwargs):
    from harnessml.core.models.wrappers.mlp import MLPModel
    defaults = dict(
        params={"hidden_dims": [32, 16], "epochs": 5, "lr": 0.01},
        mode="classifier",
    )
    defaults.update(kwargs)
    return MLPModel(**defaults)


def test_fit_predict_binary(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))


def test_save_load_roundtrip(synthetic_binary_data, tmp_path):
    from harnessml.core.models.wrappers.mlp import MLPModel
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    model.save(tmp_path / "mlp")
    loaded = MLPModel.load(tmp_path / "mlp")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_feature_importance(synthetic_binary_data):
    """MLP doesn't have feature_importances_ -- verify first layer weights are accessible."""
    X, y = synthetic_binary_data
    model = _make_model()
    model.fit(X, y)
    # Access first Linear layer weights as a proxy for feature importance
    first_layer = model._models[0][0]
    weights = first_layer.weight.detach().numpy()
    assert weights.shape[1] == X.shape[1]
    assert np.all(np.isfinite(weights))


def test_multi_seed_averaging(synthetic_binary_data):
    X, y = synthetic_binary_data
    model_1seed = _make_model(n_seeds=1, seed=0)
    model_1seed.fit(X, y)

    model_3seed = _make_model(n_seeds=3, seed=0)
    model_3seed.fit(X, y)

    probs_1 = model_1seed.predict_proba(X)
    probs_3 = model_3seed.predict_proba(X)

    # Multi-seed model should have 3 sub-models
    assert len(model_3seed._models) == 3
    # Predictions should differ (averaged across seeds)
    assert not np.allclose(probs_1, probs_3, atol=1e-6)
    # Both should still produce valid probabilities
    assert np.all((probs_3 >= 0) & (probs_3 <= 1))


def test_normalize_option(synthetic_binary_data):
    X, y = synthetic_binary_data
    model = _make_model(normalize=True)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert np.all((probs >= 0) & (probs <= 1))
    # Verify normalization stats were computed
    assert model._feature_means is not None
    assert model._feature_stds is not None
    assert len(model._feature_means) == X.shape[1]
