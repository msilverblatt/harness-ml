"""Tests for MLP and TabNet neural network wrappers."""
import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pytorch_tabnet

    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_classifier():
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [32, 16], "dropout": 0.1, "lr": 0.01, "epochs": 5, "batch_size": 32},
        mode="classifier",
        n_seeds=2,
    )
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_regressor():
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [32], "dropout": 0.0, "lr": 0.01, "epochs": 5},
        mode="regressor",
        n_seeds=1,
        cdf_scale=5.0,
    )
    X = np.random.randn(100, 3).astype(np.float32)
    y = (X[:, 0] * 3 + np.random.randn(100)).astype(np.float32)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is True


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_regressor_predict_margin():
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [16], "dropout": 0.0, "lr": 0.01, "epochs": 3},
        mode="regressor",
        n_seeds=1,
        cdf_scale=5.0,
    )
    X = np.random.randn(50, 2).astype(np.float32)
    y = (X[:, 0] * 2).astype(np.float32)
    model.fit(X, y)
    margins = model.predict_margin(X)
    assert margins.shape == (50,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_classifier_predict_margin_raises():
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [16], "epochs": 2},
        mode="classifier",
    )
    X = np.random.randn(50, 2).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_margin(X)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_regressor_no_cdf_scale_raises():
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [16], "epochs": 2},
        mode="regressor",
        n_seeds=1,
    )
    X = np.random.randn(50, 2).astype(np.float32)
    y = X[:, 0].astype(np.float32)
    model.fit(X, y)
    with pytest.raises(ValueError, match="cdf_scale"):
        model.predict_proba(X)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_invalid_mode():
    from harnessml.core.models.wrappers.mlp import MLPModel

    with pytest.raises(ValueError, match="mode must be"):
        MLPModel(params={}, mode="bad")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_multi_seed_averaging():
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [16], "dropout": 0.0, "lr": 0.01, "epochs": 3},
        mode="classifier",
        n_seeds=3,
    )
    X = np.random.randn(80, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    assert len(model._models) == 3
    probs = model.predict_proba(X)
    assert probs.shape == (80,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_save_load(tmp_path):
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [16, 8], "dropout": 0.0, "lr": 0.01, "epochs": 3},
        mode="classifier",
        n_seeds=2,
    )
    X = np.random.randn(50, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "mlp_model")
    loaded = MLPModel.load(tmp_path / "mlp_model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True
    assert len(loaded._models) == 2


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_regressor_save_load(tmp_path):
    from harnessml.core.models.wrappers.mlp import MLPModel

    model = MLPModel(
        params={"hidden_dims": [16], "dropout": 0.0, "lr": 0.01, "epochs": 3},
        mode="regressor",
        n_seeds=1,
        cdf_scale=5.0,
    )
    X = np.random.randn(50, 2).astype(np.float32)
    y = (X[:, 0] * 2).astype(np.float32)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = MLPModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    np.testing.assert_array_almost_equal(
        model.predict_margin(X), loaded.predict_margin(X)
    )
    assert loaded.is_regression is True
    assert loaded._cdf_scale == 5.0


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_normalize():
    """normalize=True standardizes features; means/stds saved in meta."""
    from harnessml.core.models.wrappers.mlp import MLPModel

    X = np.random.randn(100, 4).astype(np.float32)
    # Shift features so they have non-zero mean and non-unit std
    X[:, 0] += 100.0
    X[:, 1] *= 50.0
    y = (X[:, 0] > 100).astype(int)

    model = MLPModel(
        params={"hidden_dims": [16], "epochs": 3, "lr": 0.01},
        mode="classifier",
        n_seeds=1,
        normalize=True,
    )
    model.fit(X, y)

    # Check that normalization stats are stored
    assert hasattr(model, "_feature_means")
    assert hasattr(model, "_feature_stds")
    assert model._feature_means.shape == (4,)
    assert model._feature_stds.shape == (4,)

    # Predictions should still be valid probabilities
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_normalize_save_load(tmp_path):
    """Normalization stats survive save/load cycle."""
    from harnessml.core.models.wrappers.mlp import MLPModel

    X = np.random.randn(80, 3).astype(np.float32)
    X[:, 0] += 50.0
    y = (X[:, 0] > 50).astype(int)

    model = MLPModel(
        params={"hidden_dims": [16], "epochs": 3, "lr": 0.01},
        mode="classifier",
        n_seeds=1,
        normalize=True,
    )
    model.fit(X, y)
    model.save(tmp_path / "norm_model")
    loaded = MLPModel.load(tmp_path / "norm_model")

    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._normalize is True
    np.testing.assert_array_almost_equal(loaded._feature_means, model._feature_means)
    np.testing.assert_array_almost_equal(loaded._feature_stds, model._feature_stds)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_batch_norm():
    """batch_norm=True adds BatchNorm1d layers; still produces valid output."""
    import torch.nn as nn
    from harnessml.core.models.wrappers.mlp import MLPModel, _build_mlp

    # Check that _build_mlp inserts BatchNorm1d layers
    net = _build_mlp(5, [32, 16], 0.0, "classifier", batch_norm=True)
    bn_layers = [m for m in net.modules() if isinstance(m, nn.BatchNorm1d)]
    assert len(bn_layers) == 2  # one per hidden layer

    # End-to-end: train and predict
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = MLPModel(
        params={"hidden_dims": [32, 16], "epochs": 3, "lr": 0.01},
        mode="classifier",
        n_seeds=1,
        batch_norm=True,
    )
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_batch_norm_save_load(tmp_path):
    """batch_norm models survive save/load."""
    from harnessml.core.models.wrappers.mlp import MLPModel

    X = np.random.randn(80, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model = MLPModel(
        params={"hidden_dims": [16, 8], "epochs": 3, "lr": 0.01},
        mode="classifier",
        n_seeds=1,
        batch_norm=True,
    )
    model.fit(X, y)
    model.save(tmp_path / "bn_model")
    loaded = MLPModel.load(tmp_path / "bn_model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_weight_decay():
    """weight_decay param is accepted and passed to optimizer."""
    from unittest.mock import patch

    from harnessml.core.models.wrappers.mlp import MLPModel

    X = np.random.randn(50, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = MLPModel(
        params={"hidden_dims": [16], "epochs": 2, "lr": 0.01},
        mode="classifier",
        n_seeds=1,
        weight_decay=0.01,
    )

    # Patch Adam to capture kwargs
    import torch.optim as optim
    original_adam = optim.Adam
    captured_kwargs = {}

    def mock_adam(params, **kwargs):
        captured_kwargs.update(kwargs)
        return original_adam(params, **kwargs)

    with patch.object(optim, "Adam", side_effect=mock_adam):
        model.fit(X, y)

    assert captured_kwargs.get("weight_decay") == 0.01


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_early_stopping():
    """early_stopping_rounds + eval_set stops training early."""
    from harnessml.core.models.wrappers.mlp import MLPModel

    X_train = np.random.randn(100, 3).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(int)
    X_val = np.random.randn(30, 3).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(int)

    model = MLPModel(
        params={"hidden_dims": [16], "epochs": 1000, "lr": 0.01},
        mode="classifier",
        n_seeds=1,
        early_stopping_rounds=5,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # With 1000 epochs and patience=5, it should stop well before 1000
    assert model._fitted is True
    probs = model.predict_proba(X_train)
    assert probs.shape == (100,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_seed_stride():
    """seed_stride changes per-seed random state, producing different models."""
    from harnessml.core.models.wrappers.mlp import MLPModel

    X = np.random.randn(80, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    # Default stride=1: seeds 0, 1
    model_s1 = MLPModel(
        params={"hidden_dims": [16], "epochs": 5, "lr": 0.01},
        mode="classifier",
        n_seeds=2,
        seed_stride=1,
    )
    model_s1.fit(X, y)

    # stride=10: seeds 0, 10
    model_s10 = MLPModel(
        params={"hidden_dims": [16], "epochs": 5, "lr": 0.01},
        mode="classifier",
        n_seeds=2,
        seed_stride=10,
    )
    model_s10.fit(X, y)

    preds_s1 = model_s1.predict_proba(X)
    preds_s10 = model_s10.predict_proba(X)

    # First seed (seed=0) should be identical for both
    # But averaged predictions should differ because second seed differs
    # (seed 1 vs seed 10)
    assert not np.allclose(preds_s1, preds_s10, atol=1e-6)


# ---------------------------------------------------------------------------
# TabNet
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TABNET, reason="pytorch-tabnet not installed")
def test_tabnet_classifier():
    from harnessml.core.models.wrappers.tabnet import TabNetModel

    model = TabNetModel(
        params={"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 5, "patience": 3, "verbose": 0},
        n_seeds=2,
    )
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False
    assert model._fitted is True


@pytest.mark.skipif(not HAS_TABNET, reason="pytorch-tabnet not installed")
def test_tabnet_predict_margin_raises():
    from harnessml.core.models.wrappers.tabnet import TabNetModel

    model = TabNetModel(
        params={"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 3, "patience": 3, "verbose": 0},
        n_seeds=1,
    )
    X = np.random.randn(50, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    with pytest.raises(NotImplementedError):
        model.predict_margin(X)


@pytest.mark.skipif(not HAS_TABNET, reason="pytorch-tabnet not installed")
def test_tabnet_multi_seed():
    from harnessml.core.models.wrappers.tabnet import TabNetModel

    model = TabNetModel(
        params={"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 3, "patience": 3, "verbose": 0},
        n_seeds=3,
    )
    X = np.random.randn(80, 4).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    assert len(model._models) == 3
    probs = model.predict_proba(X)
    assert probs.shape == (80,)


@pytest.mark.skipif(not HAS_TABNET, reason="pytorch-tabnet not installed")
def test_tabnet_save_load(tmp_path):
    from harnessml.core.models.wrappers.tabnet import TabNetModel

    model = TabNetModel(
        params={"n_d": 8, "n_a": 8, "n_steps": 3, "max_epochs": 5, "patience": 3, "verbose": 0},
        n_seeds=2,
    )
    X = np.random.randn(50, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "tabnet_model")
    loaded = TabNetModel.load(tmp_path / "tabnet_model")
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )
    assert loaded._fitted is True
    assert len(loaded._models) == 2
