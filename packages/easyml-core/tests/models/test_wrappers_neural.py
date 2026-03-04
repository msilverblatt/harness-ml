"""Tests for MLP and TabNet neural network wrappers."""
import pytest
import numpy as np

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
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

    with pytest.raises(ValueError, match="mode must be"):
        MLPModel(params={}, mode="bad")


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_mlp_multi_seed_averaging():
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

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
    from easyml.core.models.wrappers.mlp import MLPModel

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


# ---------------------------------------------------------------------------
# TabNet
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_TABNET, reason="pytorch-tabnet not installed")
def test_tabnet_classifier():
    from easyml.core.models.wrappers.tabnet import TabNetModel

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
    from easyml.core.models.wrappers.tabnet import TabNetModel

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
    from easyml.core.models.wrappers.tabnet import TabNetModel

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
    from easyml.core.models.wrappers.tabnet import TabNetModel

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
