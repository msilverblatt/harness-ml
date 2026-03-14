"""Tests for TabPFN model wrapper: fit, predict, save/load, multiclass, warnings."""
import numpy as np
import pytest

try:
    from tabpfn import TabPFNClassifier
    # Verify model can actually be loaded (requires HF auth for gated models)
    _clf = TabPFNClassifier()
    _clf.fit(np.array([[0, 1], [1, 0], [0, 0], [1, 1]]), np.array([0, 1, 0, 1]))
    HAS_TABPFN = True
except Exception:
    HAS_TABPFN = False

pytestmark = pytest.mark.skipif(not HAS_TABPFN, reason="tabpfn not available (not installed or model not accessible)")


def _make_model():
    from harnessml.core.models.wrappers.tabpfn import TabPFNModel
    return TabPFNModel(params={})


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
    model.save(tmp_path / "tabpfn")

    from harnessml.core.models.wrappers.tabpfn import TabPFNModel
    loaded = TabPFNModel.load(tmp_path / "tabpfn")
    # TabPFN is pre-trained so load doesn't restore fit state;
    # re-fit and verify predictions are consistent
    loaded.fit(X, y)
    np.testing.assert_array_almost_equal(
        model.predict_proba(X), loaded.predict_proba(X)
    )


def test_multiclass():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0.5).astype(int) + (X[:, 1] > 0.5).astype(int)  # 3 classes: 0, 1, 2
    model = _make_model()
    model.fit(X, y)
    probs = model.predict_proba(X)
    n_classes = len(np.unique(y))
    assert probs.shape == (len(X), n_classes)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_large_data_warning(caplog):
    """Verify warning logged for >10k rows."""
    import logging
    np.random.seed(42)
    X = np.random.randn(10_001, 5)
    y = (X[:, 0] > 0).astype(float)
    model = _make_model()
    with caplog.at_level(logging.WARNING, logger="harnessml.core.models.wrappers.tabpfn"):
        model.fit(X, y)
    assert any("10000" in msg or "10001" in msg for msg in caplog.messages)


def test_is_regression():
    model = _make_model()
    assert model.is_regression is False
