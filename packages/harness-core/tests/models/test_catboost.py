"""Tests for CatBoost model wrapper with eval_set support."""
import pytest
import numpy as np

try:
    import catboost

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

pytestmark = pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")


def _make_data(n=100, features=3, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, features)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_fit_with_eval_set():
    """eval_set is forwarded to CatBoost via Pool objects for early stopping."""
    from harnessml.core.models.wrappers.catboost import CatBoostModel

    X, y = _make_data(200)
    X_train, y_train = X[:150], y[:150]
    X_val, y_val = X[150:], y[150:]

    model = CatBoostModel(
        params={"iterations": 100, "early_stopping_rounds": 10, "verbose": 0}
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    assert model._fitted
    probs = model.predict_proba(X_val)
    assert probs.shape == (50,)
    assert all(0 <= p <= 1 for p in probs)
    # Early stopping was accepted (model trained successfully with eval_set)
    assert model._model.tree_count_ <= 100


def test_fit_without_eval_set_strips_early_stopping():
    """early_stopping_rounds is stripped when no eval_set to avoid CatBoost error."""
    from harnessml.core.models.wrappers.catboost import CatBoostModel

    X, y = _make_data()

    # This should NOT raise even though early_stopping_rounds is set
    model = CatBoostModel(
        params={"iterations": 10, "early_stopping_rounds": 5, "verbose": 0}
    )
    model.fit(X, y)

    assert model._fitted
    probs = model.predict_proba(X)
    assert probs.shape == (100,)


def test_fit_basic_no_early_stopping():
    """Basic fit without early_stopping_rounds or eval_set works as before."""
    from harnessml.core.models.wrappers.catboost import CatBoostModel

    X, y = _make_data()
    model = CatBoostModel(params={"iterations": 10, "verbose": 0})
    model.fit(X, y)

    assert model._fitted
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
