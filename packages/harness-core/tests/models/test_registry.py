"""Tests for BaseModel protocol and ModelRegistry."""
import numpy as np
import pytest
from pathlib import Path

from harnessml.core.models.base import BaseModel
from harnessml.core.models.registry import ModelRegistry


class MockModel(BaseModel):
    def fit(self, X, y):
        self._fitted = True

    def predict_proba(self, X):
        return np.full(len(X), 0.5)

    @property
    def is_regression(self):
        return False

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "mock.txt").write_text("mock")

    @classmethod
    def load(cls, path):
        return cls()


def test_register_and_create():
    registry = ModelRegistry()
    registry.register("mock", MockModel)
    model = registry.create("mock", params={"depth": 3})
    assert isinstance(model, MockModel)
    assert model.params == {"depth": 3}


def test_create_from_config():
    from harnessml.core.schemas.contracts import ModelConfig

    registry = ModelRegistry()
    registry.register("mock", MockModel)
    config = ModelConfig(
        name="test",
        type="mock",
        mode="classifier",
        features=["x"],
        params={"depth": 3},
    )
    model = registry.create_from_config(config)
    assert isinstance(model, MockModel)
    assert model.params == {"depth": 3}


def test_contains():
    registry = ModelRegistry()
    registry.register("mock", MockModel)
    assert "mock" in registry
    assert "nonexistent" not in registry


def test_create_unknown_raises():
    registry = ModelRegistry()
    with pytest.raises(KeyError):
        registry.create("nonexistent")


def test_base_model_predict_margin_raises():
    model = MockModel()
    with pytest.raises(NotImplementedError):
        model.predict_margin(np.array([1, 2]))


def test_with_defaults_loads_builtins():
    registry = ModelRegistry.with_defaults()
    assert "logistic_regression" in registry
    assert "elastic_net" in registry


def test_mock_model_save_load(tmp_path):
    model = MockModel(params={"a": 1})
    model.fit(np.array([[1]]), np.array([0]))
    assert model._fitted is True
    model.save(tmp_path / "mock_model")
    assert (tmp_path / "mock_model" / "mock.txt").exists()
    loaded = MockModel.load(tmp_path / "mock_model")
    assert isinstance(loaded, MockModel)


def test_mock_model_predict_proba():
    model = MockModel()
    probs = model.predict_proba(np.zeros((5, 3)))
    assert probs.shape == (5,)
    np.testing.assert_array_equal(probs, 0.5)


def test_default_params_empty():
    model = MockModel()
    assert model.params == {}
    assert model._fitted is False


# --- kwargs forwarding tests ---


class MockModelWithKwargs(BaseModel):
    """Model whose constructor accepts specific keyword arguments."""

    def __init__(self, params: dict | None = None, *, mode: str = "classifier"):
        super().__init__(params)
        self.mode = mode

    def fit(self, X, y):
        self._fitted = True

    def predict_proba(self, X):
        return np.full(len(X), 0.5)

    @property
    def is_regression(self):
        return False

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


class MockModelWithVarKeyword(BaseModel):
    """Model whose constructor accepts **kwargs."""

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(params)
        self.extra = kwargs

    def fit(self, X, y):
        self._fitted = True

    def predict_proba(self, X):
        return np.full(len(X), 0.5)

    @property
    def is_regression(self):
        return False

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()


def test_create_with_kwargs():
    """create() forwards matching kwargs to the model constructor."""
    registry = ModelRegistry()
    registry.register("mock_kw", MockModelWithKwargs)
    model = registry.create("mock_kw", params={"depth": 3}, mode="regressor")
    assert isinstance(model, MockModelWithKwargs)
    assert model.params == {"depth": 3}
    assert model.mode == "regressor"


def test_create_filters_unknown_kwargs():
    """create() silently drops kwargs the constructor doesn't accept."""
    registry = ModelRegistry()
    registry.register("mock_kw", MockModelWithKwargs)
    model = registry.create("mock_kw", params={"depth": 3}, mode="regressor", unknown_arg="foo")
    assert isinstance(model, MockModelWithKwargs)
    assert model.mode == "regressor"
    # unknown_arg should have been silently dropped
    assert not hasattr(model, "unknown_arg")


def test_create_var_keyword():
    """create() forwards all kwargs when constructor has **kwargs."""
    registry = ModelRegistry()
    registry.register("mock_var", MockModelWithVarKeyword)
    model = registry.create("mock_var", params={"depth": 3}, mode="regressor", custom="value")
    assert isinstance(model, MockModelWithVarKeyword)
    assert model.extra == {"mode": "regressor", "custom": "value"}
