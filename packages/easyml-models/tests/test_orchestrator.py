"""Tests for TrainOrchestrator."""
import numpy as np
import pytest
from pathlib import Path

from easyml.models.base import BaseModel
from easyml.models.orchestrator import TrainOrchestrator
from easyml.models.registry import ModelRegistry


class MockModel(BaseModel):
    """Minimal mock model for orchestrator tests."""

    def fit(self, X, y):
        self._fitted = True
        self._n_features = X.shape[1] if X.ndim > 1 else 1

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
        m = cls()
        m._fitted = True
        return m


class FailingModel(BaseModel):
    """Mock model that always raises during fit."""

    def fit(self, X, y):
        raise RuntimeError("Training failed")

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


def test_orchestrator_trains_all(tmp_path):
    registry = ModelRegistry()
    registry.register("mock", MockModel)

    config = {
        "model_a": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
        "model_b": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
    }

    X = np.random.randn(50, 1)
    y = (X[:, 0] > 0).astype(int)

    orch = TrainOrchestrator(model_registry=registry, model_configs=config, output_dir=tmp_path)
    results = orch.train_all(X, y, feature_columns=["x"])
    assert "model_a" in results
    assert "model_b" in results


def test_orchestrator_skips_inactive(tmp_path):
    registry = ModelRegistry()
    registry.register("mock", MockModel)

    config = {
        "active_model": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
        "inactive_model": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": False},
    }

    orch = TrainOrchestrator(model_registry=registry, model_configs=config, output_dir=tmp_path)
    results = orch.train_all(np.random.randn(50, 1), (np.random.randn(50) > 0).astype(int), feature_columns=["x"])
    assert "active_model" in results
    assert "inactive_model" not in results


def test_orchestrator_saves_artifacts(tmp_path):
    registry = ModelRegistry()
    registry.register("mock", MockModel)

    config = {
        "my_model": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
    }

    orch = TrainOrchestrator(model_registry=registry, model_configs=config, output_dir=tmp_path)
    orch.train_all(np.random.randn(30, 1), (np.random.randn(30) > 0).astype(int), feature_columns=["x"])

    assert (tmp_path / "my_model" / "mock.txt").exists()
    assert (tmp_path / "my_model" / "fingerprint.json").exists()


def test_orchestrator_fingerprint_cache(tmp_path):
    registry = ModelRegistry()
    registry.register("mock", MockModel)

    config = {
        "cached": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
    }

    X = np.random.randn(30, 1)
    y = (X[:, 0] > 0).astype(int)

    orch = TrainOrchestrator(model_registry=registry, model_configs=config, output_dir=tmp_path)

    # First run: trains
    results1 = orch.train_all(X, y, feature_columns=["x"])
    assert "cached" in results1

    # Second run: should load from cache
    results2 = orch.train_all(X, y, feature_columns=["x"])
    assert "cached" in results2


def test_orchestrator_failure_policy_skip(tmp_path):
    registry = ModelRegistry()
    registry.register("mock", MockModel)
    registry.register("failing", FailingModel)

    config = {
        "good": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
        "bad": {"type": "failing", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
    }

    orch = TrainOrchestrator(
        model_registry=registry, model_configs=config, output_dir=tmp_path, failure_policy="skip"
    )
    results = orch.train_all(np.random.randn(30, 1), (np.random.randn(30) > 0).astype(int), feature_columns=["x"])
    assert "good" in results
    assert "bad" not in results


def test_orchestrator_failure_policy_raise(tmp_path):
    registry = ModelRegistry()
    registry.register("failing", FailingModel)

    config = {
        "bad": {"type": "failing", "mode": "classifier", "features": ["x"], "params": {}, "active": True},
    }

    orch = TrainOrchestrator(
        model_registry=registry, model_configs=config, output_dir=tmp_path, failure_policy="raise"
    )
    with pytest.raises(RuntimeError, match="Training failed"):
        orch.train_all(np.random.randn(30, 1), (np.random.randn(30) > 0).astype(int), feature_columns=["x"])
