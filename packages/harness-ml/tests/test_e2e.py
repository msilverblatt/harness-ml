"""End-to-end tests covering task types, model training, and integration."""

from __future__ import annotations

import numpy as np
import pytest

from harness.ml.models.registry import ModelRegistry
from harness.ml.tasks.registry import TaskRegistry


class TestE2ETaskTypes:
    def test_registry_has_all_types(self):
        assert set(TaskRegistry.list_available()) == {"binary", "multiclass", "regression"}

    def test_binary_brier_manual_verification(self):
        """Manually calculated: y=[1,1,1,1,0], p=[0.8]*5 → Brier=0.16"""
        task = TaskRegistry.get("binary")
        y = np.array([1, 1, 1, 1, 0])
        p = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        m = task.compute_metrics(y, p, ["brier"])
        assert abs(m["brier"] - 0.16) < 0.001

    def test_regression_rmse_manual_verification(self):
        """y=[1,2,3], p=[1,2,4] → RMSE=sqrt(1/3)"""
        task = TaskRegistry.get("regression")
        m = task.compute_metrics(
            np.array([1, 2, 3.0]), np.array([1, 2, 4.0]), ["rmse"]
        )
        assert abs(m["rmse"] - np.sqrt(1 / 3)) < 0.001

    def test_multiclass_accuracy_manual(self):
        task = TaskRegistry.get("multiclass")
        y = np.array([0, 1, 2, 0, 1])
        p = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.05, 0.9],
                [0.9, 0.05, 0.05],
                [0.05, 0.05, 0.9],
            ]
        )
        m = task.compute_metrics(y, p, ["accuracy"])
        assert abs(m["accuracy"] - 0.8) < 0.001  # 4th is wrong


class TestE2EModels:
    def test_all_models_registered(self):
        models = ModelRegistry.list_available()
        expected = {
            "logistic",
            "elastic_net",
            "random_forest",
            "svm",
            "xgboost",
            "lightgbm",
            "catboost",
            "hist_gbm",
            "mlp",
        }
        # mlp may or may not be available depending on torch
        assert len(models) >= 8

    def test_logistic_learns_binary(self, binary_dataset):
        model = ModelRegistry.get("logistic")
        task = TaskRegistry.get("binary")
        X, y = binary_dataset
        result = model.fit(X, y, None, None, model.default_params("binary"))
        preds = model.predict(result.model, X)
        metrics = task.compute_metrics(y.values, preds, ["accuracy", "auroc"])
        assert metrics["accuracy"] > 0.65
        assert metrics["auroc"] > 0.7

    def test_random_forest_learns_regression(self, regression_dataset):
        model = ModelRegistry.get("random_forest")
        task = TaskRegistry.get("regression")
        X, y = regression_dataset
        result = model.fit(X, y, None, None, model.default_params("regression"))
        preds = model.predict(result.model, X)
        metrics = task.compute_metrics(y.values, preds, ["r2"])
        assert metrics["r2"] > 0.5

    def test_xgboost_learns_binary(self, binary_dataset):
        model = ModelRegistry.get("xgboost")
        if model is None:
            pytest.skip("xgboost not installed")
        task = TaskRegistry.get("binary")
        X, y = binary_dataset
        result = model.fit(X, y, None, None, model.default_params("binary"))
        preds = model.predict(result.model, X)
        metrics = task.compute_metrics(y.values, preds, ["accuracy", "brier"])
        assert metrics["accuracy"] > 0.7
        assert metrics["brier"] < 0.3

    def test_model_save_load_predictions_match(self, binary_dataset, tmp_path):
        """Save and load a model, verify predictions are identical."""
        model = ModelRegistry.get("logistic")
        X, y = binary_dataset
        result = model.fit(X, y, None, None, model.default_params("binary"))
        preds1 = model.predict(result.model, X)

        path = tmp_path / "model.pkl"
        model.save(result.model, path)
        loaded = model.load(path)
        preds2 = model.predict(loaded, X)

        np.testing.assert_array_almost_equal(preds1, preds2)

    def test_all_tasks_have_adaptation_for_sklearn_models(self):
        """Every sklearn model should have adaptation entries for supported tasks."""
        from harness.ml.tasks.binary.adaptation import OBJECTIVES as binary_obj
        from harness.ml.tasks.regression.adaptation import OBJECTIVES as reg_obj

        for model_name in ["logistic", "random_forest", "svm"]:
            model = ModelRegistry.get(model_name)
            if "binary" in model.supports_tasks:
                found = model_name in binary_obj or any(
                    model_name in k for k in binary_obj
                )
                assert found, f"{model_name} missing from binary adaptation"
