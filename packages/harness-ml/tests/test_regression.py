"""Tests for the regression task type."""

import math

import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.regression.task import RegressionTask
from harness.ml.tasks.protocol import ValidationResult


@pytest.fixture
def task():
    return RegressionTask()


class TestRegressionTaskBasics:
    def test_name(self, task):
        assert task.name == "regression"

    def test_metrics_list(self, task):
        metrics = task.metrics()
        names = [m.name for m in metrics]
        assert "rmse" in names
        assert "mae" in names
        assert "r2" in names
        assert "mape" in names
        assert "median_ae" in names
        assert "explained_variance" in names

    def test_default_metrics(self, task):
        defaults = task.default_metrics()
        assert isinstance(defaults, list)
        assert len(defaults) >= 3
        all_names = [m.name for m in task.metrics()]
        for d in defaults:
            assert d in all_names


class TestValidateTarget:
    def test_valid_float_target(self, task):
        y = pd.Series([1.0, 2.5, 3.7, -1.0, 0.0])
        result = task.validate_target(y)
        assert result.is_valid

    def test_valid_integer_target(self, task):
        y = pd.Series([1, 2, 3, 4, 5])
        result = task.validate_target(y)
        assert result.is_valid

    def test_non_numeric_rejected(self, task):
        y = pd.Series(["a", "b", "c"])
        result = task.validate_target(y)
        assert not result.is_valid

    def test_all_same_value_warns(self, task):
        y = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = task.validate_target(y)
        assert result.is_valid
        assert len(result.messages) > 0


class TestValidatePredictions:
    def test_valid_numeric_predictions(self, task):
        preds = np.array([1.0, 2.0, 3.0, -1.0])
        result = task.validate_predictions(preds)
        assert result.is_valid

    def test_nan_rejected(self, task):
        preds = np.array([1.0, float("nan"), 3.0])
        result = task.validate_predictions(preds)
        assert not result.is_valid


class TestComputeMetrics:
    def test_rmse_manual(self, task):
        # y=[1,2,3], p=[1,2,4] → errors=[0,0,1] → mse=1/3 → rmse=sqrt(1/3)
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        result = task.compute_metrics(y_true, y_pred, ["rmse"])
        expected = math.sqrt(1.0 / 3.0)
        assert result["rmse"] == pytest.approx(expected, rel=1e-6)

    def test_r2_perfect(self, task):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])
        result = task.compute_metrics(y_true, y_pred, ["r2"])
        assert result["r2"] == pytest.approx(1.0, abs=1e-10)

    def test_mae_manual(self, task):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        result = task.compute_metrics(y_true, y_pred, ["mae"])
        # |1-2| + |2-2| + |3-2| = 1 + 0 + 1 = 2 → mean = 2/3
        assert result["mae"] == pytest.approx(2.0 / 3.0, rel=1e-6)

    def test_returns_dict(self, task):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        result = task.compute_metrics(y_true, y_pred, ["rmse", "mae", "r2"])
        assert isinstance(result, dict)

    def test_multiple_metrics(self, task):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.9, 5.1])
        result = task.compute_metrics(y_true, y_pred, ["rmse", "mae", "r2"])
        assert "rmse" in result
        assert "mae" in result
        assert "r2" in result
        assert result["r2"] > 0.99


class TestAdaptation:
    def test_objectives_exist(self):
        from harness.ml.tasks.regression.adaptation import OBJECTIVES
        assert isinstance(OBJECTIVES, dict)
        assert "xgboost" in OBJECTIVES
        assert "lightgbm" in OBJECTIVES
        assert "catboost" in OBJECTIVES
