"""Comprehensive tests for the binary classification task type."""

import math

import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.binary.task import BinaryTask
from harness.ml.tasks.protocol import ValidationResult


@pytest.fixture
def task():
    return BinaryTask()


class TestBinaryTaskBasics:
    def test_name(self, task):
        assert task.name == "binary"

    def test_metrics_list(self, task):
        metrics = task.metrics()
        names = [m.name for m in metrics]
        assert "brier" in names
        assert "log_loss" in names
        assert "auroc" in names
        assert "accuracy" in names
        assert "f1" in names
        assert "precision" in names
        assert "recall" in names
        assert "ece" in names

    def test_default_metrics(self, task):
        defaults = task.default_metrics()
        assert isinstance(defaults, list)
        assert len(defaults) >= 3
        all_names = [m.name for m in task.metrics()]
        for d in defaults:
            assert d in all_names


class TestValidateTarget:
    def test_valid_binary_target(self, task):
        y = pd.Series([0, 1, 0, 1, 1])
        result = task.validate_target(y)
        assert result.is_valid

    def test_invalid_values(self, task):
        y = pd.Series([0, 1, 2, 3])
        result = task.validate_target(y)
        assert not result.is_valid

    def test_float_values(self, task):
        y = pd.Series([0.0, 1.0, 0.0, 1.0])
        result = task.validate_target(y)
        assert result.is_valid

    def test_all_same_class_warning(self, task):
        y = pd.Series([1, 1, 1, 1])
        result = task.validate_target(y)
        # Should be valid but with a message
        assert result.is_valid
        assert len(result.messages) > 0

    def test_negative_values(self, task):
        y = pd.Series([-1, 0, 1])
        result = task.validate_target(y)
        assert not result.is_valid

    def test_non_integer_floats(self, task):
        y = pd.Series([0.5, 0.3, 0.7])
        result = task.validate_target(y)
        assert not result.is_valid


class TestValidatePredictions:
    def test_valid_predictions(self, task):
        preds = np.array([0.1, 0.5, 0.9, 0.0, 1.0])
        result = task.validate_predictions(preds)
        assert result.is_valid

    def test_out_of_range_high(self, task):
        preds = np.array([0.5, 1.5, 0.3])
        result = task.validate_predictions(preds)
        assert not result.is_valid

    def test_out_of_range_low(self, task):
        preds = np.array([-0.1, 0.5, 0.3])
        result = task.validate_predictions(preds)
        assert not result.is_valid


class TestComputeMetrics:
    def test_perfect_brier(self, task):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        result = task.compute_metrics(y_true, y_pred, ["brier"])
        assert result["brier"] == pytest.approx(0.0, abs=1e-10)

    def test_worst_brier(self, task):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1.0, 0.0, 1.0, 0.0])
        result = task.compute_metrics(y_true, y_pred, ["brier"])
        assert result["brier"] == pytest.approx(1.0, abs=1e-10)

    def test_manual_brier(self, task):
        """Manually verified: 4 wins predicted at 80%, 1 loss predicted at 80%."""
        y_true = np.array([1, 1, 1, 1, 0])
        y_pred = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
        result = task.compute_metrics(y_true, y_pred, ["brier"])
        # Brier = mean((y-p)^2) = (4*0.04 + 1*0.64)/5 = 0.16
        assert result["brier"] == pytest.approx(0.16, abs=0.001)

    def test_perfect_accuracy(self, task):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.0, 1.0, 0.0, 1.0])
        result = task.compute_metrics(y_true, y_pred, ["accuracy"])
        assert result["accuracy"] == pytest.approx(1.0)

    def test_auroc_single_class(self, task):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0.5, 0.6, 0.7, 0.8])
        result = task.compute_metrics(y_true, y_pred, ["auroc"])
        assert math.isnan(result["auroc"])

    def test_compute_multiple_metrics(self, task, binary_dataset):
        X, y = binary_dataset
        rng = np.random.RandomState(42)
        y_pred = np.clip(y.values + rng.randn(len(y)) * 0.3, 0, 1)
        result = task.compute_metrics(y.values, y_pred, ["brier", "accuracy", "auroc"])
        assert isinstance(result, dict)
        assert "brier" in result
        assert "accuracy" in result
        assert "auroc" in result
        # auroc can be nan if all predictions round to same class — check it's a number or nan
        assert result["auroc"] > 0.5 or math.isnan(result["auroc"])

    def test_returns_dict(self, task):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8])
        result = task.compute_metrics(y_true, y_pred, ["brier", "accuracy"])
        assert isinstance(result, dict)


class TestCalibration:
    def test_calibration_methods_exist(self, task):
        methods = task.calibration_methods()
        assert len(methods) >= 4
        names = [m.name for m in methods]
        assert "isotonic" in names
        assert "platt" in names
        assert "spline" in names
        assert "beta" in names


class TestPostprocess:
    def test_clipping(self, task):
        preds = np.array([-0.1, 0.5, 1.2, 0.0, 1.0])
        result = task.postprocess(preds, {"clip": True})
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result[1] == pytest.approx(0.5)

    def test_clip_floor(self, task):
        preds = np.array([0.01, 0.5, 0.99])
        result = task.postprocess(preds, {"clip_floor": 0.05})
        assert result.min() >= 0.05
        assert result.max() <= 0.95


class TestAdaptation:
    def test_objectives_exist(self):
        from harness.ml.tasks.binary.adaptation import OBJECTIVES
        assert isinstance(OBJECTIVES, dict)
        assert "xgboost" in OBJECTIVES
        assert "lightgbm" in OBJECTIVES
        # Check for logistic (may be "logistic" or "logistic_regression")
        assert "logistic" in OBJECTIVES or "logistic_regression" in OBJECTIVES

    def test_default_params_exist(self):
        from harness.ml.tasks.binary.adaptation import DEFAULT_PARAMS
        assert isinstance(DEFAULT_PARAMS, dict)
        assert len(DEFAULT_PARAMS) > 0
