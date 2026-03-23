"""Comprehensive tests for the binary classification task type."""

import math

import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.binary.task import BinaryTask


@pytest.fixture
def task():
    return BinaryTask()


class TestBinaryTaskBasics:
    def test_name(self, task):
        assert task.name == "binary"

    def test_metrics_list(self, task):
        metrics = task.metrics
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
        defaults = task.default_metrics
        assert isinstance(defaults, list)
        assert len(defaults) > 0
        # Default metrics should be a subset of all metrics
        all_names = [m.name for m in task.metrics]
        for d in defaults:
            assert d in all_names


class TestValidateTarget:
    def test_valid_binary_target(self, task):
        y = pd.Series([0, 1, 0, 1, 1])
        result = task.validate_target(y)
        assert result.is_valid is True

    def test_invalid_values(self, task):
        y = pd.Series([0, 1, 2, 3])
        result = task.validate_target(y)
        assert result.is_valid is False

    def test_float_values(self, task):
        y = pd.Series([0.0, 1.0, 0.0, 1.0])
        result = task.validate_target(y)
        assert result.is_valid is True

    def test_all_same_class_warning(self, task):
        y = pd.Series([1, 1, 1, 1])
        result = task.validate_target(y)
        assert result.is_valid is True
        assert len(result.messages) > 0

    def test_negative_values(self, task):
        y = pd.Series([-1, 0, 1])
        result = task.validate_target(y)
        assert result.is_valid is False

    def test_non_integer_floats(self, task):
        y = pd.Series([0.5, 0.3, 0.7])
        result = task.validate_target(y)
        assert result.is_valid is False


class TestValidatePredictions:
    def test_valid_predictions(self, task):
        preds = pd.Series([0.1, 0.5, 0.9, 0.0, 1.0])
        result = task.validate_predictions(preds)
        assert result.is_valid is True

    def test_out_of_range_high(self, task):
        preds = pd.Series([0.5, 1.5, 0.3])
        result = task.validate_predictions(preds)
        assert result.is_valid is False

    def test_out_of_range_low(self, task):
        preds = pd.Series([-0.1, 0.5, 0.3])
        result = task.validate_predictions(preds)
        assert result.is_valid is False

    def test_all_zeros(self, task):
        preds = pd.Series([0.0, 0.0, 0.0])
        result = task.validate_predictions(preds)
        assert result.is_valid is True

    def test_all_ones(self, task):
        preds = pd.Series([1.0, 1.0, 1.0])
        result = task.validate_predictions(preds)
        assert result.is_valid is True


class TestComputeMetrics:
    def test_perfect_brier(self, task):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0.0, 1.0, 0.0, 1.0])
        metrics = task.compute_metrics(y_true, y_pred)
        brier = next(m for m in metrics if m.name == "brier")
        assert brier.value == pytest.approx(0.0, abs=1e-10)

    def test_worst_brier(self, task):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([1.0, 0.0, 1.0, 0.0])
        metrics = task.compute_metrics(y_true, y_pred)
        brier = next(m for m in metrics if m.name == "brier")
        assert brier.value == pytest.approx(1.0, abs=1e-10)

    def test_perfect_accuracy(self, task):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0.0, 1.0, 0.0, 1.0])
        metrics = task.compute_metrics(y_true, y_pred)
        acc = next(m for m in metrics if m.name == "accuracy")
        assert acc.value == pytest.approx(1.0)

    def test_auroc_single_class(self, task):
        y_true = pd.Series([1, 1, 1, 1])
        y_pred = pd.Series([0.5, 0.6, 0.7, 0.8])
        metrics = task.compute_metrics(y_true, y_pred)
        auroc = next(m for m in metrics if m.name == "auroc")
        assert math.isnan(auroc.value)

    def test_compute_metrics_on_dataset(self, task, binary_dataset):
        X, y = binary_dataset
        # Simulate predictions with some noise
        rng = np.random.RandomState(42)
        y_pred = pd.Series(
            np.clip(y.values + rng.randn(len(y)) * 0.3, 0, 1),
            name="prediction",
        )
        metrics = task.compute_metrics(y, y_pred)
        metric_dict = {m.name: m.value for m in metrics}
        # With correlated predictions, auroc should be well above random
        assert metric_dict["auroc"] > 0.7
        assert 0 <= metric_dict["brier"] <= 1
        assert 0 <= metric_dict["accuracy"] <= 1

    def test_metric_higher_is_better_flags(self, task):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0.1, 0.9, 0.2, 0.8])
        metrics = task.compute_metrics(y_true, y_pred)
        metric_dict = {m.name: m for m in metrics}
        # Lower is better for loss metrics
        assert metric_dict["brier"].higher_is_better is False
        assert metric_dict["log_loss"].higher_is_better is False
        assert metric_dict["ece"].higher_is_better is False
        # Higher is better for performance metrics
        assert metric_dict["auroc"].higher_is_better is True
        assert metric_dict["accuracy"].higher_is_better is True
        assert metric_dict["f1"].higher_is_better is True


class TestCalibration:
    def test_calibration_methods_exist(self, task):
        methods = task.calibration_methods
        assert len(methods) >= 4
        names = [m.name for m in methods]
        assert "isotonic" in names
        assert "platt" in names
        assert "spline" in names
        assert "beta" in names


class TestPostprocess:
    def test_clipping(self, task):
        preds = pd.Series([-0.1, 0.5, 1.2, 0.0, 1.0])
        result = task.postprocess(preds)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.iloc[1] == pytest.approx(0.5)


class TestAdaptation:
    def test_objectives_exist(self, task):
        objectives = task.adaptation_objectives
        assert isinstance(objectives, dict)
        assert len(objectives) > 0

    def test_default_params_exist(self, task):
        params = task.default_params
        assert isinstance(params, dict)
