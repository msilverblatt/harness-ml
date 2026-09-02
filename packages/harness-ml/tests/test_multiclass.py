"""Tests for the multiclass task type."""

import numpy as np
import pandas as pd
import pytest

from harness.ml.tasks.multiclass.task import MulticlassTask
from harness.ml.tasks.protocol import ValidationResult


@pytest.fixture
def task():
    return MulticlassTask()


class TestMulticlassTaskBasics:
    def test_name(self, task):
        assert task.name == "multiclass"

    def test_metrics_list(self, task):
        metrics = task.metrics()
        names = [m.name for m in metrics]
        assert "accuracy" in names
        assert "f1_macro" in names
        assert "log_loss" in names
        assert "f1_micro" in names
        assert "f1_weighted" in names
        assert "precision_macro" in names
        assert "recall_macro" in names

    def test_default_metrics(self, task):
        defaults = task.default_metrics()
        assert isinstance(defaults, list)
        assert len(defaults) >= 3
        all_names = [m.name for m in task.metrics()]
        for d in defaults:
            assert d in all_names


class TestValidateTarget:
    def test_valid_integer_target_3_classes(self, task):
        y = pd.Series([0, 1, 2, 0, 1, 2])
        result = task.validate_target(y)
        assert result.is_valid

    def test_valid_more_classes(self, task):
        y = pd.Series([0, 1, 2, 3, 4])
        result = task.validate_target(y)
        assert result.is_valid

    def test_too_few_classes_rejected(self, task):
        y = pd.Series([0, 1, 0, 1])
        result = task.validate_target(y)
        assert not result.is_valid

    def test_string_labels_accepted(self, task):
        y = pd.Series(["cat", "dog", "bird"])
        result = task.validate_target(y)
        assert result.is_valid

    def test_non_integer_numeric_labels_rejected(self, task):
        y = pd.Series([0.5, 1.5, 2.5])
        result = task.validate_target(y)
        assert not result.is_valid

    def test_missing_labels_rejected(self, task):
        y = pd.Series(["cat", "dog", None])
        result = task.validate_target(y)
        assert not result.is_valid


class TestValidatePredictions:
    def test_valid_2d_probabilities(self, task):
        # 4 samples, 3 classes, rows sum to 1
        preds = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.33, 0.33, 0.34],
        ])
        result = task.validate_predictions(preds)
        assert result.is_valid

    def test_1d_rejected(self, task):
        preds = np.array([0.1, 0.5, 0.4])
        result = task.validate_predictions(preds)
        assert not result.is_valid

    def test_rows_not_summing_to_1_rejected(self, task):
        preds = np.array([
            [0.7, 0.2, 0.5],  # sums to 1.4
            [0.1, 0.8, 0.1],
        ])
        result = task.validate_predictions(preds)
        assert not result.is_valid


class TestComputeMetrics:
    def test_perfect_accuracy(self, task):
        # 3 classes, perfect predictions
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = task.compute_metrics(y_true, y_pred, ["accuracy"])
        assert result["accuracy"] == pytest.approx(1.0)

    def test_accuracy_4_correct_out_of_5(self, task):
        # 1 wrong out of 5 → accuracy 0.8
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([
            [1.0, 0.0, 0.0],   # correct: 0
            [0.0, 1.0, 0.0],   # correct: 1
            [0.0, 0.0, 1.0],   # correct: 2
            [1.0, 0.0, 0.0],   # correct: 0
            [1.0, 0.0, 0.0],   # wrong: predicted 0, true is 1
        ])
        result = task.compute_metrics(y_true, y_pred, ["accuracy"])
        assert result["accuracy"] == pytest.approx(0.8)

    def test_returns_dict(self, task):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        result = task.compute_metrics(y_true, y_pred, ["accuracy", "f1_macro", "log_loss"])
        assert isinstance(result, dict)

    def test_multiple_metrics(self, task):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        result = task.compute_metrics(y_true, y_pred, ["accuracy", "f1_macro", "log_loss"])
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1_macro"] == pytest.approx(1.0)
        assert result["log_loss"] < 0.2


class TestAdaptation:
    def test_objectives_exist(self):
        from harness.ml.tasks.multiclass.adaptation import OBJECTIVES
        assert isinstance(OBJECTIVES, dict)
        assert "xgboost" in OBJECTIVES
        assert "lightgbm" in OBJECTIVES
        assert "catboost" in OBJECTIVES
