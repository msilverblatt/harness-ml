"""Tests for task type protocol and data structures."""

import pytest
from harness.ml.tasks.protocol import Metric, ValidationResult


class TestMetric:
    def test_metric_creation(self):
        m = Metric(name="accuracy", value=0.95, higher_is_better=True)
        assert m.name == "accuracy"
        assert m.value == 0.95
        assert m.higher_is_better is True

    def test_metric_lower_is_better(self):
        m = Metric(name="log_loss", value=0.3, higher_is_better=False)
        assert m.higher_is_better is False

    def test_metric_nan_value(self):
        import math
        m = Metric(name="auroc", value=float("nan"), higher_is_better=True)
        assert math.isnan(m.value)


class TestValidationResult:
    def test_valid_result(self):
        vr = ValidationResult(is_valid=True, messages=[])
        assert vr.is_valid is True
        assert vr.messages == []

    def test_invalid_result_with_messages(self):
        vr = ValidationResult(
            is_valid=False,
            messages=["Target contains non-binary values"],
        )
        assert vr.is_valid is False
        assert len(vr.messages) == 1
        assert "non-binary" in vr.messages[0]

    def test_valid_result_with_warnings(self):
        vr = ValidationResult(
            is_valid=True,
            messages=["Only one class present in target"],
        )
        assert vr.is_valid is True
        assert len(vr.messages) == 1
