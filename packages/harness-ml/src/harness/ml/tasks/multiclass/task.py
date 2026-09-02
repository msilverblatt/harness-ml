"""Multiclass classification task type implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.ml.tasks.protocol import CalibrationType, Metric, ValidationResult
from harness.ml.tasks.multiclass.adaptation import OBJECTIVES
from harness.ml.tasks.multiclass.metrics import METRIC_FUNCTIONS
from harness.ml.tasks.multiclass.validation import (
    validate_predictions as _validate_predictions,
    validate_target as _validate_target,
)

_HIGHER_IS_BETTER = {
    "accuracy": True,
    "f1_macro": True,
    "f1_micro": True,
    "f1_weighted": True,
    "log_loss": False,
    "precision_macro": True,
    "recall_macro": True,
}

_DEFAULT_METRIC_NAMES = ["accuracy", "f1_macro", "f1_weighted", "log_loss"]


class MulticlassTask:
    """Multiclass classification task type."""

    name = "multiclass"

    def metrics(self) -> list[Metric]:
        return [
            Metric(name=name, higher_is_better=_HIGHER_IS_BETTER[name])
            for name in METRIC_FUNCTIONS
        ]

    def default_metrics(self) -> list[str]:
        return list(_DEFAULT_METRIC_NAMES)

    def validate_target(self, series: pd.Series) -> ValidationResult:
        return _validate_target(series)

    def validate_predictions(self, predictions: np.ndarray) -> ValidationResult:
        return _validate_predictions(predictions)

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_names: list[str],
    ) -> dict[str, float]:
        results = {}
        for name in metric_names:
            fn = METRIC_FUNCTIONS.get(name)
            if fn is None:
                continue
            try:
                results[name] = float(fn(y_true, y_pred))
            except Exception:
                results[name] = float("nan")
        return results

    def calibration_methods(self) -> list[CalibrationType]:
        return []

    def postprocess(self, predictions: np.ndarray, config: dict) -> np.ndarray:
        result = predictions.copy()
        if config.get("normalize", False):
            row_sums = result.sum(axis=1, keepdims=True)
            result = result / np.where(row_sums == 0, 1.0, row_sums)
        return result
