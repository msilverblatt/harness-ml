"""Regression task type implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness.ml.tasks.protocol import CalibrationType, Metric, ValidationResult
from harness.ml.tasks.regression.metrics import METRIC_FUNCTIONS
from harness.ml.tasks.regression.validation import (
    validate_predictions as _validate_predictions,
)
from harness.ml.tasks.regression.validation import (
    validate_target as _validate_target,
)

_HIGHER_IS_BETTER = {
    "rmse": False,
    "mae": False,
    "r2": True,
    "mape": False,
    "median_ae": False,
    "explained_variance": True,
}

_DEFAULT_METRIC_NAMES = ["rmse", "mae", "r2", "explained_variance"]


class RegressionTask:
    """Regression task type."""

    name = "regression"

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
        clip_min = config.get("clip_min")
        clip_max = config.get("clip_max")
        if clip_min is not None or clip_max is not None:
            result = np.clip(result, clip_min, clip_max)
        return result
