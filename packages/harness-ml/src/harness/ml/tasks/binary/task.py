"""Binary classification task type implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness.ml.tasks.binary.calibration import CALIBRATION_METHODS
from harness.ml.tasks.binary.metrics import METRIC_FUNCTIONS
from harness.ml.tasks.binary.validation import (
    validate_predictions as _validate_predictions,
)
from harness.ml.tasks.binary.validation import (
    validate_target as _validate_target,
)
from harness.ml.tasks.protocol import CalibrationType, Metric, ValidationResult

_HIGHER_IS_BETTER = {
    "auroc": True,
    "accuracy": True,
    "f1": True,
    "precision": True,
    "recall": True,
    "brier": False,
    "log_loss": False,
    "ece": False,
}

_DEFAULT_METRIC_NAMES = ["brier", "log_loss", "auroc", "accuracy", "ece"]


class BinaryTask:
    """Binary classification task type."""

    name = "binary"

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
        return list(CALIBRATION_METHODS)

    def postprocess(self, predictions: np.ndarray, config: dict) -> np.ndarray:
        result = predictions.copy()
        if config.get("clip", False):
            result = np.clip(result, 0.0, 1.0)
        clip_floor = config.get("clip_floor")
        if clip_floor is not None:
            result = np.clip(result, clip_floor, 1.0 - clip_floor)
        return result
