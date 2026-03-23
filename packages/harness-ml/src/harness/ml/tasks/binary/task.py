"""Binary classification task type implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.ml.tasks.protocol import CalibrationType, Metric, ValidationResult
from harness.ml.tasks.binary.adaptation import DEFAULT_PARAMS, OBJECTIVES
from harness.ml.tasks.binary.calibration import CALIBRATION_METHODS
from harness.ml.tasks.binary.metrics import METRIC_FUNCTIONS
from harness.ml.tasks.binary.validation import (
    validate_predictions as _validate_predictions,
    validate_target as _validate_target,
)

# Which metrics have "higher is better" semantics
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

_DEFAULT_METRIC_NAMES = ["brier", "log_loss", "auroc", "accuracy", "f1"]


class BinaryTask:
    """Binary classification task type."""

    @property
    def name(self) -> str:
        return "binary"

    @property
    def metrics(self) -> list[Metric]:
        return [
            Metric(name=name, higher_is_better=_HIGHER_IS_BETTER[name])
            for name in METRIC_FUNCTIONS
        ]

    @property
    def default_metrics(self) -> list[str]:
        return list(_DEFAULT_METRIC_NAMES)

    def validate_target(self, y: pd.Series) -> ValidationResult:
        return _validate_target(y)

    def validate_predictions(self, predictions: pd.Series) -> ValidationResult:
        return _validate_predictions(predictions)

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_names: list[str] | None = None,
    ) -> list[Metric]:
        names = metric_names or list(METRIC_FUNCTIONS.keys())
        results: list[Metric] = []
        for name in names:
            fn = METRIC_FUNCTIONS[name]
            value = fn(y_true, y_pred)
            results.append(
                Metric(
                    name=name,
                    value=value,
                    higher_is_better=_HIGHER_IS_BETTER[name],
                )
            )
        return results

    @property
    def calibration_methods(self) -> list[CalibrationType]:
        return list(CALIBRATION_METHODS)

    def postprocess(self, predictions: pd.Series) -> pd.Series:
        """Clip predictions to [0, 1]."""
        return predictions.clip(lower=0.0, upper=1.0)

    @property
    def adaptation_objectives(self) -> dict:
        return dict(OBJECTIVES)

    @property
    def default_params(self) -> dict:
        return dict(DEFAULT_PARAMS)
