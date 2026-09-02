"""Task type protocol and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class Metric:
    """A single evaluation metric definition or computed result."""

    name: str
    value: float = float("nan")
    higher_is_better: bool = True


@dataclass
class ValidationResult:
    """Result of validating targets or predictions."""

    is_valid: bool
    messages: list[str] = field(default_factory=list)


@dataclass
class CalibrationType:
    """A calibration method that can be applied to predictions."""

    name: str
    description: str = ""


@dataclass
class ResultSummary:
    """Summary of model evaluation results."""

    metrics: list[Metric] = field(default_factory=list)
    calibration_method: str | None = None
    additional_info: dict | None = None


@runtime_checkable
class TaskType(Protocol):
    """Protocol defining the interface for a task type."""

    @property
    def name(self) -> str:
        """Unique name for this task type (e.g. 'binary', 'regression')."""
        ...

    @property
    def metrics(self) -> list[Metric]:
        """All metrics supported by this task type."""
        ...

    @property
    def default_metrics(self) -> list[str]:
        """Names of metrics computed by default."""
        ...

    def validate_target(self, y: pd.Series) -> ValidationResult:
        """Validate that a target series is appropriate for this task type."""
        ...

    def validate_predictions(self, predictions: pd.Series) -> ValidationResult:
        """Validate that predictions are appropriate for this task type."""
        ...

    def compute_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_names: list[str] | None = None,
    ) -> list[Metric]:
        """Compute metrics for the given true and predicted values."""
        ...

    @property
    def calibration_methods(self) -> list[CalibrationType]:
        """Available calibration methods for this task type."""
        ...

    def postprocess(self, predictions: pd.Series) -> pd.Series:
        """Apply any task-specific postprocessing to raw predictions."""
        ...

    @property
    def adaptation_objectives(self) -> dict:
        """Model-specific objective/loss function mappings for this task."""
        ...

    @property
    def default_params(self) -> dict:
        """Default hyperparameters for this task type."""
        ...
