"""Model protocol and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    """Result returned from Model.fit()."""

    model: Any
    feature_importance: dict[str, float] = field(default_factory=dict)
    training_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Model(Protocol):
    """Protocol defining the interface for a model wrapper."""

    name: str
    supports_tasks: list[str]
    requires_packages: list[str]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        params: dict,
    ) -> FitResult: ...

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray: ...

    def default_params(self, task_type: str) -> dict: ...

    def param_schema(self) -> dict: ...

    def save(self, model: Any, path: Path) -> None: ...

    def load(self, path: Path) -> Any: ...

    def supports_multi_seed(self) -> bool: ...
