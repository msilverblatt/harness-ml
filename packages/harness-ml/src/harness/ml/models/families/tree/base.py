"""Base class for tree model family."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from harness.ml.models.protocol import FitResult


class TreeBase(ABC):
    """Shared implementation for sklearn tree-based models."""

    @abstractmethod
    def _create_model(self, params: dict, task_type: str) -> Any:
        """Create the underlying sklearn model with the given params."""
        ...

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        params: dict,
    ) -> FitResult:
        task_type = params.pop("_task_type", self.supports_tasks[0])
        model = self._create_model(params, task_type)
        model.fit(X_train, y_train)

        feature_importance: dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            for fname, imp in zip(X_train.columns, model.feature_importances_):
                feature_importance[fname] = float(imp)

        return FitResult(
            model=model,
            feature_importance=feature_importance,
        )

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return model.predict(X)

    def save(self, model: Any, path: Path) -> None:
        import joblib

        joblib.dump(model, path)

    def load(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)

    def supports_multi_seed(self) -> bool:
        return True
