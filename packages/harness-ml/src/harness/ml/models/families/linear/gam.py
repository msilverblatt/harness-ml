"""GAM (Generalized Additive Model) wrapper using pygam."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.ml.models.protocol import FitResult

NAME = "gam"

try:
    import pygam as _pygam  # noqa: F401
    _PYGAM_AVAILABLE = True
except ImportError:
    _PYGAM_AVAILABLE = False


class GAMModel:
    name = "gam"
    supports_tasks = ["binary", "regression"]
    requires_packages = ["pygam"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _PYGAM_AVAILABLE:
            raise ImportError(
                "pygam is not installed. Install it with: pip install pygam"
            )
        from pygam import LinearGAM, LogisticGAM

        if task_type == "regression":
            return LinearGAM(**params)
        else:
            return LogisticGAM(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        params: dict,
    ) -> FitResult:
        params = dict(params)
        task_type = params.pop("_task_type", self.supports_tasks[0])
        model = self._create_model(params, task_type)
        model.fit(X_train.values, y_train.values)

        return FitResult(model=model)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            # LogisticGAM has predict_proba
            try:
                proba = model.predict_proba(X.values)
                return proba
            except Exception:
                pass
        return model.predict(X.values)

    def default_params(self, task_type: str) -> dict:
        return {
            "_task_type": task_type,
            "max_iter": 100,
        }

    def param_schema(self) -> dict:
        return {
            "max_iter": {"type": "int", "default": 100, "min": 10, "max": 1000},
            "tol": {"type": "float", "default": 1e-4, "min": 1e-8, "max": 1e-1},
        }

    def save(self, model: Any, path: Path) -> None:
        import joblib

        joblib.dump(model, path)

    def load(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)

    def supports_multi_seed(self) -> bool:
        return False
