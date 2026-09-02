"""RealMLP model wrapper using pytabkit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.ml.models.protocol import FitResult

NAME = "realmlp"

try:
    import pytabkit as _ptk  # noqa: F401
    _PYTABKIT_AVAILABLE = True
except ImportError:
    _PYTABKIT_AVAILABLE = False


class RealMLPModel:
    name = "realmlp"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages = ["pytabkit"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _PYTABKIT_AVAILABLE:
            raise ImportError(
                "pytabkit is not installed. Install it with: pip install pytabkit"
            )
        from pytabkit import RealMLP_TD_Classifier, RealMLP_TD_Regressor

        if task_type == "regression":
            return RealMLP_TD_Regressor(**params)
        else:
            return RealMLP_TD_Classifier(**params)

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
        model.fit(X_train, y_train)

        return FitResult(model=model)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            if proba.ndim == 2:
                return proba
        return model.predict(X)

    def default_params(self, task_type: str) -> dict:
        return {
            "_task_type": task_type,
            "device": "cpu",
            "verbosity": 0,
            "n_cv": 1,
            "n_refit": 0,
            "n_threads": 1,
        }

    def param_schema(self) -> dict:
        return {
            "n_epochs": {"type": "int", "default": 256, "min": 1, "max": 2000},
            "batch_size": {"type": "int", "default": 256, "min": 8, "max": 4096},
            "n_cv": {"type": "int", "default": 1, "min": 1, "max": 10},
            "n_refit": {"type": "int", "default": 0, "min": 0, "max": 5},
            "verbosity": {"type": "int", "default": 0, "min": 0, "max": 2},
        }

    def save(self, model: Any, path: Path) -> None:
        import joblib

        joblib.dump(model, path)

    def load(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)

    def supports_multi_seed(self) -> bool:
        return True
