"""TabPFN model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.ml.models.protocol import FitResult

NAME = "tabpfn"

try:
    import tabpfn as _tabpfn  # noqa: F401
    _TABPFN_AVAILABLE = True
except ImportError:
    _TABPFN_AVAILABLE = False


class TabPFNModel:
    name = "tabpfn"
    supports_tasks = ["binary", "multiclass"]
    requires_packages = ["tabpfn"]

    def _create_model(self, params: dict) -> Any:
        if not _TABPFN_AVAILABLE:
            raise ImportError(
                "tabpfn is not installed. Install it with: pip install tabpfn"
            )
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(**params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None,
        y_val: pd.Series | None,
        params: dict,
    ) -> FitResult:
        params = dict(params)
        params.pop("_task_type", None)

        # TabPFN has limits on dataset size; subsample if needed
        max_samples = params.pop("max_samples", 10000)
        max_features = params.pop("max_features", 100)

        model = self._create_model(params)

        X_fit = X_train
        y_fit = y_train

        # Subsample rows if exceeding limit
        if len(X_fit) > max_samples:
            idx = np.random.RandomState(42).choice(
                len(X_fit), max_samples, replace=False
            )
            X_fit = X_fit.iloc[idx]
            y_fit = y_fit.iloc[idx]

        # Subsample columns if exceeding limit
        if X_fit.shape[1] > max_features:
            X_fit = X_fit.iloc[:, :max_features]

        model.fit(X_fit, y_fit)

        return FitResult(model=model)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def default_params(self, task_type: str) -> dict:
        return {
            "_task_type": task_type,
            "ignore_pretraining_limits": True,
            "inference_precision": "autocast",
        }

    def param_schema(self) -> dict:
        return {
            "n_estimators": {"type": "int", "default": 4, "min": 1, "max": 32},
            "max_samples": {"type": "int", "default": 10000, "min": 100, "max": 100000},
            "max_features": {"type": "int", "default": 100, "min": 1, "max": 500},
        }

    def save(self, model: Any, path: Path) -> None:
        import joblib

        joblib.dump(model, path)

    def load(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)

    def supports_multi_seed(self) -> bool:
        return False
