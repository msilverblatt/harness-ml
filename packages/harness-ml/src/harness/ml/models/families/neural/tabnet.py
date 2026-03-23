"""TabNet model wrapper using pytorch-tabnet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.ml.models.protocol import FitResult

NAME = "tabnet"

try:
    import pytorch_tabnet as _tabnet  # noqa: F401
    _TABNET_AVAILABLE = True
except ImportError:
    _TABNET_AVAILABLE = False


class TabNetModel:
    name = "tabnet"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages = ["pytorch_tabnet"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _TABNET_AVAILABLE:
            raise ImportError(
                "pytorch-tabnet is not installed. Install it with: pip install pytorch-tabnet"
            )
        from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

        if task_type == "regression":
            return TabNetRegressor(**params)
        else:
            return TabNetClassifier(**params)

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

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val.values, y_val.values.reshape(-1, 1)
                                       if task_type == "regression"
                                       else y_val.values)]
            fit_kwargs["eval_name"] = ["val"]
            fit_kwargs["eval_metric"] = (
                ["rmse"] if task_type == "regression" else ["auc"]
            )

        X_np = X_train.values
        y_np = y_train.values
        if task_type == "regression":
            y_np = y_np.reshape(-1, 1)

        model.fit(X_np, y_np, **fit_kwargs)

        feature_importance: dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            for fname, imp in zip(X_train.columns, model.feature_importances_):
                feature_importance[fname] = float(imp)

        return FitResult(model=model, feature_importance=feature_importance)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        X_np = X.values
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_np)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            if proba.ndim == 2:
                return proba
        preds = model.predict(X_np)
        return preds.squeeze()

    def default_params(self, task_type: str) -> dict:
        return {
            "_task_type": task_type,
            "n_d": 8,
            "n_a": 8,
            "n_steps": 3,
            "gamma": 1.3,
            "verbose": 0,
        }

    def param_schema(self) -> dict:
        return {
            "n_d": {"type": "int", "default": 8, "min": 4, "max": 64},
            "n_a": {"type": "int", "default": 8, "min": 4, "max": 64},
            "n_steps": {"type": "int", "default": 3, "min": 1, "max": 10},
            "gamma": {"type": "float", "default": 1.3, "min": 1.0, "max": 2.0},
            "lambda_sparse": {
                "type": "float",
                "default": 1e-3,
                "min": 0.0,
                "max": 1.0,
            },
        }

    def save(self, model: Any, path: Path) -> None:
        import joblib

        joblib.dump(model, path)

    def load(self, path: Path) -> Any:
        import joblib

        return joblib.load(path)

    def supports_multi_seed(self) -> bool:
        return True
