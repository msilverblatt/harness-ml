"""NGBoost model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from harness.ml.models.families.boosting.base import BoostingBase
from harness.ml.models.protocol import FitResult

NAME = "ngboost"

try:
    import ngboost as _ngboost  # noqa: F401
    _NGBOOST_AVAILABLE = True
except ImportError:
    _NGBOOST_AVAILABLE = False


class NGBoostModel(BoostingBase):
    name = "ngboost"
    supports_tasks = ["binary", "regression"]
    requires_packages = ["ngboost"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _NGBOOST_AVAILABLE:
            raise ImportError(
                "ngboost is not installed. Install it with: pip install ngboost"
            )
        from ngboost import NGBClassifier, NGBRegressor

        if task_type == "regression":
            return NGBRegressor(**params)
        else:
            return NGBClassifier(**params)

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
            fit_kwargs["X_val"] = X_val
            fit_kwargs["Y_val"] = y_val

        model.fit(X_train, y_train, **fit_kwargs)

        feature_importance: dict[str, float] = {}
        if hasattr(model, "feature_importances_"):
            fi = np.asarray(model.feature_importances_)
            # NGBoost returns shape (1, n_features); flatten to 1D
            if fi.ndim > 1:
                fi = fi.mean(axis=0)
            for fname, imp in zip(X_train.columns, fi):
                feature_importance[fname] = float(imp)

        return FitResult(model=model, feature_importance=feature_importance)

    def predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                return proba[:, 1]
            return proba
        return model.predict(X)

    def default_params(self, task_type: str) -> dict:
        return {
            "_task_type": task_type,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "verbose": False,
        }

    def param_schema(self) -> dict:
        return {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 2000},
            "learning_rate": {
                "type": "float",
                "default": 0.1,
                "min": 1e-4,
                "max": 1.0,
            },
            "minibatch_frac": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 1.0,
            },
            "col_sample": {
                "type": "float",
                "default": 1.0,
                "min": 0.1,
                "max": 1.0,
            },
        }
