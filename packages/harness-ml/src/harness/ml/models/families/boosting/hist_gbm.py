"""Histogram-based Gradient Boosting model wrapper (sklearn)."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.boosting.base import BoostingBase

NAME = "hist_gbm"


class HistGBMModel(BoostingBase):
    name = "hist_gbm"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages: list[str] = []

    def _create_model(self, params: dict, task_type: str) -> Any:
        if task_type == "regression":
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(**params)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(**params)

    def fit(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any | None,
        y_val: Any | None,
        params: dict,
    ) -> Any:
        # HistGradientBoosting uses validation_fraction / early_stopping natively,
        # not an eval_set kwarg, so we override fit to avoid passing eval_set.
        from harness.ml.models.protocol import FitResult

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

    def default_params(self, task_type: str) -> dict:
        base: dict[str, Any] = {
            "_task_type": task_type,
            "max_iter": 100,
            "max_depth": None,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "random_state": 42,
        }
        return base

    def param_schema(self) -> dict:
        return {
            "max_iter": {"type": "int", "default": 100, "min": 10, "max": 2000},
            "max_depth": {"type": "int", "default": None, "min": 1, "max": 15},
            "learning_rate": {"type": "float", "default": 0.1, "min": 1e-4, "max": 1.0},
            "max_leaf_nodes": {"type": "int", "default": 31, "min": 2, "max": 256},
            "l2_regularization": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
            "min_samples_leaf": {"type": "int", "default": 20, "min": 1, "max": 200},
        }
