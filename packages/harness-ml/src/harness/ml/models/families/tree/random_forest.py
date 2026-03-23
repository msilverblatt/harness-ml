"""Random forest model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.tree.base import TreeBase

NAME = "random_forest"


class RandomForestModel(TreeBase):
    name = "random_forest"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages: list[str] = []

    def _create_model(self, params: dict, task_type: str) -> Any:
        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**params)
        else:
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**params)

    def default_params(self, task_type: str) -> dict:
        return {
            "_task_type": task_type,
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        }

    def param_schema(self) -> dict:
        return {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 2000},
            "max_depth": {"type": "int", "default": 10, "min": 1, "max": 100},
            "min_samples_split": {"type": "int", "default": 5, "min": 2, "max": 50},
            "min_samples_leaf": {"type": "int", "default": 2, "min": 1, "max": 50},
            "max_features": {
                "type": "str",
                "default": "sqrt",
                "choices": ["sqrt", "log2", None],
            },
        }
