"""Logistic regression model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.linear.base import LinearBase

NAME = "logistic"


class LogisticModel(LinearBase):
    name = "logistic"
    supports_tasks = ["binary", "multiclass"]
    requires_packages: list[str] = []

    def _create_model(self, params: dict, task_type: str) -> Any:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(**params)

    def default_params(self, task_type: str) -> dict:
        base = {
            "_task_type": task_type,
            "max_iter": 1000,
            "solver": "lbfgs",
        }
        if task_type == "multiclass":
            base["multi_class"] = "multinomial"
        return base

    def param_schema(self) -> dict:
        return {
            "C": {"type": "float", "default": 1.0, "min": 1e-6, "max": 1e6},
            "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 10000},
            "solver": {
                "type": "str",
                "default": "lbfgs",
                "choices": ["lbfgs", "liblinear", "saga"],
            },
            "penalty": {
                "type": "str",
                "default": "l2",
                "choices": ["l1", "l2", "elasticnet", None],
            },
        }
