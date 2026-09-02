"""SVM model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.kernel.base import KernelBase

NAME = "svm"


class SVMModel(KernelBase):
    name = "svm"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages: list[str] = []

    def _create_model(self, params: dict, task_type: str) -> Any:
        if task_type == "regression":
            from sklearn.svm import SVR

            return SVR(**params)
        else:
            from sklearn.svm import SVC

            return SVC(probability=True, **params)

    def default_params(self, task_type: str) -> dict:
        base: dict[str, Any] = {
            "_task_type": task_type,
            "C": 1.0,
            "kernel": "rbf",
        }
        if task_type == "regression":
            base["epsilon"] = 0.1
        return base

    def param_schema(self) -> dict:
        return {
            "C": {"type": "float", "default": 1.0, "min": 1e-6, "max": 1e6},
            "kernel": {
                "type": "str",
                "default": "rbf",
                "choices": ["linear", "poly", "rbf", "sigmoid"],
            },
            "gamma": {
                "type": "str",
                "default": "scale",
                "choices": ["scale", "auto"],
            },
            "epsilon": {"type": "float", "default": 0.1, "min": 0.0, "max": 10.0},
        }
