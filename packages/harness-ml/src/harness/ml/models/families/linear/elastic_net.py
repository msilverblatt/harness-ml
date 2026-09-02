"""Elastic net model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.linear.base import LinearBase

NAME = "elastic_net"


class ElasticNetModel(LinearBase):
    name = "elastic_net"
    supports_tasks = ["binary", "regression"]
    requires_packages: list[str] = []

    def _create_model(self, params: dict, task_type: str) -> Any:
        if task_type == "regression":
            from sklearn.linear_model import ElasticNet

            return ElasticNet(**params)
        else:
            from sklearn.linear_model import LogisticRegression

            l1_ratio = params.pop("l1_ratio", 0.5)
            return LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                **params,
            )

    def default_params(self, task_type: str) -> dict:
        base: dict[str, Any] = {"_task_type": task_type}
        if task_type == "regression":
            base["alpha"] = 1.0
            base["l1_ratio"] = 0.5
            base["max_iter"] = 1000
        else:
            base["C"] = 1.0
            base["l1_ratio"] = 0.5
            base["max_iter"] = 1000
        return base

    def param_schema(self) -> dict:
        return {
            "alpha": {"type": "float", "default": 1.0, "min": 1e-6, "max": 1e6},
            "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "C": {"type": "float", "default": 1.0, "min": 1e-6, "max": 1e6},
            "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 10000},
        }
