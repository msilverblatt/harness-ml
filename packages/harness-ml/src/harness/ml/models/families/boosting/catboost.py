"""CatBoost model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.boosting.base import BoostingBase

NAME = "catboost"

try:
    import catboost as _cb  # noqa: F401
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False


class CatBoostModel(BoostingBase):
    name = "catboost"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages = ["catboost"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _CATBOOST_AVAILABLE:
            raise ImportError(
                "catboost is not installed. Install it with: pip install catboost"
            )
        import catboost as cb

        if task_type == "regression":
            return cb.CatBoostRegressor(**params)
        elif task_type == "multiclass":
            return cb.CatBoostClassifier(loss_function="MultiClass", **params)
        else:
            return cb.CatBoostClassifier(loss_function="Logloss", **params)

    def default_params(self, task_type: str) -> dict:
        base: dict[str, Any] = {
            "_task_type": task_type,
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": 0,
        }
        return base

    def param_schema(self) -> dict:
        return {
            "iterations": {"type": "int", "default": 100, "min": 10, "max": 2000},
            "depth": {"type": "int", "default": 6, "min": 1, "max": 16},
            "learning_rate": {"type": "float", "default": 0.1, "min": 1e-4, "max": 1.0},
            "l2_leaf_reg": {"type": "float", "default": 3.0, "min": 0.0, "max": 100.0},
            "bagging_temperature": {"type": "float", "default": 1.0, "min": 0.0, "max": 10.0},
            "subsample": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
        }
