"""LightGBM model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.boosting.base import BoostingBase

NAME = "lightgbm"

try:
    import lightgbm as _lgb  # noqa: F401
    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False


class LightGBMModel(BoostingBase):
    name = "lightgbm"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages = ["lightgbm"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _LIGHTGBM_AVAILABLE:
            raise ImportError(
                "lightgbm is not installed. Install it with: pip install lightgbm"
            )
        import lightgbm as lgb

        if task_type == "regression":
            return lgb.LGBMRegressor(**params)
        elif task_type == "multiclass":
            return lgb.LGBMClassifier(objective="multiclass", **params)
        else:
            return lgb.LGBMClassifier(objective="binary", **params)

    def default_params(self, task_type: str) -> dict:
        base: dict[str, Any] = {
            "_task_type": task_type,
            "n_estimators": 100,
            "max_depth": -1,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": -1,
        }
        return base

    def param_schema(self) -> dict:
        return {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 2000},
            "max_depth": {"type": "int", "default": -1, "min": -1, "max": 15},
            "learning_rate": {"type": "float", "default": 0.1, "min": 1e-4, "max": 1.0},
            "num_leaves": {"type": "int", "default": 31, "min": 2, "max": 256},
            "subsample": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            "colsample_bytree": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            "reg_alpha": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
            "reg_lambda": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
        }
