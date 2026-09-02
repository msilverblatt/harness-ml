"""XGBoost model wrapper."""

from __future__ import annotations

from typing import Any

from harness.ml.models.families.boosting.base import BoostingBase

NAME = "xgboost"

try:
    import xgboost as _xgb  # noqa: F401
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False


class XGBoostModel(BoostingBase):
    name = "xgboost"
    supports_tasks = ["binary", "multiclass", "regression"]
    requires_packages = ["xgboost"]

    def _create_model(self, params: dict, task_type: str) -> Any:
        if not _XGBOOST_AVAILABLE:
            raise ImportError(
                "xgboost is not installed. Install it with: pip install xgboost"
            )
        import xgboost as xgb

        if task_type == "regression":
            return xgb.XGBRegressor(**params)
        elif task_type == "multiclass":
            return xgb.XGBClassifier(objective="multi:softprob", **params)
        else:
            return xgb.XGBClassifier(objective="binary:logistic", **params)

    def default_params(self, task_type: str) -> dict:
        base: dict[str, Any] = {
            "_task_type": task_type,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "verbosity": 0,
        }
        return base

    def param_schema(self) -> dict:
        return {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 2000},
            "max_depth": {"type": "int", "default": 6, "min": 1, "max": 15},
            "learning_rate": {"type": "float", "default": 0.1, "min": 1e-4, "max": 1.0},
            "subsample": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            "colsample_bytree": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            "reg_alpha": {"type": "float", "default": 0.0, "min": 0.0, "max": 100.0},
            "reg_lambda": {"type": "float", "default": 1.0, "min": 0.0, "max": 100.0},
        }
