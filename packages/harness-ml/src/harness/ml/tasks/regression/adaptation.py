"""Model adaptation parameters for regression."""

from __future__ import annotations

# Model-specific objective/loss function mappings for regression
OBJECTIVES: dict[str, dict] = {
    "xgboost": {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    },
    "lightgbm": {
        "objective": "regression",
        "metric": "rmse",
    },
    "catboost": {
        "loss_function": "RMSE",
    },
    "hist_gbm": {},
    "logistic_regression": {},  # Not applicable but listed for completeness
    "elastic_net": {},
    "mlp": {
        "loss": "mse",
        "output_dim": 1,
        "output_activation": "none",
    },
    "random_forest": {},
    "svm": {},
}
