"""Model adaptation parameters for binary classification."""

from __future__ import annotations

# Model-specific objective/loss function mappings for binary classification
OBJECTIVES: dict[str, dict] = {
    "xgboost": {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
    },
    "lightgbm": {
        "objective": "binary",
        "metric": "binary_logloss",
    },
    "catboost": {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
    },
    "logistic_regression": {
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "random_forest": {
        "criterion": "gini",
    },
    "svm": {
        "probability": True,
    },
    "hist_gbm": {
        "loss": "log_loss",
    },
    "mlp": {
        "loss": "log_loss",
    },
}

# Default hyperparameters for binary classification tasks
DEFAULT_PARAMS: dict[str, object] = {
    "threshold": 0.5,
    "pos_label": 1,
}
