"""Model adaptation parameters for multiclass classification."""

from __future__ import annotations

# Model-specific objective/loss function mappings for multiclass classification
OBJECTIVES: dict[str, dict] = {
    "xgboost": {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
    },
    "lightgbm": {
        "objective": "multiclass",
        "metric": "multi_logloss",
    },
    "catboost": {
        "loss_function": "MultiClass",
    },
    "hist_gbm": {
        "loss": "log_loss",
    },
    "logistic_regression": {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 1000,
    },
    "elastic_net": {},
    "mlp": {
        "loss": "cross_entropy",
        "output_activation": "softmax",
    },
    "random_forest": {
        "criterion": "gini",
    },
    "svm": {
        "probability": True,
        "decision_function_shape": "ovr",
    },
}
