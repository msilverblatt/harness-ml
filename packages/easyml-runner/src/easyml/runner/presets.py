"""Model preset definitions for common model types.

Presets provide sensible default configurations so users can add models
without specifying every parameter.  Explicit overrides always win.
"""
from __future__ import annotations

import copy
from typing import Any


_PRESETS: dict[str, dict[str, Any]] = {
    "xgboost_classifier": {
        "type": "xgboost",
        "mode": "classifier",
        "params": {
            "max_depth": 4,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "early_stopping_rounds": 50,
        },
    },
    "xgboost_regressor": {
        "type": "xgboost_regression",
        "mode": "regressor",
        "prediction_type": "margin",
        "params": {
            "max_depth": 3,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    "logistic_regression": {
        "type": "logistic_regression",
        "mode": "classifier",
        "params": {
            "C": 1.0,
            "max_iter": 1000,
        },
    },
    "catboost_classifier": {
        "type": "catboost",
        "mode": "classifier",
        "params": {
            "depth": 6,
            "learning_rate": 0.03,
            "iterations": 1000,
            "verbose": 0,
        },
    },
    "lightgbm_classifier": {
        "type": "lightgbm",
        "mode": "classifier",
        "params": {
            "max_depth": 5,
            "learning_rate": 0.01,
            "n_estimators": 1000,
            "verbose": -1,
        },
    },
    "mlp_classifier": {
        "type": "mlp",
        "mode": "classifier",
        "n_seeds": 3,
        "params": {
            "hidden_layers": [128, 64],
            "dropout": 0.3,
            "learning_rate": 0.001,
            "epochs": 100,
        },
    },
    "mlp_regressor": {
        "type": "mlp",
        "mode": "regressor",
        "prediction_type": "margin",
        "n_seeds": 5,
        "params": {
            "hidden_layers": [128, 64],
            "dropout": 0.4,
            "learning_rate": 0.001,
            "epochs": 150,
        },
    },
}


def list_presets() -> list[str]:
    """Return available preset names."""
    return sorted(_PRESETS.keys())


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of preset config.

    Raises ``KeyError`` if *name* is not a known preset.
    """
    if name not in _PRESETS:
        raise KeyError(
            f"Unknown preset '{name}'. Available: {', '.join(sorted(_PRESETS))}"
        )
    return copy.deepcopy(_PRESETS[name])


def apply_preset(
    name: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get a preset and merge *overrides* on top.

    Top-level keys from *overrides* replace the preset value.  For the
    ``params`` key, the preset params are updated (not replaced) with
    override params so callers can tweak individual hyperparameters
    without re-specifying the whole dict.
    """
    result = get_preset(name)
    if not overrides:
        return result

    for key, value in overrides.items():
        if key == "params" and isinstance(value, dict) and "params" in result:
            result["params"].update(value)
        else:
            result[key] = value

    return result
