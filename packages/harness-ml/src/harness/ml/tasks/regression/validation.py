"""Validation functions for regression targets and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.ml.tasks.protocol import ValidationResult


def validate_target(y: pd.Series) -> ValidationResult:
    """Validate that a target series is numeric.

    Parameters
    ----------
    y : pd.Series
        Target series to validate.

    Returns
    -------
    ValidationResult
        Validation result with messages for any issues found.
    """
    messages: list[str] = []

    try:
        arr = np.asarray(y, dtype=float)
    except (ValueError, TypeError):
        return ValidationResult(
            is_valid=False,
            messages=["Target contains non-numeric values"],
        )

    if np.isnan(arr).any():
        return ValidationResult(
            is_valid=False,
            messages=["Target contains NaN values"],
        )

    if np.all(arr == arr[0]):
        messages.append("All target values are identical")

    return ValidationResult(is_valid=True, messages=messages)


def validate_predictions(predictions: np.ndarray) -> ValidationResult:
    """Validate that predictions are numeric with no NaN values.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted values to validate.

    Returns
    -------
    ValidationResult
        Validation result with messages for any issues found.
    """
    try:
        arr = np.asarray(predictions, dtype=float)
    except (ValueError, TypeError):
        return ValidationResult(
            is_valid=False,
            messages=["Predictions contain non-numeric values"],
        )

    if np.isnan(arr).any():
        return ValidationResult(
            is_valid=False,
            messages=["Predictions contain NaN values"],
        )

    return ValidationResult(is_valid=True, messages=[])
