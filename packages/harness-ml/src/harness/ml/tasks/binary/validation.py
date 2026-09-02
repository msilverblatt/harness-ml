"""Validation functions for binary classification targets and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness.ml.tasks.protocol import ValidationResult


def validate_target(y: pd.Series) -> ValidationResult:
    """Validate that a target series contains only binary (0/1) values.

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
    arr = np.asarray(y, dtype=float)

    # Check for NaN
    if np.isnan(arr).any():
        return ValidationResult(
            is_valid=False,
            messages=["Target contains NaN values"],
        )

    # Check that all values are 0 or 1
    unique_vals = set(np.unique(arr))
    if not unique_vals.issubset({0.0, 1.0}):
        return ValidationResult(
            is_valid=False,
            messages=[
                f"Target contains non-binary values: {sorted(unique_vals - {0.0, 1.0})}"
            ],
        )

    # Warn if only one class present
    if len(unique_vals) < 2:
        messages.append("Only one class present in target")

    return ValidationResult(is_valid=True, messages=messages)


def validate_predictions(predictions: pd.Series) -> ValidationResult:
    """Validate that predictions are probabilities in [0, 1].

    Parameters
    ----------
    predictions : pd.Series
        Predicted probabilities to validate.

    Returns
    -------
    ValidationResult
        Validation result with messages for any issues found.
    """
    arr = np.asarray(predictions, dtype=float)

    if np.isnan(arr).any():
        return ValidationResult(
            is_valid=False,
            messages=["Predictions contain NaN values"],
        )

    if arr.min() < 0.0 or arr.max() > 1.0:
        return ValidationResult(
            is_valid=False,
            messages=[
                f"Predictions out of range [0, 1]: min={arr.min():.4f}, max={arr.max():.4f}"
            ],
        )

    return ValidationResult(is_valid=True, messages=[])
