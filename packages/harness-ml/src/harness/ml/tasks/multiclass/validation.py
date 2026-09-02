"""Validation functions for multiclass targets and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from harness.ml.tasks.protocol import ValidationResult


def validate_target(y: pd.Series) -> ValidationResult:
    """Validate integer or string class labels with at least three classes."""
    if y.isna().any():
        return ValidationResult(
            is_valid=False,
            messages=["Target contains missing values"],
        )

    arr = np.asarray(y)
    if np.issubdtype(arr.dtype, np.number):
        try:
            arr_float = arr.astype(float)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                messages=["Target contains invalid numeric values"],
            )
        if not np.isfinite(arr_float).all():
            return ValidationResult(
                is_valid=False,
                messages=["Target contains non-finite values"],
            )
        if not np.all(arr_float == arr_float.astype(int)):
            return ValidationResult(
                is_valid=False,
                messages=["Numeric targets must contain integer class labels"],
            )
    elif not all(isinstance(value, str) for value in arr):
        return ValidationResult(
            is_valid=False,
            messages=["Target labels must be consistently strings or integers"],
        )

    try:
        n_classes = len(np.unique(arr))
    except TypeError:
        return ValidationResult(
            is_valid=False,
            messages=["Target labels must be mutually comparable"],
        )
    if n_classes < 3:
        return ValidationResult(
            is_valid=False,
            messages=[
                f"Multiclass task requires at least 3 unique classes, found {n_classes}"
            ],
        )

    return ValidationResult(is_valid=True, messages=[])


def validate_predictions(predictions: np.ndarray) -> ValidationResult:
    """Validate that predictions are a 2D array of probabilities with rows summing to ~1.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class probabilities to validate.

    Returns
    -------
    ValidationResult
        Validation result with messages for any issues found.
    """
    arr = np.asarray(predictions, dtype=float)

    if arr.ndim != 2:
        return ValidationResult(
            is_valid=False,
            messages=[
                f"Predictions must be a 2D array of class probabilities, got shape {arr.shape}"
            ],
        )

    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad = np.where(~np.isclose(row_sums, 1.0, atol=1e-3))[0]
        return ValidationResult(
            is_valid=False,
            messages=[
                f"Prediction rows must sum to 1.0; {len(bad)} row(s) do not"
            ],
        )

    return ValidationResult(is_valid=True, messages=[])
