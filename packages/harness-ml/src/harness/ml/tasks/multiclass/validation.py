"""Validation functions for multiclass targets and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from harness.ml.tasks.protocol import ValidationResult


def validate_target(y: pd.Series) -> ValidationResult:
    """Validate that a target series contains integer class labels with at least 3 classes.

    Parameters
    ----------
    y : pd.Series
        Target series to validate.

    Returns
    -------
    ValidationResult
        Validation result with messages for any issues found.
    """
    try:
        arr = np.asarray(y)
        # Check that values are integer-like
        arr_float = arr.astype(float)
        if not np.all(arr_float == arr_float.astype(int)):
            return ValidationResult(
                is_valid=False,
                messages=["Target must contain integer class labels"],
            )
        arr_int = arr_float.astype(int)
    except (ValueError, TypeError):
        return ValidationResult(
            is_valid=False,
            messages=["Target contains non-numeric values"],
        )

    n_classes = len(np.unique(arr_int))
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
                f"Prediction rows must sum to 1.0; {len(bad)} row(s) do not "
                f"(e.g. row {bad[0]} sums to {row_sums[bad[0]]:.4f})"
            ],
        )

    return ValidationResult(is_valid=True, messages=[])
