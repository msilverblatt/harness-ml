"""Metric computation functions for regression."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


def compute_median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute error."""
    return float(median_absolute_error(y_true, y_pred))


def compute_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance score."""
    return float(explained_variance_score(y_true, y_pred))


METRIC_FUNCTIONS = {
    "rmse": compute_rmse,
    "mae": compute_mae,
    "r2": compute_r2,
    "mape": compute_mape,
    "median_ae": compute_median_ae,
    "explained_variance": compute_explained_variance,
}
