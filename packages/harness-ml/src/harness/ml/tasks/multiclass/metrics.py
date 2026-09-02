"""Metric computation functions for multiclass classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Accuracy using argmax of class probabilities."""
    y_class = np.argmax(y_pred, axis=1)
    return float(accuracy_score(y_true, y_class))


def compute_f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 score."""
    y_class = np.argmax(y_pred, axis=1)
    return float(f1_score(y_true, y_class, average="macro", zero_division=0))


def compute_f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Micro-averaged F1 score."""
    y_class = np.argmax(y_pred, axis=1)
    return float(f1_score(y_true, y_class, average="micro", zero_division=0))


def compute_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted F1 score."""
    y_class = np.argmax(y_pred, axis=1)
    return float(f1_score(y_true, y_class, average="weighted", zero_division=0))


def compute_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multiclass log loss."""
    return float(log_loss(y_true, y_pred))


def compute_precision_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged precision."""
    y_class = np.argmax(y_pred, axis=1)
    return float(precision_score(y_true, y_class, average="macro", zero_division=0))


def compute_recall_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged recall."""
    y_class = np.argmax(y_pred, axis=1)
    return float(recall_score(y_true, y_class, average="macro", zero_division=0))


METRIC_FUNCTIONS = {
    "accuracy": compute_accuracy,
    "f1_macro": compute_f1_macro,
    "f1_micro": compute_f1_micro,
    "f1_weighted": compute_f1_weighted,
    "log_loss": compute_log_loss,
    "precision_macro": compute_precision_macro,
    "recall_macro": compute_recall_macro,
}
