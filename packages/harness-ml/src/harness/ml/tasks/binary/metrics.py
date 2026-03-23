"""Metric computation functions for binary classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_brier(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Brier score (mean squared error of probabilities)."""
    return float(brier_score_loss(y_true, y_pred))


def compute_log_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Logarithmic loss. Returns nan if only one class present."""
    if len(y_true.unique()) < 2:
        return float("nan")
    return float(log_loss(y_true, y_pred))


def compute_auroc(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Area under the ROC curve. Returns nan if only one class present."""
    if len(y_true.unique()) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def compute_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Accuracy at threshold 0.5."""
    y_binary = (y_pred >= 0.5).astype(int)
    return float(accuracy_score(y_true, y_binary))


def compute_f1(y_true: pd.Series, y_pred: pd.Series) -> float:
    """F1 score at threshold 0.5."""
    y_binary = (y_pred >= 0.5).astype(int)
    return float(f1_score(y_true, y_binary, zero_division=0))


def compute_precision(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Precision at threshold 0.5."""
    y_binary = (y_pred >= 0.5).astype(int)
    return float(precision_score(y_true, y_binary, zero_division=0))


def compute_recall(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Recall at threshold 0.5."""
    y_binary = (y_pred >= 0.5).astype(int)
    return float(recall_score(y_true, y_binary, zero_division=0))


def compute_ece(y_true: pd.Series, y_pred: pd.Series, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins, weighted by bin size."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_pred_arr >= low) & (y_pred_arr < high)
        else:
            # Last bin includes right edge
            mask = (y_pred_arr >= low) & (y_pred_arr <= high)

        bin_count = mask.sum()
        if bin_count == 0:
            continue

        bin_acc = y_true_arr[mask].mean()
        bin_conf = y_pred_arr[mask].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


METRIC_FUNCTIONS = {
    "brier": compute_brier,
    "log_loss": compute_log_loss,
    "auroc": compute_auroc,
    "accuracy": compute_accuracy,
    "f1": compute_f1,
    "precision": compute_precision,
    "recall": compute_recall,
    "ece": compute_ece,
}
