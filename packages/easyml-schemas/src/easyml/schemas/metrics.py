"""Built-in metrics — probability, classification, regression, and ensemble diagnostics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss as _sklearn_log_loss,
)


# ---------------------------------------------------------------------------
# Probability metrics
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between true labels and predicted probabilities."""
    return float(brier_score_loss(y_true, y_prob))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Logarithmic loss (cross-entropy) between true labels and probabilities."""
    return float(_sklearn_log_loss(y_true, y_prob))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """Accuracy after thresholding predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — weighted average of per-bin |mean_pred - actual_freq|."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    if total == 0:
        return 0.0

    weighted_error = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            # Include right edge in last bin
            mask = (y_prob >= lo) & (y_prob <= hi)

        count = mask.sum()
        if count == 0:
            continue

        mean_pred = y_prob[mask].mean()
        actual_freq = y_true[mask].mean()
        weighted_error += (count / total) * abs(mean_pred - actual_freq)

    return float(weighted_error)


def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Return per-bin calibration statistics as a list of dicts."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)

        count = int(mask.sum())
        if count == 0:
            continue

        table.append({
            "bin_start": float(lo),
            "bin_end": float(hi),
            "mean_predicted": float(y_prob[mask].mean()),
            "actual_accuracy": float(y_true[mask].mean()),
            "count": count,
        })

    return table
