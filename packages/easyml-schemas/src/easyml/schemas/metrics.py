"""Built-in metrics — probability, classification, regression, and ensemble diagnostics."""
from __future__ import annotations

import numpy as np
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss as _sklearn_log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
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


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Classification metrics (additional)
# ---------------------------------------------------------------------------

def auc_roc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area under ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def f1(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """F1 score after thresholding predicted probabilities."""
    y_pred = (y_prob >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Ensemble diagnostics
# ---------------------------------------------------------------------------

def model_correlations(predictions: dict[str, np.ndarray]) -> dict[str, float]:
    """Pearson correlation for all model prediction pairs.

    Returns dict with keys like ``"model_a|model_b"`` and float correlation
    values.
    """
    names = sorted(predictions.keys())
    result: dict[str, float] = {}
    for a, b in combinations(names, 2):
        corr = float(np.corrcoef(predictions[a], predictions[b])[0, 1])
        result[f"{a}|{b}"] = corr
    return result


# Dispatch table mapping metric name strings to callables.
_METRIC_DISPATCH: dict[str, callable] = {
    "brier": brier_score,
    "log_loss": log_loss,
    "accuracy": accuracy,
    "ece": ece,
    "rmse": rmse,
    "mae": mae,
    "r_squared": r_squared,
    "auc_roc": auc_roc,
    "f1": f1,
}


def model_audit(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Per-model metric evaluation.

    Returns ``{model_name: {metric_name: value}}`` for each model in
    *predictions* and each metric name in *metrics*.
    """
    result: dict[str, dict[str, float]] = {}
    for model_name, y_prob in predictions.items():
        model_metrics: dict[str, float] = {}
        for metric_name in metrics:
            fn = _METRIC_DISPATCH[metric_name]
            model_metrics[metric_name] = fn(y_true, y_prob)
        result[model_name] = model_metrics
    return result
