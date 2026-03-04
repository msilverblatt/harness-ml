"""Built-in metrics — probability, classification, regression, and ensemble diagnostics."""
from __future__ import annotations

import numpy as np
from collections.abc import Callable
from itertools import combinations

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix as _sklearn_confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss as _sklearn_log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# MetricRegistry — task-type dispatch
# ---------------------------------------------------------------------------

class MetricRegistry:
    """Registry of metrics organized by ML task type."""

    _metrics: dict[str, dict[str, Callable]] = {}

    @classmethod
    def register(
        cls,
        task: str,
        name: str,
        fn: Callable,
        *,
        requires: list[str] | None = None,
    ) -> None:
        """Register a metric function for a task type.

        Args:
            task: Task type (binary, multiclass, regression, ranking,
                  survival, probabilistic)
            name: Metric name
            fn: Metric function taking (y_true, y_pred_or_prob, **kwargs)
            requires: Optional list of required kwargs beyond y_true/y_pred
        """
        if task not in cls._metrics:
            cls._metrics[task] = {}
        cls._metrics[task][name] = fn

    @classmethod
    def get(cls, task: str, name: str) -> Callable | None:
        """Retrieve a single metric function by task and name."""
        return cls._metrics.get(task, {}).get(name)

    @classmethod
    def compute_all(
        cls, task: str, y_true, y_pred, **kwargs
    ) -> dict[str, float]:
        """Compute all registered metrics for a task type."""
        results: dict[str, float] = {}
        for name, fn in cls._metrics.get(task, {}).items():
            try:
                val = fn(y_true, y_pred, **kwargs)
                # Some metrics return dicts (confusion_matrix) — skip float cast
                if isinstance(val, dict):
                    results[name] = val  # type: ignore[assignment]
                else:
                    results[name] = float(val)
            except Exception:
                results[name] = float("nan")
        return results

    @classmethod
    def list_metrics(cls, task: str | None = None) -> dict[str, list[str]]:
        """List available metrics, optionally filtered by task."""
        if task:
            return {task: list(cls._metrics.get(task, {}).keys())}
        return {t: list(m.keys()) for t, m in cls._metrics.items()}


# ---------------------------------------------------------------------------
# Probability / binary classification metrics
# ---------------------------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    """Mean squared error between true labels and predicted probabilities."""
    return float(brier_score_loss(y_true, y_prob))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    """Logarithmic loss (cross-entropy) between true labels and probabilities."""
    return float(_sklearn_log_loss(y_true, y_prob))


def accuracy(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """Accuracy after thresholding predicted probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(accuracy_score(y_true, y_pred))


def ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, **kwargs
) -> float:
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


def auc_roc(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    """Area under ROC curve."""
    return float(roc_auc_score(y_true, y_prob))


def f1(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """F1 score after thresholding predicted probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))


def precision(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """Precision after thresholding predicted probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(precision_score(y_true, y_pred, zero_division=0))


def recall(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """Recall (sensitivity) after thresholding predicted probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=0))


def mcc(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """Matthews correlation coefficient after thresholding."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(matthews_corrcoef(y_true, y_pred))


def auc_pr(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    """Area under precision-recall curve (average precision)."""
    return float(average_precision_score(y_true, y_prob))


def specificity(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """Specificity = TN / (TN + FP)."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true_arr = np.asarray(y_true)
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    denom = tn + fp
    if denom == 0:
        return 0.0
    return float(tn / denom)


def binary_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> dict:
    """Confusion matrix as dict with tp, tn, fp, fn."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true_arr = np.asarray(y_true)
    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def cohen_kappa(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, **kwargs
) -> float:
    """Cohen's kappa after thresholding predicted probabilities."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return float(cohen_kappa_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Multiclass classification metrics
# ---------------------------------------------------------------------------

def accuracy_multi(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Accuracy for multiclass predictions (argmax of probabilities or labels)."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(accuracy_score(y_true, y_pred_arr))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Macro-averaged F1 for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(f1_score(y_true, y_pred_arr, average="macro", zero_division=0))


def f1_micro(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Micro-averaged F1 for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(f1_score(y_true, y_pred_arr, average="micro", zero_division=0))


def f1_weighted(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Weighted-averaged F1 for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(f1_score(y_true, y_pred_arr, average="weighted", zero_division=0))


def precision_macro(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Macro-averaged precision for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(precision_score(y_true, y_pred_arr, average="macro", zero_division=0))


def recall_macro(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Macro-averaged recall for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(recall_score(y_true, y_pred_arr, average="macro", zero_division=0))


def confusion_matrix_multi(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs
) -> dict:
    """NxN confusion matrix returned as nested dict."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    cm = _sklearn_confusion_matrix(y_true, y_pred_arr)
    labels = sorted(set(np.asarray(y_true).ravel()) | set(y_pred_arr.ravel()))
    return {
        "matrix": cm.tolist(),
        "labels": [int(l) for l in labels],
    }


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> dict:
    """Per-class precision, recall, f1-score, support as parsed dict."""
    from sklearn.metrics import classification_report

    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return classification_report(
        y_true, y_pred_arr, output_dict=True, zero_division=0
    )


def log_loss_multi(y_true: np.ndarray, y_prob: np.ndarray, **kwargs) -> float:
    """Log loss for multiclass (expects probability matrix)."""
    return float(_sklearn_log_loss(y_true, y_prob))


def mcc_multi(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Matthews correlation coefficient for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(matthews_corrcoef(y_true, y_pred_arr))


def cohen_kappa_multi(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Cohen's kappa for multiclass."""
    y_pred_arr = np.asarray(y_pred)
    if y_pred_arr.ndim == 2:
        y_pred_arr = y_pred_arr.argmax(axis=1)
    return float(cohen_kappa_score(y_true, y_pred_arr))


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """R-squared (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean absolute percentage error (handles zeros by using sklearn impl)."""
    return float(mean_absolute_percentage_error(y_true, y_pred))


def median_ae(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Median absolute error."""
    return float(median_absolute_error(y_true, y_pred))


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Explained variance score."""
    return float(explained_variance_score(y_true, y_pred))


def mean_bias(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean bias (mean of y_pred - y_true)."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def quantile_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5, **kwargs
) -> float:
    """Pinball (quantile) loss for a given quantile."""
    q = kwargs.get("quantile", quantile)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residual = y_true_arr - y_pred_arr
    return float(np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual)))


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


# ---------------------------------------------------------------------------
# Backward-compatible dispatch table
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Register all metrics with MetricRegistry
# ---------------------------------------------------------------------------

# Binary classification
for _name, _fn in [
    ("brier", brier_score),
    ("log_loss", log_loss),
    ("accuracy", accuracy),
    ("ece", ece),
    ("auc_roc", auc_roc),
    ("f1", f1),
    ("precision", precision),
    ("recall", recall),
    ("mcc", mcc),
    ("auc_pr", auc_pr),
    ("specificity", specificity),
    ("confusion_matrix", binary_confusion_matrix),
    ("cohen_kappa", cohen_kappa),
]:
    MetricRegistry.register("binary", _name, _fn)

# Multiclass classification
for _name, _fn in [
    ("accuracy_multi", accuracy_multi),
    ("f1_macro", f1_macro),
    ("f1_micro", f1_micro),
    ("f1_weighted", f1_weighted),
    ("precision_macro", precision_macro),
    ("recall_macro", recall_macro),
    ("confusion_matrix_multi", confusion_matrix_multi),
    ("per_class_report", per_class_report),
    ("log_loss_multi", log_loss_multi),
    ("mcc_multi", mcc_multi),
    ("cohen_kappa_multi", cohen_kappa_multi),
]:
    MetricRegistry.register("multiclass", _name, _fn)

# Regression
for _name, _fn in [
    ("rmse", rmse),
    ("mae", mae),
    ("r_squared", r_squared),
    ("mape", mape),
    ("median_ae", median_ae),
    ("explained_variance", explained_variance),
    ("mean_bias", mean_bias),
    ("quantile_loss", quantile_loss),
]:
    MetricRegistry.register("regression", _name, _fn)
