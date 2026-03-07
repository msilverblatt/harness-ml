"""Built-in metrics — probability, classification, regression, and ensemble diagnostics."""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    ndcg_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    confusion_matrix as _sklearn_confusion_matrix,
)
from sklearn.metrics import (
    log_loss as _sklearn_log_loss,
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
# Ranking metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, **kwargs) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    k_val = kwargs.get("k", k)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    # sklearn ndcg_score expects 2D arrays
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(1, -1)
        y_pred_arr = y_pred_arr.reshape(1, -1)
    return float(ndcg_score(y_true_arr, y_pred_arr, k=k_val))


def mean_average_precision(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean Average Precision — average of AP across queries.

    Expects y_true as binary relevance, y_pred as scores. For single-query
    data, returns the average precision for that query.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    # If 2D (multiple queries), compute AP per row and average
    if y_true_arr.ndim == 2:
        aps = []
        for row_true, row_pred in zip(y_true_arr, y_pred_arr):
            if row_true.sum() > 0:
                aps.append(float(average_precision_score(row_true, row_pred)))
        return float(np.mean(aps)) if aps else 0.0
    # Single query
    return float(average_precision_score(y_true_arr, y_pred_arr))


def mrr(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Mean Reciprocal Rank.

    For single-query: y_true is binary relevance, y_pred is predicted scores.
    Returns 1/rank of the first relevant item by predicted score order.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.ndim == 2:
        rrs = []
        for row_true, row_pred in zip(y_true_arr, y_pred_arr):
            order = np.argsort(-row_pred)
            for rank, idx in enumerate(order, 1):
                if row_true[idx] > 0:
                    rrs.append(1.0 / rank)
                    break
            else:
                rrs.append(0.0)
        return float(np.mean(rrs)) if rrs else 0.0
    # Single query
    order = np.argsort(-y_pred_arr)
    for rank, idx in enumerate(order, 1):
        if y_true_arr[idx] > 0:
            return 1.0 / rank
    return 0.0


def precision_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, **kwargs
) -> float:
    """Precision in the top-k predicted items."""
    k_val = kwargs.get("k", k)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    top_k_idx = np.argsort(-y_pred_arr)[:k_val]
    return float(y_true_arr[top_k_idx].sum() / k_val) if k_val > 0 else 0.0


def recall_at_k(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, **kwargs
) -> float:
    """Recall in the top-k predicted items."""
    k_val = kwargs.get("k", k)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    total_relevant = y_true_arr.sum()
    if total_relevant == 0:
        return 0.0
    top_k_idx = np.argsort(-y_pred_arr)[:k_val]
    return float(y_true_arr[top_k_idx].sum() / total_relevant)


def spearman_rank_corr(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs
) -> float:
    """Spearman rank correlation coefficient."""
    from scipy.stats import spearmanr

    corr, _ = spearmanr(y_true, y_pred)
    return float(corr)


# ---------------------------------------------------------------------------
# Survival metrics
# ---------------------------------------------------------------------------

def concordance_index(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs
) -> float:
    """Harrell's concordance index (C-index).

    Args:
        y_true: Observed survival times.
        y_pred: Predicted risk scores (higher = higher risk = shorter survival).
        event: Boolean/int array indicating whether the event was observed
               (1 = event, 0 = censored). Passed via kwargs.
    """
    event = kwargs.get("event")
    times = np.asarray(y_true, dtype=float)
    risk = np.asarray(y_pred, dtype=float)
    if event is not None:
        event_arr = np.asarray(event, dtype=bool)
    else:
        event_arr = np.ones(len(times), dtype=bool)

    concordant = 0
    discordant = 0
    tied_risk = 0
    n = len(times)

    for i in range(n):
        if not event_arr[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] > times[i]:
                if risk[j] < risk[i]:
                    concordant += 1
                elif risk[j] > risk[i]:
                    discordant += 1
                else:
                    tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return float((concordant + 0.5 * tied_risk) / total)


def time_dependent_brier(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs
) -> float:
    """Brier score adapted for censored survival data at a given time horizon.

    Args:
        y_true: Observed survival times.
        y_pred: Predicted survival probabilities at time_horizon
                (prob of surviving past time_horizon).
        event: Boolean/int array indicating event observed. Passed via kwargs.
        time_horizon: The time point at which to evaluate. Passed via kwargs.
    """
    event = kwargs.get("event")
    time_horizon = kwargs.get("time_horizon")
    if time_horizon is None:
        # Default to median observed time
        time_horizon = float(np.median(y_true))

    times = np.asarray(y_true, dtype=float)
    surv_prob = np.asarray(y_pred, dtype=float)
    if event is not None:
        event_arr = np.asarray(event, dtype=bool)
    else:
        event_arr = np.ones(len(times), dtype=bool)

    n = len(times)
    bs = 0.0
    count = 0
    for i in range(n):
        if times[i] <= time_horizon and event_arr[i]:
            # Event occurred before horizon: true outcome = 0 (did not survive)
            bs += surv_prob[i] ** 2
            count += 1
        elif times[i] > time_horizon:
            # Survived past horizon: true outcome = 1
            bs += (1.0 - surv_prob[i]) ** 2
            count += 1
        # Censored before horizon: skip (no information)

    if count == 0:
        return 0.0
    return float(bs / count)


def cumulative_incidence_auc(
    y_true: np.ndarray, y_pred: np.ndarray, **kwargs
) -> float:
    """AUC for cumulative incidence at a given time horizon.

    Args:
        y_true: Observed survival times.
        y_pred: Predicted cumulative incidence (1 - survival probability)
                at time_horizon.
        event: Boolean/int array indicating event observed. Passed via kwargs.
        time_horizon: The time point at which to evaluate. Passed via kwargs.
    """
    event = kwargs.get("event")
    time_horizon = kwargs.get("time_horizon")
    if time_horizon is None:
        time_horizon = float(np.median(y_true))

    times = np.asarray(y_true, dtype=float)
    pred_inc = np.asarray(y_pred, dtype=float)
    if event is not None:
        event_arr = np.asarray(event, dtype=bool)
    else:
        event_arr = np.ones(len(times), dtype=bool)

    # Binary label: did the event occur before or at time_horizon?
    binary_label = ((times <= time_horizon) & event_arr).astype(int)

    # Filter out censored-before-horizon cases (ambiguous)
    usable = (times > time_horizon) | event_arr
    if usable.sum() < 2 or binary_label[usable].sum() == 0:
        return float("nan")

    return float(roc_auc_score(binary_label[usable], pred_inc[usable]))


# ---------------------------------------------------------------------------
# Probabilistic metrics
# ---------------------------------------------------------------------------

def crps(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Continuous Ranked Probability Score.

    For deterministic point predictions, CRPS reduces to MAE.
    For ensemble predictions (y_pred is 2D: samples x observations), computes
    the full CRPS via the representation:
        CRPS = E|X - y| - 0.5 * E|X - X'|
    where X, X' are independent draws from the predictive distribution.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_pred_arr.ndim == 1:
        # Deterministic prediction: CRPS = MAE
        return float(np.mean(np.abs(y_pred_arr - y_true_arr)))

    # Ensemble predictions: shape (n_samples, n_obs)
    n_obs = y_pred_arr.shape[1]
    scores = np.zeros(n_obs)
    for j in range(n_obs):
        ens = y_pred_arr[:, j]
        # E|X - y|
        term1 = np.mean(np.abs(ens - y_true_arr[j]))
        # E|X - X'| via double sum
        diffs = np.abs(ens[:, None] - ens[None, :])
        term2 = np.mean(diffs)
        scores[j] = term1 - 0.5 * term2
    return float(np.mean(scores))


def pit_histogram_data(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10, **kwargs
) -> dict:
    """PIT (Probability Integral Transform) histogram bin counts.

    For binary/probabilistic predictions: PIT value = predicted prob of
    the observed outcome. Returns histogram of PIT values.

    Args:
        y_true: True outcomes (0/1 for binary, or continuous values).
        y_pred: Predicted probabilities or CDF values at y_true.
        n_bins: Number of histogram bins.
    """
    n_bins_val = kwargs.get("n_bins", n_bins)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    # For binary outcomes, PIT value is p if y=1 else 1-p
    if set(np.unique(y_true_arr)).issubset({0.0, 1.0}):
        pit_values = np.where(y_true_arr == 1, y_pred_arr, 1.0 - y_pred_arr)
    else:
        # Assume y_pred are already CDF values at y_true
        pit_values = y_pred_arr

    counts, bin_edges = np.histogram(pit_values, bins=n_bins_val, range=(0.0, 1.0))
    return {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist(),
    }


def sharpness(y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
    """Sharpness — variance of predicted probabilities.

    Higher sharpness means the model is more confident (predictions farther
    from 0.5 on average for binary classification).
    """
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.var(y_pred_arr))


def coverage_at_level(
    y_true: np.ndarray, y_pred: np.ndarray, level: float = 0.9, **kwargs
) -> float:
    """Fraction of true values within prediction intervals.

    Args:
        y_true: True values.
        y_pred: 2D array of shape (n, 2) with [lower, upper] bounds per obs,
                or pass lower and upper via kwargs.
        level: Nominal coverage level (used for documentation/naming).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    lower = kwargs.get("lower")
    upper = kwargs.get("upper")

    if lower is not None and upper is not None:
        lower_arr = np.asarray(lower, dtype=float)
        upper_arr = np.asarray(upper, dtype=float)
    elif y_pred_arr.ndim == 2 and y_pred_arr.shape[1] == 2:
        lower_arr = y_pred_arr[:, 0]
        upper_arr = y_pred_arr[:, 1]
    else:
        return float("nan")

    covered = ((y_true_arr >= lower_arr) & (y_true_arr <= upper_arr)).mean()
    return float(covered)


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

# Ranking
for _name, _fn in [
    ("ndcg_at_k", ndcg_at_k),
    ("mean_average_precision", mean_average_precision),
    ("mrr", mrr),
    ("precision_at_k", precision_at_k),
    ("recall_at_k", recall_at_k),
    ("spearman_rank_corr", spearman_rank_corr),
]:
    MetricRegistry.register("ranking", _name, _fn)

# Survival
for _name, _fn in [
    ("concordance_index", concordance_index),
    ("time_dependent_brier", time_dependent_brier),
    ("cumulative_incidence_auc", cumulative_incidence_auc),
]:
    MetricRegistry.register("survival", _name, _fn, requires=["event", "time_horizon"])

# Probabilistic
for _name, _fn in [
    ("crps", crps),
    ("pit_histogram_data", pit_histogram_data),
    ("sharpness", sharpness),
    ("coverage_at_level", coverage_at_level),
]:
    MetricRegistry.register("probabilistic", _name, _fn)
