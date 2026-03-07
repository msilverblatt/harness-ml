"""Diagnostics and metrics computation for model evaluation.

Provides Brier score, ECE, calibration curve, and pooled/per-fold
metrics computation for backtest evaluation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities).

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.

    Returns
    -------
    float
        Brier score. Lower is better. Range: [0, 1].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins predictions into n_bins equal-width bins and computes the
    weighted average of |mean_predicted - mean_actual| per bin.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        ECE. Lower is better.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    if total == 0:
        return 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            # Include right edge in last bin
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_pred = y_prob[mask].mean()
        avg_actual = y_true[mask].mean()
        ece += (n_in_bin / total) * abs(avg_pred - avg_actual)

    return float(ece)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve (reliability diagram data).

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    tuple of (mean_predicted, mean_actual, bin_counts)
        Arrays of length <= n_bins (bins with no samples are excluded).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    mean_predicted = []
    mean_actual = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        mean_predicted.append(float(y_prob[mask].mean()))
        mean_actual.append(float(y_true[mask].mean()))
        bin_counts.append(int(n_in_bin))

    return (
        np.array(mean_predicted),
        np.array(mean_actual),
        np.array(bin_counts),
    )


def evaluate_fold_predictions(
    preds: pd.DataFrame,
    actuals: dict[str, int],
    fold_id: int,
    target_column: str = "result",
) -> list[dict]:
    """Compute per-model metrics for a single fold.

    Parameters
    ----------
    preds : pd.DataFrame
        Prediction DataFrame with prob_{model_name} columns and
        a 'result' column (binary outcome). If 'result' is not present,
        uses actuals dict with matchup keys.
    actuals : dict[str, int]
        Mapping of matchup identifier -> binary outcome. Used as
        fallback if 'result' column is not in preds.
    fold_id : int
        Fold identifier for the output.

    Returns
    -------
    list of dict
        Each dict has: model, fold, accuracy, brier_score, ece, log_loss.
    """
    # Determine ground truth
    if target_column in preds.columns:
        y_true = preds[target_column].values.astype(float)
    elif actuals:
        # Use actuals dict — assumes index or row order matches
        y_true = np.array(list(actuals.values()), dtype=float)
    else:
        raise ValueError(f"No ground truth: '{target_column}' column missing and actuals is empty")

    # Find prob_* columns
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]

    results = []
    for col in prob_cols:
        model_name = col.replace("prob_", "", 1)
        y_prob = preds[col].values.astype(float)

        # Skip if all NaN
        valid = ~np.isnan(y_prob)
        if valid.sum() == 0:
            continue

        y_t = y_true[valid]
        y_p = y_prob[valid]

        brier = compute_brier_score(y_t, y_p)
        ece = compute_ece(y_t, y_p)
        accuracy = _compute_accuracy(y_t, y_p)
        logloss = _compute_log_loss(y_t, y_p)

        results.append({
            "model": model_name,
            "fold": fold_id,
            "accuracy": accuracy,
            "brier_score": brier,
            "ece": ece,
            "log_loss": logloss,
        })

    return results


def compute_pooled_metrics(
    fold_predictions: list[pd.DataFrame],
    target_column: str = "result",
) -> dict[str, dict]:
    """Compute pooled metrics across all folds.

    Pooled = computed on concatenated predictions (more stable than
    averaging per-fold metrics).

    Parameters
    ----------
    fold_predictions : list of pd.DataFrame
        Each DataFrame has prob_{model_name} columns and a 'result' column.

    Returns
    -------
    dict mapping model_name -> metric dict
        Each metric dict has: accuracy, brier_score, ece, log_loss, n_samples.
    """
    if not fold_predictions:
        return {}

    combined = pd.concat(fold_predictions, ignore_index=True)

    if target_column not in combined.columns:
        raise ValueError(f"'{target_column}' column required in prediction DataFrames")

    y_true = combined[target_column].values.astype(float)

    prob_cols = [c for c in combined.columns if c.startswith("prob_")]

    metrics: dict[str, dict] = {}
    for col in prob_cols:
        model_name = col.replace("prob_", "", 1)
        y_prob = combined[col].values.astype(float)

        valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
        if valid.sum() == 0:
            continue

        y_t = y_true[valid]
        y_p = y_prob[valid]

        metrics[model_name] = {
            "accuracy": _compute_accuracy(y_t, y_p),
            "brier_score": compute_brier_score(y_t, y_p),
            "ece": compute_ece(y_t, y_p),
            "log_loss": _compute_log_loss(y_t, y_p),
            "n_samples": int(valid.sum()),
        }

    return metrics


def compute_model_agreement(preds_df: pd.DataFrame) -> np.ndarray:
    """Compute fraction of individual models agreeing with ensemble direction.

    For each row, counts how many prob_* models (excluding prob_logreg_seed
    and prob_ensemble) agree with the ensemble on prediction direction
    (both > 0.5 or both <= 0.5).

    Parameters
    ----------
    preds_df : pd.DataFrame
        Predictions with prob_* columns including prob_ensemble.

    Returns
    -------
    np.ndarray
        Agreement fraction per row (0.0 to 1.0).
    """
    if "prob_ensemble" not in preds_df.columns:
        return np.ones(len(preds_df))

    ensemble_direction = preds_df["prob_ensemble"].values > 0.5

    # Find individual model columns (exclude ensemble and logreg_seed)
    exclude = {"prob_ensemble", "prob_logreg_seed"}
    model_cols = [
        c for c in preds_df.columns
        if c.startswith("prob_") and c not in exclude
    ]

    if not model_cols:
        return np.ones(len(preds_df))

    n_rows = len(preds_df)
    agreement_count = np.zeros(n_rows)

    n_available = np.zeros(n_rows)
    for col in model_cols:
        values = preds_df[col].values
        valid = ~np.isnan(values.astype(float))
        model_direction = values > 0.5
        agreement_count += ((model_direction == ensemble_direction) & valid).astype(float)
        n_available += valid.astype(float)

    # Avoid division by zero: if no models available, return 1.0
    n_available = np.maximum(n_available, 1.0)
    return agreement_count / n_available


def evaluate_fold_predictions_multiclass(
    preds: pd.DataFrame,
    fold_id: int,
    target_column: str = "result",
) -> list[dict]:
    """Compute per-model metrics for a single fold (multiclass).

    Finds per-model probability columns matching ``prob_{model}_c{i}``
    pattern, reconstructs probability matrices, and computes accuracy,
    log_loss, and f1_macro.

    Parameters
    ----------
    preds : pd.DataFrame
        Prediction DataFrame with ``prob_{model}_c{i}`` columns and
        a target column with integer class labels.
    fold_id : int
        Fold identifier for the output.
    target_column : str
        Name of the column containing ground truth class labels.

    Returns
    -------
    list of dict
        Each dict has: model, fold, accuracy, log_loss, f1_macro.
    """
    import re
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import log_loss as sklearn_log_loss

    if target_column not in preds.columns:
        raise ValueError(f"No ground truth: '{target_column}' column missing")

    y_true = preds[target_column].values.astype(int)

    # Group prob columns by model: prob_{model}_c{i}
    pattern = re.compile(r"^prob_(.+)_c(\d+)$")
    model_cols: dict[str, dict[int, str]] = {}
    for col in preds.columns:
        m = pattern.match(col)
        if m:
            model_name = m.group(1)
            class_idx = int(m.group(2))
            model_cols.setdefault(model_name, {})[class_idx] = col

    results = []
    for model_name, idx_to_col in sorted(model_cols.items()):
        n_classes = max(idx_to_col.keys()) + 1
        if set(idx_to_col.keys()) != set(range(n_classes)):
            continue  # incomplete class columns

        ordered_cols = [idx_to_col[i] for i in range(n_classes)]
        prob_matrix = preds[ordered_cols].values.astype(float)

        # Skip rows with NaN
        valid = ~np.isnan(prob_matrix).any(axis=1)
        if valid.sum() == 0:
            continue

        y_t = y_true[valid]
        y_p = prob_matrix[valid]
        y_pred = y_p.argmax(axis=1)

        accuracy = float(accuracy_score(y_t, y_pred))
        try:
            ll = float(sklearn_log_loss(y_t, y_p))
        except Exception:
            ll = float("nan")
        try:
            f1m = float(f1_score(y_t, y_pred, average="macro"))
        except Exception:
            f1m = float("nan")

        results.append({
            "model": model_name,
            "fold": fold_id,
            "accuracy": accuracy,
            "log_loss": ll,
            "f1_macro": f1m,
        })

    return results


def compute_pooled_metrics_multiclass(
    fold_predictions: list[pd.DataFrame],
    target_column: str = "result",
) -> dict[str, dict]:
    """Compute pooled multiclass metrics across all folds.

    Parameters
    ----------
    fold_predictions : list of pd.DataFrame
        Each DataFrame has ``prob_{model}_c{i}`` columns and a target column.
    target_column : str
        Name of the column containing ground truth class labels.

    Returns
    -------
    dict mapping model_name -> metric dict
        Each metric dict has: accuracy, log_loss, f1_macro, n_samples.
    """
    import re
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics import log_loss as sklearn_log_loss

    if not fold_predictions:
        return {}

    combined = pd.concat(fold_predictions, ignore_index=True)

    if target_column not in combined.columns:
        raise ValueError(f"'{target_column}' column required in prediction DataFrames")

    y_true = combined[target_column].values.astype(int)

    pattern = re.compile(r"^prob_(.+)_c(\d+)$")
    model_cols: dict[str, dict[int, str]] = {}
    for col in combined.columns:
        m = pattern.match(col)
        if m:
            model_name = m.group(1)
            class_idx = int(m.group(2))
            model_cols.setdefault(model_name, {})[class_idx] = col

    metrics: dict[str, dict] = {}
    for model_name, idx_to_col in sorted(model_cols.items()):
        n_classes = max(idx_to_col.keys()) + 1
        if set(idx_to_col.keys()) != set(range(n_classes)):
            continue

        ordered_cols = [idx_to_col[i] for i in range(n_classes)]
        prob_matrix = combined[ordered_cols].values.astype(float)

        valid = ~np.isnan(prob_matrix).any(axis=1) & ~np.isnan(y_true.astype(float))
        if valid.sum() == 0:
            continue

        y_t = y_true[valid]
        y_p = prob_matrix[valid]
        y_pred = y_p.argmax(axis=1)

        m: dict = {
            "accuracy": float(accuracy_score(y_t, y_pred)),
            "n_samples": int(valid.sum()),
        }
        try:
            m["log_loss"] = float(sklearn_log_loss(y_t, y_p))
        except Exception:
            pass
        try:
            m["f1_macro"] = float(f1_score(y_t, y_pred, average="macro"))
        except Exception:
            pass

        metrics[model_name] = m

    return metrics


def _compute_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute accuracy from probabilities (threshold 0.5)."""
    predictions = (y_prob >= 0.5).astype(float)
    return float(np.mean(predictions == y_true))


def _compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute log loss (binary cross-entropy).

    Clips probabilities to avoid log(0).
    """
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(
        y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)
    ))


# ---------------------------------------------------------------------------
# SHAP integration
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X: pd.DataFrame | np.ndarray,
    method: str = "auto",
) -> dict:
    """Compute SHAP values for a trained model.

    Parameters
    ----------
    model
        Trained model. Must have ``predict`` or ``predict_proba``.
    X : DataFrame or array
        Feature data to explain.
    method : str
        One of ``"auto"``, ``"tree"``, ``"kernel"``, ``"linear"``.
        ``"auto"`` tries TreeExplainer first, then falls back to
        KernelExplainer.

    Returns
    -------
    dict
        Keys: ``values`` (array), ``feature_names`` (list),
        ``base_value`` (float).  If shap is not installed, returns
        ``{"error": "shap package not installed..."}``.
    """
    try:
        import shap
    except ImportError:
        return {
            "error": "shap package not installed. Install with: pip install shap",
        }

    feature_names = (
        list(X.columns) if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    if method == "auto":
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            predict_fn = (
                model.predict_proba if hasattr(model, "predict_proba")
                else model.predict
            )
            explainer = shap.KernelExplainer(predict_fn, shap.sample(X, min(100, len(X))))
    elif method == "tree":
        explainer = shap.TreeExplainer(model)
    elif method == "kernel":
        predict_fn = (
            model.predict_proba if hasattr(model, "predict_proba")
            else model.predict
        )
        explainer = shap.KernelExplainer(predict_fn, shap.sample(X, min(100, len(X))))
    elif method == "linear":
        explainer = shap.LinearExplainer(model, X)
    else:
        return {"error": f"Unknown SHAP method: {method!r}. Use auto/tree/kernel/linear."}

    shap_values = explainer.shap_values(X)

    base_value = explainer.expected_value
    if np.isscalar(base_value):
        base_value = float(base_value)
    elif hasattr(base_value, "__len__"):
        base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
    else:
        base_value = float(base_value)

    return {
        "values": shap_values,
        "feature_names": feature_names,
        "base_value": base_value,
    }


# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------

def roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute ROC curve data points.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.

    Returns
    -------
    dict
        Keys: ``fpr`` (list), ``tpr`` (list), ``thresholds`` (list),
        ``auc`` (float).
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc),
    }


def pr_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute precision-recall curve data points.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.

    Returns
    -------
    dict
        Keys: ``precision`` (list), ``recall`` (list),
        ``thresholds`` (list), ``average_precision`` (float).
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "average_precision": float(avg_precision),
    }


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------

def permutation_importance_data(
    model,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    scoring: str = "accuracy",
) -> dict:
    """Compute permutation importance for a trained model.

    Parameters
    ----------
    model
        Trained model with a ``predict`` method.
    X : DataFrame or array
        Feature data.
    y : array-like
        True labels.
    n_repeats : int
        Number of permutation repeats.
    scoring : str
        Scoring metric (e.g., ``"accuracy"``, ``"neg_brier_score"``).

    Returns
    -------
    dict
        Keys: ``importances_mean`` (list), ``importances_std`` (list),
        ``feature_names`` (list).
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, scoring=scoring, random_state=42,
    )

    feature_names = (
        list(X.columns) if hasattr(X, "columns")
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    return {
        "importances_mean": result.importances_mean.tolist(),
        "importances_std": result.importances_std.tolist(),
        "feature_names": feature_names,
    }
