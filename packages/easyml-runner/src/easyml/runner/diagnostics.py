"""Diagnostics and metrics computation for model evaluation.

Provides Brier score, ECE, calibration curve, and pooled/per-season
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


def evaluate_season_predictions(
    preds: pd.DataFrame,
    actuals: dict[str, int],
    season: int,
) -> list[dict]:
    """Compute per-model metrics for a single season.

    Parameters
    ----------
    preds : pd.DataFrame
        Prediction DataFrame with prob_{model_name} columns and
        a 'result' column (binary outcome). If 'result' is not present,
        uses actuals dict with matchup keys.
    actuals : dict[str, int]
        Mapping of matchup identifier -> binary outcome. Used as
        fallback if 'result' column is not in preds.
    season : int
        Season identifier for the output.

    Returns
    -------
    list of dict
        Each dict has: model, season, accuracy, brier_score, ece, log_loss.
    """
    # Determine ground truth
    if "result" in preds.columns:
        y_true = preds["result"].values.astype(float)
    elif actuals:
        # Use actuals dict — assumes index or row order matches
        y_true = np.array(list(actuals.values()), dtype=float)
    else:
        raise ValueError("No ground truth: 'result' column missing and actuals is empty")

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
            "season": season,
            "accuracy": accuracy,
            "brier_score": brier,
            "ece": ece,
            "log_loss": logloss,
        })

    return results


def compute_pooled_metrics(
    season_predictions: list[pd.DataFrame],
) -> dict[str, dict]:
    """Compute pooled metrics across all seasons.

    Pooled = computed on concatenated predictions (more stable than
    averaging per-season metrics).

    Parameters
    ----------
    season_predictions : list of pd.DataFrame
        Each DataFrame has prob_{model_name} columns and a 'result' column.

    Returns
    -------
    dict mapping model_name -> metric dict
        Each metric dict has: accuracy, brier_score, ece, log_loss, n_samples.
    """
    if not season_predictions:
        return {}

    combined = pd.concat(season_predictions, ignore_index=True)

    if "result" not in combined.columns:
        raise ValueError("'result' column required in prediction DataFrames")

    y_true = combined["result"].values.astype(float)

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
