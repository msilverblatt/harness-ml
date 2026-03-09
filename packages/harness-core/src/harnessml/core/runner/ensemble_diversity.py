"""Ensemble diversity metrics for measuring model agreement/disagreement."""
from __future__ import annotations

import numpy as np


def compute_diversity(
    predictions: np.ndarray,
    method: str = "disagreement",
    labels: np.ndarray | None = None,
    threshold: float = 0.5,
) -> float:
    """Compute ensemble diversity metric.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (n_samples, n_models) of probability predictions.
    method : str
        One of "disagreement", "q_statistic", "kappa", "correlation".
    labels : np.ndarray, optional
        True labels, required for q_statistic and kappa.
    threshold : float
        Decision threshold for converting probabilities to binary.

    Returns
    -------
    float
        Diversity score. Higher = more diverse.
    """
    if method == "disagreement":
        return _disagreement(predictions, threshold)
    elif method == "q_statistic":
        if labels is None:
            raise ValueError("q_statistic requires labels")
        return _q_statistic(predictions, labels, threshold)
    elif method == "kappa":
        if labels is None:
            raise ValueError("kappa requires labels")
        return _kappa(predictions, labels, threshold)
    elif method == "correlation":
        return _correlation(predictions)
    else:
        raise ValueError(f"Unknown diversity method: {method}")


def _disagreement(predictions, threshold):
    """Average pairwise disagreement rate."""
    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0
    binary = (predictions >= threshold).astype(int)
    total = 0.0
    count = 0
    for i in range(n_models):
        for j in range(i + 1, n_models):
            total += np.mean(binary[:, i] != binary[:, j])
            count += 1
    return total / count if count > 0 else 0.0


def _q_statistic(predictions, labels, threshold):
    """Average pairwise Yule's Q statistic. Lower = more diverse."""
    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0
    binary = (predictions >= threshold).astype(int)
    correct = (binary == labels.reshape(-1, 1)).astype(int)
    total = 0.0
    count = 0
    for i in range(n_models):
        for j in range(i + 1, n_models):
            n11 = np.sum(correct[:, i] * correct[:, j])
            n00 = np.sum((1 - correct[:, i]) * (1 - correct[:, j]))
            n10 = np.sum(correct[:, i] * (1 - correct[:, j]))
            n01 = np.sum((1 - correct[:, i]) * correct[:, j])
            denom = n11 * n00 + n10 * n01
            if denom == 0:
                q = 0.0
            else:
                q = (n11 * n00 - n10 * n01) / denom
            total += q
            count += 1
    return total / count if count > 0 else 0.0


def _kappa(predictions, labels, threshold):
    """Average pairwise Cohen's kappa. Lower = more diverse."""
    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0
    binary = (predictions >= threshold).astype(int)
    total = 0.0
    count = 0
    for i in range(n_models):
        for j in range(i + 1, n_models):
            agree = np.mean(binary[:, i] == binary[:, j])
            p_i = np.mean(binary[:, i])
            p_j = np.mean(binary[:, j])
            chance = p_i * p_j + (1 - p_i) * (1 - p_j)
            if abs(1 - chance) < 1e-10:
                k = 1.0
            else:
                k = (agree - chance) / (1 - chance)
            total += k
            count += 1
    return total / count if count > 0 else 0.0


def _correlation(predictions):
    """Average pairwise Pearson correlation of predictions. Lower = more diverse."""
    n_models = predictions.shape[1]
    if n_models < 2:
        return 0.0
    corr_matrix = np.corrcoef(predictions.T)
    total = 0.0
    count = 0
    for i in range(n_models):
        for j in range(i + 1, n_models):
            total += corr_matrix[i, j]
            count += 1
    return total / count if count > 0 else 0.0
