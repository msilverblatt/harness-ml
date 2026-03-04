"""Backtest runner — evaluate ensemble predictions across temporal folds.

Computes pooled and per-fold metrics from pre-computed per-fold predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def _brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error between predictions and binary labels."""
    return float(np.mean((y_pred - y_true) ** 2))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions (threshold = 0.5)."""
    return float(np.mean((y_pred >= 0.5) == y_true))


def _log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary cross-entropy loss."""
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def _ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_pred >= lo) & (y_pred < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


_METRIC_REGISTRY: dict[str, callable] = {
    "brier": _brier_score,
    "accuracy": _accuracy,
    "log_loss": _log_loss,
    "ece": _ece,
}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for backtest results.

    Attributes
    ----------
    pooled_metrics : dict[str, float]
        Metrics computed across all folds pooled together.
    per_fold_metrics : dict[int, dict[str, float]]
        Per-fold metric dicts keyed by fold id.
    pooled_y_true : np.ndarray
        Concatenated ground truth across all folds.
    pooled_y_pred : np.ndarray
        Concatenated predictions across all folds.
    """

    pooled_metrics: dict[str, float] = field(default_factory=dict)
    per_fold_metrics: dict[int, dict[str, float]] = field(default_factory=dict)
    pooled_y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    pooled_y_pred: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BacktestRunner:
    """Run backtesting across pre-computed per-fold predictions.

    Parameters
    ----------
    metrics : list[str]
        List of metric names to compute. Supported: ``"brier"``,
        ``"accuracy"``, ``"log_loss"``, ``"ece"``.
    """

    def __init__(self, metrics: list[str] | None = None) -> None:
        self.metrics = metrics or ["brier", "accuracy"]

        # Validate metric names
        for m in self.metrics:
            if m not in _METRIC_REGISTRY:
                raise ValueError(
                    f"Unknown metric {m!r}. Supported: {list(_METRIC_REGISTRY.keys())}"
                )

    def run(
        self,
        per_fold_data: dict[int, dict],
    ) -> BacktestResult:
        """Compute pooled and per-fold metrics.

        Parameters
        ----------
        per_fold_data : dict[int, dict]
            Mapping of ``fold_id -> {"preds": dict[str, ndarray], "y": ndarray}``.
            Each ``preds`` dict maps model names to prediction arrays. When
            multiple models are present, predictions are averaged to produce
            the ensemble prediction for that fold.

        Returns
        -------
        BacktestResult
        """
        all_y: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        per_fold_metrics: dict[int, dict[str, float]] = {}

        for fold_id in sorted(per_fold_data.keys()):
            fold = per_fold_data[fold_id]
            y_true = np.asarray(fold["y"])
            model_preds = fold["preds"]

            # Average predictions across models for ensemble
            y_pred = np.mean(
                [np.asarray(p) for p in model_preds.values()],
                axis=0,
            )

            all_y.append(y_true)
            all_preds.append(y_pred)

            # Per-fold metrics
            fold_metrics = {}
            for m in self.metrics:
                fold_metrics[m] = _METRIC_REGISTRY[m](y_true, y_pred)
            per_fold_metrics[fold_id] = fold_metrics

        # Pooled metrics
        pooled_y = np.concatenate(all_y)
        pooled_preds = np.concatenate(all_preds)
        pooled_metrics = {}
        for m in self.metrics:
            pooled_metrics[m] = _METRIC_REGISTRY[m](pooled_y, pooled_preds)

        return BacktestResult(
            pooled_metrics=pooled_metrics,
            per_fold_metrics=per_fold_metrics,
            pooled_y_true=pooled_y,
            pooled_y_pred=pooled_preds,
        )
