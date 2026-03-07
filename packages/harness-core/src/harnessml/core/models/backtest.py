"""Backtest runner — evaluate ensemble predictions across temporal folds.

Computes pooled and per-fold metrics from pre-computed per-fold predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from harnessml.core.schemas.metrics import MetricRegistry

# ---------------------------------------------------------------------------
# Metric lookup — delegates to MetricRegistry (binary task type)
# ---------------------------------------------------------------------------

# Aliases that map to canonical MetricRegistry names
_METRIC_ALIASES: dict[str, str] = {
    "auc": "auc_roc",
    "r2": "r_squared",
    "brier_score": "brier",
}


def _get_metric_fn(name: str, task_type: str = "binary"):
    """Look up a metric function from MetricRegistry, resolving aliases."""
    canonical = _METRIC_ALIASES.get(name, name)
    fn = MetricRegistry.get(task_type, canonical)
    if fn is None and task_type != "binary":
        fn = MetricRegistry.get("binary", canonical)  # fallback
    if fn is None:
        available = list(MetricRegistry.list_metrics(task_type).get(task_type, []))
        raise ValueError(
            f"Unknown metric {name!r}. Available {task_type} metrics: {available}"
        )
    return fn


def _METRIC_REGISTRY_lookup(name: str, y_true, y_pred, task_type: str = "binary") -> float:
    """Compute a named metric via MetricRegistry."""
    fn = _get_metric_fn(name, task_type=task_type)
    val = fn(y_true, y_pred)
    return float(val)


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
        ``"accuracy"``, ``"log_loss"``, ``"ece"``, ``"auc_roc"``
        (alias ``"auc"``), ``"f1"``, ``"precision"``, ``"recall"``.
    """

    def __init__(self, metrics: list[str] | None = None, task_type: str = "binary") -> None:
        self.metrics = metrics or ["brier", "accuracy"]
        self.task_type = task_type

        # Validate metric names up front
        for m in self.metrics:
            _get_metric_fn(m, task_type=self.task_type)  # raises ValueError if unknown

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
                fold_metrics[m] = _METRIC_REGISTRY_lookup(m, y_true, y_pred, task_type=self.task_type)
            per_fold_metrics[fold_id] = fold_metrics

        # Pooled metrics
        pooled_y = np.concatenate(all_y)
        pooled_preds = np.concatenate(all_preds)
        pooled_metrics = {}
        for m in self.metrics:
            pooled_metrics[m] = _METRIC_REGISTRY_lookup(m, pooled_y, pooled_preds, task_type=self.task_type)

        return BacktestResult(
            pooled_metrics=pooled_metrics,
            per_fold_metrics=per_fold_metrics,
            pooled_y_true=pooled_y,
            pooled_y_pred=pooled_preds,
        )
