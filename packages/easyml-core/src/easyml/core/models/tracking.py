"""Tracking callback protocol for MLflow / Weights & Biases integration.

Provides abstract hooks that fire at key pipeline events.  The default
implementations are no-ops so users only override what they need.
"""
from __future__ import annotations

from abc import ABC


class TrackingCallback(ABC):
    """Abstract base for pipeline tracking integrations.

    All methods have default no-op implementations so subclasses can
    override only the hooks they care about.
    """

    def on_model_trained(
        self,
        model_name: str,
        metrics: dict[str, float],
        duration_s: float,
    ) -> None:
        """Called after a single model finishes training.

        Parameters
        ----------
        model_name : str
            Name of the model that was trained.
        metrics : dict[str, float]
            Training-time metrics (loss, etc.).
        duration_s : float
            Wall-clock training time in seconds.
        """

    def on_backtest_complete(self, metrics: dict[str, float]) -> None:
        """Called after backtesting finishes.

        Parameters
        ----------
        metrics : dict[str, float]
            Pooled backtest metrics.
        """

    def on_experiment_logged(
        self,
        experiment_id: str,
        verdict: str,
        metrics: dict[str, float],
    ) -> None:
        """Called when an experiment result is logged.

        Parameters
        ----------
        experiment_id : str
            Experiment identifier.
        verdict : str
            One of ``"keep"``, ``"revert"``, ``"partial"``.
        metrics : dict[str, float]
            Final experiment metrics.
        """
