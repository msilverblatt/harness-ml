"""Training orchestrator — iterates model configs, trains, and saves artifacts.

Handles:
- Active/inactive filtering
- Fingerprint-based cache skipping
- Configurable failure policy (skip vs raise)
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from easyml.core.models.base import BaseModel
from easyml.core.models.fingerprint import Fingerprint
from easyml.core.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class TrainOrchestrator:
    """Orchestrate training of multiple models from config.

    Parameters
    ----------
    model_registry : ModelRegistry
        Registry mapping model type strings to classes.
    model_configs : dict[str, dict]
        Model name -> config dict.  Each config must have at least:
        ``type``, ``mode``, ``features``, ``params``, ``active``.
    output_dir : Path
        Directory where per-model artifacts are written.
    failure_policy : str
        ``"skip"`` (default) logs a warning on model failure and continues.
        ``"raise"`` re-raises the exception.
    use_fingerprint : bool
        When True, check fingerprints before training and skip unchanged models.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_configs: dict[str, dict],
        output_dir: Path | str,
        failure_policy: str = "skip",
        use_fingerprint: bool = True,
    ) -> None:
        self.model_registry = model_registry
        self.model_configs = model_configs
        self.output_dir = Path(output_dir)
        self.failure_policy = failure_policy
        self.use_fingerprint = use_fingerprint

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: list[str],
    ) -> dict[str, BaseModel]:
        """Train all active models.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features).
        y : np.ndarray
            Target array (n_samples,).
        feature_columns : list[str]
            Column names corresponding to X columns — used for feature
            subsetting and fingerprinting.

        Returns
        -------
        dict[str, BaseModel]
            Mapping of model_name -> trained model (only active, successfully
            trained models).
        """
        results: dict[str, BaseModel] = {}

        for model_name, config in self.model_configs.items():
            if not config.get("active", True):
                logger.info("Skipping inactive model: %s", model_name)
                continue

            try:
                model = self._train_one(model_name, config, X, y, feature_columns)
                results[model_name] = model
            except Exception:
                if self.failure_policy == "raise":
                    raise
                logger.warning(
                    "Model %s failed, skipping (failure_policy='skip')",
                    model_name,
                    exc_info=True,
                )

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one(
        self,
        model_name: str,
        config: dict,
        X: np.ndarray,
        y: np.ndarray,
        feature_columns: list[str],
    ) -> BaseModel:
        """Train a single model, with optional fingerprint check."""
        model_dir = self.output_dir / model_name
        fp_path = model_dir / "fingerprint.json"

        # Build fingerprint
        data_hash = hashlib.sha256(
            np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(y).tobytes()
        ).hexdigest()[:16]
        fp = Fingerprint.compute(
            config_dict=config,
            data_hash=data_hash,
            data_size=float(len(y)),
        )

        # Check cache
        if self.use_fingerprint and fp.matches(fp_path):
            logger.info("Fingerprint match — loading cached model: %s", model_name)
            model_cls = self.model_registry._registry[config["type"]]
            return model_cls.load(model_dir)

        # Select features for this model
        model_features = config.get("features", feature_columns)
        feat_indices = [feature_columns.index(f) for f in model_features if f in feature_columns]
        X_sub = X[:, feat_indices] if feat_indices else X

        # Create and train
        model = self.model_registry.create(config["type"], params=config.get("params"))
        logger.info("Training model: %s", model_name)
        model.fit(X_sub, y)

        # Save
        model.save(model_dir)
        fp.save(fp_path)

        return model
