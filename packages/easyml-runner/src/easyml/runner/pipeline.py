"""PipelineRunner — wire library APIs from YAML config.

Orchestrates model training, backtesting, and evaluation by loading
ProjectConfig and delegating to easyml-models components.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from easyml.models.backtest import BacktestRunner
from easyml.models.cv import LeaveOneSeasonOut
from easyml.models.orchestrator import TrainOrchestrator
from easyml.models.registry import ModelRegistry
from easyml.runner.schema import ProjectConfig
from easyml.runner.validator import validate_project

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrate training and backtesting from YAML-driven config.

    Parameters
    ----------
    project_dir : str | Path
        Root project directory.
    config_dir : str | Path
        Path to the config directory containing pipeline.yaml, etc.
    variant : str | None
        Optional variant suffix (e.g. "w" for women's).
    overlay : dict | None
        Optional overlay dict to merge on top of config.
    """

    def __init__(
        self,
        project_dir: str | Path,
        config_dir: str | Path,
        variant: str | None = None,
        overlay: dict | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.config_dir = Path(config_dir)
        self.variant = variant
        self.overlay = overlay

        self.config: ProjectConfig | None = None
        self._registry: ModelRegistry | None = None
        self._df: pd.DataFrame | None = None
        self._feature_columns: list[str] | None = None
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._seasons: np.ndarray | None = None

    def load(self) -> None:
        """Validate config and set up ModelRegistry."""
        result = validate_project(
            self.config_dir,
            overlay=self.overlay,
            variant=self.variant,
        )
        if not result.valid:
            raise ValueError(
                f"Config validation failed:\n{result.format()}"
            )
        self.config = result.config
        self._registry = ModelRegistry.with_defaults()
        self._load_data()

    def _load_data(self) -> None:
        """Read matchup_features.parquet and extract features, y, and seasons."""
        features_dir = Path(self.config.data.features_dir)
        parquet_path = features_dir / "matchup_features.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"matchup_features.parquet not found at {parquet_path}"
            )

        self._df = pd.read_parquet(parquet_path)

        # Determine feature columns from model configs
        all_features: set[str] = set()
        for model_def in self.config.models.values():
            if model_def.active:
                all_features.update(model_def.features)

        # Filter to columns that actually exist in the data
        available = set(self._df.columns)
        self._feature_columns = sorted(f for f in all_features if f in available)

        if not self._feature_columns:
            raise ValueError(
                "No declared model features found in matchup_features.parquet. "
                f"Available columns: {sorted(available)}"
            )

        self._X = self._df[self._feature_columns].values.astype(np.float64)
        self._y = self._df["result"].values.astype(np.float64)
        self._seasons = self._df["season"].values

    def train(self, run_id: str | None = None) -> dict[str, Any]:
        """Train all active models on the full dataset.

        Parameters
        ----------
        run_id : str | None
            Optional run identifier for artifact naming.

        Returns
        -------
        dict
            Result dict with status, models_trained.
        """
        if self.config is None:
            raise RuntimeError("Call load() before train()")

        output_dir = self.project_dir / "models"
        if run_id:
            output_dir = output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build model configs dict for TrainOrchestrator
        model_configs = self._build_model_configs()

        orchestrator = TrainOrchestrator(
            model_registry=self._registry,
            model_configs=model_configs,
            output_dir=output_dir,
            failure_policy="skip",
            use_fingerprint=False,
        )

        trained = orchestrator.train_all(
            X=self._X,
            y=self._y,
            feature_columns=self._feature_columns,
        )

        return {
            "status": "success",
            "models_trained": list(trained.keys()),
        }

    def backtest(self) -> dict[str, Any]:
        """Run LOSO cross-validation across seasons.

        Per fold: train each active model on train split, predict on test split.
        Collect per-fold predictions, pass to BacktestRunner.

        Returns
        -------
        dict
            Result dict with status, metrics, per_fold.
        """
        if self.config is None:
            raise RuntimeError("Call load() before backtest()")

        bt_config = self.config.backtest
        cv = LeaveOneSeasonOut(min_train_folds=bt_config.min_train_folds)
        folds = cv.split(None, fold_ids=self._seasons)

        if not folds:
            raise ValueError("No valid folds for backtesting. Check seasons and min_train_folds.")

        model_configs = self._build_model_configs()
        active_model_names = [
            name for name, cfg in model_configs.items() if cfg.get("active", True)
        ]

        # Per-fold: train and predict
        per_fold_data: dict[int, dict] = {}

        for fold in folds:
            X_train = self._X[fold.train_idx]
            y_train = self._y[fold.train_idx]
            X_test = self._X[fold.test_idx]
            y_test = self._y[fold.test_idx]

            fold_preds: dict[str, np.ndarray] = {}

            for model_name in active_model_names:
                cfg = model_configs[model_name]
                model_features = cfg.get("features", self._feature_columns)

                # Get feature indices for this model
                feat_indices = [
                    self._feature_columns.index(f)
                    for f in model_features
                    if f in self._feature_columns
                ]
                if not feat_indices:
                    logger.warning(
                        "Model %s has no available features in fold %d, skipping",
                        model_name, fold.fold_id,
                    )
                    continue

                X_train_sub = X_train[:, feat_indices]
                X_test_sub = X_test[:, feat_indices]

                model = self._registry.create(cfg["type"], params=cfg.get("params"))
                model.fit(X_train_sub, y_train)

                preds = model.predict_proba(X_test_sub)
                fold_preds[model_name] = preds

            per_fold_data[fold.fold_id] = {
                "preds": fold_preds,
                "y": y_test,
            }

        # Run BacktestRunner
        bt_runner = BacktestRunner(metrics=bt_config.metrics)
        bt_result = bt_runner.run(per_fold_data)

        return {
            "status": "success",
            "metrics": bt_result.pooled_metrics,
            "per_fold": {
                fold_id: metrics
                for fold_id, metrics in bt_result.per_fold_metrics.items()
            },
            "models_trained": active_model_names,
        }

    def run_full(self) -> dict[str, Any]:
        """Run load + train + backtest."""
        self.load()
        self.train()
        result = self.backtest()
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_model_configs(self) -> dict[str, dict]:
        """Convert ModelDef objects to plain dicts for TrainOrchestrator."""
        return {
            name: {
                "type": model_def.type,
                "features": model_def.features,
                "params": model_def.params,
                "active": model_def.active,
                "mode": model_def.mode,
                "n_seeds": model_def.n_seeds,
            }
            for name, model_def in self.config.models.items()
        }
