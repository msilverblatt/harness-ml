"""PipelineRunner — wire library APIs from YAML config.

Orchestrates model training, backtesting, and evaluation by loading
ProjectConfig and delegating to easyml-models components.  Supports
real ensemble backtesting with stacked meta-learner and per-model
feature subsets, training season filtering, and regressor models.
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
from easyml.runner.meta_learner import train_meta_learner_loso
from easyml.runner.postprocessing import apply_ensemble_postprocessing
from easyml.runner.schema import ModelDef, ProjectConfig
from easyml.runner.training import (
    predict_single_model,
    train_single_model,
)
from easyml.runner.validator import validate_project

logger = logging.getLogger(__name__)

# Column name mappings: mm-style -> easyml-style
_COLUMN_RENAMES = {
    "TeamAWon": "result",
    "TeamAMargin": "margin",
    "Season": "season",
}


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
        """Read matchup_features.parquet and normalize column names."""
        features_dir = Path(self.config.data.features_dir)
        parquet_path = features_dir / "matchup_features.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(
                f"matchup_features.parquet not found at {parquet_path}"
            )

        self._df = pd.read_parquet(parquet_path)
        self._normalize_columns()

    def _normalize_columns(self) -> None:
        """Auto-detect mm-style columns and rename to easyml convention."""
        for old_name, new_name in _COLUMN_RENAMES.items():
            if old_name in self._df.columns and new_name not in self._df.columns:
                self._df = self._df.rename(columns={old_name: new_name})

    def _get_active_models(self) -> dict[str, ModelDef]:
        """Return active models, excluding those in ensemble.exclude_models."""
        exclude = set(self.config.ensemble.exclude_models)
        return {
            name: model_def
            for name, model_def in self.config.models.items()
            if model_def.active and name not in exclude
        }

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

        # Determine feature columns from active models
        all_features: set[str] = set()
        for model_def in self.config.models.values():
            if model_def.active:
                all_features.update(model_def.features)
        available = set(self._df.columns)
        feature_columns = sorted(f for f in all_features if f in available)

        if not feature_columns:
            raise ValueError(
                "No declared model features found in matchup_features.parquet."
            )

        X = self._df[feature_columns].values.astype(np.float64)
        y = self._df["result"].values.astype(np.float64)

        orchestrator = TrainOrchestrator(
            model_registry=self._registry,
            model_configs=model_configs,
            output_dir=output_dir,
            failure_policy="skip",
            use_fingerprint=False,
        )

        trained = orchestrator.train_all(
            X=X,
            y=y,
            feature_columns=feature_columns,
        )

        return {
            "status": "success",
            "models_trained": list(trained.keys()),
        }

    def predict(
        self, season: int, run_id: str | None = None
    ) -> pd.DataFrame:
        """Generate predictions for a target season.

        1. Load trained models from models_dir (or train fresh)
        2. Build all pairwise matchups from team features + seeds
        3. Predict each matchup with each model
        4. Train production meta-learner on backtest predictions
        5. Apply ensemble post-processing
        6. Return predictions DataFrame with prob_* and prob_ensemble columns

        Parameters
        ----------
        season : int
            Target season to predict.
        run_id : str | None
            Optional run identifier for locating model artifacts.

        Returns
        -------
        pd.DataFrame
            Prediction DataFrame with prob_{model_name} and prob_ensemble columns.
        """
        if self.config is None:
            raise RuntimeError("Call load() before predict()")

        active_models = self._get_active_models()
        if not active_models:
            raise ValueError("No active models for prediction.")

        # Train all active models on data before the target season
        trained_models = {}
        for model_name, model_def in active_models.items():
            try:
                model, feature_cols, metrics = train_single_model(
                    model_name=model_name,
                    model_def=model_def,
                    train_df=self._df,
                    registry=self._registry,
                    target_season=season,
                )
                trained_models[model_name] = (model, feature_cols, metrics)
            except Exception:
                logger.exception(
                    "Failed to train %s for prediction season %d",
                    model_name, season,
                )
                continue

        if not trained_models:
            raise ValueError("No models could be trained for prediction.")

        # Get test data for the target season
        test_mask = self._df["season"] == season
        test_df = self._df[test_mask].copy()

        if len(test_df) == 0:
            logger.warning("No data found for season %d", season)
            return pd.DataFrame()

        # Build predictions DataFrame
        preds_df = test_df[["season"]].copy()
        if "result" in test_df.columns:
            preds_df["result"] = test_df["result"].values

        # Add diff_seed_num if available
        if "diff_seed_num" in test_df.columns:
            preds_df["diff_seed_num"] = test_df["diff_seed_num"].values
        else:
            preds_df["diff_seed_num"] = np.zeros(len(test_df))

        # Add meta_features if configured
        meta_feature_names = self.config.ensemble.meta_features
        for feat_name in meta_feature_names:
            if feat_name in test_df.columns:
                preds_df[feat_name] = test_df[feat_name].values

        # Generate predictions from each model
        for model_name, (model, feature_cols, metrics) in trained_models.items():
            try:
                cdf_scale = metrics.get("cdf_scale")
                probs = predict_single_model(
                    model=model,
                    model_def=active_models[model_name],
                    test_df=test_df,
                    feature_columns=feature_cols,
                    cdf_scale=cdf_scale,
                )
                preds_df[f"prob_{model_name}"] = probs
            except Exception:
                logger.exception(
                    "Failed to predict %s for season %d",
                    model_name, season,
                )
                continue

        # Check we got at least one model prediction
        prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
        if not prob_cols:
            return preds_df

        # Train meta-learner on backtest data and apply ensemble
        ensemble_config = self.config.ensemble.model_dump()
        active_model_names = [
            name for name in active_models
            if f"prob_{name}" in preds_df.columns
        ]

        if ensemble_config["method"] == "stacked" and "result" in self._df.columns:
            try:
                # Generate backtest predictions for meta-learner training
                bt_data = self._generate_backtest_for_meta(season, active_models)
                if bt_data:
                    meta, cal, pre_cals = self._train_production_meta(
                        bt_data, ensemble_config, active_model_names,
                    )
                    preds_df = apply_ensemble_postprocessing(
                        preds_df, meta, cal, ensemble_config,
                        pre_calibrators=pre_cals,
                    )
                else:
                    # Fall back to simple average
                    preds_df["prob_ensemble"] = preds_df[prob_cols].mean(axis=1)
            except Exception:
                logger.exception("Meta-learner training failed, falling back to average")
                preds_df["prob_ensemble"] = preds_df[prob_cols].mean(axis=1)
        else:
            # Simple average
            preds_df["prob_ensemble"] = preds_df[prob_cols].mean(axis=1)

        return preds_df

    def _generate_backtest_for_meta(
        self,
        target_season: int,
        active_models: dict[str, ModelDef],
    ) -> dict[int, pd.DataFrame]:
        """Generate backtest predictions for training the production meta-learner.

        Trains on data before target_season, using LOSO within that data.
        """
        # Use all seasons before target for meta-learner training
        available_df = self._df[self._df["season"] < target_season]
        if len(available_df) == 0:
            return {}

        seasons = sorted(available_df["season"].unique())
        season_data: dict[int, pd.DataFrame] = {}

        for holdout in seasons:
            preds_df = self._generate_season_predictions(holdout, active_models)
            if preds_df is not None and len(preds_df) > 0:
                season_data[holdout] = preds_df

        return season_data

    def _train_production_meta(
        self,
        season_data: dict[int, pd.DataFrame],
        ensemble_config: dict,
        active_model_names: list[str],
    ) -> tuple:
        """Train the production meta-learner on all backtest seasons."""
        all_dfs = [season_data[s] for s in sorted(season_data.keys())]
        combined = pd.concat(all_dfs, ignore_index=True)

        available_models = [
            name for name in active_model_names
            if f"prob_{name}" in combined.columns
        ]

        if not available_models:
            raise ValueError("No model predictions for meta-learner")

        model_preds = {
            name: combined[f"prob_{name}"].values
            for name in available_models
        }

        y_true = combined["result"].values.astype(float)
        seed_diffs = combined["diff_seed_num"].values.astype(float)
        season_labels = combined["season"].values

        extra_features = None
        meta_feature_names = ensemble_config.get("meta_features", [])
        if meta_feature_names:
            extra_features = {}
            for feat_name in meta_feature_names:
                if feat_name in combined.columns:
                    extra_features[feat_name] = combined[feat_name].values

        meta, cal, pre_cals = train_meta_learner_loso(
            y_true=y_true,
            model_preds=model_preds,
            seed_diffs=seed_diffs,
            season_labels=season_labels,
            model_names=available_models,
            ensemble_config=ensemble_config,
            extra_features=extra_features if extra_features else None,
        )

        return meta, cal, pre_cals

    def backtest(self) -> dict[str, Any]:
        """Run real ensemble backtesting with LOSO cross-validation.

        Two-pass approach:
          Pass 1: Per holdout season, train all active models on everything
                  except that season, predict that season's matchups.
          Pass 2: Train LOSO meta-learner (one per held-out season), apply
                  ensemble post-processing.

        Returns
        -------
        dict
            Result dict with status, metrics, per_fold, models_trained.
        """
        if self.config is None:
            raise RuntimeError("Call load() before backtest()")

        bt_config = self.config.backtest
        ensemble_config = self.config.ensemble.model_dump()
        active_models = self._get_active_models()

        if not active_models:
            raise ValueError("No active models to backtest.")

        # Determine backtest seasons
        seasons = bt_config.seasons
        if not seasons:
            seasons = sorted(self._df["season"].unique())

        # Pass 1: per-season OOF predictions
        season_data: dict[int, pd.DataFrame] = {}
        for holdout in seasons:
            preds_df = self._generate_season_predictions(holdout, active_models)
            if preds_df is not None and len(preds_df) > 0:
                season_data[holdout] = preds_df

        if not season_data:
            raise ValueError("No valid holdout seasons produced predictions.")

        active_model_names = list(active_models.keys())

        # Pass 2: meta-learner + post-processing
        if ensemble_config["method"] == "stacked":
            self._apply_stacked_ensemble(season_data, ensemble_config, active_model_names)
        else:
            # Simple average
            for holdout in season_data:
                prob_cols = [
                    c for c in season_data[holdout].columns
                    if c.startswith("prob_")
                ]
                if prob_cols:
                    season_data[holdout] = season_data[holdout].copy()
                    season_data[holdout]["prob_ensemble"] = (
                        season_data[holdout][prob_cols].mean(axis=1)
                    )

        return self._compute_backtest_metrics(season_data, active_model_names)

    def run_full(self) -> dict[str, Any]:
        """Run load + train + backtest."""
        self.load()
        self.train()
        result = self.backtest()
        return result

    # ------------------------------------------------------------------
    # Pass 1: Generate per-season predictions
    # ------------------------------------------------------------------

    def _generate_season_predictions(
        self,
        holdout_season: int,
        active_models: dict[str, ModelDef],
    ) -> pd.DataFrame | None:
        """Train all active models on non-holdout data, predict holdout.

        Returns a DataFrame with prob_{model_name} columns, plus
        diff_seed_num and any meta_features columns.
        """
        test_mask = self._df["season"] == holdout_season
        train_df = self._df[~test_mask].copy()
        test_df = self._df[test_mask].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            return None

        preds_df = test_df[["season"]].copy()
        if "result" in test_df.columns:
            preds_df["result"] = test_df["result"].values

        # Add diff_seed_num if available
        if "diff_seed_num" in test_df.columns:
            preds_df["diff_seed_num"] = test_df["diff_seed_num"].values
        else:
            preds_df["diff_seed_num"] = np.zeros(len(test_df))

        # Add meta_features if configured
        meta_feature_names = self.config.ensemble.meta_features
        for feat_name in meta_feature_names:
            if feat_name in test_df.columns:
                preds_df[feat_name] = test_df[feat_name].values

        # Train and predict each model
        for model_name, model_def in active_models.items():
            try:
                model, feature_cols, metrics = train_single_model(
                    model_name=model_name,
                    model_def=model_def,
                    train_df=train_df,
                    registry=self._registry,
                )

                cdf_scale = metrics.get("cdf_scale")
                probs = predict_single_model(
                    model=model,
                    model_def=model_def,
                    test_df=test_df,
                    feature_columns=feature_cols,
                    cdf_scale=cdf_scale,
                )

                preds_df[f"prob_{model_name}"] = probs

            except Exception:
                logger.exception(
                    "Failed to train/predict %s for season %d",
                    model_name, holdout_season,
                )
                continue

        # Check we got at least one model
        prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
        if not prob_cols:
            return None

        return preds_df

    # ------------------------------------------------------------------
    # Pass 2: Stacked ensemble
    # ------------------------------------------------------------------

    def _apply_stacked_ensemble(
        self,
        season_data: dict[int, pd.DataFrame],
        ensemble_config: dict,
        active_model_names: list[str],
    ) -> None:
        """Apply stacked meta-learner via LOSO to all holdout seasons.

        For each holdout season, train the meta-learner on all OTHER
        holdout seasons' predictions, then predict the holdout.
        """
        holdout_seasons = sorted(season_data.keys())

        for holdout in holdout_seasons:
            # Train meta-learner on all seasons except this one
            train_seasons = [s for s in holdout_seasons if s != holdout]

            if not train_seasons:
                # Only one season — fall back to simple average
                prob_cols = [
                    c for c in season_data[holdout].columns
                    if c.startswith("prob_")
                ]
                if prob_cols:
                    season_data[holdout] = season_data[holdout].copy()
                    season_data[holdout]["prob_ensemble"] = (
                        season_data[holdout][prob_cols].mean(axis=1)
                    )
                continue

            meta, cal, pre_cals = self._train_meta_for_season(
                season_data, ensemble_config, active_model_names,
                holdout, train_seasons,
            )

            season_data[holdout] = apply_ensemble_postprocessing(
                season_data[holdout],
                meta,
                cal,
                ensemble_config,
                pre_calibrators=pre_cals,
            )

    def _train_meta_for_season(
        self,
        season_data: dict[int, pd.DataFrame],
        ensemble_config: dict,
        active_model_names: list[str],
        holdout: int,
        train_seasons: list[int],
    ) -> tuple:
        """Train meta-learner on train_seasons' predictions.

        Uses train_meta_learner_loso with nested CV on the training
        seasons' predictions.

        Returns (meta_learner, calibrator, pre_calibrators).
        """
        # Collect training data from non-holdout seasons
        train_dfs = [season_data[s] for s in train_seasons]
        train_combined = pd.concat(train_dfs, ignore_index=True)

        # Determine which model columns are actually present
        available_models = [
            name for name in active_model_names
            if f"prob_{name}" in train_combined.columns
        ]

        if not available_models:
            raise ValueError("No model predictions available for meta-learner training")

        # Build model_preds dict
        model_preds = {
            name: train_combined[f"prob_{name}"].values
            for name in available_models
        }

        y_true = train_combined["result"].values.astype(float)
        seed_diffs = train_combined["diff_seed_num"].values.astype(float)
        season_labels = train_combined["season"].values

        # Extra features
        extra_features = None
        meta_feature_names = ensemble_config.get("meta_features", [])
        if meta_feature_names:
            extra_features = {}
            for feat_name in meta_feature_names:
                if feat_name in train_combined.columns:
                    extra_features[feat_name] = train_combined[feat_name].values

        meta, cal, pre_cals = train_meta_learner_loso(
            y_true=y_true,
            model_preds=model_preds,
            seed_diffs=seed_diffs,
            season_labels=season_labels,
            model_names=available_models,
            ensemble_config=ensemble_config,
            extra_features=extra_features if extra_features else None,
        )

        return meta, cal, pre_cals

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_backtest_metrics(
        self,
        season_data: dict[int, pd.DataFrame],
        active_model_names: list[str],
    ) -> dict[str, Any]:
        """Compute pooled and per-fold metrics from season_data.

        Uses BacktestRunner from easyml-models.
        """
        bt_config = self.config.backtest

        per_fold_data: dict[int, dict] = {}
        for season_id, df in sorted(season_data.items()):
            if "prob_ensemble" not in df.columns:
                continue

            per_fold_data[season_id] = {
                "preds": {"ensemble": df["prob_ensemble"].values},
                "y": df["result"].values.astype(float),
            }

        if not per_fold_data:
            return {
                "status": "success",
                "metrics": {},
                "per_fold": {},
                "models_trained": active_model_names,
            }

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
