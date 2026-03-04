"""PipelineRunner — wire library APIs from YAML config.

Orchestrates model training, backtesting, and evaluation by loading
ProjectConfig and delegating to easyml-models components.  Supports
real ensemble backtesting with stacked meta-learner and per-model
feature subsets, training season filtering, and regressor models.
Provider models (whose outputs become features for downstream models)
are trained in dependency order using topological wave sorting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from easyml.core.models.backtest import BacktestRunner
from easyml.core.models.orchestrator import TrainOrchestrator
from easyml.core.models.registry import ModelRegistry
from easyml.core.runner.cv_strategies import generate_cv_folds
from easyml.core.runner.dag import build_provider_map, infer_dependencies, topological_waves
from easyml.core.runner.fingerprint import compute_fingerprint
from easyml.core.runner.prediction_cache import PredictionCache
from easyml.core.runner.meta_learner import train_meta_learner_loso
from easyml.core.runner.postprocessing import apply_ensemble_postprocessing
from easyml.core.runner.schema import ModelDef, ProjectConfig
from easyml.core.runner.stage_guards import PipelineGuards
from easyml.core.runner.training import (
    predict_single_model,
    train_single_model,
)
from easyml.core.runner.validator import validate_project

logger = logging.getLogger(__name__)

# Column name mappings: mm-style -> easyml-style
_COLUMN_RENAMES = {
    "TeamAWon": "result",
    "TeamAMargin": "margin",
    "Season": "season",
}


# -----------------------------------------------------------------------
# ProviderContext — in-memory store for provider outputs during a fold
# -----------------------------------------------------------------------

@dataclass
class ProviderContext:
    """In-memory store for provider model outputs during a single fold.

    Matchup-level providers store raw prediction arrays keyed by
    column name and split (train/test).  Team-level providers store
    team-season DataFrames that get differenced into matchup pairs
    at injection time.
    """

    # {provider_name: {col_name: {"train": array, "test": array}}}
    matchup: dict[str, dict[str, dict[str, np.ndarray]]] = field(
        default_factory=dict
    )

    # {provider_name: DataFrame with team-season rows}
    team: dict[str, pd.DataFrame] = field(default_factory=dict)

    def store_matchup(
        self,
        model_name: str,
        provides: list[str],
        train_values: np.ndarray,
        test_values: np.ndarray,
    ) -> None:
        """Store matchup-level provider predictions for both splits."""
        self.matchup[model_name] = {}
        for col in provides:
            self.matchup[model_name][col] = {
                "train": train_values,
                "test": test_values,
            }

    def store_team(
        self,
        model_name: str,
        team_df: pd.DataFrame,
    ) -> None:
        """Store team-level provider predictions (team-season rows)."""
        self.team[model_name] = team_df

    def inject(
        self,
        df: pd.DataFrame,
        deps: set[str],
        models: dict[str, ModelDef],
        split: str,
    ) -> pd.DataFrame:
        """Inject upstream provider features into a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame to inject features into.
        deps : set[str]
            Provider model names this consumer depends on.
        models : dict[str, ModelDef]
            All model definitions (to look up provides/provides_level).
        split : str
            "train" or "test" — which stored predictions to use.

        Returns
        -------
        pd.DataFrame
            DataFrame with provider feature columns added.
        """
        if not deps:
            return df

        df = df.copy()

        for dep_name in sorted(deps):
            dep_def = models[dep_name]

            if dep_def.provides_level == "team" and dep_name in self.team:
                df = self._inject_team(df, dep_name, dep_def)
            elif dep_name in self.matchup:
                for col in dep_def.provides:
                    if col in self.matchup[dep_name]:
                        values = self.matchup[dep_name][col][split]
                        # Matchup-level: value IS the diff already
                        df[col] = values
                        df[f"diff_{col}"] = values

        return df

    def _inject_team(
        self,
        df: pd.DataFrame,
        provider_name: str,
        provider_def: ModelDef,
    ) -> pd.DataFrame:
        """Inject team-level features via TeamA/TeamB lookup + differencing."""
        team_df = self.team[provider_name]

        # Detect team ID and season column names
        team_id_col = next(
            (c for c in ("team_id", "TeamID") if c in team_df.columns),
            None,
        )
        season_col = next(
            (c for c in ("season", "Season") if c in team_df.columns),
            None,
        )
        if not team_id_col or not season_col:
            logger.warning(
                "Team provider %s output missing team_id/season columns, "
                "skipping injection",
                provider_name,
            )
            return df

        # Detect matchup team columns
        team_a_col = next(
            (c for c in ("TeamA", "team_a") if c in df.columns), None
        )
        team_b_col = next(
            (c for c in ("TeamB", "team_b") if c in df.columns), None
        )
        df_season_col = "season" if "season" in df.columns else "Season"

        if not team_a_col or not team_b_col:
            logger.warning(
                "Cannot inject team-level features: missing team columns"
            )
            return df

        for col in provider_def.provides:
            if col not in team_df.columns:
                continue

            lookup = team_df.set_index([team_id_col, season_col])[col].to_dict()

            a_vals = [
                lookup.get((row[team_a_col], row[df_season_col]), 0.0)
                for _, row in df.iterrows()
            ]
            b_vals = [
                lookup.get((row[team_b_col], row[df_season_col]), 0.0)
                for _, row in df.iterrows()
            ]

            df[f"diff_{col}"] = np.array(a_vals) - np.array(b_vals)

        return df


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
        config_dir: str | Path | None = None,
        variant: str | None = None,
        overlay: dict | None = None,
        enable_guards: bool = True,
        config: ProjectConfig | None = None,
        prediction_cache: PredictionCache | None = None,
        run_dir: str | Path | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.config_dir = Path(config_dir) if config_dir else None
        self.variant = variant
        self.overlay = overlay
        self.enable_guards = enable_guards
        self.run_dir = Path(run_dir) if run_dir else None

        self.config: ProjectConfig | None = config
        self._registry: ModelRegistry | None = None
        self._df: pd.DataFrame | None = None
        self._team_df: pd.DataFrame | None = None
        self._guards: PipelineGuards | None = None
        self._pred_cache = prediction_cache
        self._cache_stats: dict[str, int] = {"hits": 0, "misses": 0}
        self._failed_models: set[str] = set()
        self._model_cdf_scales: dict[str, list[float]] = {}

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return a copy of the cache hit/miss stats."""
        return dict(self._cache_stats)

    def load(self) -> None:
        """Validate config and set up ModelRegistry.

        If a ProjectConfig was provided at construction time, uses it
        directly without loading YAML from disk.
        """
        if self.config is None:
            # Load from YAML files on disk
            if self.config_dir is None:
                raise ValueError(
                    "Either config_dir or config must be provided"
                )
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
        self._guards = PipelineGuards(
            self.config.data,
            self.project_dir,
            enabled=self.enable_guards,
        )

        # Validate model features against declared feature registry
        if self.config.features:
            from easyml.core.runner.feature_utils import validate_model_features
            for model_name, model_def in self.config.models.items():
                warnings = validate_model_features(model_def, self.config.features, model_name)
                for w in warnings:
                    logger.warning(w)

        # Validate model types against model registry
        from easyml.core.runner.feature_utils import validate_registry_coverage
        registry_warnings = validate_registry_coverage(self.config, self._registry)
        for w in registry_warnings:
            logger.warning(w)

        self._load_data()

    def _load_data(self) -> None:
        """Read features DataFrame and normalize column names.

        Uses get_features_df() which resolves views if configured,
        otherwise falls back to the features parquet file.

        Also loads team features if any model has
        ``provides_level="team"``.
        """
        from easyml.core.runner.data_utils import get_features_df

        self._df = get_features_df(self.project_dir, self.config.data)
        self._normalize_columns()

        # If declarative features are configured, compute them via FeatureStore
        if self.config.data.feature_defs:
            from easyml.core.runner.feature_store import FeatureStore

            store = FeatureStore(self.project_dir, self.config.data)

            # Pre-compute all registered features.  For team features this
            # triggers pairwise derivative generation (diff_*, ratio_*) and
            # registers the derivatives in the store's internal registry.
            store.compute_all()

            # Determine which features are needed by active models
            needed: set[str] = set()
            for name, model in self._get_active_models().items():
                needed.update(model.features)
                if model.feature_sets:
                    needed.update(store.resolve_sets(model.feature_sets))

            # Now that derivatives are registered, resolve everything needed
            store_features = {f.name for f in store.available()}
            computable = needed & store_features

            if computable:
                feature_df = store.compute_all(list(computable))
                for col in feature_df.columns:
                    if col not in self._df.columns:
                        self._df[col] = feature_df[col].values

        # Load team features if any model has provides_level="team"
        has_team_providers = any(
            m.provides and m.provides_level == "team"
            for m in self.config.models.values()
        )
        if has_team_providers and self.config.data.team_features_path:
            team_path = Path(self.config.data.team_features_path)
            if not team_path.is_absolute():
                team_path = self.project_dir / team_path
            if team_path.exists():
                self._team_df = pd.read_parquet(team_path)
                logger.info("Loaded team features: %s", team_path)
            else:
                logger.warning(
                    "Team features path configured but not found: %s",
                    team_path,
                )

        # Apply feature injections if configured
        if self.config.injections:
            self._apply_injections()

        # Apply interaction features if configured
        if self.config.interactions:
            from easyml.core.runner.matchups import compute_interactions
            self._df = compute_interactions(self._df, self.config.interactions)

    def _normalize_columns(self) -> None:
        """Auto-detect mm-style columns and rename to easyml convention."""
        for old_name, new_name in _COLUMN_RENAMES.items():
            if old_name in self._df.columns and new_name not in self._df.columns:
                self._df = self._df.rename(columns={old_name: new_name})

    def _apply_injections(self) -> None:
        """Apply configured feature injections to the loaded data."""
        from easyml.core.runner.feature_utils import inject_features

        for inj_name, inj_def in self.config.injections.items():
            if "{season}" in (inj_def.path_pattern or ""):
                # Per-season injection
                for season in self._df["season"].unique():
                    mask = self._df["season"] == season
                    season_df = self._df[mask].copy()
                    injected = inject_features(season_df, inj_def, season=int(season))
                    for col in inj_def.columns:
                        if col in injected.columns:
                            self._df.loc[mask, col] = injected[col].values
            else:
                self._df = inject_features(self._df, inj_def)

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

        Models are trained in dependency order: providers first,
        then consumers.  When fingerprint caching is active,
        retraining a provider automatically invalidates all
        downstream dependents (upstream fingerprints are included
        in each model's cache key).

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

        if self._guards:
            self._guards.guard_train()

        output_dir = self.project_dir / "models"
        if run_id:
            output_dir = output_dir / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        active_models = self._get_active_models()
        if not active_models:
            raise ValueError("No active models to train.")

        # Compute dependency waves for training order
        pmap = build_provider_map(active_models)
        deps = infer_dependencies(active_models, pmap)
        waves = topological_waves(deps)

        context = ProviderContext()
        trained_names: list[str] = []
        # Track fingerprints so dependents can include upstream hashes
        model_fingerprints: dict[str, str] = {}

        for wave in waves:
            for model_name in wave:
                model_def = active_models[model_name]
                model_deps = deps.get(model_name, set())

                # Inject provider features from upstream models
                model_train_df = context.inject(
                    self._df, model_deps, active_models, "train",
                )

                try:
                    model, feature_cols, metrics = train_single_model(
                        model_name=model_name,
                        model_def=model_def,
                        train_df=model_train_df,
                        registry=self._registry,
                    )

                    # Compute fingerprint including upstream dependencies
                    from easyml.core.runner.fingerprint import save_fingerprint
                    upstream_fps = {
                        dep: model_fingerprints[dep]
                        for dep in model_deps
                        if dep in model_fingerprints
                    }
                    fp = compute_fingerprint(
                        model_config=model_def.model_dump(),
                        upstream_fingerprints=upstream_fps or None,
                    )
                    model_fingerprints[model_name] = fp
                    save_fingerprint(output_dir, model_name, fp)

                    # If this model provides features, predict on full
                    # data and store for downstream consumers
                    if model_def.provides and model_def.provides_level == "matchup":
                        train_preds = predict_single_model(
                            model=model,
                            model_def=model_def,
                            test_df=model_train_df,
                            feature_columns=feature_cols,
                            cdf_scale=metrics.get("cdf_scale"),
                        )
                        context.store_matchup(
                            model_name, model_def.provides,
                            train_preds, train_preds,
                        )

                    trained_names.append(model_name)

                except Exception:
                    logger.exception(
                        "Failed to train %s", model_name,
                    )
                    continue

        return {
            "status": "success",
            "models_trained": trained_names,
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

        if self._guards:
            self._guards.guard_predict()

        active_models = self._get_active_models()
        if not active_models:
            raise ValueError("No active models for prediction.")

        # Resolve feature_sets if feature declarations are configured
        if self.config.features:
            from easyml.core.runner.feature_utils import resolve_model_features
            for model_name, model_def in list(active_models.items()):
                if model_def.feature_sets:
                    resolved = resolve_model_features(model_def, self.config.features)
                    active_models[model_name] = model_def.model_copy(
                        update={"features": resolved}
                    )

        # Compute dependency order for training
        pmap = build_provider_map(active_models)
        deps = infer_dependencies(active_models, pmap)
        waves = topological_waves(deps)

        # Split data
        train_df = self._df[self._df["season"] < season].copy()
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

        # Train and predict in wave order (providers before consumers)
        context = ProviderContext()
        trained_count = 0

        for wave in waves:
            for model_name in wave:
                model_def = active_models[model_name]
                model_deps = deps.get(model_name, set())

                model_train_df = context.inject(
                    train_df, model_deps, active_models, "train",
                )
                model_test_df = context.inject(
                    test_df, model_deps, active_models, "test",
                )

                try:
                    model, feature_cols, metrics = train_single_model(
                        model_name=model_name,
                        model_def=model_def,
                        train_df=model_train_df,
                        registry=self._registry,
                        target_season=season,
                    )

                    cdf_scale = metrics.get("cdf_scale")
                    probs = predict_single_model(
                        model=model,
                        model_def=model_def,
                        test_df=model_test_df,
                        feature_columns=feature_cols,
                        cdf_scale=cdf_scale,
                    )

                    # Store provider outputs for downstream consumers
                    if model_def.provides and model_def.provides_level == "matchup":
                        train_preds = predict_single_model(
                            model=model,
                            model_def=model_def,
                            test_df=model_train_df,
                            feature_columns=feature_cols,
                            cdf_scale=cdf_scale,
                        )
                        context.store_matchup(
                            model_name, model_def.provides,
                            train_preds, probs,
                        )

                    if model_def.include_in_ensemble:
                        preds_df[f"prob_{model_name}"] = probs

                    trained_count += 1

                except Exception:
                    logger.exception(
                        "Failed to train/predict %s for season %d",
                        model_name, season,
                    )
                    continue

        if trained_count == 0:
            raise ValueError("No models could be trained for prediction.")

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
        """Run real ensemble backtesting with cross-validation.

        Two-pass approach:
          Pass 1: Per CV fold, train all active models on the training
                  seasons, predict the test season's matchups.
          Pass 2: Train LOSO meta-learner (one per held-out season), apply
                  ensemble post-processing.

        Returns
        -------
        dict
            Result dict with status, metrics, per_fold, models_trained.
        """
        if self.config is None:
            raise RuntimeError("Call load() before backtest()")

        if self._guards:
            self._guards.guard_backtest()

        bt_config = self.config.backtest
        ensemble_config = self.config.ensemble.model_dump()
        active_models = self._get_active_models()

        if not active_models:
            raise ValueError("No active models to backtest.")

        self._failed_models = set()
        self._model_cdf_scales = {}

        # Generate CV folds from strategy
        cv_folds = generate_cv_folds(self._df, bt_config)

        # Pass 1: per-fold OOF predictions
        season_data: dict[int, pd.DataFrame] = {}
        for train_seasons, test_season in cv_folds:
            preds_df = self._generate_season_predictions_from_fold(
                train_seasons, test_season, active_models
            )
            if preds_df is not None and len(preds_df) > 0:
                season_data[test_season] = preds_df

        if not season_data:
            raise ValueError("No valid holdout seasons produced predictions.")

        # Only models with include_in_ensemble=True participate in ensemble
        active_model_names = [
            name for name, m in active_models.items()
            if m.include_in_ensemble
        ]

        # Pass 2: meta-learner + post-processing
        meta_coefficients = None
        if ensemble_config["method"] == "stacked":
            meta_coefficients = self._apply_stacked_ensemble(season_data, ensemble_config, active_model_names)
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

        result = self._compute_backtest_metrics(season_data, active_model_names)

        if meta_coefficients is not None:
            result["meta_coefficients"] = meta_coefficients

        # Generate reporting artifacts
        result = self._generate_report(result, season_data)

        failed = sorted(self._failed_models)
        result["models_failed"] = failed
        if failed:
            result["models_trained"] = [
                m for m in result.get("models_trained", [])
                if m not in self._failed_models
            ]

        if self._model_cdf_scales:
            result["model_cdf_scales"] = {
                name: sum(vals) / len(vals)
                for name, vals in self._model_cdf_scales.items()
            }

        return result

    def _generate_report(
        self,
        result: dict[str, Any],
        season_data: dict[int, pd.DataFrame],
    ) -> dict[str, Any]:
        """Generate diagnostics report, pick log, and markdown report.

        If ``run_dir`` is set, exports all artifacts to that directory.
        """
        from easyml.core.runner.reporting import (
            build_diagnostics_report,
            build_pick_log,
            export_backtest_artifacts,
            generate_markdown_report,
        )

        # Build per-season diagnostics
        diagnostics_df = build_diagnostics_report(season_data)

        # Build combined pick log across seasons
        pick_logs = []
        for season, df in sorted(season_data.items()):
            if "prob_ensemble" in df.columns:
                pick_logs.append(build_pick_log(df, season))
        pick_log = pd.concat(pick_logs, ignore_index=True) if pick_logs else pd.DataFrame()

        # Wrap pooled metrics for generate_markdown_report
        pooled_for_report = {"ensemble": result.get("metrics", {})}

        # Extract meta-learner coefficients if available
        meta_coefficients = result.get("meta_coefficients")
        if meta_coefficients is not None:
            pooled_for_report["meta_coefficients"] = meta_coefficients

        report_md = generate_markdown_report(
            pooled_for_report,
            diagnostics_df=diagnostics_df if len(diagnostics_df) > 0 else None,
            pick_log=pick_log if len(pick_log) > 0 else None,
            meta_coefficients=meta_coefficients,
        )

        result["report"] = report_md
        result["diagnostics"] = diagnostics_df.to_dict(orient="records") if len(diagnostics_df) > 0 else []

        # Export artifacts to run_dir if configured
        if self.run_dir is not None:
            export_backtest_artifacts(
                run_dir=self.run_dir,
                season_data=season_data,
                pooled_metrics=pooled_for_report,
                diagnostics_df=diagnostics_df,
                pick_log=pick_log,
                report_md=report_md,
            )
            result["run_dir"] = str(self.run_dir)

        return result

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

        Delegates to _generate_season_predictions_from_fold with all
        seasons except the holdout as training seasons.
        """
        all_seasons = sorted(self._df["season"].unique())
        train_seasons = [s for s in all_seasons if s != holdout_season]
        return self._generate_season_predictions_from_fold(
            train_seasons, holdout_season, active_models
        )

    def _generate_season_predictions_from_fold(
        self,
        train_seasons: list[int],
        test_season: int,
        active_models: dict[str, ModelDef],
    ) -> pd.DataFrame | None:
        """Train on specified seasons, predict test_season.

        Models are trained in dependency order: providers first,
        then consumers that use provider outputs as features.
        Within each wave, models are independent and can train
        in any order.

        Parameters
        ----------
        train_seasons : list[int]
            Seasons to include in training data.
        test_season : int
            Season to predict.
        active_models : dict[str, ModelDef]
            Active model definitions.

        Returns
        -------
        pd.DataFrame | None
            DataFrame with prob_{model_name} columns (only for
            models with include_in_ensemble=True), plus
            diff_seed_num and any meta_features columns,
            or None if no predictions could be generated.
        """
        train_mask = self._df["season"].isin(train_seasons)
        test_mask = self._df["season"] == test_season
        train_df = self._df[train_mask].copy()
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

        # Resolve feature_sets if feature declarations are configured
        resolved_models = dict(active_models)
        if self.config.features:
            from easyml.core.runner.feature_utils import resolve_model_features
            for model_name, model_def in active_models.items():
                if model_def.feature_sets:
                    resolved = resolve_model_features(model_def, self.config.features)
                    resolved_models[model_name] = model_def.model_copy(
                        update={"features": resolved}
                    )

        # Compute dependency waves for training order
        pmap = build_provider_map(resolved_models)
        deps = infer_dependencies(resolved_models, pmap)
        waves = topological_waves(deps)

        context = ProviderContext()
        model_fingerprints: dict[str, str] = {}

        # Train and predict in wave order (providers before consumers)
        for wave in waves:
            for model_name in wave:
                model_def = resolved_models[model_name]
                model_deps = deps.get(model_name, set())

                # Compute fingerprint (includes upstream provider fingerprints)
                upstream_fps = {
                    dep: model_fingerprints[dep]
                    for dep in model_deps
                    if dep in model_fingerprints
                }
                fp = compute_fingerprint(
                    model_config=model_def.model_dump(),
                    upstream_fingerprints=upstream_fps or None,
                )
                model_fingerprints[model_name] = fp

                # Check prediction cache (non-provider models only)
                if (
                    self._pred_cache is not None
                    and not model_def.provides
                ):
                    cached = self._pred_cache.lookup(
                        model_name, test_season, fp,
                    )
                    if cached is not None and "prediction" in cached.columns:
                        self._cache_stats["hits"] += 1
                        if model_def.include_in_ensemble:
                            preds_df[f"prob_{model_name}"] = (
                                cached["prediction"].values
                            )
                        continue

                # Inject provider features from upstream models
                model_train_df = context.inject(
                    train_df, model_deps, resolved_models, "train",
                )
                model_test_df = context.inject(
                    test_df, model_deps, resolved_models, "test",
                )

                try:
                    model, feature_cols, metrics = train_single_model(
                        model_name=model_name,
                        model_def=model_def,
                        train_df=model_train_df,
                        registry=self._registry,
                    )

                    cdf_scale = metrics.get("cdf_scale")
                    if cdf_scale is not None:
                        self._model_cdf_scales.setdefault(model_name, []).append(cdf_scale)
                    probs = predict_single_model(
                        model=model,
                        model_def=model_def,
                        test_df=model_test_df,
                        feature_columns=feature_cols,
                        cdf_scale=cdf_scale,
                    )

                    # If this model provides features, predict on train
                    # data too and store outputs for downstream consumers
                    if model_def.provides:
                        if model_def.provides_level == "matchup":
                            train_preds = predict_single_model(
                                model=model,
                                model_def=model_def,
                                test_df=model_train_df,
                                feature_columns=feature_cols,
                                cdf_scale=cdf_scale,
                            )
                            context.store_matchup(
                                model_name,
                                model_def.provides,
                                train_preds,
                                probs,
                            )
                        elif model_def.provides_level == "team":
                            # Team-level providers need special handling
                            # with team_df and per-team predictions
                            logger.info(
                                "Team-level provider %s — team feature "
                                "injection requires team_df",
                                model_name,
                            )

                    # Store in prediction cache (non-provider models only)
                    if (
                        self._pred_cache is not None
                        and not model_def.provides
                    ):
                        cache_df = pd.DataFrame({"prediction": probs})
                        self._pred_cache.store(
                            model_name, test_season, fp, cache_df,
                        )
                        self._cache_stats["misses"] += 1

                    # Only add to ensemble predictions if included
                    if model_def.include_in_ensemble:
                        preds_df[f"prob_{model_name}"] = probs

                except Exception:
                    logger.exception(
                        "Failed to train/predict %s for season %d",
                        model_name, test_season,
                    )
                    self._failed_models.add(model_name)
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
    ) -> dict[str, float] | None:
        """Apply stacked meta-learner via LOSO to all holdout seasons.

        For each holdout season, train the meta-learner on all OTHER
        holdout seasons' predictions, then predict the holdout.

        Returns the meta-learner coefficients from the last fold,
        or None if no meta-learner was trained.
        """
        holdout_seasons = sorted(season_data.keys())
        last_meta = None

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
            last_meta = meta

            season_data[holdout] = apply_ensemble_postprocessing(
                season_data[holdout],
                meta,
                cal,
                ensemble_config,
                pre_calibrators=pre_cals,
            )

        # Extract coefficients from the last trained meta-learner
        if last_meta is not None:
            try:
                return last_meta.get_coefficients()
            except Exception:
                return None
        return None

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
