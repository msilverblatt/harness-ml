"""Programmatic project builder API.

Provides a fluent Python interface for building ML pipeline configs
without manually writing YAML.  YAML becomes an optional persistence
format rather than the required authoring interface.

Example
-------
>>> project = Project("my_project")
>>> project.set_data(features_dir="data/features")
>>> project.add_model("logreg_seed", "logistic_regression", features=["diff_prior"])
>>> project.configure_backtest(fold_values=[2015, 2016, 2017, 2018, 2019])
>>> results = project.backtest()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from easyml.core.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    ModelDef,
    ProjectConfig,
    SourceDecl,
)

logger = logging.getLogger(__name__)


@dataclass
class _LeakageWarning:
    """A leakage concern found during validation."""

    model_name: str
    feature: str
    source_name: str
    temporal_safety: str
    note: str


class Project:
    """Programmatic project builder.

    Build configs via method calls instead of YAML editing.
    Validates features against actual data, checks leakage at build time,
    and can serialize to YAML for reproducibility.

    Parameters
    ----------
    project_dir : str | Path
        Root project directory (must contain data/features/{features_file}).
    """

    def __init__(self, project_dir: str | Path) -> None:
        self.project_dir = Path(project_dir)

        # Config components — built up via method calls
        self._data = DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
        )
        self._models: dict[str, ModelDef] = {}
        self._ensemble = EnsembleDef(method="stacked")
        self._backtest = BacktestConfig(
            cv_strategy="leave_one_out",
            metrics=["brier", "accuracy", "ece", "log_loss"],
        )
        self._sources: dict[str, SourceDecl] = {}

        # Data awareness
        self._data_columns: set[str] | None = None
        self._data_profile = None

    # ------------------------------------------------------------------
    # Data configuration
    # ------------------------------------------------------------------

    def set_data(
        self,
        features_dir: str = "data/features",
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        features_file: str = "features.parquet",
        task: str = "classification",
        target_column: str = "result",
        key_columns: list[str] | None = None,
        time_column: str | None = None,
        exclude_columns: list[str] | None = None,
    ) -> "Project":
        """Configure data directories and ML problem definition.

        Parameters
        ----------
        features_file : str
            Name of the features parquet file (relative to features_dir).
        task : str
            ML task type: classification, regression, ranking.
        target_column : str
            Column to predict.
        key_columns : list[str] | None
            Row identifier columns.
        time_column : str | None
            Column for temporal CV splits.
        exclude_columns : list[str] | None
            Columns to never use as features.
        """
        self._data = DataConfig(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            features_dir=features_dir,
            features_file=features_file,
            task=task,
            target_column=target_column,
            key_columns=key_columns or [],
            time_column=time_column,
            exclude_columns=exclude_columns or [],
        )
        # Reset data awareness cache when data config changes
        self._data_columns = None
        self._data_profile = None
        return self

    def _ensure_data_loaded(self) -> None:
        """Lazily load data column names from parquet."""
        if self._data_columns is not None:
            return

        parquet_path = self.project_dir / self._data.features_dir / self._data.features_file
        if not parquet_path.exists():
            logger.warning("Data file not found: %s", parquet_path)
            self._data_columns = set()
            return

        import pandas as pd

        # Read only metadata — no need to load full data
        pf = pd.read_parquet(parquet_path, columns=[])
        # Actually, read_parquet with empty columns list doesn't give us column names
        # Read the schema instead
        import pyarrow.parquet as pq

        schema = pq.read_schema(parquet_path)
        self._data_columns = set(schema.names)

    def available_features(self, prefix: str | None = None) -> list[str]:
        """List feature columns available in the dataset.

        Parameters
        ----------
        prefix : str | None
            Filter to columns starting with this prefix (e.g., "diff_").

        Returns
        -------
        list[str]
            Sorted list of available column names.
        """
        self._ensure_data_loaded()
        cols = self._data_columns or set()
        if prefix:
            cols = {c for c in cols if c.startswith(prefix)}
        return sorted(cols)

    def profile(self):
        """Profile the dataset. Returns a DataProfile object."""
        if self._data_profile is None:
            from easyml.core.runner.data_profiler import profile_dataset

            parquet_path = (
                self.project_dir / self._data.features_dir / self._data.features_file
            )
            self._data_profile = profile_dataset(parquet_path)
        return self._data_profile

    # ------------------------------------------------------------------
    # Source / leakage tracking
    # ------------------------------------------------------------------

    def add_source(
        self,
        name: str,
        *,
        temporal_safety: str,
        outputs: list[str],
        module: str = "",
        function: str = "",
        category: str = "unknown",
        leakage_notes: str = "",
    ) -> "Project":
        """Declare a data source with temporal safety classification.

        Parameters
        ----------
        name : str
            Source identifier.
        temporal_safety : str
            One of: "pre_event", "post_event", "mixed", "unknown".
        outputs : list[str]
            Column names this source produces.
        leakage_notes : str
            Human-readable explanation of the temporal safety assessment.
        """
        self._sources[name] = SourceDecl(
            module=module or f"sources.{name}",
            function=function or f"load_{name}",
            category=category,
            temporal_safety=temporal_safety,
            outputs=outputs,
            leakage_notes=leakage_notes,
        )
        return self

    def _build_feature_lineage(self) -> dict[str, tuple[str, str, str]]:
        """Build feature→(source_name, temporal_safety, leakage_notes) map."""
        lineage: dict[str, tuple[str, str, str]] = {}
        for source_name, source_decl in self._sources.items():
            for col in source_decl.outputs:
                lineage[col] = (
                    source_name,
                    source_decl.temporal_safety,
                    source_decl.leakage_notes,
                )
                # Also map diff_ variants
                lineage[f"diff_{col}"] = (
                    source_name,
                    source_decl.temporal_safety,
                    source_decl.leakage_notes,
                )
        return lineage

    def check_leakage(self) -> list[_LeakageWarning]:
        """Check all models for potential data leakage.

        Returns
        -------
        list[_LeakageWarning]
            Any leakage concerns found. Empty list means all clear.
        """
        if not self._sources:
            return []

        lineage = self._build_feature_lineage()
        warnings: list[_LeakageWarning] = []

        for model_name, model_def in self._models.items():
            for feat in model_def.features:
                if feat not in lineage:
                    continue  # Unknown source — can't check
                source_name, safety, notes = lineage[feat]
                if safety in ("post_event", "mixed", "unknown"):
                    warnings.append(
                        _LeakageWarning(
                            model_name=model_name,
                            feature=feat,
                            source_name=source_name,
                            temporal_safety=safety,
                            note=notes,
                        )
                    )

        return warnings

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def add_model(
        self,
        name: str,
        type: str | None = None,
        features: list[str] | None = None,
        params: dict[str, Any] | None = None,
        mode: str | None = None,
        active: bool = True,
        n_seeds: int | None = None,
        pre_calibration: str | None = None,
        cdf_scale: float | None = None,
        train_folds: str | None = None,
        provides: list[str] | None = None,
        provides_level: str | None = None,
        include_in_ensemble: bool = True,
        provider_isolation: str | None = None,
        preset: str | None = None,
    ) -> "Project":
        """Add a model to the pipeline.

        Feature validation is deferred to build() so that provider models
        can be added in any order (provider-generated features are resolved
        after all models are declared).

        Parameters
        ----------
        preset : str | None
            Name of a model preset (e.g., "xgboost_classifier").  The preset
            provides default values for type, mode, params, etc.  Explicit
            keyword arguments override preset values.
        provides : list[str] | None
            Feature columns this model outputs for downstream models.
        provides_level : str
            "instance" for per-instance outputs, "entity" for per-entity outputs
            that get differenced into pairwise features.
        include_in_ensemble : bool
            If False, model trains and provides features but its prob_*
            column is excluded from the meta-learner.
        provider_isolation : str
            "none" or "per_fold". "per_fold" retrains provider per
            training fold for leak-free features.
        """
        if preset is not None:
            from easyml.core.runner.presets import apply_preset

            # Build overrides from explicit kwargs
            overrides: dict[str, Any] = {}
            if type is not None:
                overrides["type"] = type
            if features is not None:
                overrides["features"] = features
            if params is not None:
                overrides["params"] = params
            if mode is not None:
                overrides["mode"] = mode
            if n_seeds is not None:
                overrides["n_seeds"] = n_seeds
            if train_folds is not None:
                overrides["train_folds"] = train_folds
            if pre_calibration is not None:
                overrides["pre_calibration"] = pre_calibration
            if cdf_scale is not None:
                overrides["cdf_scale"] = cdf_scale
            if provides is not None:
                overrides["provides"] = provides
            if provides_level is not None:
                overrides["provides_level"] = provides_level
            if provider_isolation is not None:
                overrides["provider_isolation"] = provider_isolation
            overrides["active"] = active
            overrides["include_in_ensemble"] = include_in_ensemble

            config = apply_preset(preset, overrides)
            self._models[name] = ModelDef(**config)
        else:
            if type is None:
                raise ValueError("Either 'type' or 'preset' must be specified.")
            self._models[name] = ModelDef(
                type=type,
                features=features or [],
                params=params or {},
                mode=mode or "classifier",
                active=active,
                n_seeds=n_seeds or 1,
                pre_calibration=pre_calibration,
                cdf_scale=cdf_scale,
                train_folds=train_folds or "all",
                provides=provides or [],
                provides_level=provides_level or "instance",
                include_in_ensemble=include_in_ensemble,
                provider_isolation=provider_isolation or "none",
            )
        return self

    def remove_model(self, name: str) -> "Project":
        """Remove a model from the config."""
        self._models.pop(name, None)
        return self

    def exclude_model(self, name: str) -> "Project":
        """Add a model to the ensemble exclude list (keep config, skip in backtest)."""
        excludes = list(self._ensemble.exclude_models)
        if name not in excludes:
            excludes.append(name)
        self._ensemble = self._ensemble.model_copy(
            update={"exclude_models": excludes}
        )
        return self

    # ------------------------------------------------------------------
    # Ensemble configuration
    # ------------------------------------------------------------------

    def configure_ensemble(
        self,
        method: str = "stacked",
        C: float | None = None,
        temperature: float = 1.0,
        calibration: str = "spline",
        spline_n_bins: int = 20,
        spline_prob_max: float = 0.985,
        logit_adjustments: list[dict] | None = None,
        pre_calibration: dict[str, str] | None = None,
        exclude_models: list[str] | None = None,
    ) -> "Project":
        """Configure the ensemble strategy."""
        meta_learner = {}
        if C is not None:
            meta_learner["C"] = C

        self._ensemble = EnsembleDef(
            method=method,
            meta_learner=meta_learner,
            temperature=temperature,
            calibration=calibration,
            spline_n_bins=spline_n_bins,
            spline_prob_max=spline_prob_max,
            logit_adjustments=logit_adjustments or [],
            pre_calibration=pre_calibration or {},
            exclude_models=exclude_models or list(self._ensemble.exclude_models),
        )
        return self

    # ------------------------------------------------------------------
    # Backtest configuration
    # ------------------------------------------------------------------

    def configure_backtest(
        self,
        cv_strategy: str = "leave_one_out",
        fold_values: list[int] | None = None,
        metrics: list[str] | None = None,
        min_train_folds: int = 1,
        min_train_initial: int = 3,
    ) -> "Project":
        """Configure the backtest strategy.

        If *fold_values* is not provided and data is available, auto-detects
        holdout-eligible fold values from the parquet file, reserving the
        first *min_train_initial* as training-only.

        Parameters
        ----------
        min_train_initial : int
            When auto-detecting, the earliest N fold values are reserved for
            training and excluded from the holdout list.  Ignored when
            *fold_values* is provided explicitly.
        """
        if fold_values is None:
            fold_values = self._auto_detect_folds(min_train_initial=min_train_initial)

        self._backtest = BacktestConfig(
            cv_strategy=cv_strategy,
            fold_values=fold_values or [],
            metrics=metrics or ["brier", "accuracy", "ece", "log_loss"],
            min_train_folds=min_train_folds,
        )
        return self

    def _auto_detect_folds(self, min_train_initial: int = 3) -> list[int]:
        """Detect holdout-eligible fold values from the data.

        Uses the configured time_column if set, otherwise falls back to
        heuristic column name detection.

        Returns all unique fold values after reserving the first
        *min_train_initial* for training.
        """
        parquet_path = (
            self.project_dir / self._data.features_dir / self._data.features_file
        )
        if not parquet_path.exists():
            return []

        import pandas as pd

        # Use configured time_column if available, otherwise heuristic
        time_col = None
        if self._data.time_column:
            try:
                df = pd.read_parquet(parquet_path, columns=[self._data.time_column])
                time_col = self._data.time_column
            except (KeyError, Exception):
                pass

        if time_col is None:
            # Broader heuristic fallback
            for candidate in ["Season", "season", "year", "Year", "period", "Period", "date"]:
                try:
                    df = pd.read_parquet(parquet_path, columns=[candidate])
                    time_col = candidate
                    break
                except (KeyError, Exception):
                    continue

        if time_col is None:
            return []

        all_values = sorted(df[time_col].dropna().unique().astype(int).tolist())
        if len(all_values) <= min_train_initial:
            return []
        return all_values[min_train_initial:]

    # ------------------------------------------------------------------
    # Build / validate
    # ------------------------------------------------------------------

    def build(self) -> ProjectConfig:
        """Build and validate the ProjectConfig.

        Performs two-pass feature validation so provider-generated features
        are recognized. Checks for dependency cycles and leakage.

        Raises ValueError if validation fails, cycles exist, or leakage
        is detected.
        """
        if not self._models:
            raise ValueError("No models configured. Call add_model() first.")

        # Two-pass feature validation
        self._validate_features()

        # Check for dependency cycles
        self._validate_dependency_graph()

        config = ProjectConfig(
            data=self._data,
            models=self._models,
            ensemble=self._ensemble,
            backtest=self._backtest,
            sources=self._sources if self._sources else None,
        )

        # Check leakage
        leakage = self.check_leakage()
        if leakage:
            msgs = []
            for w in leakage:
                msgs.append(
                    f"  {w.model_name}.{w.feature}: source={w.source_name}, "
                    f"temporal_safety={w.temporal_safety} — {w.note}"
                )
            raise ValueError(
                "Leakage detected in model features:\n" + "\n".join(msgs)
            )

        return config

    def _validate_features(self) -> None:
        """Two-pass feature validation.

        Pass 1: Collect all provider-generated columns.
        Pass 2: Validate each model's features against data + provider columns.
        """
        self._ensure_data_loaded()
        if not self._data_columns:
            return  # No data available to validate against

        # Pass 1: collect provider-generated columns
        provider_columns: set[str] = set()
        for model_def in self._models.values():
            for col in model_def.provides:
                provider_columns.add(col)
                provider_columns.add(f"diff_{col}")

        valid_columns = self._data_columns | provider_columns

        # Pass 2: validate each model's features
        for model_name, model_def in self._models.items():
            missing = [f for f in model_def.features if f not in valid_columns]
            if missing:
                available_diff = sorted(
                    c for c in self._data_columns if c.startswith("diff_")
                )
                raise ValueError(
                    f"Model '{model_name}': features not found in data: {missing}\n"
                    f"Available diff_ features ({len(available_diff)}): "
                    f"{available_diff[:10]}..."
                )

    def _validate_dependency_graph(self) -> None:
        """Check for cycles in the model dependency graph."""
        from easyml.core.runner.dag import (
            build_provider_map,
            infer_dependencies,
            topological_waves,
        )

        has_providers = any(m.provides for m in self._models.values())
        if not has_providers:
            return

        provider_map = build_provider_map(self._models)
        deps = infer_dependencies(self._models, provider_map)
        topological_waves(deps)  # raises ValueError on cycle

        # Team-level provider validation is handled at pipeline runtime

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------

    def backtest(self) -> dict[str, Any]:
        """Build config and run backtest.

        Returns dict with status, metrics, per_fold, models_trained.
        """
        config = self.build()

        from easyml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=self.project_dir,
            config_dir=str(self.project_dir / "config"),  # Needed for guards path
            config=config,
        )
        runner.load()
        return runner.backtest()

    def train(self, run_id: str | None = None) -> dict[str, Any]:
        """Build config and run training."""
        config = self.build()

        from easyml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=self.project_dir,
            config_dir=str(self.project_dir / "config"),
            config=config,
        )
        runner.load()
        return runner.train(run_id=run_id)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_to_yaml(self, config_dir: str | Path | None = None) -> Path:
        """Save current config to YAML files for reproducibility.

        Parameters
        ----------
        config_dir : str | Path | None
            Target directory. Defaults to {project_dir}/config.

        Returns
        -------
        Path
            The config directory path.
        """
        config = self.build()
        config_dir = Path(config_dir or self.project_dir / "config")
        config_dir.mkdir(parents=True, exist_ok=True)

        _write_yaml(
            config_dir / "pipeline.yaml",
            {
                "data": config.data.model_dump(mode="json"),
                "backtest": config.backtest.model_dump(mode="json"),
            },
        )

        _write_yaml(
            config_dir / "models.yaml",
            {"models": {k: v.model_dump(mode="json") for k, v in config.models.items()}},
        )

        _write_yaml(
            config_dir / "ensemble.yaml",
            {"ensemble": config.ensemble.model_dump(mode="json")},
        )

        if config.sources:
            _write_yaml(
                config_dir / "sources.yaml",
                {"sources": {k: v.model_dump(mode="json") for k, v in config.sources.items()}},
            )

        return config_dir


def _write_yaml(path: Path, data: dict) -> None:
    """Write a dict as YAML."""
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
