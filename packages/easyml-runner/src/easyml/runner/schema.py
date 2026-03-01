"""Project-level Pydantic models for YAML-driven orchestration.

These schemas define the shape of a project's configuration — models,
ensemble, backtest, features, sources, experiments, guardrails, and
MCP server definitions.  They live in the runner package (not
easyml-schemas) because they are orchestration concerns.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator


# -----------------------------------------------------------------------
# Feature & source declarations
# -----------------------------------------------------------------------

class FeatureDecl(BaseModel):
    """Declares a feature pointing to a Python module."""

    module: str
    function: str
    category: str
    level: Literal["team", "matchup"]
    columns: list[str]
    nan_strategy: str = "median"


class SourceDecl(BaseModel):
    """Declares a data source."""

    module: str
    function: str
    category: str
    temporal_safety: Literal["pre_tournament", "post_tournament", "mixed", "unknown"]
    outputs: list[str]
    leakage_notes: str = ""


# -----------------------------------------------------------------------
# Features config (pipeline-level feature settings)
# -----------------------------------------------------------------------

class FeaturesConfig(BaseModel):
    """Pipeline-level feature computation settings."""

    first_season: int = 2003
    momentum_window: int = 10


# -----------------------------------------------------------------------
# Data config
# -----------------------------------------------------------------------

class DataConfig(BaseModel):
    """Data directory layout."""

    raw_dir: str
    processed_dir: str
    features_dir: str
    gender: str = "M"
    predictions_dir: str | None = None
    survival_dir: str | None = None
    outputs_dir: str | None = None


# -----------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------

# Known model types — extensible via register_type() / unregister_type()
_KNOWN_MODEL_TYPES: set[str] = {
    "xgboost",
    "xgboost_regression",
    "catboost",
    "lightgbm",
    "random_forest",
    "logistic_regression",
    "elastic_net",
    "mlp",
    "tabnet",
    "gnn",
    "survival",
}


class ModelDef(BaseModel):
    """Single model definition."""

    type: str
    features: list[str] = []
    feature_sets: list[str] = []
    params: dict[str, Any] = {}
    active: bool = True
    mode: Literal["classifier", "regressor"] = "classifier"
    n_seeds: int = 1
    prediction_type: str | None = None
    train_seasons: str = "all"
    pre_calibration: str | None = None
    cdf_scale: float | None = None
    training_filter: dict[str, Any] | None = None

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        if v not in _KNOWN_MODEL_TYPES:
            raise ValueError(
                f"Unknown model type {v!r}. "
                f"Known types: {sorted(_KNOWN_MODEL_TYPES)}. "
                f"Use ModelDef.register_type() to add custom types."
            )
        return v

    @classmethod
    def register_type(cls, type_name: str) -> None:
        """Register a custom model type so it passes validation."""
        _KNOWN_MODEL_TYPES.add(type_name)

    @classmethod
    def unregister_type(cls, type_name: str) -> None:
        """Remove a custom model type from the known set."""
        _KNOWN_MODEL_TYPES.discard(type_name)


# -----------------------------------------------------------------------
# Ensemble definition
# -----------------------------------------------------------------------

class EnsembleDef(BaseModel):
    """Ensemble configuration."""

    method: Literal["stacked", "average"]
    meta_learner: dict[str, Any] = {}
    pre_calibration: dict[str, str] = {}
    calibration: str = "spline"
    spline_prob_max: float = 0.985
    spline_n_bins: int = 20
    meta_features: list[str] = []
    seed_compression: float = 0.0
    seed_compression_threshold: int = 4
    temperature: float = 1.0
    clip_floor: float = 0.0
    availability_adjustment: float = 0.1
    exclude_models: list[str] = []


# -----------------------------------------------------------------------
# Backtest config
# -----------------------------------------------------------------------

_CV_STRATEGIES = {"leave_one_season_out", "expanding_window", "sliding_window", "purged_kfold"}


class BacktestConfig(BaseModel):
    """Backtest configuration."""

    cv_strategy: str
    seasons: list[int] = []
    metrics: list[str] = ["brier", "accuracy", "ece", "logloss"]
    min_train_folds: int = 1

    @field_validator("cv_strategy")
    @classmethod
    def _validate_cv_strategy(cls, v: str) -> str:
        if v not in _CV_STRATEGIES:
            raise ValueError(
                f"Invalid cv_strategy {v!r}. "
                f"Must be one of: {sorted(_CV_STRATEGIES)}"
            )
        return v


# -----------------------------------------------------------------------
# Experiment definition
# -----------------------------------------------------------------------

class ExperimentDef(BaseModel):
    """Experiment protocol configuration."""

    naming_pattern: str | None = None
    log_path: str | None = None
    experiments_dir: str | None = None
    do_not_retry_path: str | None = None


# -----------------------------------------------------------------------
# Guardrail definition
# -----------------------------------------------------------------------

class GuardrailDef(BaseModel):
    """Guardrail configuration."""

    feature_leakage_denylist: list[str] = []
    critical_paths: list[str] = []
    naming_pattern: str | None = None
    rate_limit_seconds: int | None = None


# -----------------------------------------------------------------------
# Server / MCP tool definitions
# -----------------------------------------------------------------------

class ServerToolDef(BaseModel):
    """One MCP tool definition."""

    command: str
    args: list[str] = []
    guardrails: list[str] = []
    description: str | None = None
    timeout: int | None = None


class ServerDef(BaseModel):
    """MCP server configuration."""

    name: str
    tools: dict[str, ServerToolDef] = {}
    inspection: list[str] = []


# -----------------------------------------------------------------------
# Top-level project config
# -----------------------------------------------------------------------

class ProjectConfig(BaseModel):
    """Top-level project configuration, assembled from YAML files."""

    data: DataConfig
    models: dict[str, ModelDef]
    ensemble: EnsembleDef
    backtest: BacktestConfig
    feature_config: FeaturesConfig | None = None
    features: dict[str, FeatureDecl] | None = None
    sources: dict[str, SourceDecl] | None = None
    experiments: ExperimentDef | None = None
    guardrails: GuardrailDef | None = None
    server: ServerDef | None = None
