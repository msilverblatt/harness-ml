"""Shared contracts and metrics for EasyML."""

from easyml.schemas.core import (
    ArtifactDecl,
    EnsembleConfig,
    ExperimentResult,
    FeatureMeta,
    Fold,
    GuardrailViolation,
    ModelConfig,
    PipelineConfig,
    RunManifest,
    SourceMeta,
    StageConfig,
    TemporalFilter,
)
from easyml.schemas.metrics import (
    accuracy,
    auc_roc,
    brier_score,
    calibration_table,
    ece,
    f1,
    log_loss,
    mae,
    model_audit,
    model_correlations,
    r_squared,
    rmse,
)

__all__ = [
    # Core schemas
    "ArtifactDecl",
    "EnsembleConfig",
    "ExperimentResult",
    "FeatureMeta",
    "Fold",
    "GuardrailViolation",
    "ModelConfig",
    "PipelineConfig",
    "RunManifest",
    "SourceMeta",
    "StageConfig",
    "TemporalFilter",
    # Metrics
    "accuracy",
    "auc_roc",
    "brier_score",
    "calibration_table",
    "ece",
    "f1",
    "log_loss",
    "mae",
    "model_audit",
    "model_correlations",
    "r_squared",
    "rmse",
]
