"""Backward-compat shim — re-exports from easyml.core.schemas.contracts."""

from easyml.core.schemas.contracts import (  # noqa: F401
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
