"""YAML-driven orchestration layer for easyml."""

from easyml.runner.fingerprint import (
    compute_fingerprint,
    compute_meta_fingerprint,
    is_cached,
    load_meta_cache,
    save_fingerprint,
    save_meta_cache,
)
from easyml.runner.loaders import load_features, load_sources
from easyml.runner.pipeline import PipelineRunner
from easyml.runner.scaffold import scaffold_project
from easyml.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    ExperimentDef,
    FeatureDecl,
    GuardrailDef,
    ModelDef,
    ProjectConfig,
    ServerDef,
    ServerToolDef,
    SourceDecl,
)
from easyml.runner.server_gen import GeneratedServer, ToolSpec, generate_server
from easyml.runner.training import (
    predict_single_model,
    train_single_model,
)
from easyml.runner.validator import ValidationResult, validate_project

__all__ = [
    "BacktestConfig",
    "DataConfig",
    "EnsembleDef",
    "ExperimentDef",
    "FeatureDecl",
    "GeneratedServer",
    "GuardrailDef",
    "ModelDef",
    "PipelineRunner",
    "ProjectConfig",
    "ServerDef",
    "ServerToolDef",
    "SourceDecl",
    "ToolSpec",
    "ValidationResult",
    "compute_fingerprint",
    "compute_meta_fingerprint",
    "generate_server",
    "is_cached",
    "load_features",
    "load_meta_cache",
    "load_sources",
    "predict_single_model",
    "save_fingerprint",
    "save_meta_cache",
    "scaffold_project",
    "train_single_model",
    "validate_project",
]
