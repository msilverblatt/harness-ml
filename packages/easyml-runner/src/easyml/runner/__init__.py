"""YAML-driven orchestration layer for easyml."""

from easyml.runner.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SplineCalibrator,
    build_calibrator,
    temperature_scale,
)
from easyml.runner.diagnostics import (
    compute_brier_score,
    compute_calibration_curve,
    compute_ece,
    compute_pooled_metrics,
    evaluate_season_predictions,
)
from easyml.runner.fingerprint import (
    compute_fingerprint,
    compute_meta_fingerprint,
    is_cached,
    load_meta_cache,
    save_fingerprint,
    save_meta_cache,
)
from easyml.runner.loaders import load_features, load_sources
from easyml.runner.matchups import (
    generate_pairwise_matchups,
    predict_all_matchups,
)
from easyml.runner.meta_learner import (
    StackedEnsemble,
    train_meta_learner_loso,
)
from easyml.runner.pipeline import PipelineRunner
from easyml.runner.postprocessing import apply_ensemble_postprocessing
from easyml.runner.run_manager import RunManager
from easyml.runner.scaffold import scaffold_project
from easyml.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    ExperimentDef,
    FeatureDecl,
    FeaturesConfig,
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
    "FeaturesConfig",
    "GeneratedServer",
    "GuardrailDef",
    "IsotonicCalibrator",
    "ModelDef",
    "PipelineRunner",
    "PlattCalibrator",
    "ProjectConfig",
    "RunManager",
    "ServerDef",
    "ServerToolDef",
    "SourceDecl",
    "SplineCalibrator",
    "StackedEnsemble",
    "ToolSpec",
    "ValidationResult",
    "apply_ensemble_postprocessing",
    "build_calibrator",
    "compute_brier_score",
    "compute_calibration_curve",
    "compute_ece",
    "compute_fingerprint",
    "compute_meta_fingerprint",
    "compute_pooled_metrics",
    "evaluate_season_predictions",
    "generate_pairwise_matchups",
    "generate_server",
    "is_cached",
    "load_features",
    "load_meta_cache",
    "load_sources",
    "predict_all_matchups",
    "predict_single_model",
    "save_fingerprint",
    "save_meta_cache",
    "scaffold_project",
    "temperature_scale",
    "train_meta_learner_loso",
    "train_single_model",
    "validate_project",
]
