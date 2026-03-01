"""YAML-driven orchestration layer for easyml."""

from easyml.runner.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SplineCalibrator,
    build_calibrator,
    temperature_scale,
)
from easyml.runner.cv_strategies import generate_cv_folds
from easyml.runner.dag import (
    build_provider_map,
    detect_cycle,
    infer_dependencies,
    topological_waves,
)
from easyml.runner.data_profiler import DataProfile, profile_dataset
from easyml.runner.diagnostics import (
    compute_brier_score,
    compute_calibration_curve,
    compute_ece,
    compute_model_agreement,
    compute_pooled_metrics,
    evaluate_season_predictions,
)
from easyml.runner.experiment import (
    ChangeSet,
    ExperimentResult,
    compute_deltas,
    detect_experiment_changes,
    format_change_summary,
    format_delta_table,
    load_baseline_metrics,
)
from easyml.runner.feature_utils import (
    group_features_by_category,
    inject_features,
    resolve_model_features,
    validate_model_features,
    validate_registry_coverage,
)
from easyml.runner.fingerprint import (
    compute_feature_schema,
    compute_fingerprint,
    compute_meta_fingerprint,
    is_cached,
    load_meta_cache,
    save_fingerprint,
    save_meta_cache,
)
from easyml.runner.loaders import load_features, load_sources
from easyml.runner.matchups import (
    compute_interactions,
    generate_pairwise_matchups,
    predict_all_matchups,
)
from easyml.runner.meta_learner import (
    StackedEnsemble,
    train_meta_learner_loso,
)
from easyml.runner.pipeline import PipelineRunner
from easyml.runner.project import Project
from easyml.runner.postprocessing import apply_ensemble_postprocessing
from easyml.runner.reporting import (
    build_diagnostics_report,
    build_pick_log,
    export_backtest_artifacts,
    generate_markdown_report,
)
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
    InjectionDef,
    InteractionDef,
    ModelDef,
    ProjectConfig,
    ServerDef,
    ServerToolDef,
    SourceDecl,
)
from easyml.runner.server_gen import GeneratedServer, ToolSpec, generate_server
from easyml.runner.stage_guards import PipelineGuards
from easyml.runner.training import (
    predict_single_model,
    train_single_model,
)
from easyml.runner.validator import ValidationResult, validate_project

__all__ = [
    "BacktestConfig",
    "build_provider_map",
    "ChangeSet",
    "DataProfile",
    "DataConfig",
    "EnsembleDef",
    "ExperimentDef",
    "ExperimentResult",
    "FeatureDecl",
    "FeaturesConfig",
    "GeneratedServer",
    "GuardrailDef",
    "InjectionDef",
    "InteractionDef",
    "IsotonicCalibrator",
    "ModelDef",
    "PipelineGuards",
    "PipelineRunner",
    "Project",
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
    "build_diagnostics_report",
    "build_pick_log",
    "detect_cycle",
    "compute_brier_score",
    "compute_calibration_curve",
    "compute_deltas",
    "compute_ece",
    "compute_feature_schema",
    "compute_fingerprint",
    "compute_interactions",
    "compute_meta_fingerprint",
    "compute_model_agreement",
    "compute_pooled_metrics",
    "detect_experiment_changes",
    "evaluate_season_predictions",
    "export_backtest_artifacts",
    "format_change_summary",
    "format_delta_table",
    "generate_cv_folds",
    "infer_dependencies",
    "generate_markdown_report",
    "generate_pairwise_matchups",
    "generate_server",
    "group_features_by_category",
    "inject_features",
    "is_cached",
    "load_baseline_metrics",
    "load_features",
    "load_meta_cache",
    "load_sources",
    "predict_all_matchups",
    "profile_dataset",
    "predict_single_model",
    "resolve_model_features",
    "save_fingerprint",
    "save_meta_cache",
    "scaffold_project",
    "temperature_scale",
    "topological_waves",
    "train_meta_learner_loso",
    "train_single_model",
    "validate_model_features",
    "validate_project",
    "validate_registry_coverage",
]
