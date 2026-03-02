"""YAML-driven orchestration layer for easyml."""

from easyml.runner.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SplineCalibrator,
    build_calibrator,
    temperature_scale,
)
from easyml.runner import config_writer
from easyml.runner.cv_strategies import generate_cv_folds
from easyml.runner.dag import (
    build_provider_map,
    detect_cycle,
    infer_dependencies,
    topological_waves,
)
from easyml.runner.data_ingest import (
    IngestResult,
    drop_duplicates,
    fill_nulls,
    ingest_dataset,
    rename_columns,
    validate_dataset,
)
from easyml.runner.data_utils import (
    get_feature_columns,
    get_features_path,
    load_data_config,
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
    auto_log_result,
    auto_next_id,
    compute_deltas,
    detect_experiment_changes,
    format_change_summary,
    format_delta_table,
    format_sweep_summary,
    load_baseline_metrics,
    promote_experiment,
    run_sweep,
    save_frozen_config,
)
from easyml.runner.feature_discovery import (
    compute_feature_correlations,
    compute_feature_importance,
    detect_redundant_features,
    format_discovery_report,
    suggest_feature_groups,
    suggest_features,
)
from easyml.runner.feature_engine import (
    FeatureResult,
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
from easyml.runner.pipeline_planner import PipelinePlan, PipelineStep, plan_execution
from easyml.runner.prediction_cache import PredictionCache
from easyml.runner.presets import apply_preset, get_preset, list_presets
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
    CastStep,
    DataConfig,
    DeriveStep,
    DistinctStep,
    EnsembleDef,
    ExperimentDef,
    FeatureDecl,
    FeaturesConfig,
    FilterStep,
    GroupByStep,
    GuardrailDef,
    InjectionDef,
    InteractionDef,
    JoinStep,
    ModelDef,
    ProjectConfig,
    SelectStep,
    ServerDef,
    ServerToolDef,
    SortStep,
    SourceDecl,
    TransformStep,
    UnionStep,
    UnpivotStep,
    ViewDef,
)
from easyml.runner.server_gen import GeneratedServer, ToolSpec, generate_server
from easyml.runner.stage_guards import PipelineGuards
from easyml.runner.sweep import expand_sweep
from easyml.runner.training import (
    predict_single_model,
    train_single_model,
)
from easyml.runner.transformation_tester import (
    TransformationReport,
    TransformationResult,
    run_transformation_tests,
)
from easyml.runner.validator import ValidationResult, validate_project

__all__ = [
    # --- Classes ---
    "BacktestConfig",
    "CastStep",
    "ChangeSet",
    "DataConfig",
    "DeriveStep",
    "DistinctStep",
    "DataProfile",
    "EnsembleDef",
    "ExperimentDef",
    "ExperimentResult",
    "FeatureDecl",
    "FeatureResult",
    "FeaturesConfig",
    "FilterStep",
    "GeneratedServer",
    "GroupByStep",
    "GuardrailDef",
    "IngestResult",
    "InjectionDef",
    "InteractionDef",
    "IsotonicCalibrator",
    "JoinStep",
    "ModelDef",
    "PipelineGuards",
    "PipelinePlan",
    "PipelineRunner",
    "PipelineStep",
    "PlattCalibrator",
    "PredictionCache",
    "Project",
    "ProjectConfig",
    "RunManager",
    "SelectStep",
    "ServerDef",
    "ServerToolDef",
    "SortStep",
    "SourceDecl",
    "SplineCalibrator",
    "StackedEnsemble",
    "ToolSpec",
    "TransformStep",
    "TransformationReport",
    "TransformationResult",
    "UnionStep",
    "UnpivotStep",
    "ValidationResult",
    "ViewDef",
    # --- Functions ---
    "apply_ensemble_postprocessing",
    "apply_preset",
    "auto_log_result",
    "auto_next_id",
    "build_calibrator",
    "build_diagnostics_report",
    "build_pick_log",
    "build_provider_map",
    "compute_brier_score",
    "compute_calibration_curve",
    "compute_deltas",
    "compute_ece",
    "compute_feature_correlations",
    "compute_feature_importance",
    "compute_feature_schema",
    "compute_fingerprint",
    "compute_interactions",
    "compute_meta_fingerprint",
    "compute_model_agreement",
    "compute_pooled_metrics",
    "detect_cycle",
    "detect_experiment_changes",
    "drop_duplicates",
    "detect_redundant_features",
    "evaluate_season_predictions",
    "expand_sweep",
    "export_backtest_artifacts",
    "fill_nulls",
    "format_change_summary",
    "format_delta_table",
    "format_discovery_report",
    "format_sweep_summary",
    "generate_cv_folds",
    "generate_markdown_report",
    "generate_pairwise_matchups",
    "generate_server",
    "get_feature_columns",
    "get_features_path",
    "get_preset",
    "group_features_by_category",
    "infer_dependencies",
    "ingest_dataset",
    "inject_features",
    "is_cached",
    "list_presets",
    "load_baseline_metrics",
    "load_data_config",
    "load_features",
    "load_meta_cache",
    "load_sources",
    "plan_execution",
    "predict_all_matchups",
    "predict_single_model",
    "profile_dataset",
    "promote_experiment",
    "rename_columns",
    "resolve_model_features",
    "run_sweep",
    "run_transformation_tests",
    "save_fingerprint",
    "save_frozen_config",
    "save_meta_cache",
    "scaffold_project",
    "suggest_feature_groups",
    "suggest_features",
    "temperature_scale",
    "topological_waves",
    "train_meta_learner_loso",
    "train_single_model",
    "validate_dataset",
    "validate_model_features",
    "validate_project",
    "validate_registry_coverage",
    # --- Modules (tool surfaces) ---
    "config_writer",
]
