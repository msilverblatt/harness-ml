"""Pipeline runner, experiment management, and orchestration for EasyML."""

from easyml.core.runner.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SplineCalibrator,
    build_calibrator,
    temperature_scale,
)
from easyml.core.runner import config_writer
from easyml.core.runner.cv_strategies import generate_cv_folds
from easyml.core.runner.dag import (
    build_provider_map,
    detect_cycle,
    infer_dependencies,
    topological_waves,
)
from easyml.core.runner.data_ingest import (
    IngestResult,
    drop_duplicates,
    fill_nulls,
    ingest_dataset,
    rename_columns,
    validate_dataset,
)
from easyml.core.runner.data_utils import (
    get_feature_columns,
    get_features_df,
    get_features_path,
    load_data_config,
)
from easyml.core.runner.data_profiler import DataProfile, profile_dataset
from easyml.core.runner.diagnostics import (
    compute_brier_score,
    compute_calibration_curve,
    compute_ece,
    compute_model_agreement,
    compute_pooled_metrics,
    evaluate_fold_predictions,
)
from easyml.core.runner.experiment import (
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
from easyml.core.runner.experiment_manager import (
    ChangeReport,
    ExperimentError,
    ExperimentManager,
)
from easyml.core.runner.exploration import (
    AxisDef,
    ExplorationSpace,
    run_exploration,
)
from easyml.core.runner.feature_discovery import (
    compute_feature_correlations,
    compute_feature_importance,
    detect_redundant_features,
    format_discovery_report,
    suggest_feature_groups,
    suggest_features,
)
from easyml.core.runner.feature_engine import (
    FeatureResult,
)
from easyml.core.runner.feature_utils import (
    group_features_by_category,
    inject_features,
    resolve_model_features,
    validate_model_features,
    validate_registry_coverage,
)
from easyml.core.runner.fingerprint import (
    compute_feature_schema,
    compute_fingerprint,
    compute_meta_fingerprint,
    is_cached,
    load_meta_cache,
    save_fingerprint,
    save_meta_cache,
)
from easyml.core.runner.loaders import load_features, load_sources
from easyml.core.runner.matchups import (
    compute_interactions,
    generate_pairwise_matchups,
    predict_all_matchups,
)
from easyml.core.runner.meta_learner import (
    StackedEnsemble,
    train_meta_learner_loso,
)
from easyml.core.runner.pipeline import PipelineRunner
from easyml.core.runner.pipeline_planner import PipelinePlan, PipelineStep, plan_execution
from easyml.core.runner.prediction_cache import PredictionCache
from easyml.core.runner.presets import apply_preset, get_preset, list_presets
from easyml.core.runner.project import Project
from easyml.core.runner.postprocessing import apply_ensemble_postprocessing
from easyml.core.runner.reporting import (
    build_diagnostics_report,
    build_pick_log,
    export_backtest_artifacts,
    generate_markdown_report,
)
from easyml.core.runner.run_manager import RunManager
from easyml.core.runner.scaffold import scaffold_project
from easyml.core.runner.schema import (
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
from easyml.core.runner.server_gen import GeneratedServer, ToolSpec, generate_server
from easyml.core.runner.stage_guards import PipelineGuards
from easyml.core.runner.sweep import expand_sweep
from easyml.core.runner.training import (
    predict_single_model,
    train_single_model,
)
from easyml.core.runner.transformation_tester import (
    TransformationReport,
    TransformationResult,
    run_transformation_tests,
)
from easyml.core.runner.validator import ValidationResult, validate_project

__all__ = [
    # --- Classes ---
    "AxisDef",
    "BacktestConfig",
    "CastStep",
    "ChangeReport",
    "ChangeSet",
    "DataConfig",
    "DeriveStep",
    "DistinctStep",
    "DataProfile",
    "EnsembleDef",
    "ExperimentDef",
    "ExperimentError",
    "ExperimentManager",
    "ExperimentResult",
    "ExplorationSpace",
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
    "evaluate_fold_predictions",
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
    "get_features_df",
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
    "run_exploration",
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
