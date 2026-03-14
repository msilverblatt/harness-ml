"""Config writer -- pure functions for YAML config mutations.

Every function: load -> modify -> validate -> write -> return markdown confirmation.
Used by both Claude Code plugin and MCP server tools.

This package re-exports all public functions from its submodules so that
``from harnessml.core.runner.config_writer import X`` continues to work.
"""

# Shared helpers (also re-exported for tests that use _load_yaml / _save_yaml)
from harnessml.core.runner.config_writer._helpers import (  # noqa: F401
    _LOWER_IS_BETTER,
    _expand_dot_keys,
    _get_config_dir,
    _get_freshness_tracker,
    _get_source_registry,
    _invalidate_view_cache,
    _load_yaml,
    _persist_feature_defs,
    _save_yaml,
)

# Project initialization
from harnessml.core.runner.config_writer._init import scaffold_init  # noqa: F401

# Data operations
from harnessml.core.runner.config_writer.data import (  # noqa: F401
    add_dataset,
    add_source,
    add_target,
    available_features,
    configure_denylist,
    configure_exclude_columns,
    derive_column,
    detect_outliers,
    drop_rows,
    feature_store_status,
    fetch_url,
    inspect_data,
    list_sources,
    list_targets,
    profile_data,
    restore_full_data,
    sample_data,
    set_active_target,
    update_data_config,
)

# Experiment operations
from harnessml.core.runner.config_writer.experiments import (  # noqa: F401
    compare_experiments,
    experiment_create,
    log_experiment_result,
    promote_experiment,
    promote_exploration_trial,
    quick_run_experiment,
    run_experiment,
    show_journal,
    write_overlay,
)

# Feature operations
from harnessml.core.runner.config_writer.features import (  # noqa: F401
    add_feature,
    add_features_batch,
    auto_search_features,
    discover_features,
    test_feature_transformations,
)

# Model operations
from harnessml.core.runner.config_writer.models import (  # noqa: F401
    add_model,
    configure_ensemble,
    remove_model,
    show_model,
    show_models,
    show_presets,
    update_model,
)

# Pipeline / run / diagnostics operations
from harnessml.core.runner.config_writer.pipeline import (  # noqa: F401
    _detect_run_metrics,
    _format_backtest_result,
    _format_comparison_table,
    _load_run_metrics,
    _load_run_metrics_raw,
    check_guardrails,
    compare_runs,
    configure_backtest,
    explain_model,
    format_target_comparison,
    inspect_predictions,
    list_runs,
    run_backtest,
    run_exploration,
    run_predict,
    show_config,
    show_diagnostics,
    show_run,
)

# Source management
from harnessml.core.runner.config_writer.sources import (  # noqa: F401
    check_freshness,
    refresh_all_sources,
    refresh_source,
    validate_source_data,
)

# View operations
from harnessml.core.runner.config_writer.views import (  # noqa: F401
    add_view,
    list_views,
    preview_view,
    remove_view,
    set_features_view,
    update_view,
    view_dag,
)
