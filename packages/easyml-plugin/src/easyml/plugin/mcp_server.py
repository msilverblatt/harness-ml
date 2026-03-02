"""EasyML MCP server — 6 category-based tools with action dispatch.

Tools are grouped by domain. Each tool takes an `action` parameter
to select the operation, plus action-specific parameters.
All tools accept project_dir (defaults to cwd) and return markdown.
"""
from __future__ import annotations

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("easyml")


def _resolve_project_dir(project_dir: str | None, *, allow_missing: bool = False) -> Path:
    """Resolve project directory from param or cwd."""
    if project_dir:
        p = Path(project_dir).resolve()
    else:
        p = Path.cwd()

    if not allow_missing:
        config_dir = p / "config"
        if not config_dir.exists():
            raise ValueError(
                f"No config/ directory found at {p}. "
                f"Is this an easyml project? Run scaffold_project() first."
            )
    return p


# -----------------------------------------------------------------------
# 1. manage_models — add, remove, list, presets
# -----------------------------------------------------------------------


@mcp.tool()
def manage_models(
    action: str,
    name: str | None = None,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    params: str | None = None,
    active: bool = True,
    include_in_ensemble: bool = True,
    project_dir: str | None = None,
) -> str:
    """Manage models in the project.

    Actions:
      - "add": Add a model. Requires name + (model_type or preset).
        Optional: features, params (JSON string), active, include_in_ensemble.
      - "remove": Remove a model. Requires name.
      - "list": List all models with type, status, feature count.
      - "presets": Show available model presets.
    """
    from easyml.runner import config_writer as cw

    if action == "add":
        if not name:
            return "**Error**: 'name' is required for add action."
        parsed_params = json.loads(params) if params else None
        return cw.add_model(
            _resolve_project_dir(project_dir),
            name,
            model_type=model_type,
            preset=preset,
            features=features,
            params=parsed_params,
            active=active,
            include_in_ensemble=include_in_ensemble,
        )
    elif action == "remove":
        if not name:
            return "**Error**: 'name' is required for remove action."
        return cw.remove_model(_resolve_project_dir(project_dir), name)
    elif action == "list":
        return cw.show_models(_resolve_project_dir(project_dir))
    elif action == "presets":
        return cw.show_presets()
    else:
        return f"**Error**: Unknown action '{action}'. Use: add, remove, list, presets."


# -----------------------------------------------------------------------
# 2. manage_data — add, validate, fill_nulls, drop_duplicates, rename,
#                  profile, list_features
# -----------------------------------------------------------------------


@mcp.tool()
def manage_data(
    action: str,
    data_path: str | None = None,
    join_on: list[str] | None = None,
    prefix: str | None = None,
    auto_clean: bool = True,
    column: str | None = None,
    strategy: str = "median",
    value: float | None = None,
    columns: list[str] | None = None,
    mapping: str | None = None,
    category: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage data in the project's feature store.

    Actions:
      - "add": Ingest a dataset (CSV/parquet/Excel). Requires data_path.
        Optional: join_on, prefix, auto_clean.
      - "validate": Preview a dataset without ingesting. Requires data_path.
      - "fill_nulls": Fill nulls in a column. Requires column.
        Optional: strategy (median/mean/mode/zero/value), value.
      - "drop_duplicates": Drop duplicate rows.
        Optional: columns (subset to check).
      - "rename": Rename columns. Requires mapping (JSON string of
        {"old_name": "new_name"} pairs).
      - "profile": Profile the features dataset. Optional: category.
      - "list_features": List available feature columns. Optional: prefix.
    """
    from easyml.runner import config_writer as cw

    if action == "add":
        if not data_path:
            return "**Error**: 'data_path' is required for add action."
        return cw.add_dataset(
            _resolve_project_dir(project_dir),
            data_path,
            join_on=join_on,
            prefix=prefix,
            auto_clean=auto_clean,
        )
    elif action == "validate":
        if not data_path:
            return "**Error**: 'data_path' is required for validate action."
        from easyml.runner.data_ingest import validate_dataset
        return validate_dataset(_resolve_project_dir(project_dir), data_path)
    elif action == "fill_nulls":
        if not column:
            return "**Error**: 'column' is required for fill_nulls action."
        from easyml.runner.data_ingest import fill_nulls
        return fill_nulls(
            _resolve_project_dir(project_dir),
            column,
            strategy=strategy,
            value=value,
        )
    elif action == "drop_duplicates":
        from easyml.runner.data_ingest import drop_duplicates
        return drop_duplicates(
            _resolve_project_dir(project_dir),
            columns=columns,
        )
    elif action == "rename":
        if not mapping:
            return "**Error**: 'mapping' is required for rename action (JSON string)."
        from easyml.runner.data_ingest import rename_columns
        parsed = json.loads(mapping)
        return rename_columns(_resolve_project_dir(project_dir), parsed)
    elif action == "profile":
        return cw.profile_data(_resolve_project_dir(project_dir), category=category)
    elif action == "list_features":
        return cw.available_features(_resolve_project_dir(project_dir), prefix=prefix)
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: add, validate, fill_nulls, drop_duplicates, rename, profile, list_features."
        )


# -----------------------------------------------------------------------
# 3. manage_features — add, add_batch, test_transformations, discover
# -----------------------------------------------------------------------


@mcp.tool()
def manage_features(
    action: str,
    name: str | None = None,
    formula: str | None = None,
    description: str = "",
    features: str | None = None,
    test_interactions: bool = True,
    top_n: int = 20,
    method: str = "xgboost",
    project_dir: str | None = None,
) -> str:
    """Create and analyze features.

    Actions:
      - "add": Create a feature from a formula. Requires name, formula.
        Optional: description. Formulas support math ops (+, -, *, /, **),
        functions (log, sqrt, cbrt, abs), and @-references to other features.
      - "add_batch": Create multiple features. Requires features (JSON array
        of {name, formula, description?} objects). Handles @-references
        between features via topological ordering.
      - "test_transformations": Test math transformations on features.
        Requires features (JSON array of column names).
        Optional: test_interactions.
      - "discover": Run feature discovery (correlations, importance,
        redundancy, groupings). Optional: top_n, method (xgboost/mutual_info).
    """
    from easyml.runner import config_writer as cw

    if action == "add":
        if not name or not formula:
            return "**Error**: 'name' and 'formula' are required for add action."
        return cw.add_feature(
            _resolve_project_dir(project_dir),
            name,
            formula,
            description=description,
        )
    elif action == "add_batch":
        if not features:
            return "**Error**: 'features' (JSON array) is required for add_batch action."
        parsed = json.loads(features)
        return cw.add_features_batch(_resolve_project_dir(project_dir), parsed)
    elif action == "test_transformations":
        if not features:
            return "**Error**: 'features' (JSON array of column names) is required."
        parsed = json.loads(features)
        return cw.test_feature_transformations(
            _resolve_project_dir(project_dir),
            parsed,
            test_interactions=test_interactions,
        )
    elif action == "discover":
        return cw.discover_features(
            _resolve_project_dir(project_dir),
            top_n=top_n,
            method=method,
        )
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: add, add_batch, test_transformations, discover."
        )


# -----------------------------------------------------------------------
# 4. manage_experiments — create, write_overlay, run, promote
# -----------------------------------------------------------------------


@mcp.tool()
def manage_experiments(
    action: str,
    experiment_id: str | None = None,
    description: str = "",
    hypothesis: str = "",
    overlay: str | None = None,
    primary_metric: str = "brier",
    variant: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage ML experiments.

    Actions:
      - "create": Create a new experiment. Requires description.
        Optional: hypothesis.
      - "write_overlay": Write overlay YAML to an experiment. Requires
        experiment_id, overlay (JSON string of config changes).
      - "run": Run experiment (backtest with overlay vs baseline). Requires
        experiment_id. Optional: primary_metric, variant.
      - "promote": Promote experiment config to production. Requires
        experiment_id. Optional: primary_metric.
    """
    from easyml.runner import config_writer as cw

    if action == "create":
        if not description:
            return "**Error**: 'description' is required for create action."
        return cw.experiment_create(
            _resolve_project_dir(project_dir),
            description,
            hypothesis=hypothesis,
        )
    elif action == "write_overlay":
        if not experiment_id:
            return "**Error**: 'experiment_id' is required for write_overlay action."
        if not overlay:
            return "**Error**: 'overlay' (JSON string) is required for write_overlay action."
        parsed = json.loads(overlay)
        return cw.write_overlay(
            _resolve_project_dir(project_dir),
            experiment_id,
            parsed,
        )
    elif action == "run":
        if not experiment_id:
            return "**Error**: 'experiment_id' is required for run action."
        return cw.run_experiment(
            _resolve_project_dir(project_dir),
            experiment_id,
            primary_metric=primary_metric,
            variant=variant,
        )
    elif action == "promote":
        if not experiment_id:
            return "**Error**: 'experiment_id' is required for promote action."
        return cw.promote_experiment(
            _resolve_project_dir(project_dir),
            experiment_id,
            primary_metric=primary_metric,
        )
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: create, write_overlay, run, promote."
        )


# -----------------------------------------------------------------------
# 5. configure — ensemble, backtest, show
# -----------------------------------------------------------------------


@mcp.tool()
def configure(
    action: str,
    project_name: str | None = None,
    task: str | None = None,
    target_column: str | None = None,
    key_columns: list[str] | None = None,
    time_column: str | None = None,
    method: str | None = None,
    temperature: float | None = None,
    exclude_models: list[str] | None = None,
    cv_strategy: str | None = None,
    seasons: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Configure project settings.

    Actions:
      - "init": Initialize a new easyml project.
        Optional: project_name, task, target_column, key_columns, time_column.
      - "ensemble": Update ensemble config.
        Optional: method, temperature, exclude_models.
      - "backtest": Update backtest config.
        Optional: cv_strategy, seasons, metrics, min_train_folds.
      - "show": Show the full resolved project configuration.
    """
    if action == "init":
        from easyml.runner import config_writer as cw
        return cw.scaffold_init(
            _resolve_project_dir(project_dir, allow_missing=True),
            project_name,
            task=task or "classification",
            target_column=target_column or "result",
            key_columns=key_columns,
            time_column=time_column,
        )

    from easyml.runner import config_writer as cw

    if action == "ensemble":
        return cw.configure_ensemble(
            _resolve_project_dir(project_dir),
            method=method,
            temperature=temperature,
            exclude_models=exclude_models,
        )
    elif action == "backtest":
        return cw.configure_backtest(
            _resolve_project_dir(project_dir),
            cv_strategy=cv_strategy,
            seasons=seasons,
            metrics=metrics,
            min_train_folds=min_train_folds,
        )
    elif action == "show":
        return cw.show_config(_resolve_project_dir(project_dir))
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: init, ensemble, backtest, show."
        )


# -----------------------------------------------------------------------
# 6. pipeline — run_backtest, list_runs, show_run
# -----------------------------------------------------------------------


@mcp.tool()
def pipeline(
    action: str,
    experiment_id: str | None = None,
    variant: str | None = None,
    run_id: str | None = None,
    season: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Run and inspect pipeline executions.

    Actions:
      - "run_backtest": Run a full backtest. Returns metrics, meta-learner
        weights, per-season breakdown. Optional: experiment_id (applies
        overlay), variant.
      - "predict": Generate predictions for a target season. Requires season.
        Optional: run_id, variant.
      - "list_runs": List all pipeline runs with status.
      - "show_run": Show results from a run. Optional: run_id (defaults
        to most recent).
    """
    from easyml.runner import config_writer as cw

    if action == "run_backtest":
        return cw.run_backtest(
            _resolve_project_dir(project_dir),
            experiment_id=experiment_id,
            variant=variant,
        )
    elif action == "predict":
        if season is None:
            return "**Error**: 'season' is required for predict action."
        return cw.run_predict(
            _resolve_project_dir(project_dir),
            season,
            run_id=run_id,
            variant=variant,
        )
    elif action == "list_runs":
        return cw.list_runs(_resolve_project_dir(project_dir))
    elif action == "show_run":
        return cw.show_run(
            _resolve_project_dir(project_dir),
            run_id=run_id,
        )
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: run_backtest, predict, list_runs, show_run."
        )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
