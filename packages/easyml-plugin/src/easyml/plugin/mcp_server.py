"""EasyML MCP server — 6 category-based tools with action dispatch.

Tools are grouped by domain. Each tool takes an `action` parameter
to select the operation, plus action-specific parameters.
All tools accept project_dir (defaults to cwd) and return markdown.
"""
from __future__ import annotations

import functools
import json
import traceback
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("easyml")


def _safe_tool(fn):
    """Wrap a tool function so unhandled exceptions become markdown errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except json.JSONDecodeError as e:
            return f"**Error**: Invalid JSON input: {e}"
        except ValueError as e:
            return f"**Error**: {e}"
        except Exception as e:
            return f"**Error**: Unexpected error in `{fn.__name__}`: {e}"
    return wrapper


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
@_safe_tool
def manage_models(
    action: str,
    name: str | None = None,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    params: str | dict | None = None,
    active: bool = True,
    include_in_ensemble: bool = True,
    mode: str | None = None,
    prediction_type: str | None = None,
    purge: bool = False,
    project_dir: str | None = None,
) -> str:
    """Manage models in the project.

    Actions:
      - "add": Add a model. Requires name + (model_type or preset).
        Optional: features, params (JSON string), active, include_in_ensemble,
        mode (e.g. "classifier", "regressor"), prediction_type (e.g. "margin").
      - "update": Update an existing model in place. Requires name.
        Optional: features, params (JSON string), active, include_in_ensemble,
        mode, prediction_type. Merges params with existing.
      - "remove": Disable a model (sets active=false, include_in_ensemble=false).
        Requires name. Pass purge=True to delete the entry permanently.
      - "list": List all models with type, status, feature count.
      - "presets": Show available model presets.
    """
    from easyml.runner import config_writer as cw

    if action == "add":
        if not name:
            return "**Error**: 'name' is required for add action."
        parsed_params = json.loads(params) if isinstance(params, str) else params
        return cw.add_model(
            _resolve_project_dir(project_dir),
            name,
            model_type=model_type,
            preset=preset,
            features=features,
            params=parsed_params,
            active=active,
            include_in_ensemble=include_in_ensemble,
            mode=mode,
            prediction_type=prediction_type,
        )
    elif action == "update":
        if not name:
            return "**Error**: 'name' is required for update action."
        parsed_params = json.loads(params) if isinstance(params, str) else params
        # Forward booleans only when they differ from the MCP default (True),
        # so omitting them doesn't accidentally overwrite the model's value.
        return cw.update_model(
            _resolve_project_dir(project_dir),
            name,
            features=features,
            params=parsed_params,
            active=False if active is False else None,
            include_in_ensemble=False if include_in_ensemble is False else None,
            mode=mode,
            prediction_type=prediction_type,
        )
    elif action == "remove":
        if not name:
            return "**Error**: 'name' is required for remove action."
        return cw.remove_model(_resolve_project_dir(project_dir), name, purge=purge)
    elif action == "list":
        return cw.show_models(_resolve_project_dir(project_dir))
    elif action == "presets":
        return cw.show_presets()
    else:
        return f"**Error**: Unknown action '{action}'. Use: add, update, remove, list, presets."


# -----------------------------------------------------------------------
# 2. manage_data — add, validate, fill_nulls, drop_duplicates, rename,
#                  profile, list_features, status, list_sources
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
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
    mapping: str | dict | None = None,
    category: str | None = None,
    # View management parameters:
    name: str | None = None,
    source: str | None = None,
    steps: str | list | None = None,
    description: str = "",
    format: str = "auto",
    n_rows: int = 5,
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
      - "status": Quick overview of the feature store (row/column count,
        target distribution, time range, source count).
      - "list_sources": List all ingested data sources from the registry.
      - "add_source": Register a raw data source. Requires name, data_path.
        Optional: format.
      - "add_view": Declare a view (transform chain). Requires name, source.
        Optional: steps (JSON array), description.
      - "update_view": Update an existing view in place. Requires name.
        Optional: source, steps (JSON array), description. Only provided
        fields are merged.

        Available step ops for views:
          filter:    {op: filter, expr: "col > 0"}
          select:    {op: select, columns: ["a", "b"]}
          derive:    {op: derive, columns: {new_col: "a - b"}}
          group_by:  {op: group_by, keys: ["k"], aggs: {col: "mean"}}
          join:      {op: join, other: "view_name", on: {left_key: right_key}, prefix: "a_"}
          union:     {op: union, other: "view_name"}
          unpivot:   {op: unpivot, id_columns: [...], unpivot_columns: {col: [src1, src2]}}
          sort:      {op: sort, by: ["col"], ascending: true}
          head:      {op: head, keys: ["group_col"], n: 5, order_by: "col", position: "last"}
          rolling:   {op: rolling, keys: ["group_col"], order_by: "col", window: 5,
                      aggs: {new_col: "source_col:mean"}}
          cast:      {op: cast, columns: {col: "int"}}
          distinct:  {op: distinct, columns: ["col"]}
          rank:      {op: rank, columns: {rank_col: "value_col"}, keys: ["group"], ascending: false}
          isin:      {op: isin, column: "col", values: ["a", "b"]}
          cond_agg:  {op: cond_agg, keys: ["k"], aggs: {new_col: "src:sum:condition_expr"}}

      - "remove_view": Remove a view. Requires name.
      - "list_views": List all views with descriptions and dependency info.
      - "preview_view": Materialize a view and show schema + first N rows.
        Requires name. Optional: n_rows.
      - "set_features_view": Set which view becomes the prediction table.
        Requires name.
      - "view_dag": Show the full view dependency graph.
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
        parsed = json.loads(mapping) if isinstance(mapping, str) else mapping
        return rename_columns(_resolve_project_dir(project_dir), parsed)
    elif action == "profile":
        return cw.profile_data(_resolve_project_dir(project_dir), category=category)
    elif action == "list_features":
        return cw.available_features(_resolve_project_dir(project_dir), prefix=prefix)
    elif action == "status":
        return cw.feature_store_status(_resolve_project_dir(project_dir))
    elif action == "list_sources":
        return cw.list_sources(_resolve_project_dir(project_dir))
    elif action == "add_source":
        if not name:
            return "**Error**: 'name' is required for add_source."
        if not data_path:
            return "**Error**: 'data_path' is required for add_source."
        return cw.add_source(
            _resolve_project_dir(project_dir),
            name,
            data_path,
            format=format,
        )
    elif action == "add_view":
        if not name:
            return "**Error**: 'name' is required for add_view."
        if not source:
            return "**Error**: 'source' is required for add_view."
        parsed_steps = json.loads(steps) if isinstance(steps, str) else (steps or [])
        return cw.add_view(
            _resolve_project_dir(project_dir),
            name,
            source,
            parsed_steps,
            description=description,
        )
    elif action == "update_view":
        if not name:
            return "**Error**: 'name' is required for update_view."
        parsed_steps = None
        if steps is not None:
            parsed_steps = json.loads(steps) if isinstance(steps, str) else steps
        return cw.update_view(
            _resolve_project_dir(project_dir),
            name,
            source=source,
            steps=parsed_steps,
            description=description if description else None,
        )
    elif action == "remove_view":
        if not name:
            return "**Error**: 'name' is required for remove_view."
        return cw.remove_view(_resolve_project_dir(project_dir), name)
    elif action == "list_views":
        return cw.list_views(_resolve_project_dir(project_dir))
    elif action == "preview_view":
        if not name:
            return "**Error**: 'name' is required for preview_view."
        return cw.preview_view(_resolve_project_dir(project_dir), name, n_rows=n_rows)
    elif action == "set_features_view":
        if not name:
            return "**Error**: 'name' is required for set_features_view."
        return cw.set_features_view(_resolve_project_dir(project_dir), name)
    elif action == "view_dag":
        return cw.view_dag(_resolve_project_dir(project_dir))
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: add, validate, fill_nulls, drop_duplicates, rename, profile, "
            "list_features, status, list_sources, add_source, add_view, update_view, "
            "remove_view, list_views, preview_view, set_features_view, view_dag."
        )


# -----------------------------------------------------------------------
# 3. manage_features — add, add_batch, test_transformations, discover
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def manage_features(
    action: str,
    name: str | None = None,
    formula: str | None = None,
    description: str = "",
    type: str | None = None,
    source: str | None = None,
    column: str | None = None,
    condition: str | None = None,
    pairwise_mode: str = "diff",
    category: str = "general",
    features: str | list | None = None,
    test_interactions: bool = True,
    top_n: int = 20,
    method: str = "xgboost",
    project_dir: str | None = None,
) -> str:
    """Create and analyze features.

    Actions:
      - "add": Create a feature. Requires name + at least one of: type,
        formula, source, or condition.
        Types: "team" (entity-level, auto-generates pairwise), "pairwise"
        (matchup-level formula), "matchup" (context column), "regime" (boolean flag).
        Optional: type, source, column, formula, condition, pairwise_mode
        (diff/ratio/both/none), category, description.
        If type is omitted, it is inferred: formula -> pairwise, condition -> regime,
        source -> team.
      - "add_batch": Create multiple features. Requires features (JSON array
        of {name, formula?, type?, source?, column?, condition?, pairwise_mode?,
        category?, description?} objects). Handles @-references between features
        via topological ordering.
      - "test_transformations": Test math transformations on features.
        Requires features (JSON array of column names).
        Optional: test_interactions.
      - "discover": Run feature discovery (correlations, importance,
        redundancy, groupings). Optional: top_n, method (xgboost/mutual_info).
    """
    from easyml.runner import config_writer as cw

    if action == "add":
        if not name:
            return "**Error**: 'name' is required for add action."
        if not formula and not type and not source and not condition:
            return (
                "**Error**: Provide at least one of: type, formula, source, or condition."
            )
        return cw.add_feature(
            _resolve_project_dir(project_dir),
            name,
            formula,
            type=type,
            source=source,
            column=column,
            condition=condition,
            pairwise_mode=pairwise_mode,
            category=category,
            description=description,
        )
    elif action == "add_batch":
        if not features:
            return "**Error**: 'features' (JSON array) is required for add_batch action."
        parsed = json.loads(features) if isinstance(features, str) else features
        return cw.add_features_batch(_resolve_project_dir(project_dir), parsed)
    elif action == "test_transformations":
        if not features:
            return "**Error**: 'features' (JSON array of column names) is required."
        parsed = json.loads(features) if isinstance(features, str) else features
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
# 4. manage_experiments — create, write_overlay, run, promote, quick_run, explore
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
def manage_experiments(
    action: str,
    experiment_id: str | None = None,
    description: str = "",
    hypothesis: str = "",
    overlay: str | dict | None = None,
    primary_metric: str = "brier",
    variant: str | None = None,
    search_space: str | dict | None = None,
    trial: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage ML experiments.

    Actions:
      - "create": Create a new experiment. Requires description.
        Optional: hypothesis.
      - "write_overlay": Write overlay YAML to an experiment. Requires
        experiment_id, overlay (JSON string of config changes).
        Keys support dot-notation ("models.xgb.params.lr": 0.01) or
        dict values to set whole blocks ("models.new_model": {"type": ...}).
      - "run": Run experiment (backtest with overlay vs baseline). Requires
        experiment_id. Optional: primary_metric, variant.
      - "promote": Promote experiment config to production. Requires
        experiment_id. Optional: primary_metric.
      - "quick_run": Create, configure, and run an experiment in one call.
        Requires description, overlay (JSON string). Optional: hypothesis,
        primary_metric.
      - "explore": Run Bayesian exploration over a search space. Requires
        search_space (JSON with axes, budget, primary_metric). Runs
        Optuna-driven trials, returns full report with best config,
        all trials ranked, and parameter importance.
      - "promote_trial": Promote a trial from an exploration run as a new
        experiment. Requires experiment_id (exploration ID, e.g. 'expl-002').
        Optional: trial (int, defaults to best trial), primary_metric, hypothesis.
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
        parsed = json.loads(overlay) if isinstance(overlay, str) else overlay
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
    elif action == "quick_run":
        if not description:
            return "**Error**: 'description' is required for quick_run action."
        if not overlay:
            return "**Error**: 'overlay' (JSON string) is required for quick_run action."
        return cw.quick_run_experiment(
            _resolve_project_dir(project_dir),
            description,
            overlay,
            hypothesis=hypothesis,
            primary_metric=primary_metric,
        )
    elif action == "explore":
        if not search_space:
            return "**Error**: 'search_space' (JSON with axes + budget) is required for explore action."
        parsed = json.loads(search_space) if isinstance(search_space, str) else search_space
        return cw.run_exploration(
            _resolve_project_dir(project_dir),
            parsed,
        )
    elif action == "promote_trial":
        if not experiment_id:
            return "**Error**: 'experiment_id' (exploration ID, e.g. 'expl-002') is required for promote_trial."
        return cw.promote_exploration_trial(
            _resolve_project_dir(project_dir),
            experiment_id,
            trial=trial,
            primary_metric=primary_metric,
            hypothesis=hypothesis,
        )
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: create, write_overlay, run, promote, quick_run, explore, promote_trial."
        )


# -----------------------------------------------------------------------
# 5. configure — init, ensemble, backtest, show, check_guardrails
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
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
    calibration: str | None = None,
    pre_calibration: str | dict | None = None,
    cv_strategy: str | None = None,
    seasons: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
    add_columns: list[str] | None = None,
    remove_columns: list[str] | None = None,
    project_dir: str | None = None,
) -> str:
    """Configure project settings.

    Actions:
      - "init": Initialize a new easyml project.
        Optional: project_name, task, target_column, key_columns, time_column.
      - "ensemble": Update ensemble config.
        Optional: method, temperature, exclude_models,
        calibration ('spline'/'isotonic'/'platt'/'none' — post-ensemble calibration),
        pre_calibration (JSON dict of {model_name: method} for per-model calibration
        applied before the meta-learner, e.g. '{"xgb_core": "platt"}').
      - "backtest": Update backtest config.
        Optional: cv_strategy, seasons, metrics, min_train_folds.
      - "show": Show the full resolved project configuration.
      - "check_guardrails": Run configured guardrails (feature leakage,
        naming conventions, model config). Returns pass/fail report.
      - "exclude_columns": Add/remove columns from data.exclude_columns.
        These columns are never used as features or in feature discovery.
        Use for regression target columns or leaky outcome columns.
        Optional: add_columns (list), remove_columns (list).
      - "set_denylist": Add/remove columns from the feature leakage denylist.
        The denylist is checked by check_guardrails() to catch models using
        forbidden columns. Optional: add_columns (list), remove_columns (list).
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
        parsed_pre_cal = None
        if pre_calibration is not None:
            parsed_pre_cal = json.loads(pre_calibration) if isinstance(pre_calibration, str) else pre_calibration
        return cw.configure_ensemble(
            _resolve_project_dir(project_dir),
            method=method,
            temperature=temperature,
            exclude_models=exclude_models,
            calibration=calibration,
            pre_calibration=parsed_pre_cal,
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
    elif action == "check_guardrails":
        return cw.check_guardrails(_resolve_project_dir(project_dir))
    elif action == "exclude_columns":
        return cw.configure_exclude_columns(
            _resolve_project_dir(project_dir),
            add_columns=add_columns,
            remove_columns=remove_columns,
        )
    elif action == "set_denylist":
        return cw.configure_denylist(
            _resolve_project_dir(project_dir),
            add_columns=add_columns,
            remove_columns=remove_columns,
        )
    else:
        return (
            f"**Error**: Unknown action '{action}'. "
            "Use: init, ensemble, backtest, show, check_guardrails, "
            "exclude_columns, set_denylist."
        )


# -----------------------------------------------------------------------
# 6. pipeline — run_backtest, predict, diagnostics, list_runs, show_run
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
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
      - "diagnostics": Show per-model diagnostics (brier, accuracy, ECE,
        log_loss, agreement, calibration). Optional: run_id.
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
    elif action == "diagnostics":
        return cw.show_diagnostics(
            _resolve_project_dir(project_dir),
            run_id=run_id,
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
            "Use: run_backtest, predict, diagnostics, list_runs, show_run."
        )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
