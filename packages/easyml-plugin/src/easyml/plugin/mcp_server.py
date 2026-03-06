"""EasyML MCP server — thin dispatcher with hot-reloadable handlers.

Tools are grouped by domain. Each tool takes an `action` parameter
to select the operation, plus action-specific parameters.
All tools accept project_dir (defaults to cwd) and return markdown.
"""
from __future__ import annotations

import asyncio
import functools
import importlib
import json
import os

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("easyml")
_DEV_MODE = os.environ.get("EASYML_DEV", "0") == "1"


def _load_handler(module_name: str):
    """Load (and optionally hot-reload) a handler module."""
    mod = importlib.import_module(f"easyml.plugin.handlers.{module_name}")
    if _DEV_MODE:
        importlib.reload(mod)
    return mod


def _safe_tool(fn):
    """Wrap a tool function so unhandled exceptions become markdown errors."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        except json.JSONDecodeError as e:
            return f"**Error**: Invalid JSON input: {e}"
        except ValueError as e:
            return f"**Error**: {e}"
        except Exception as e:
            return f"**Error**: Unexpected error in `{fn.__name__}`: {e}"
    return wrapper


# -----------------------------------------------------------------------
# 1. manage_models
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def manage_models(
    action: str,
    ctx: Context,
    name: str | None = None,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    params: str | dict | None = None,
    active: bool | None = None,
    include_in_ensemble: bool | None = None,
    mode: str | None = None,
    prediction_type: str | None = None,
    cdf_scale: float | None = None,
    zero_fill_features: list[str] | None = None,
    items: str | list | None = None,
    purge: bool = False,
    project_dir: str | None = None,
) -> str:
    """Manage models in the project.

    Actions:
      - "add": Add a model. Requires name + (model_type or preset).
        Optional: features, params (JSON string), active, include_in_ensemble,
        mode (e.g. "classifier", "regressor"), prediction_type (e.g. "margin"),
        cdf_scale (float, scales regressor output to a probability via CDF),
        zero_fill_features (list of feature columns to fill with 0 before
        NaN row removal during training).
      - "update": Update an existing model in place. Requires name.
        Optional: features, params (JSON string), active, include_in_ensemble,
        mode, prediction_type, cdf_scale, zero_fill_features. Merges params
        with existing.
        Pass active=true or include_in_ensemble=true to explicitly re-enable.
      - "remove": Disable a model (sets active=false, include_in_ensemble=false).
        Requires name. Pass purge=True to delete the entry permanently.
      - "list": List all models with type, status, feature count.
      - "presets": Show available model presets.
      - "add_batch": Add multiple models. Requires items (JSON array of model
        configs, each with name + model_type or preset + optional fields).
      - "update_batch": Update multiple models. Requires items (JSON array of
        model configs, each with name + optional update fields).
      - "remove_batch": Remove multiple models. Requires items (JSON array of
        {name, purge?} objects).
      - "clone": Clone an existing model with a new name. Requires name (source)
        and items (JSON with {new_name, ...overrides}).
    """
    return _load_handler("models").dispatch(
        action,
        ctx=ctx,
        name=name,
        model_type=model_type,
        preset=preset,
        features=features,
        params=params,
        active=active,
        include_in_ensemble=include_in_ensemble,
        mode=mode,
        prediction_type=prediction_type,
        cdf_scale=cdf_scale,
        zero_fill_features=zero_fill_features,
        items=items,
        purge=purge,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 2. manage_data
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def manage_data(
    action: str,
    ctx: Context,
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
    # Drop rows parameters:
    condition: str | None = None,
    # Derive column parameters:
    expression: str | None = None,
    group_by: str | None = None,
    dtype: str | None = None,
    # Batch parameters:
    sources: str | list | None = None,
    views: str | list | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage data in the project's feature store.

    Actions:
      - "add": Ingest a dataset (CSV/parquet/Excel). Requires data_path.
        Without join_on: APPENDS rows to existing data (use for adding more
        records of the same schema). With join_on: MERGES columns from the
        new dataset onto existing rows by matching on the specified key
        columns (use for enriching with new features). Optional: prefix
        (column name prefix for merged columns), auto_clean (default true).
      - "validate": Preview a dataset without ingesting. Requires data_path.
      - "fill_nulls": Fill nulls in a column. Requires column.
        Optional: strategy (median/mean/mode/zero/value), value.
      - "drop_duplicates": Drop duplicate rows.
        Optional: columns (subset to check).
      - "drop_rows": Drop rows from the feature store. Use column +
        condition="null" to drop rows with NaN in that column, or provide a
        pandas query expression as condition to drop matching rows. Examples:
        condition="null", column="future_return_20d"; condition="value < 0".
      - "rename": Rename columns. Requires mapping (JSON string of
        {"old_name": "new_name"} pairs).
      - "derive_column": Derive a new column from a pandas expression.
        Requires name, expression. Supports arithmetic ("close - open"),
        shifts with groupby ("close.shift(-1) / close - 1"), boolean
        thresholds ("(value > 0).astype(int)"), and datetime accessors
        ("date.dt.year"). Optional: group_by (for .shift() etc.), dtype.
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
      - "add_sources_batch": Register multiple data sources. Requires sources
        (JSON array of {name, data_path, format?} objects).
      - "fill_nulls_batch": Fill nulls in multiple columns. Requires columns
        (JSON array of {column, strategy?, value?} objects).
      - "add_views_batch": Declare multiple views. Requires views (JSON array
        of {name, source, steps?, description?} objects).
      - "check_freshness": Check freshness of all registered sources. Returns
        stale sources with last-fetched timestamps.
      - "refresh": Fetch a single source using its adapter, validate schema,
        and update freshness. Requires name.
      - "refresh_all": Fetch all stale sources in topological (dependency)
        order with per-source progress reporting.
      - "validate_source": Load a file source and validate against its schema
        definition. Requires name.
    """
    return _load_handler("data").dispatch(
        action,
        ctx=ctx,
        data_path=data_path,
        join_on=join_on,
        prefix=prefix,
        auto_clean=auto_clean,
        column=column,
        condition=condition,
        strategy=strategy,
        value=value,
        columns=columns,
        mapping=mapping,
        expression=expression,
        group_by=group_by,
        dtype=dtype,
        category=category,
        name=name,
        source=source,
        steps=steps,
        description=description,
        format=format,
        n_rows=n_rows,
        sources=sources,
        views=views,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 3. manage_features
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def manage_features(
    action: str,
    ctx: Context,
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
    search_types: str | list | None = None,
    project_dir: str | None = None,
) -> str:
    """Create and analyze features.

    Actions:
      - "add": Create a feature. Requires name + at least one of: type,
        formula, source, or condition.
        Types: "entity" (entity-level, auto-generates pairwise), "pairwise"
        (instance-level formula), "instance" (context column), "regime" (boolean flag).
        Optional: type, source, column, formula, condition, pairwise_mode
        (diff/ratio/both/none), category, description.
        If type is omitted, it is inferred: formula -> pairwise, condition -> regime,
        source -> entity.
      - "add_batch": Create multiple features. Requires features (JSON array
        of {name, formula?, type?, source?, column?, condition?, pairwise_mode?,
        category?, description?} objects). Handles @-references between features
        via topological ordering.
      - "test_transformations": Test math transformations on features.
        Requires features (JSON array of column names).
        Optional: test_interactions.
      - "discover": Run feature discovery (correlations, importance,
        redundancy, groupings). Optional: top_n, method (xgboost/mutual_info).
      - "diversity": Analyze feature diversity across models. Returns
        overlap matrix, diversity score, redundant pairs, and suggestions.
      - "auto_search": Systematically search for new candidate features via
        interactions (pairwise arithmetic), lags, and rolling means.
        Optional: features (JSON array of column names to search over;
        defaults to all feature columns), search_types (JSON array from
        ["interactions", "lags", "rolling"]; defaults to all), top_n.
    """
    return await _load_handler("features").dispatch(
        action,
        ctx=ctx,
        name=name,
        formula=formula,
        description=description,
        type=type,
        source=source,
        column=column,
        condition=condition,
        pairwise_mode=pairwise_mode,
        category=category,
        features=features,
        test_interactions=test_interactions,
        top_n=top_n,
        method=method,
        search_types=search_types,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 4. manage_experiments
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def manage_experiments(
    action: str,
    ctx: Context,
    experiment_id: str | None = None,
    experiment_ids: list[str] | None = None,
    description: str = "",
    hypothesis: str = "",
    overlay: str | dict | None = None,
    primary_metric: str = "brier",
    variant: str | None = None,
    search_space: str | dict | None = None,
    trial: int | None = None,
    detail: str | None = None,
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
        Optional: detail ("summary" returns best trial only,
        "full" (default) returns all trials + param importance).
      - "promote_trial": Promote a trial from an exploration run as a new
        experiment. Requires experiment_id (exploration ID, e.g. 'expl-002').
        Optional: trial (int, defaults to best trial), primary_metric, hypothesis.
      - "compare": Compare two experiments side by side. Requires
        experiment_ids (list of 2 experiment IDs).
    """
    return await _load_handler("experiments").dispatch(
        action,
        ctx=ctx,
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        description=description,
        hypothesis=hypothesis,
        overlay=overlay,
        primary_metric=primary_metric,
        variant=variant,
        search_space=search_space,
        trial=trial,
        detail=detail,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 5. configure
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def configure(
    action: str,
    ctx: Context,
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
    prior_feature: str | None = None,
    spline_prob_max: float | None = None,
    spline_n_bins: int | None = None,
    cv_strategy: str | None = None,
    fold_values: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
    fold_column: str | None = None,
    add_columns: list[str] | None = None,
    remove_columns: list[str] | None = None,
    detail: str | None = None,
    section: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Configure project settings.

    Actions:
      - "init": Initialize a new easyml project.
        Optional: project_name, task, target_column, key_columns, time_column.
      - "update_data": Update data config post-init.
        Optional: target_column, key_columns, time_column.
      - "ensemble": Update ensemble config.
        Optional: method ('stacked'/'average'/'weighted'), temperature, exclude_models,
        calibration ('spline'/'isotonic'/'platt'/'none' — post-ensemble calibration),
        pre_calibration (JSON dict of {model_name: method} for per-model calibration
        applied before the meta-learner, e.g. '{"xgb_core": "platt"}'),
        prior_feature (str, column name for the privileged feature passed to the
        meta-learner alongside model predictions; when None, zeros are used),
        spline_prob_max (float, upper clip for spline calibration, default 0.985),
        spline_n_bins (int, number of bins for spline calibration, default 20).
      - "backtest": Update backtest config.
        Optional: cv_strategy, fold_values, metrics, min_train_folds, fold_column.
      - "show": Show the full resolved project configuration.
        Optional: detail ("summary" for key settings only, "full" (default) for
        everything), section (e.g. "models", "ensemble", "backtest" to show only
        that block).
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
    return _load_handler("config").dispatch(
        action,
        ctx=ctx,
        project_name=project_name,
        task=task,
        target_column=target_column,
        key_columns=key_columns,
        time_column=time_column,
        method=method,
        temperature=temperature,
        exclude_models=exclude_models,
        calibration=calibration,
        pre_calibration=pre_calibration,
        prior_feature=prior_feature,
        spline_prob_max=spline_prob_max,
        spline_n_bins=spline_n_bins,
        cv_strategy=cv_strategy,
        fold_values=fold_values,
        metrics=metrics,
        min_train_folds=min_train_folds,
        fold_column=fold_column,
        add_columns=add_columns,
        remove_columns=remove_columns,
        detail=detail,
        section=section,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 6. pipeline
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def pipeline(
    action: str,
    ctx: Context,
    experiment_id: str | None = None,
    variant: str | None = None,
    run_id: str | None = None,
    run_ids: list[str] | None = None,
    fold_value: int | None = None,
    detail: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Run and inspect pipeline executions.

    Actions:
      - "run_backtest": Run a full backtest. Returns metrics, meta-learner
        weights, per-fold breakdown. Optional: experiment_id (applies
        overlay), variant.
      - "predict": Generate predictions for a target fold. Requires fold_value.
        Optional: run_id, variant.
      - "diagnostics": Show per-model diagnostics (brier, accuracy, ECE,
        log_loss, agreement, calibration). Optional: run_id.
        Optional: detail ("summary" for condensed output, "full" (default)
        for all metrics).
      - "list_runs": List all pipeline runs with status.
      - "show_run": Show results from a run. Optional: run_id (defaults
        to most recent).
        Optional: detail ("summary" for condensed output, "full" (default)
        for complete results).
      - "compare_runs": Compare metrics from two runs side by side.
        Requires run_ids (list of 2 run IDs).
    """
    import asyncio
    result = _load_handler("pipeline").dispatch(
        action,
        ctx=ctx,
        experiment_id=experiment_id,
        variant=variant,
        run_id=run_id,
        run_ids=run_ids,
        fold_value=fold_value,
        detail=detail,
        project_dir=project_dir,
    )
    if asyncio.iscoroutine(result):
        result = await result
    return result


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
