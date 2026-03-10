"""HarnessML MCP server — thin dispatcher with hot-reloadable handlers.

Tools are grouped by domain. Each tool takes an `action` parameter
to select the operation, plus action-specific parameters.
All tools accept project_dir (defaults to cwd) and return markdown.
"""
from __future__ import annotations

import asyncio
import atexit
import functools
import importlib
import json
import os
import threading
import time as _time

from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("harnessml")
_DEV_MODE = os.environ.get("HARNESS_DEV", "0") == "1"

_emitter = None
_studio_started = False
_STUDIO_PORT = int(os.environ.get("HARNESS_STUDIO_PORT", "8421"))
_init_lock = threading.Lock()


def _is_studio_running() -> bool:
    """Check if Studio is already running via HTTP health check."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(
            f"http://localhost:{_STUDIO_PORT}/api/health", timeout=2,
        )
        return resp.status == 200
    except Exception:
        return False


def _kill_studio_on_port():
    """Kill ALL processes listening on the Studio port."""
    import signal
    import subprocess as _sp

    try:
        result = _sp.run(
            ["lsof", "-ti", f":{_STUDIO_PORT}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            for pid_str in result.stdout.strip().split("\n"):
                try:
                    pid = int(pid_str.strip())
                    # Don't kill ourselves
                    if pid != os.getpid():
                        os.kill(pid, signal.SIGKILL)
                except (ValueError, ProcessLookupError, PermissionError, OSError):
                    pass
    except Exception:
        pass

    # Also clean up PID file
    from pathlib import Path
    pid_path = Path.home() / ".harnessml" / "studio.pid"
    pid_path.unlink(missing_ok=True)


def _start_studio():
    """Launch Studio as a child subprocess (dies with MCP server).

    Kills any existing process on the Studio port first.
    Uses start_new_session=False so Studio dies when the MCP server exits.
    """
    global _studio_started
    if _studio_started:
        return
    _studio_started = True

    import logging
    import subprocess
    import sys
    from pathlib import Path

    logger = logging.getLogger("harnessml.studio")

    # Kill everything on the port — no zombies
    _kill_studio_on_port()

    import time as _t
    _t.sleep(0.5)

    pid_path = Path.home() / ".harnessml" / "studio.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "harnessml.studio.cli", "--port", str(_STUDIO_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        pid_path.write_text(str(proc.pid))
        logger.debug("Started Studio (pid=%d) on port %d", proc.pid, _STUDIO_PORT)

        def _cleanup_studio():
            """Kill Studio and clean up PID file on exit."""
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            finally:
                pid_path.unlink(missing_ok=True)

        atexit.register(_cleanup_studio)
    except Exception:
        logger.debug("Failed to launch Studio subprocess (non-fatal)", exc_info=True)


def _get_emitter(project_dir: str | None = None):
    global _emitter
    if _emitter is not None:
        return _emitter
    with _init_lock:
        if _emitter is not None:
            return _emitter
        from harnessml.plugin.event_emitter import create_emitter
        _emitter = create_emitter()

        # Auto-start Studio as detached subprocess
        if _emitter.enabled:
            _start_studio()

    return _emitter


def _load_handler(module_name: str):
    """Load (and optionally hot-reload) a handler module."""
    mod = importlib.import_module(f"harnessml.plugin.handlers.{module_name}")
    if _DEV_MODE:
        importlib.reload(mod)
    return mod


_studio_url_logged = False
_studio_url_lock = threading.Lock()


def _check_studio_url_once() -> bool:
    """Log the Studio URL exactly once. Returns True if this call did the logging."""
    global _studio_url_logged
    if _studio_url_logged:
        return False
    with _studio_url_lock:
        if _studio_url_logged:
            return False
        _studio_url_logged = True
        import sys
        print(f"Harness Studio → http://localhost:{_STUDIO_PORT}", file=sys.stderr)
        return True


def _safe_tool(fn):
    """Wrap a tool function so unhandled exceptions become markdown errors."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        _check_studio_url_once()
        tool_name = fn.__name__
        action = kwargs.get("action", "")
        proj_dir = kwargs.get("project_dir") or os.getcwd()
        clean_params = {k: v for k, v in kwargs.items() if k != "ctx"}

        # Emit "running" event immediately so Studio shows the call in-progress
        emitter = _get_emitter(project_dir=proj_dir)
        emitter.set_project(proj_dir)
        emitter.set_current(tool_name, action)
        emitter.emit(
            tool=tool_name, action=action, params=clean_params,
            result="", duration_ms=0, status="running",
        )

        # Set the active emitter so handlers can access it via _common.get_active_emitter()
        from harnessml.plugin.handlers._common import set_active_emitter
        set_active_emitter(emitter)

        start = _time.monotonic()
        try:
            result = fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            elapsed = int((_time.monotonic() - start) * 1000)
            emitter.clear_current()
            emitter.emit(
                tool=tool_name, action=action, params=clean_params,
                result=result[:20000] if isinstance(result, str) else str(result)[:20000],
                duration_ms=elapsed, status="success",
            )
            return result
        except json.JSONDecodeError as e:
            error_result = f"**Error**: Invalid JSON input: {e}"
            elapsed = int((_time.monotonic() - start) * 1000)
            emitter.emit(
                tool=tool_name, action=action, params=clean_params,
                result=error_result[:20000], duration_ms=elapsed, status="error",
            )
            return error_result
        except ValueError as e:
            error_result = f"**Error**: {e}"
            elapsed = int((_time.monotonic() - start) * 1000)
            emitter.emit(
                tool=tool_name, action=action, params=clean_params,
                result=error_result[:20000], duration_ms=elapsed, status="error",
            )
            return error_result
        except Exception as e:
            error_result = f"**Error**: Unexpected error in `{fn.__name__}`: {e}"
            elapsed = int((_time.monotonic() - start) * 1000)
            emitter.emit(
                tool=tool_name, action=action, params=clean_params,
                result=error_result[:20000], duration_ms=elapsed, status="error",
            )
            return error_result
    return wrapper


# -----------------------------------------------------------------------
# 1. models
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def models(
    action: str,
    ctx: Context,
    name: str | None = None,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    append_features: list[str] | None = None,
    remove_features: list[str] | None = None,
    params: str | dict | None = None,
    active: bool | None = None,
    include_in_ensemble: bool | None = None,
    mode: str | None = None,
    prediction_type: str | None = None,
    cdf_scale: float | None = None,
    zero_fill_features: list[str] | None = None,
    class_weight: str | dict | None = None,
    items: str | list | None = None,
    purge: bool = False,
    replace_params: bool = False,
    project_dir: str | None = None,
) -> str:
    """Manage models in the project.

    Actions:
      - "add": Add a model. Requires name + (model_type or preset).
        Optional: features, params (JSON string), active, include_in_ensemble,
        mode (e.g. "classifier", "regressor"), prediction_type (e.g. "margin"),
        cdf_scale (float, scales regressor output to a probability via CDF),
        zero_fill_features (list of feature columns to fill with 0 before
        NaN row removal during training),
        class_weight ("balanced" or JSON dict mapping class labels to weights,
        e.g. {"0": 1.0, "1": 2.5}).
      - "update": Update an existing model in place. Requires name.
        Optional: features, params (JSON string), active, include_in_ensemble,
        mode, prediction_type, cdf_scale, zero_fill_features, class_weight. Merges params
        with existing by default. Pass replace_params=true to fully replace
        the params dict instead of merging.
        Pass active=true or include_in_ensemble=true to explicitly re-enable.
        append_features: list of features to add to the existing list (skips duplicates).
        remove_features: list of features to remove from the existing list.
      - "remove": Disable a model (sets active=false, include_in_ensemble=false).
        Requires name. Pass purge=True to delete the entry permanently.
      - "list": List all models with type, status, feature count.
      - "show": Show full configuration for a single model (type, features,
        params, status). Requires name.
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
        append_features=append_features,
        remove_features=remove_features,
        params=params,
        active=active,
        include_in_ensemble=include_in_ensemble,
        mode=mode,
        prediction_type=prediction_type,
        cdf_scale=cdf_scale,
        zero_fill_features=zero_fill_features,
        class_weight=class_weight,
        items=items,
        purge=purge,
        replace_params=replace_params,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 2. data
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def data(
    action: str,
    ctx: Context,
    data_path: str | None = None,
    join_on: list[str] | None = None,
    prefix: str | None = None,
    auto_clean: bool = False,
    column: str | None = None,
    strategy: str = "median",
    value: float | None = None,
    columns: list[str] | list[dict] | None = None,
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
    # Sample parameters:
    fraction: float | None = None,
    stratify_column: str | None = None,
    seed: int | None = None,
    # Batch parameters:
    sources: str | list | None = None,
    views: str | list | None = None,
    # Upload parameters:
    files: list[str] | str | None = None,
    folder_id: str | None = None,
    folder_name: str | None = None,
    dataset_slug: str | None = None,
    title: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage data in the project's feature store.

    Actions:
      - "add": Ingest a dataset (CSV/parquet/Excel). Requires data_path.
        Without join_on: APPENDS rows to existing data (use for adding more
        records of the same schema). With join_on: MERGES columns from the
        new dataset onto existing rows by matching on the specified key
        columns (use for enriching with new features). Optional: prefix
        (column name prefix for merged columns), auto_clean (default false;
        set true to auto-fill nulls and drop duplicates).
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
      - "inspect": Detailed data inspection. Without column: shows shape,
        all columns with dtypes, null counts, and null percentages. With
        column: shows detailed statistics (mean/std/quartiles for numeric,
        value counts for categorical). Use this for EDA before modeling.
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
      - "sample": Downsample the feature store for fast iteration. Saves backup
        as features_full.parquet. Requires fraction (0.0-1.0). Optional:
        stratify_column (preserve class ratios), seed.
      - "restore": Restore full feature store from backup (features_full.parquet).
      - "fetch_url": Download a file from a URL to the raw data directory.
        Requires data_path (the URL). Optional: name (filename to save as,
        auto-detected from URL if omitted).
      - "upload_drive": Upload file(s) to Google Drive. Requires files (list of
        paths). Optional: folder_id, folder_name (creates new folder), name.
        Returns file IDs and Colab URLs for notebooks.
      - "upload_kaggle": Upload file(s) as a Kaggle dataset. Requires files
        (list of paths), dataset_slug (e.g. "username/dataset-name").
        Optional: title.
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
        fraction=fraction,
        stratify_column=stratify_column,
        seed=seed,
        sources=sources,
        views=views,
        files=files,
        folder_id=folder_id,
        folder_name=folder_name,
        dataset_slug=dataset_slug,
        title=title,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 3. features
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def features(
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
# 4. experiments
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def experiments(
    action: str,
    ctx: Context,
    experiment_id: str | None = None,
    experiment_ids: list[str] | None = None,
    description: str = "",
    hypothesis: str = "",
    conclusion: str = "",
    overlay: str | dict | None = None,
    primary_metric: str = "brier",
    variant: str | None = None,
    search_space: str | dict | None = None,
    trial: int | None = None,
    detail: str | None = None,
    last_n: int | None = None,
    verdict: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage ML experiments.

    Actions:
      - "create": Create a new experiment. Requires description and
        hypothesis (what you expect and why).
      - "write_overlay": Write overlay YAML to an experiment. Requires
        experiment_id, overlay (JSON string of config changes).
        Keys support dot-notation ("models.xgb.params.lr": 0.01) or
        dict values to set whole blocks ("models.new_model": {"type": ...}).
      - "run": Run experiment (backtest with overlay vs baseline). Requires
        experiment_id. Optional: primary_metric, variant.
      - "promote": Promote experiment config to production. Requires
        experiment_id. Optional: primary_metric.
      - "quick_run": Create, configure, and run an experiment in one call.
        Requires description, overlay (JSON string), hypothesis.
        Optional: primary_metric.
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
      - "journal": Show the experiment journal — a history of experiments
        with descriptions, metrics, and verdicts. Optional: last_n (default 20).
      - "log_result": Manually log an experiment result. Requires experiment_id.
        Optional: description, hypothesis, conclusion (what was learned), verdict.
    """
    return await _load_handler("experiments").dispatch(
        action,
        ctx=ctx,
        experiment_id=experiment_id,
        experiment_ids=experiment_ids,
        description=description,
        hypothesis=hypothesis,
        conclusion=conclusion,
        overlay=overlay,
        primary_metric=primary_metric,
        variant=variant,
        search_space=search_space,
        trial=trial,
        detail=detail,
        last_n=last_n,
        verdict=verdict,
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
    name: str | None = None,
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
      - "init": Initialize a new harnessml project.
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
      - "add_target": Add a named target profile.
        Requires: name, target_column. Optional: task (default "binary"),
        metrics (list of metric names or JSON string).
      - "list_targets": List all named target profiles with details.
      - "set_target": Set a named target profile as the active target.
        Requires: name. Updates data.target_column, data.task, and
        backtest.metrics (if the profile defines metrics).
      - "studio": Get the Harness Studio dashboard URL for this session.
        Studio auto-starts with the MCP server — no setup needed.
    """
    return _load_handler("config").dispatch(
        action,
        ctx=ctx,
        name=name,
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
# 6. competitions
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def competitions(
    action: str,
    ctx: Context,
    config: str | None = None,
    name: str | None = None,
    n_sims: int | None = None,
    seed: int | None = None,
    pool_size: int | None = None,
    n_brackets: int | None = None,
    picks: str | None = None,
    actuals: str | None = None,
    adjustments: str | None = None,
    format_type: str | None = None,
    output_dir: str | None = None,
    top_n: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Manage competition simulations, brackets, and scoring.

    Actions:
      - "create": Create a competition from config. Requires config (JSON
        with format, n_participants, seeding, scoring, rounds, etc.).
        Optional: name (defaults to "default").
      - "list_formats": Show available competition formats (single_elimination,
        double_elimination, round_robin, swiss, group_knockout).
      - "simulate": Run Monte Carlo simulations. Requires name.
        Optional: n_sims (default 10000), seed (default 42).
      - "standings": Get standings distributions from simulation.
        Optional: name, top_n.
      - "round_probs": Entity progression probabilities per round.
        Optional: name, top_n.
      - "generate_brackets": Generate pool-aware brackets. Requires pool_size.
        Optional: name, n_brackets, n_sims, seed.
      - "score_bracket": Score picks vs actuals. Requires picks (JSON),
        actuals (JSON). Optional: name.
      - "adjust": Apply probability adjustments. Requires adjustments (JSON
        with entity_multipliers, probability_overrides, external_weight).
        Optional: name.
      - "explain": Generate pick explanations for generated brackets.
        Optional: name.
      - "profiles": Entity profiles sorted by champion probability.
        Optional: name, top_n.
      - "confidence": Pre-competition diagnostics (model disagreement).
        Optional: name.
      - "export": Export results. Requires output_dir.
        Optional: name, format_type (json/markdown/csv).
      - "list_strategies": Show available bracket generation strategies.
    """
    return _load_handler("competitions").dispatch(
        action,
        ctx=ctx,
        config=config,
        name=name,
        n_sims=n_sims,
        seed=seed,
        pool_size=pool_size,
        n_brackets=n_brackets,
        picks=picks,
        actuals=actuals,
        adjustments=adjustments,
        format_type=format_type,
        output_dir=output_dir,
        top_n=top_n,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 7. notebook
# -----------------------------------------------------------------------


@mcp.tool()
@_safe_tool
async def notebook(
    action: str,
    ctx: Context,
    type: str | None = None,
    content: str | None = None,
    tags: str | None = None,
    query: str | None = None,
    entry_id: str | None = None,
    reason: str | None = None,
    experiment_id: str | None = None,
    page: int | None = None,
    per_page: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Project notebook for persistent learnings across sessions.

    Actions:
      - "write": Add an entry. Requires type + content.
        type: theory | finding | research | decision | plan | note
        content: the entry text
        tags: optional JSON list of tags, e.g. '["model:xgb"]'
        experiment_id: optional link to an experiment
      - "read": Read entries (newest first, excludes struck).
        type: filter by entry type
        tags: filter by tags (JSON list)
        page: page number (default 1)
        per_page: entries per page (default 10)
      - "search": Full-text search. Requires query.
      - "strike": Hide an entry with a reason. Requires entry_id + reason.
      - "summary": Get current theory, plan, recent findings, and entity index.
        Call this at session start.
    """
    return _load_handler("notebook").dispatch(
        action,
        ctx=ctx,
        type=type,
        content=content,
        tags=tags,
        query=query,
        entry_id=entry_id,
        reason=reason,
        experiment_id=experiment_id,
        page=page,
        per_page=per_page,
        project_dir=project_dir,
    )


# -----------------------------------------------------------------------
# 8. pipeline
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
    name: str | None = None,
    top_n: int | None = None,
    mode: str | None = None,
    destination: str | None = None,
    output_path: str | None = None,
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
        Requires run_ids (list of 2 run IDs). Shows deltas with direction
        indicators.
      - "compare_latest": Compare the two most recent runs automatically.
        No parameters required. Shows deltas with direction indicators.
      - "compare_targets": Run backtests for all configured target profiles and
        show side-by-side comparison. No additional params required — uses
        target profiles from configure(action='add_target').
      - "explain": SHAP-based feature importance for a trained model. Requires
        shap package. Optional: name (model name, defaults to first), top_n
        (default 10), run_id (defaults to most recent).
      - "inspect_predictions": Inspect predictions from a backtest run.
        Optional: run_id (defaults to most recent), mode ("worst" for most
        confident wrong, "best" for most confident correct, "uncertain" for
        closest to 0.5), top_n (default 10).
      - "export_notebook": Generate a Jupyter notebook from the project config.
        Requires destination ("colab", "kaggle", or "local"). Optional: output_path.
      - "progress": Show workflow phase completion status — which exploration
        phases are done (feature discovery, model diversity, tuning readiness).
        No parameters required.
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
        name=name,
        top_n=top_n,
        mode=mode,
        destination=destination,
        output_path=output_path,
        project_dir=project_dir,
    )
    if asyncio.iscoroutine(result):
        result = await result
    return result


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

# Start Studio immediately at import time so it's available before
# any tool call. Uses cwd as default project_dir.
_get_emitter(project_dir=os.getcwd())

def main():
    """Entry point for the harness-ml MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
