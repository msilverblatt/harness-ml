"""Config writer — pure functions for YAML config mutations.

Every function: load -> modify -> validate -> write -> return markdown confirmation.
Used by both Claude Code plugin and MCP server tools.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _save_yaml(path: Path, data: dict) -> None:
    """Save data to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def _get_config_dir(project_dir: Path) -> Path:
    """Resolve the config directory from project dir."""
    config_dir = project_dir / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    return config_dir


def _invalidate_view_cache(project_dir: Path, name: str) -> None:
    """Remove cached parquet files for a view (and downstream dependents).

    The fingerprint-based cache is self-invalidating on next resolve, but this
    proactively removes stale files so they don't consume disk space.
    """
    cache_dir = Path(project_dir) / "data" / "views" / ".cache"
    if not cache_dir.exists():
        return
    for f in cache_dir.glob(f"{name}_*.parquet"):
        f.unlink()


# -----------------------------------------------------------------------
# Model tools
# -----------------------------------------------------------------------

def add_model(
    project_dir: Path,
    name: str,
    *,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
    active: bool = True,
    include_in_ensemble: bool = True,
    mode: str | None = None,
    prediction_type: str | None = None,
    cdf_scale: float | None = None,
    zero_fill_features: list[str] | None = None,
) -> str:
    """Add a model to models.yaml.

    Either model_type or preset must be specified. If preset is given,
    applies the preset defaults and merges any overrides.

    Returns markdown confirmation.
    """
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    if "models" not in data:
        data["models"] = {}

    if name in data["models"]:
        return f"**Error**: Model `{name}` already exists. Use a different name or remove it first."

    model_def: dict = {}

    if preset:
        from easyml.core.runner.presets import apply_preset
        model_def = apply_preset(preset, overrides=params or {})
    elif model_type:
        model_def["type"] = model_type
        if mode is not None:
            model_def["mode"] = mode
        if prediction_type is not None:
            model_def["prediction_type"] = prediction_type
        if params:
            model_def["params"] = params
    else:
        return "**Error**: Either `model_type` or `preset` must be specified."

    if features:
        model_def["features"] = features
    model_def["active"] = active
    model_def["include_in_ensemble"] = include_in_ensemble
    if cdf_scale is not None:
        model_def["cdf_scale"] = cdf_scale
    if zero_fill_features is not None:
        model_def["zero_fill_features"] = zero_fill_features

    data["models"][name] = model_def
    _save_yaml(models_path, data)

    n_features = len(model_def.get("features", []))
    type_str = model_def.get("type", "unknown")
    preset_str = f" (preset: {preset})" if preset else ""

    return (
        f"**Added model**: `{name}`\n"
        f"- Type: {type_str}{preset_str}\n"
        f"- Features: {n_features}\n"
        f"- Active: {active}\n"
        f"- Include in ensemble: {include_in_ensemble}"
    )


def remove_model(project_dir: Path, name: str, *, purge: bool = False) -> str:
    """Disable or permanently delete a model from models.yaml.

    By default (purge=False) sets active=False and include_in_ensemble=False.
    Pass purge=True to delete the model entry entirely.
    """
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        return f"**Error**: Model `{name}` not found."

    if purge:
        del models[name]
        _save_yaml(models_path, data)
        return f"**Removed model**: `{name}`"

    models[name]["active"] = False
    models[name]["include_in_ensemble"] = False
    _save_yaml(models_path, data)
    return (
        f"**Disabled model**: `{name}`\n"
        f"- Set `active: false`, `include_in_ensemble: false`\n"
        f"- Use `purge=True` to delete permanently."
    )


def update_model(
    project_dir: Path,
    name: str,
    *,
    features: list[str] | None = None,
    params: dict | None = None,
    active: bool | None = None,
    include_in_ensemble: bool | None = None,
    mode: str | None = None,
    prediction_type: str | None = None,
    cdf_scale: float | None = None,
    zero_fill_features: list[str] | None = None,
) -> str:
    """Update an existing model in models.yaml.

    Only provided fields are merged — None values keep the existing value.
    For params, does a dict merge (update, not replace).
    """
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        return f"**Error**: Model `{name}` not found. Available: {sorted(models.keys())}"

    model_def = models[name]

    if features is not None:
        model_def["features"] = features
    if params is not None:
        existing_params = model_def.get("params", {})
        existing_params.update(params)
        model_def["params"] = existing_params
    if active is not None:
        model_def["active"] = active
    if include_in_ensemble is not None:
        model_def["include_in_ensemble"] = include_in_ensemble
    if mode is not None:
        model_def["mode"] = mode
    if prediction_type is not None:
        model_def["prediction_type"] = prediction_type
    if cdf_scale is not None:
        model_def["cdf_scale"] = cdf_scale
    if zero_fill_features is not None:
        model_def["zero_fill_features"] = zero_fill_features

    _save_yaml(models_path, data)

    status = "active" if model_def.get("active", True) else "inactive"
    n_feat = len(model_def.get("features", []))
    in_ens = "yes" if model_def.get("include_in_ensemble", True) else "no"

    return (
        f"**Updated model**: `{name}`\n"
        f"- Type: {model_def.get('type', '?')}\n"
        f"- Status: {status}\n"
        f"- Features: {n_feat}\n"
        f"- Ensemble: {in_ens}\n"
        f"- Params: {model_def.get('params', {})}"
    )


def show_models(project_dir: Path) -> str:
    """List all models with type, status, and feature count."""
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if not models:
        return "No models configured."

    lines = ["## Models\n"]
    lines.append("| Name | Type | Status | Features | Ensemble |")
    lines.append("|------|------|--------|----------|----------|")

    for name, m in sorted(models.items()):
        status = "active" if m.get("active", True) else "inactive"
        n_feat = len(m.get("features", []))
        in_ens = "yes" if m.get("include_in_ensemble", True) else "no"
        lines.append(f"| {name} | {m.get('type', '?')} | {status} | {n_feat} | {in_ens} |")

    return "\n".join(lines)


def show_model(project_dir: Path, name: str) -> str:
    """Show full configuration for a single model."""
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        available = sorted(models.keys())
        return f"**Error**: Model `{name}` not found. Available: {', '.join(available) or '(none)'}"

    model_config = models[name]

    lines = [f"## Model: `{name}`\n"]
    lines.append(f"- **Type**: {model_config.get('type', '?')}")
    lines.append(f"- **Active**: {model_config.get('active', True)}")
    lines.append(f"- **In Ensemble**: {model_config.get('include_in_ensemble', True)}")

    features = model_config.get("features", [])
    lines.append(f"\n### Features ({len(features)})\n")
    for f in features:
        lines.append(f"- {f}")

    params = model_config.get("params", {})
    if params:
        lines.append(f"\n### Parameters\n")
        lines.append(f"```yaml\n{yaml.dump(params, default_flow_style=False)}```")

    # Show any other config keys
    skip_keys = {"type", "active", "include_in_ensemble", "features", "params"}
    extras = {k: v for k, v in model_config.items() if k not in skip_keys}
    if extras:
        lines.append(f"\n### Other Settings\n")
        lines.append(f"```yaml\n{yaml.dump(extras, default_flow_style=False)}```")

    return "\n".join(lines)


def show_presets() -> str:
    """List available model presets."""
    from easyml.core.runner.presets import get_preset, list_presets

    preset_names = list_presets()
    if not preset_names:
        return "No presets available."

    lines = ["## Model Presets\n"]
    lines.append("| Name | Type | Mode |")
    lines.append("|------|------|------|")
    for name in preset_names:
        preset = get_preset(name)
        lines.append(
            f"| {name} | {preset.get('type', '?')} | {preset.get('mode', '?')} |"
        )

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Ensemble tools
# -----------------------------------------------------------------------

def configure_ensemble(
    project_dir: Path,
    *,
    method: str | None = None,
    temperature: float | None = None,
    exclude_models: list[str] | None = None,
    calibration: str | None = None,
    pre_calibration: dict | None = None,
    prior_feature: str | None = None,
    spline_prob_max: float | None = None,
    spline_n_bins: int | None = None,
    **kwargs,
) -> str:
    """Update ensemble.yaml configuration.

    Parameters
    ----------
    calibration : str | None
        Post-ensemble calibration method: 'spline', 'isotonic', 'platt', 'none'.
    pre_calibration : dict | None
        Per-model pre-calibration applied before the meta-learner.
        Maps model_name -> method, e.g. {"xgb_core": "platt"}.
        Valid methods: 'platt', 'isotonic', 'spline', 'none'.
    """
    config_dir = _get_config_dir(Path(project_dir))
    ensemble_path = config_dir / "ensemble.yaml"
    data = _load_yaml(ensemble_path)

    if "ensemble" not in data:
        data["ensemble"] = {}

    ens = data["ensemble"]

    if method is not None:
        ens["method"] = method
    if temperature is not None:
        ens["temperature"] = temperature
    if exclude_models is not None:
        ens["exclude_models"] = exclude_models
    if calibration is not None:
        ens["calibration"] = calibration
    if pre_calibration is not None:
        ens["pre_calibration"] = pre_calibration
    if prior_feature is not None:
        ens["prior_feature"] = prior_feature
    if spline_prob_max is not None:
        ens["spline_prob_max"] = spline_prob_max
    if spline_n_bins is not None:
        ens["spline_n_bins"] = spline_n_bins

    for key, val in kwargs.items():
        ens[key] = val

    _save_yaml(ensemble_path, data)

    pre_cal_models = list(ens.get("pre_calibration", {}).keys())
    return (
        f"**Updated ensemble config**\n"
        f"- Method: {ens.get('method', 'average')}\n"
        f"- Temperature: {ens.get('temperature', 1.0)}\n"
        f"- Calibration: {ens.get('calibration', 'spline')}\n"
        f"- Pre-calibration: {pre_cal_models or 'none'}"
    )


# -----------------------------------------------------------------------
# Backtest tools
# -----------------------------------------------------------------------

def configure_backtest(
    project_dir: Path,
    *,
    cv_strategy: str | None = None,
    fold_values: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
    fold_column: str | None = None,
) -> str:
    """Update backtest section of pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "backtest" not in data:
        data["backtest"] = {}

    bt = data["backtest"]

    if cv_strategy is not None:
        from easyml.core.runner.schema import _CV_STRATEGY_ALIASES
        bt["cv_strategy"] = _CV_STRATEGY_ALIASES.get(cv_strategy, cv_strategy)
    if fold_values is not None:
        bt["fold_values"] = fold_values
    if metrics is not None:
        bt["metrics"] = metrics
    if min_train_folds is not None:
        bt["min_train_folds"] = min_train_folds
    if fold_column is not None:
        bt["fold_column"] = fold_column

    _save_yaml(pipeline_path, data)

    fold_col_str = f"\n- Fold column: {bt['fold_column']}" if "fold_column" in bt else ""
    return (
        f"**Updated backtest config**\n"
        f"- CV strategy: {bt.get('cv_strategy', 'N/A')}\n"
        f"- Fold values: {bt.get('fold_values', [])}\n"
        f"- Metrics: {bt.get('metrics', [])}"
        f"{fold_col_str}"
    )


def update_data_config(
    project_dir: Path,
    *,
    target_column: str | None = None,
    key_columns: list[str] | None = None,
    time_column: str | None = None,
) -> str:
    """Update data section of pipeline.yaml (target_column, key_columns, time_column)."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}

    d = data["data"]
    updates = []

    if target_column is not None:
        d["target_column"] = target_column
        updates.append(f"- Target column: {target_column}")
    if key_columns is not None:
        d["key_columns"] = key_columns
        updates.append(f"- Key columns: {key_columns}")
    if time_column is not None:
        d["time_column"] = time_column
        updates.append(f"- Time column: {time_column}")

    if not updates:
        return "**No changes** — provide at least one of: target_column, key_columns, time_column."

    _save_yaml(pipeline_path, data)

    return "**Updated data config**\n" + "\n".join(updates)


def configure_exclude_columns(
    project_dir: Path,
    *,
    add_columns: list[str] | None = None,
    remove_columns: list[str] | None = None,
) -> str:
    """Add or remove columns from data.exclude_columns in pipeline.yaml.

    Excluded columns are never used as model features or in feature discovery.
    Use this to mark regression target columns (e.g. 'margin') or ID columns
    that should not be treated as predictive features.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    current = list(data.get("data", {}).get("exclude_columns", []))
    current_set = set(current)

    added, removed = [], []
    if add_columns:
        for col in add_columns:
            if col not in current_set:
                current.append(col)
                current_set.add(col)
                added.append(col)

    if remove_columns:
        for col in remove_columns:
            if col in current_set:
                current.remove(col)
                current_set.discard(col)
                removed.append(col)

    if "data" not in data:
        data["data"] = {}
    data["data"]["exclude_columns"] = current
    _save_yaml(pipeline_path, data)

    lines = ["**Updated `data.exclude_columns`**"]
    if added:
        lines.append(f"- Added: {added}")
    if removed:
        lines.append(f"- Removed: {removed}")
    lines.append(f"- Current list: {current}")
    return "\n".join(lines)


def configure_denylist(
    project_dir: Path,
    *,
    add_columns: list[str] | None = None,
    remove_columns: list[str] | None = None,
) -> str:
    """Add or remove columns from the guardrails feature leakage denylist.

    The denylist is checked by check_guardrails() — any model whose feature
    list contains a denied column causes a FAIL.
    """
    project_dir = Path(project_dir)
    sources_path = _get_config_dir(project_dir) / "sources.yaml"
    data = _load_yaml(sources_path)

    if "guardrails" not in data:
        data["guardrails"] = {}
    current = list(data["guardrails"].get("feature_leakage_denylist", []))
    current_set = set(current)

    added, removed = [], []
    if add_columns:
        for col in add_columns:
            if col not in current_set:
                current.append(col)
                current_set.add(col)
                added.append(col)

    if remove_columns:
        for col in remove_columns:
            if col in current_set:
                current.remove(col)
                current_set.discard(col)
                removed.append(col)

    data["guardrails"]["feature_leakage_denylist"] = current
    _save_yaml(sources_path, data)

    lines = ["**Updated `guardrails.feature_leakage_denylist`**"]
    if added:
        lines.append(f"- Added: {added}")
    if removed:
        lines.append(f"- Removed: {removed}")
    lines.append(f"- Current denylist: {current}")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Target profile tools
# -----------------------------------------------------------------------

def add_target(
    project_dir: Path,
    name: str,
    *,
    column: str,
    task: str = "binary",
    metrics: list[str] | None = None,
) -> str:
    """Add a named target profile to data.targets in pipeline.yaml.

    Returns markdown confirmation.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}
    if "targets" not in data["data"]:
        data["data"]["targets"] = {}

    target_def: dict = {"column": column, "task": task}
    if metrics:
        target_def["metrics"] = metrics

    data["data"]["targets"][name] = target_def
    _save_yaml(pipeline_path, data)

    lines = [f"**Added target profile `{name}`**"]
    lines.append(f"- Column: `{column}`")
    lines.append(f"- Task: `{task}`")
    if metrics:
        lines.append(f"- Metrics: {metrics}")
    return "\n".join(lines)


def list_targets(project_dir: Path) -> str:
    """List all named target profiles from pipeline.yaml.

    Returns markdown-formatted list.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    targets = data.get("data", {}).get("targets", {})
    active_column = data.get("data", {}).get("target_column", "result")
    active_task = data.get("data", {}).get("task", "classification")

    if not targets:
        return (
            f"**No named target profiles defined.**\n"
            f"- Default target: `{active_column}` (task: `{active_task}`)\n"
            f"- Use `add_target` to define named profiles."
        )

    lines = ["**Target Profiles**\n"]
    for name, tgt in targets.items():
        col = tgt.get("column", "?")
        task = tgt.get("task", "binary")
        metrics = tgt.get("metrics", [])
        active_marker = " **(active)**" if col == active_column and task == active_task else ""
        line = f"- **{name}**{active_marker}: column=`{col}`, task=`{task}`"
        if metrics:
            line += f", metrics={metrics}"
        lines.append(line)

    return "\n".join(lines)


def set_active_target(project_dir: Path, name: str) -> str:
    """Set a named target profile as the active target.

    Updates data.target_column, data.task, and optionally backtest.metrics.
    Returns markdown confirmation.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    targets = data.get("data", {}).get("targets", {})
    if name not in targets:
        available = ", ".join(sorted(targets.keys())) if targets else "(none defined)"
        return f"**Error**: Unknown target `{name}`. Available targets: {available}"

    tgt = targets[name]
    data["data"]["target_column"] = tgt["column"]
    data["data"]["task"] = tgt.get("task", "binary")

    metrics = tgt.get("metrics", [])
    if metrics:
        if "backtest" not in data:
            data["backtest"] = {}
        data["backtest"]["metrics"] = metrics

    _save_yaml(pipeline_path, data)

    lines = [f"**Activated target profile `{name}`**"]
    lines.append(f"- target_column: `{tgt['column']}`")
    lines.append(f"- task: `{tgt.get('task', 'binary')}`")
    if metrics:
        lines.append(f"- backtest.metrics updated to: {metrics}")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Data tools
# -----------------------------------------------------------------------

def add_dataset(
    project_dir: Path,
    data_path: str,
    *,
    join_on: list[str] | None = None,
    prefix: str | None = None,
    features_dir: str | None = None,
    auto_clean: bool = True,
) -> str:
    """Add a new dataset by merging into the features parquet.

    Uses DataPipeline when sources are configured in DataConfig,
    falls back to direct ingest otherwise.
    """
    from easyml.core.runner.data_utils import load_data_config

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = None

    # Use DataPipeline if config has sources configured
    if config is not None and config.sources:
        from easyml.core.runner.data_pipeline import DataPipeline
        from easyml.core.runner.schema import SourceConfig

        pipeline = DataPipeline(project_dir, config)
        name = Path(data_path).stem
        source = SourceConfig(name=name, path=data_path, join_on=join_on)
        pipeline.config.sources[name] = source
        result = pipeline.refresh(sources=[name])
        cols = result.columns_added.get(name, [])
        lines = [f"## Ingested: {name}\n"]
        lines.append(f"- **Columns added**: {len(cols)}")
        if cols:
            cols_preview = ", ".join(cols[:10])
            if len(cols) > 10:
                cols_preview += f", ... (+{len(cols) - 10} more)"
            lines.append(f"- **Columns**: {cols_preview}")
        lines.append("- **Source registered** in pipeline config")
        if result.errors:
            lines.append(f"\n### Errors\n")
            for src, err in result.errors.items():
                lines.append(f"- {src}: {err}")
        return "\n".join(lines)

    # Fallback to direct ingest
    from easyml.core.runner.data_ingest import ingest_dataset
    result = ingest_dataset(
        project_dir=project_dir,
        data_path=data_path,
        join_on=join_on,
        prefix=prefix,
        features_dir=features_dir,
        auto_clean=auto_clean,
    )
    return result.format_summary()


def derive_column(
    project_dir: Path,
    name: str,
    expression: str,
    *,
    group_by: str | None = None,
    dtype: str | None = None,
) -> str:
    """Derive a new column from a pandas expression and save to the feature store."""
    from easyml.core.runner.data_ingest import derive_column as _derive

    return _derive(
        Path(project_dir),
        name,
        expression,
        group_by=group_by,
        dtype=dtype,
    )


def drop_rows(
    project_dir: Path,
    *,
    column: str | None = None,
    condition: str = "null",
) -> str:
    """Drop rows from the feature store by condition."""
    from easyml.core.runner.data_ingest import drop_rows as _drop

    return _drop(
        Path(project_dir),
        column=column,
        condition=condition,
    )


def sample_data(project_dir: Path, *, fraction=0.1, stratify_column=None, seed=42) -> str:
    """Sample the feature store for fast iteration."""
    from easyml.core.runner.data_ingest import sample_data as _sample
    return _sample(Path(project_dir), fraction=fraction, stratify_column=stratify_column, seed=seed)


def restore_full_data(project_dir: Path) -> str:
    """Restore the full feature store from backup."""
    from easyml.core.runner.data_ingest import restore_full_data as _restore
    return _restore(Path(project_dir))


def inspect_data(project_dir: Path, *, column: str | None = None) -> str:
    """Inspect the features dataset.

    Without column: overview of all columns (shape, dtypes, null counts).
    With column: detailed statistics for that specific column.
    """
    from easyml.core.runner.data_utils import get_features_df, load_data_config
    from easyml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    if column is not None:
        return _inspect_column(df, column)
    return _inspect_overview(df)


def _inspect_overview(df) -> str:
    """Return a markdown overview of all columns in the DataFrame."""
    n_rows, n_cols = df.shape
    lines = [
        f"## Data Overview: {n_rows:,} rows x {n_cols} columns\n",
        "| Column | Dtype | Nulls | Null% |",
        "|--------|-------|-------|-------|",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0.0
        lines.append(f"| {col} | {dtype} | {null_count:,} | {null_pct:.1f}% |")
    return "\n".join(lines)


def _inspect_column(df, column: str) -> str:
    """Return detailed statistics for a single column."""
    if column not in df.columns:
        available = ", ".join(f"`{c}`" for c in sorted(df.columns))
        return f"**Error**: Column `{column}` not found. Available columns: {available}"

    series = df[column]
    dtype = str(series.dtype)
    non_null = int(series.notna().sum())
    n_unique = int(series.nunique())
    n_rows = len(series)

    lines = [
        f"## Column: `{column}`\n",
        f"- **Dtype**: {dtype}",
        f"- **Non-null**: {non_null:,} / {n_rows:,}",
        f"- **Unique values**: {n_unique:,}",
    ]

    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        lines.append("")
        lines.append("### Statistics")
        lines.append(f"- **Mean**: {desc['mean']:.4f}")
        lines.append(f"- **Std**: {desc['std']:.4f}")
        lines.append(f"- **Min**: {desc['min']:.4f}")
        lines.append(f"- **25%**: {desc['25%']:.4f}")
        lines.append(f"- **50%**: {desc['50%']:.4f}")
        lines.append(f"- **75%**: {desc['75%']:.4f}")
        lines.append(f"- **Max**: {desc['max']:.4f}")

    if n_unique <= 20:
        lines.append("")
        lines.append("### Value Counts")
        vc = series.value_counts(dropna=False)
        for val, count in vc.items():
            pct = count / n_rows * 100 if n_rows > 0 else 0.0
            label = "NaN" if pd.isna(val) else str(val)
            lines.append(f"- `{label}`: {count:,} ({pct:.1f}%)")
    elif n_unique > 20:
        lines.append("")
        lines.append("### Sample Values (first 10)")
        for val in series.dropna().unique()[:10]:
            lines.append(f"- `{val}`")

    return "\n".join(lines)


def profile_data(project_dir: Path, category: str | None = None) -> str:
    """Profile the features dataset."""
    from easyml.core.runner.data_utils import get_features_df, load_data_config
    from easyml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    from easyml.core.runner.data_profiler import profile_dataset

    profile = profile_dataset(config=config, df=df)

    if category:
        return profile.format_columns(category=category)
    return profile.format_summary()


def available_features(
    project_dir: Path,
    prefix: str | None = None,
    type_filter: str | None = None,
) -> str:
    """List available feature columns from the dataset.

    If the project uses the declarative feature store, shows features
    grouped by type. Otherwise falls back to column listing.
    """
    from easyml.core.runner.data_utils import get_features_df, load_data_config
    from easyml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    # Check for declarative feature store
    if config.feature_defs:
        from easyml.core.runner.feature_store import FeatureStore
        from easyml.core.runner.schema import FeatureType

        store = FeatureStore(project_dir, config)
        ft = FeatureType(type_filter) if type_filter else None
        features = store.available(type_filter=ft)

        if not features:
            return "No declarative features registered."

        lines = [f"## Declarative Features ({len(features)})\n"]
        by_type: dict[str, list] = {}
        for f in features:
            by_type.setdefault(f.type.value, []).append(f)

        for ft_name, feats in by_type.items():
            lines.append(f"### {ft_name.title()} ({len(feats)})")
            for f in feats:
                lines.append(f"- `{f.name}` — {f.description or f.category}")
            lines.append("")

        return "\n".join(lines)

    # Fallback: flat column listing
    import pandas as pd

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."
    cols = sorted(df.columns)
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]

    if not cols:
        return "No features found."

    lines = [f"## Available Features ({len(cols)} columns)\n"]
    for col in cols:
        lines.append(f"- `{col}`")
    return "\n".join(lines)


def feature_store_status(project_dir: Path) -> str:
    """Quick overview of the feature store state.

    Returns: row count, column count, target distribution,
    time column range, source count.
    """
    from easyml.core.runner.data_utils import get_features_df, load_data_config
    from easyml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    import pandas as pd

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature store found. Ingest data first with `manage_data(action='add')` or set a features_view."

    n_rows = len(df)
    n_cols = len(df.columns)

    lines = ["## Feature Store Status\n"]
    lines.append(f"- **Rows**: {n_rows}")
    lines.append(f"- **Columns**: {n_cols}")

    # Show source info
    if config.features_view:
        lines.append(f"- **Source**: view `{config.features_view}`")
    else:
        import os
        from datetime import datetime
        from easyml.core.runner.data_utils import get_features_path
        parquet_path = get_features_path(project_dir, config)
        lines.append(f"- **File**: `{parquet_path.relative_to(project_dir)}`")
        if parquet_path.exists():
            mtime = os.path.getmtime(parquet_path)
            lines.append(f"- **Last modified**: {datetime.fromtimestamp(mtime).isoformat()}")

    # Target column distribution
    target_col = config.target_column
    if target_col and target_col in df.columns:
        dist = df[target_col].value_counts()
        lines.append(f"\n### Target Distribution (`{target_col}`)\n")
        for val, count in dist.items():
            pct = count / n_rows * 100
            lines.append(f"- {val}: {count} ({pct:.1f}%)")
    elif target_col:
        lines.append(f"\n*Target column `{target_col}` not found in data.*")

    # Time column range
    time_col = config.time_column
    if time_col and time_col in df.columns:
        lines.append(f"\n### Time Range (`{time_col}`)\n")
        lines.append(f"- Min: {df[time_col].min()}")
        lines.append(f"- Max: {df[time_col].max()}")
        lines.append(f"- Unique values: {df[time_col].nunique()}")

    # Source count
    registry_path = project_dir / "data" / "source_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        n_sources = len(registry.get("sources", []))
        lines.append(f"\n- **Ingested sources**: {n_sources}")

    return "\n".join(lines)


def list_sources(project_dir: Path) -> str:
    """List ingested data sources from the source registry.

    Reads data/source_registry.json and returns a summary of
    each source: name, path, columns added, row count.
    """
    project_dir = Path(project_dir)
    registry_path = project_dir / "data" / "source_registry.json"

    if not registry_path.exists():
        return "**No sources registered.** Ingest data with `manage_data(action='add')` first."

    registry = json.loads(registry_path.read_text())
    sources = registry.get("sources", [])

    if not sources:
        return "**No sources registered.** Ingest data with `manage_data(action='add')` first."

    lines = [f"## Data Sources ({len(sources)} registered)\n"]
    lines.append("| # | Name | Path | Columns Added | Rows | Bootstrap |")
    lines.append("|---|------|------|---------------|------|-----------|")

    for i, src in enumerate(sources, 1):
        name = src.get("name", "unknown")
        path = src.get("path", "—")
        cols = src.get("columns_added", [])
        rows = src.get("rows", "—")
        bootstrap = "Yes" if src.get("is_bootstrap") else "No"
        lines.append(f"| {i} | {name} | `{path}` | {len(cols)} | {rows} | {bootstrap} |")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Feature tools
# -----------------------------------------------------------------------

def _persist_feature_defs(project_dir: Path, config) -> None:
    """Write feature_defs back to pipeline.yaml so the runner picks them up."""
    config_dir = _get_config_dir(project_dir)
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}

    if config.feature_defs:
        data["data"]["feature_defs"] = {
            name: feat.model_dump(mode="json")
            for name, feat in config.feature_defs.items()
        }
    elif "feature_defs" in data.get("data", {}):
        del data["data"]["feature_defs"]

    _save_yaml(pipeline_path, data)


def add_feature(
    project_dir: Path,
    name: str,
    formula: str | None = None,
    *,
    type: str | None = None,
    source: str | None = None,
    column: str | None = None,
    condition: str | None = None,
    pairwise_mode: str = "diff",
    category: str = "general",
    description: str = "",
) -> str:
    """Create a new feature via the declarative FeatureStore.

    All features go through the FeatureStore. If type is not specified,
    it is inferred: formula -> pairwise, condition -> regime, source -> team.
    """
    project_dir = Path(project_dir)

    from easyml.core.runner.feature_store import FeatureStore
    from easyml.core.runner.schema import FeatureDef, FeatureType, PairwiseMode
    from easyml.core.runner.data_utils import load_data_config

    config = load_data_config(project_dir)
    store = FeatureStore(project_dir, config)

    # Infer type if not specified
    if type is None:
        if formula is not None:
            type = "pairwise"
        elif condition is not None:
            type = "regime"
        elif source is not None:
            type = "entity"
        else:
            raise ValueError(
                "Must provide type, formula, condition, or source."
            )

    feature_type = FeatureType(type)
    pw_mode = PairwiseMode(pairwise_mode)

    feature_def = FeatureDef(
        name=name,
        type=feature_type,
        source=source,
        column=column,
        formula=formula,
        condition=condition,
        pairwise_mode=pw_mode,
        category=category,
        description=description,
    )

    result = store.add(feature_def)

    # Persist feature_defs to pipeline.yaml
    store.save_registry()
    _persist_feature_defs(project_dir, config)

    # Format response
    lines = [f"## Added {type} feature: {name}\n"]
    if description:
        lines.append(f"_{description}_\n")

    if feature_type == FeatureType.ENTITY:
        cache_entry = store._cache._entries.get(name)
        if cache_entry and cache_entry.derivatives:
            lines.append("**Auto-generated pairwise:**")
            matchup_df = store._load_matchup_data()
            target_col = config.target_column
            for deriv in cache_entry.derivatives:
                try:
                    deriv_series = store.compute(deriv)
                    corr = 0.0
                    if target_col in matchup_df.columns:
                        corr = float(deriv_series.corr(matchup_df[target_col].astype(float)))
                        if not isinstance(corr, float) or corr != corr:
                            corr = 0.0
                    lines.append(f"- `{deriv}` (r={corr:+.4f})")
                except Exception:
                    lines.append(f"- `{deriv}`")

    lines.append(f"\n- **Correlation**: {result.correlation:+.4f}")
    lines.append(f"- **Null rate**: {result.null_rate:.1%}")
    if result.stats:
        for k, v in result.stats.items():
            lines.append(f"- **{k.title()}**: {v:.4f}")
    lines.append(f"- **Category**: {category}")

    # Check for redundant formulas
    if formula and config.feature_defs:
        warnings = []
        for fname, fdef in config.feature_defs.items():
            if fname != name and getattr(fdef, "formula", None) == formula:
                warnings.append(
                    f"**Warning**: Formula is identical to existing feature `{fname}`"
                )
        if warnings:
            lines.append("")
            lines.extend(warnings)

    return "\n".join(lines)


def add_features_batch(
    project_dir: Path,
    features: list[dict],
) -> str:
    """Create multiple features via the declarative FeatureStore.

    Each dict can include: name, formula, type, source, column, condition,
    pairwise_mode, category, description. Handles @-references between
    features via topological ordering.
    """
    from easyml.core.runner.feature_store import FeatureStore
    from easyml.core.runner.schema import FeatureDef, FeatureType, PairwiseMode
    from easyml.core.runner.data_utils import load_data_config
    from easyml.core.runner.feature_engine import _topological_sort_features

    project_dir = Path(project_dir)
    config = load_data_config(project_dir)
    store = FeatureStore(project_dir, config)

    # Resolve dependency order
    feature_names = {f["name"] for f in features}
    ordered = _topological_sort_features(features, feature_names)

    results = []
    for feat_dict in ordered:
        feat_name = feat_dict["name"]
        feat_formula = feat_dict.get("formula")
        feat_type = feat_dict.get("type")
        feat_condition = feat_dict.get("condition")
        feat_source = feat_dict.get("source")

        # Infer type if not specified
        if feat_type is None:
            if feat_formula is not None:
                feat_type = "pairwise"
            elif feat_condition is not None:
                feat_type = "regime"
            elif feat_source is not None:
                feat_type = "entity"
            else:
                feat_type = "pairwise"

        feature_def = FeatureDef(
            name=feat_name,
            type=FeatureType(feat_type),
            formula=feat_formula,
            source=feat_source,
            column=feat_dict.get("column"),
            condition=feat_condition,
            pairwise_mode=PairwiseMode(feat_dict.get("pairwise_mode", "diff")),
            category=feat_dict.get("category", "general"),
            description=feat_dict.get("description", ""),
        )
        result = store.add(feature_def)
        results.append(result)

    # Persist all at once
    store.save_registry()
    _persist_feature_defs(project_dir, config)

    lines = [f"## Created {len(results)} Features\n"]
    for r in results:
        lines.append(f"- **{r.column_added}** (corr={r.correlation:+.4f})")
    return "\n".join(lines)


def test_feature_transformations(
    project_dir: Path,
    features: list[str],
    *,
    test_interactions: bool = True,
) -> str:
    """Test mathematical transformations of features."""
    from easyml.core.runner.data_utils import get_features_df, load_data_config
    from easyml.core.runner.transformation_tester import run_transformation_tests

    project_dir = Path(project_dir)

    # Load config and features DataFrame
    feat_defs = None
    try:
        config = load_data_config(project_dir)
        if config.feature_defs:
            feat_defs = dict(config.feature_defs)
    except Exception:
        config = None

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    report = run_transformation_tests(
        project_dir=project_dir,
        features=features,
        test_interactions=test_interactions,
        feature_defs=feat_defs,
        df=df,
    )
    return report.format_summary()


def discover_features(
    project_dir: Path,
    *,
    top_n: int = 20,
    method: str = "xgboost",
    on_progress: callable | None = None,
) -> str:
    """Run feature discovery analysis."""
    from easyml.core.runner.data_utils import get_feature_columns, get_features_df, load_data_config
    from easyml.core.runner.schema import DataConfig

    def _report(step, total, msg):
        if on_progress is not None:
            on_progress(step, total, msg)

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    import pandas as pd

    _report(0, 5, "Loading feature data...")
    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    # Get feature columns and feature_defs from config if available
    feature_cols = None
    feat_defs = None
    pipeline_data = _load_yaml(_get_config_dir(project_dir) / "pipeline.yaml")
    backtest_data = pipeline_data.get("backtest", {})
    fold_col = backtest_data.get("fold_column")
    if config is not None:
        feature_cols = get_feature_columns(df, config, fold_column=fold_col)
        if config.feature_defs:
            feat_defs = dict(config.feature_defs)

    # Exclude denylist columns from feature discovery
    _sources_data = _load_yaml(_get_config_dir(project_dir) / "sources.yaml")
    _denylist = set(_sources_data.get("guardrails", {}).get("feature_leakage_denylist", []))
    if _denylist and feature_cols:
        feature_cols = [c for c in feature_cols if c not in _denylist]

    from easyml.core.runner.feature_discovery import (
        compute_feature_correlations,
        compute_feature_importance,
        detect_redundant_features,
        format_discovery_report,
        suggest_feature_groups,
    )

    _report(1, 5, "Computing feature correlations...")
    correlations = compute_feature_correlations(
        df, top_n=top_n, feature_columns=feature_cols, feature_defs=feat_defs,
    )
    _report(2, 5, "Computing feature importance (method=%s)..." % method)
    importance = compute_feature_importance(
        df, method=method, top_n=top_n, feature_columns=feature_cols, feature_defs=feat_defs,
    )
    _report(3, 5, "Detecting redundant features...")
    redundant = detect_redundant_features(df, feature_columns=feature_cols)
    _report(4, 5, "Suggesting feature groups...")
    groups = suggest_feature_groups(df, feature_columns=feature_cols, feature_defs=feat_defs)

    return format_discovery_report(correlations, importance, redundant, groups)


def auto_search_features(
    project_dir: Path,
    features: list[str] | None = None,
    *,
    search_types: list[str] | None = None,
    top_n: int = 20,
) -> str:
    """Run automated feature search over given columns.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    features : list[str] | None
        Column names to search over. If None, uses all feature columns.
    search_types : list[str] | None
        Which search types to run: "interactions", "lags", "rolling".
        Defaults to all three.
    top_n : int
        Number of top results to return.

    Returns
    -------
    str
        Markdown-formatted report of top candidates.
    """
    from easyml.core.runner.auto_search import auto_search, format_auto_search_report
    from easyml.core.runner.data_utils import get_feature_columns, get_features_df, load_data_config
    from easyml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    # Resolve feature columns
    pipeline_data = _load_yaml(_get_config_dir(project_dir) / "pipeline.yaml")
    backtest_data = pipeline_data.get("backtest", {})
    fold_col = backtest_data.get("fold_column")
    if features:
        feature_cols = [c for c in features if c in df.columns]
        missing = [c for c in features if c not in df.columns]
        if missing:
            logger.warning("Columns not found in dataset, skipping: %s", missing)
    else:
        feature_cols = get_feature_columns(df, config, fold_column=fold_col)

    if not feature_cols:
        return "**Error**: No feature columns found to search over."

    target_col = config.target_column

    results = auto_search(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        search_types=search_types,
        top_n=top_n,
    )
    return format_auto_search_report(results)


# -----------------------------------------------------------------------
# Experiment tools
# -----------------------------------------------------------------------

def experiment_create(
    project_dir: Path,
    description: str,
    *,
    hypothesis: str = "",
) -> str:
    """Create a new experiment directory with auto-generated ID."""
    config_dir = _get_config_dir(Path(project_dir))

    experiments_dir = Path(project_dir) / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    from easyml.core.runner.experiment import auto_next_id

    exp_id = auto_next_id(experiments_dir)

    exp_dir = experiments_dir / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Write empty overlay
    overlay_path = exp_dir / "overlay.yaml"
    overlay_path.write_text(
        yaml.dump({"description": description}, default_flow_style=False)
    )

    # Write hypothesis if provided
    if hypothesis:
        (exp_dir / "hypothesis.txt").write_text(hypothesis)

    return (
        f"**Created experiment**: `{exp_id}`\n"
        f"- Directory: `{exp_dir}`\n"
        f"- Overlay: `{overlay_path}`\n"
        f"- Description: {description}"
    )


def _expand_dot_keys(flat: dict) -> dict:
    """Expand a flat dict with dot-notation keys into nested dicts.

    Keys that don't contain dots are left as-is.  Keys that do contain dots
    are expanded into nested structures using ``set_nested_key``.

    Example::

        _expand_dot_keys({"models.xgb.features": ["a", "b"]})
        # -> {"models": {"xgb": {"features": ["a", "b"]}}}
    """
    from easyml.core.runner.sweep import set_nested_key

    result: dict = {}
    for key, value in flat.items():
        if "." in key:
            set_nested_key(result, key, value)
        else:
            result[key] = value
    return result


def write_overlay(
    project_dir: Path,
    experiment_id: str,
    overlay: dict,
) -> str:
    """Write an overlay YAML to an experiment directory.

    Dot-notation keys (e.g. ``models.xgb_core.features``) are expanded
    into nested dicts before writing so that ``deep_merge`` can apply them.
    """
    experiments_dir = Path(project_dir) / "experiments"
    exp_dir = experiments_dir / experiment_id

    if not exp_dir.exists():
        return f"**Error**: Experiment directory not found: {exp_dir}"

    nested_overlay = _expand_dot_keys(overlay)
    overlay_path = exp_dir / "overlay.yaml"
    overlay_path.write_text(
        yaml.dump(nested_overlay, default_flow_style=False, sort_keys=False)
    )

    return (
        f"**Overlay written**: `{overlay_path}`\n"
        f"- Keys: {list(overlay.keys())}"
    )


def show_config(project_dir: Path) -> str:
    """Show the resolved project configuration."""
    config_dir = _get_config_dir(Path(project_dir))
    from easyml.core.runner.validator import validate_project

    result = validate_project(str(config_dir))

    if not result.valid:
        return f"**Config validation failed**:\n{result.format()}"

    config = result.config
    lines = ["## Project Configuration\n"]

    # Models
    lines.append(f"### Models ({len(config.models)})\n")
    for name, m in sorted(config.models.items()):
        status = "active" if m.active else "inactive"
        lines.append(f"- **{name}**: {m.type} ({status}, {len(m.features)} features)")

    # Ensemble
    lines.append(f"\n### Ensemble\n")
    lines.append(f"- Method: {config.ensemble.method}")
    lines.append(f"- Temperature: {config.ensemble.temperature}")

    # Backtest
    lines.append(f"\n### Backtest\n")
    lines.append(f"- Strategy: {config.backtest.cv_strategy}")
    lines.append(f"- Fold values: {config.backtest.fold_values}")
    lines.append(f"- Metrics: {config.backtest.metrics}")

    return "\n".join(lines)


def check_guardrails(project_dir: Path) -> str:
    """Run configured guardrails and return a status report.

    Checks:
    - Feature leakage: model features vs denylist
    - Naming conventions: experiment IDs vs pattern
    - Model configuration: active models have features

    Returns markdown report with pass/fail per guardrail.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    models_data = _load_yaml(config_dir / "models.yaml")
    sources_data = _load_yaml(config_dir / "sources.yaml")

    guardrail_config = sources_data.get("guardrails", {})
    models = models_data.get("models", {})

    results = []

    # 1. Feature leakage check
    denylist = set(guardrail_config.get("feature_leakage_denylist", []))
    if denylist:
        violations = []
        for model_name, model_def in models.items():
            model_features = set(model_def.get("features", []))
            found = denylist & model_features
            if found:
                violations.append((model_name, sorted(found)))

        if violations:
            details = "; ".join(
                f"`{m}` uses {cols}" for m, cols in violations
            )
            results.append(("Feature Leakage", "FAIL", details))
        else:
            results.append(("Feature Leakage", "PASS", f"No denied columns found (denylist: {len(denylist)} entries)"))
    else:
        results.append(("Feature Leakage", "SKIP", "No denylist configured"))

    # 1b. Auto-detect leaky features (name patterns + correlation)
    try:
        from easyml.core.guardrails.inventory import detect_leaky_columns
        from easyml.core.runner.data_utils import get_feature_columns, get_features_df, load_data_config

        config = load_data_config(project_dir)
        df = get_features_df(project_dir, config)
        feature_cols = get_feature_columns(df, config)
        leaky = detect_leaky_columns(
            feature_cols, target_column=config.target_column, df=df, corr_threshold=0.90
        )
        if leaky:
            results.append(("Auto-Leakage Detection", "WARN", f"Suspicious columns: {leaky}"))
        else:
            results.append(("Auto-Leakage Detection", "PASS", "No suspicious patterns or high correlations"))
    except Exception:
        results.append(("Auto-Leakage Detection", "SKIP", "Could not load features for analysis"))

    # 2. Naming convention check
    naming_pattern = guardrail_config.get("naming_pattern")
    if naming_pattern:
        import re
        experiments_dir = project_dir / "experiments"
        if experiments_dir.exists():
            bad_names = []
            for d in experiments_dir.iterdir():
                if d.is_dir() and not re.match(naming_pattern, d.name):
                    bad_names.append(d.name)
            if bad_names:
                results.append(("Naming Convention", "FAIL", f"Invalid names: {bad_names}"))
            else:
                results.append(("Naming Convention", "PASS", f"All experiment names match `{naming_pattern}`"))
        else:
            results.append(("Naming Convention", "SKIP", "No experiments directory"))
    else:
        results.append(("Naming Convention", "SKIP", "No naming_pattern configured"))

    # 3. Model configuration check
    active_models = [n for n, m in models.items() if m.get("active", True)]
    featureless = [n for n in active_models if not models[n].get("features")]
    if featureless:
        results.append(("Model Config", "WARN", f"Models without features: {featureless}"))
    else:
        results.append(("Model Config", "PASS", f"{len(active_models)} active models, all with features"))

    # 4. Critical paths check
    critical_paths = guardrail_config.get("critical_paths", [])
    if critical_paths:
        results.append(("Critical Paths", "INFO", f"Protected: {critical_paths}"))
    else:
        results.append(("Critical Paths", "SKIP", "No critical paths configured"))

    # Format report
    n_fail = sum(1 for _, status, _ in results if status == "FAIL")
    n_pass = sum(1 for _, status, _ in results if status == "PASS")
    n_warn = sum(1 for _, status, _ in results if status == "WARN")

    lines = ["## Guardrail Report\n"]
    lines.append(f"**Summary**: {n_pass} passed, {n_fail} failed, {n_warn} warnings\n")

    lines.append("| Guardrail | Status | Details |")
    lines.append("|-----------|--------|---------|")
    for name, status, details in results:
        lines.append(f"| {name} | {status} | {details} |")

    if n_fail > 0:
        lines.append(f"\n**{n_fail} guardrail(s) failed.** Fix violations before proceeding.")

    return "\n".join(lines)


def scaffold_init(
    project_dir: Path,
    project_name: str | None = None,
    *,
    task: str = "classification",
    target_column: str = "result",
    key_columns: list[str] | None = None,
    time_column: str | None = None,
) -> str:
    """Initialize a new easyml project via scaffold.

    Returns markdown confirmation or error.
    """
    project_dir = Path(project_dir)

    try:
        from easyml.core.runner.scaffold import scaffold_project

        scaffold_project(
            project_dir,
            project_name,
            task=task,
            target_column=target_column,
            key_columns=key_columns,
            time_column=time_column,
        )
    except FileExistsError:
        return f"**Error**: Directory `{project_dir}` already exists and is not empty."
    except Exception as exc:
        return f"**Error**: Failed to initialize project: {exc}"

    name = project_name or project_dir.name
    lines = [
        f"**Initialized project**: `{name}`",
        f"- Directory: `{project_dir}`",
        f"- Task: {task}",
        f"- Target column: {target_column}",
    ]
    if key_columns:
        lines.append(f"- Key columns: {key_columns}")
    if time_column:
        lines.append(f"- Time column: {time_column}")
    lines.append(f"\nConfig files created in `{project_dir}/config/`")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Run tools
# -----------------------------------------------------------------------

def list_runs(project_dir: Path) -> str:
    """List all pipeline runs with key metrics."""
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_path = config_dir / "pipeline.yaml"
    pipeline_data = _load_yaml(pipeline_path)

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "No outputs_dir configured."

    from easyml.core.runner.run_manager import RunManager

    abs_outputs = project_dir / outputs_dir
    mgr = RunManager(abs_outputs)
    runs = mgr.list_runs()

    if not runs:
        return "No runs found."

    # Try to detect metrics from the most recent run
    metric_keys = _detect_run_metrics(runs[0]["path"])

    lines = ["## Pipeline Runs\n"]
    if metric_keys:
        header = "| Run ID | " + " | ".join(k.title() for k in metric_keys) + " | Current |"
        sep = "|--------|" + "|".join("-------" for _ in metric_keys) + "|---------|"
        lines.extend([header, sep])
        for r in runs:
            metrics = _load_run_metrics(r["path"])
            vals = " | ".join(metrics.get(k, "—") for k in metric_keys)
            current = " ✓" if r["is_current"] else ""
            lines.append(f"| {r['run_id']} | {vals} | {current} |")
    else:
        for r in runs:
            marker = " **(current)**" if r["is_current"] else ""
            lines.append(f"- `{r['run_id']}`{marker}")

    return "\n".join(lines)


def _detect_run_metrics(run_path) -> list[str]:
    """Detect which metrics exist from a run's pooled_metrics.json."""
    metrics_file = Path(run_path) / "pooled_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        return list(data.keys())
    return []


def _load_run_metrics(run_path) -> dict[str, str]:
    """Load and format metrics from a run directory."""
    metrics_file = Path(run_path) / "pooled_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        return {k: f"{v:.4f}" if isinstance(v, float) else str(v) for k, v in data.items()}
    return {}


def _format_backtest_result(result: dict, run_id: str | None = None) -> str:
    """Format backtest results as structured markdown."""
    lines = ["## Backtest Results\n"]

    if run_id:
        lines.append(f"- **Run ID**: `{run_id}`")

    status = result.get("status", "unknown")
    lines.append(f"- **Status**: {status}")

    if status != "success":
        error = result.get("error", "Unknown error")
        lines.append(f"- **Error**: {error}")
        return "\n".join(lines)

    # Ensemble metrics
    metrics = result.get("metrics", {})
    if metrics:
        lines.append("\n### Ensemble Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, val in sorted(metrics.items()):
            lines.append(f"| {name} | {val:.4f} |")

    # Per-model metrics (if available in per_fold)
    models_trained = result.get("models_trained", [])
    if models_trained:
        lines.append(f"\n### Models Trained ({len(models_trained)})\n")
        for m in models_trained:
            lines.append(f"- `{m}`")

    models_failed = result.get("models_failed", [])
    if models_failed:
        lines.append(f"\n### Models Failed ({len(models_failed)})\n")
        for m in models_failed:
            lines.append(f"- `{m}` (failed during backtest — check logs)")

    # Meta-learner coefficients
    meta_coeff = result.get("meta_coefficients")
    if meta_coeff:
        lines.append("\n### Meta-Learner Weights\n")
        lines.append("| Model | Weight |")
        lines.append("|-------|--------|")
        for name, weight in sorted(meta_coeff.items(), key=lambda x: -abs(x[1])):
            lines.append(f"| {name} | {weight:+.4f} |")

    # Regression model CDF scales
    cdf_scales = result.get("model_cdf_scales", {})
    if cdf_scales:
        lines.append("\n### Regression Model CDF Scales\n")
        lines.append("| Model | Avg CDF Scale |")
        lines.append("|-------|--------------|")
        for name, scale in sorted(cdf_scales.items()):
            lines.append(f"| {name} | {scale:.3f} |")

    # Per-fold breakdown
    per_fold = result.get("per_fold", {})
    if per_fold:
        lines.append(f"\n### Per-Fold Breakdown ({len(per_fold)} folds)\n")
        lines.append("| Fold | Brier | Accuracy |")
        lines.append("|------|-------|----------|")
        for fold_id, fold_metrics in sorted(per_fold.items()):
            brier = fold_metrics.get("brier", fold_metrics.get("brier_score", "N/A"))
            acc = fold_metrics.get("accuracy", "N/A")
            brier_str = f"{brier:.4f}" if isinstance(brier, (int, float)) else str(brier)
            acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)
            lines.append(f"| {fold_id} | {brier_str} | {acc_str} |")

    return "\n".join(lines)


def run_backtest(
    project_dir: Path,
    *,
    experiment_id: str | None = None,
    variant: str | None = None,
    on_progress=None,
) -> str:
    """Run a full backtest and return formatted results.

    Parameters
    ----------
    project_dir : Path
        Root project directory.
    experiment_id : str | None
        If provided, applies the experiment's overlay before running.
    variant : str | None
        Config variant (e.g., "w" for women's).

    Returns
    -------
    str
        Markdown-formatted backtest results.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    # Load experiment overlay if specified
    overlay = None
    if experiment_id:
        exp_dir = project_dir / "experiments" / experiment_id
        overlay_path = exp_dir / "overlay.yaml"
        if overlay_path.exists():
            overlay = _load_yaml(overlay_path)

    # Create run directory
    run_dir = None
    run_id = None
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if outputs_dir:
        from easyml.core.runner.run_manager import RunManager
        mgr = RunManager(project_dir / outputs_dir)
        run_dir = mgr.new_run()
        run_id = run_dir.name

    try:
        from easyml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
            overlay=overlay,
            run_dir=run_dir,
        )
        runner.load()
        result = runner.backtest(on_progress=on_progress)

        return _format_backtest_result(result, run_id=run_id)
    except Exception as exc:
        return f"**Backtest failed**: {exc}"


def run_predict(
    project_dir: Path,
    fold_value: int,
    *,
    run_id: str | None = None,
    variant: str | None = None,
) -> str:
    """Generate predictions for a target fold.

    Trains on all historical data before the target fold,
    then predicts every row in the target fold.

    Returns markdown-formatted prediction summary.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    try:
        from easyml.core.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
        )
        runner.load()
        preds_df = runner.predict(fold_value, run_id=run_id)

        if preds_df is None or len(preds_df) == 0:
            return f"**No predictions**: No data found for fold {fold_value}."

        prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
        lines = [f"## Predictions for Fold {fold_value}\n"]
        lines.append(f"- **Rows predicted**: {len(preds_df)}")
        lines.append(f"- **Models**: {len(prob_cols)}")

        if "prob_ensemble" in preds_df.columns:
            ens_probs = preds_df["prob_ensemble"]
            lines.append(f"\n### Ensemble Probability Distribution\n")
            lines.append(f"- Mean: {ens_probs.mean():.4f}")
            lines.append(f"- Std: {ens_probs.std():.4f}")
            lines.append(f"- Min: {ens_probs.min():.4f}")
            lines.append(f"- Max: {ens_probs.max():.4f}")

            preds_df["_confidence"] = (preds_df["prob_ensemble"] - 0.5).abs()
            top = preds_df.nlargest(10, "_confidence")

            lines.append(f"\n### Top 10 Most Confident Predictions\n")
            lines.append("| Row | Ensemble Prob | Confidence |")
            lines.append("|-----|---------------|------------|")
            for idx, row in top.iterrows():
                lines.append(
                    f"| {idx} | {row['prob_ensemble']:.4f} | {row['_confidence']:.4f} |"
                )

        return "\n".join(lines)

    except Exception as exc:
        return f"**Prediction failed**: {exc}"


def run_experiment(
    project_dir: Path,
    experiment_id: str,
    *,
    primary_metric: str = "brier",
    variant: str | None = None,
    on_progress=None,
) -> str:
    """Run a full experiment: backtest with overlay, compare to baseline.

    Steps:
    1. Load experiment overlay and run backtest
    2. Load baseline results (from last run or re-run without overlay)
    3. Compute deltas
    4. Auto-log results
    5. Return comprehensive comparison

    Returns
    -------
    str
        Markdown-formatted experiment results with comparison.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    # Load experiment overlay
    exp_dir = project_dir / "experiments" / experiment_id
    if not exp_dir.exists():
        return f"**Error**: Experiment '{experiment_id}' not found."

    overlay_path = exp_dir / "overlay.yaml"
    overlay = _load_yaml(overlay_path) if overlay_path.exists() else {}

    # Load hypothesis
    hypothesis = ""
    hyp_path = exp_dir / "hypothesis.txt"
    if hyp_path.exists():
        hypothesis = hyp_path.read_text().strip()

    try:
        from easyml.core.runner.pipeline import PipelineRunner

        # Run experiment backtest (with overlay)
        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
            overlay=overlay,
        )
        runner.load()
        exp_result = runner.backtest(on_progress=on_progress)

        # Run baseline backtest (without overlay)
        baseline_runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
        )
        baseline_runner.load()
        baseline_result = baseline_runner.backtest(on_progress=on_progress)

        # Build comparison
        exp_metrics = exp_result.get("metrics", {})
        base_metrics = baseline_result.get("metrics", {})

        lines = [f"## Experiment: `{experiment_id}`\n"]
        if hypothesis:
            lines.append(f"**Hypothesis**: {hypothesis}\n")

        # Overlay changes
        if overlay:
            lines.append("### Changes Applied\n")
            lines.append(f"```yaml\n{yaml.dump(overlay, default_flow_style=False)}```\n")

        # Delta table
        lines.append("### Results Comparison\n")
        lines.append("| Metric | Baseline | Experiment | Delta |")
        lines.append("|--------|----------|------------|-------|")

        _LOWER_IS_BETTER = {"brier", "brier_score", "ece", "log_loss"}

        overall_verdict = "neutral"
        primary_delta = 0.0

        for metric in sorted(set(list(exp_metrics.keys()) + list(base_metrics.keys()))):
            base_val = base_metrics.get(metric, float("nan"))
            exp_val = exp_metrics.get(metric, float("nan"))
            delta = exp_val - base_val

            base_str = f"{base_val:.4f}" if not isinstance(base_val, float) or base_val == base_val else "N/A"
            exp_str = f"{exp_val:.4f}" if not isinstance(exp_val, float) or exp_val == exp_val else "N/A"
            delta_str = f"{delta:+.4f}" if delta == delta else "N/A"

            lines.append(f"| {metric} | {base_str} | {exp_str} | {delta_str} |")

            if metric == primary_metric or metric == f"{primary_metric}_score":
                primary_delta = delta
                if metric in _LOWER_IS_BETTER:
                    overall_verdict = "improved" if delta < 0 else ("regressed" if delta > 0 else "neutral")
                else:
                    overall_verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "neutral")

        lines.append(f"\n**Verdict**: {overall_verdict} (primary metric: {primary_metric}, delta: {primary_delta:+.4f})")

        # Per-fold deltas
        exp_per_fold = exp_result.get("per_fold", {})
        base_per_fold = baseline_result.get("per_fold", {})
        if exp_per_fold and base_per_fold:
            common_folds = sorted(
                set(exp_per_fold.keys()) & set(base_per_fold.keys())
            )
            if common_folds:
                lines.append("\n### Per-Fold Deltas\n")
                lines.append("| Fold | Brier (base) | Brier (exp) | Delta | Acc (base) | Acc (exp) | Delta |")
                lines.append("|------|-------------|-------------|-------|------------|-----------|-------|")
                for s in common_folds:
                    eb = exp_per_fold[s].get("brier", exp_per_fold[s].get("brier_score"))
                    bb = base_per_fold[s].get("brier", base_per_fold[s].get("brier_score"))
                    ea = exp_per_fold[s].get("accuracy")
                    ba = base_per_fold[s].get("accuracy")
                    bd = (eb - bb) if eb is not None and bb is not None else None
                    ad = (ea - ba) if ea is not None and ba is not None else None
                    bb_s = f"{bb:.4f}" if bb is not None else "-"
                    eb_s = f"{eb:.4f}" if eb is not None else "-"
                    bd_s = f"{bd:+.4f}" if bd is not None else "-"
                    ba_s = f"{ba:.4f}" if ba is not None else "-"
                    ea_s = f"{ea:.4f}" if ea is not None else "-"
                    ad_s = f"{ad:+.4f}" if ad is not None else "-"
                    lines.append(f"| {s} | {bb_s} | {eb_s} | {bd_s} | {ba_s} | {ea_s} | {ad_s} |")

        # Experiment backtest details
        lines.append("\n---\n")
        lines.append(_format_backtest_result(exp_result))

        # Save results to experiment dir
        try:
            results = {
                "experiment_id": experiment_id,
                "metrics": exp_metrics,
                "baseline_metrics": base_metrics,
                "verdict": overall_verdict,
                "primary_metric": primary_metric,
                "primary_delta": primary_delta,
            }
            (exp_dir / "results.json").write_text(json.dumps(results, indent=2))

            # Auto-log
            from easyml.core.runner.experiment import auto_log_result
            auto_log_result(
                log_path=project_dir / "EXPERIMENT_LOG.md",
                experiment_id=experiment_id,
                hypothesis=hypothesis,
                changes=yaml.dump(overlay, default_flow_style=True) if overlay else "",
                metrics=exp_metrics,
                baseline_metrics=base_metrics,
                verdict=overall_verdict,
            )
        except Exception as log_exc:
            lines.append(f"\n*Warning: Failed to log results: {log_exc}*")

        return "\n".join(lines)

    except Exception as exc:
        return f"**Experiment failed**: {exc}"


def quick_run_experiment(
    project_dir: Path,
    description: str,
    overlay: str | dict,
    *,
    hypothesis: str = "",
    primary_metric: str = "brier",
    on_progress=None,
) -> str:
    """Create, configure, and run an experiment in a single call.

    Combines experiment_create + write_overlay + run_experiment.
    Returns the combined results or error at any step.
    """
    if not description:
        return "**Error**: 'description' is required for quick_run."

    project_dir = Path(project_dir)

    # Step 1: Create experiment
    create_result = experiment_create(project_dir, description, hypothesis=hypothesis)
    if "Error" in create_result:
        return create_result

    # Extract experiment ID from create result
    import re
    id_match = re.search(r'(exp-\d+)', create_result, re.IGNORECASE)
    if not id_match:
        return f"**Error**: Could not extract experiment ID from creation result.\n\n{create_result}"
    experiment_id = id_match.group(1)

    # Step 2: Write overlay
    try:
        parsed_overlay = json.loads(overlay) if isinstance(overlay, str) else overlay
    except json.JSONDecodeError as e:
        return f"**Error**: Invalid overlay JSON: {e}"

    overlay_result = write_overlay(project_dir, experiment_id, parsed_overlay)
    if "Error" in overlay_result:
        return f"**Error** writing overlay:\n\n{overlay_result}"

    # Step 3: Run experiment
    run_result = run_experiment(
        project_dir,
        experiment_id,
        primary_metric=primary_metric,
        on_progress=on_progress,
    )

    # Combine results
    lines = [
        f"## Quick Run: {experiment_id}\n",
        f"**Description:** {description}",
    ]
    if hypothesis:
        lines.append(f"**Hypothesis:** {hypothesis}")
    lines.append(f"\n### Overlay\n\n{overlay_result}")
    lines.append(f"\n### Results\n\n{run_result}")

    return "\n".join(lines)


def show_run(
    project_dir: Path,
    run_id: str | None = None,
) -> str:
    """Show results from a pipeline run.

    Parameters
    ----------
    run_id : str | None
        Run to show. If None, shows the most recent run.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured."

    outputs_path = project_dir / outputs_dir

    if run_id:
        run_dir = outputs_path / run_id
    else:
        # Find most recent run
        if not outputs_path.exists():
            return "**Error**: No runs found."
        runs = sorted(
            [d for d in outputs_path.iterdir() if d.is_dir() and d.name != "current"],
            key=lambda d: d.name,
            reverse=True,
        )
        if not runs:
            return "**Error**: No runs found."
        run_dir = runs[0]

    if not run_dir.exists():
        return f"**Error**: Run '{run_id}' not found."

    lines = [f"## Run: `{run_dir.name}`\n"]

    # Read report.md if available
    report_path = run_dir / "report.md"
    if report_path.exists():
        lines.append(report_path.read_text())
        return "\n".join(lines)

    # Read pooled_metrics.json if available
    metrics_path = run_dir / "pooled_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        lines.append("### Metrics\n")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for name, val in sorted(metrics.items()):
            if isinstance(val, (int, float)):
                lines.append(f"| {name} | {val:.4f} |")
        return "\n".join(lines)

    # List available artifacts
    artifacts = sorted(f.name for f in run_dir.rglob("*") if f.is_file())
    if artifacts:
        lines.append("### Artifacts\n")
        for a in artifacts[:20]:
            lines.append(f"- `{a}`")
        if len(artifacts) > 20:
            lines.append(f"- ... +{len(artifacts) - 20} more")
    else:
        lines.append("No artifacts found in this run.")

    return "\n".join(lines)


def show_diagnostics(
    project_dir: Path,
    run_id: str | None = None,
) -> str:
    """Show per-model diagnostics from a backtest run.

    Reads prediction artifacts from the run directory and computes
    per-model metrics (brier, accuracy, ECE, log_loss) plus model
    agreement and calibration summary.

    Returns markdown report.
    """
    import numpy as np
    import pandas as pd

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured in pipeline.yaml."

    outputs_path = project_dir / outputs_dir

    if run_id:
        run_dir = outputs_path / run_id
    else:
        if not outputs_path.exists():
            return "**Error**: No runs found."
        runs = sorted(
            [d for d in outputs_path.iterdir() if d.is_dir() and d.name != "current"],
            key=lambda d: d.name,
            reverse=True,
        )
        if not runs:
            return "**Error**: No runs found."
        run_dir = runs[0]

    if not run_dir.exists():
        return f"**Error**: Run '{run_id}' not found."

    preds_dir = run_dir / "predictions"
    preds_path = preds_dir / "predictions.parquet"
    if preds_path.exists():
        preds_df = pd.read_parquet(preds_path)
    elif preds_dir.exists():
        # Predictions may be split per-fold (e.g. 2024_probabilities.parquet)
        fold_files = sorted(preds_dir.glob("*_probabilities.parquet"))
        if fold_files:
            preds_df = pd.concat(
                [pd.read_parquet(f) for f in fold_files],
                ignore_index=True,
            )
        else:
            diag_path = run_dir / "diagnostics" / "predictions.parquet"
            if diag_path.exists():
                preds_df = pd.read_parquet(diag_path)
            else:
                return f"**Error**: No predictions found in run `{run_dir.name}`."
    else:
        diag_path = run_dir / "diagnostics" / "predictions.parquet"
        if diag_path.exists():
            preds_df = pd.read_parquet(diag_path)
        else:
            return f"**Error**: No predictions found in run `{run_dir.name}`."

    if "result" not in preds_df.columns:
        return "**Error**: Predictions file missing 'result' column for evaluation."

    y_true = preds_df["result"].values.astype(float)
    prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]

    if not prob_cols:
        return "**Error**: No prob_* columns found in predictions."

    from easyml.core.runner.diagnostics import (
        compute_brier_score,
        compute_calibration_curve,
        compute_ece,
        compute_model_agreement,
    )

    model_metrics = []
    for col in sorted(prob_cols):
        model_name = col.replace("prob_", "", 1)
        y_prob = preds_df[col].values.astype(float)

        valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
        if valid.sum() == 0:
            continue

        y_t = y_true[valid]
        y_p = y_prob[valid]

        brier = compute_brier_score(y_t, y_p)
        ece = compute_ece(y_t, y_p)
        accuracy = float(np.mean((y_p >= 0.5).astype(float) == y_t))
        eps = 1e-15
        y_clipped = np.clip(y_p, eps, 1.0 - eps)
        log_loss = float(-np.mean(
            y_t * np.log(y_clipped) + (1 - y_t) * np.log(1 - y_clipped)
        ))

        model_metrics.append({
            "model": model_name,
            "brier": brier,
            "accuracy": accuracy,
            "ece": ece,
            "log_loss": log_loss,
            "n_samples": int(valid.sum()),
        })

    lines = [f"## Diagnostics: `{run_dir.name}`\n"]
    lines.append(f"- Predictions: {len(preds_df)} rows")
    lines.append(f"- Models: {len(model_metrics)}\n")

    lines.append("### Per-Model Metrics\n")
    lines.append("| Model | Brier | Accuracy | ECE | Log Loss | N |")
    lines.append("|-------|-------|----------|-----|----------|---|")
    for m in sorted(model_metrics, key=lambda x: x["brier"]):
        lines.append(
            f"| {m['model']} | {m['brier']:.4f} | {m['accuracy']:.4f} "
            f"| {m['ece']:.4f} | {m['log_loss']:.4f} | {m['n_samples']} |"
        )

    agreement = compute_model_agreement(preds_df)
    mean_agreement = float(np.mean(agreement))
    lines.append(f"\n### Model Agreement\n")
    lines.append(f"- Mean agreement with ensemble: {mean_agreement:.4f}")

    lines.append(f"\n### Calibration Summary\n")
    for m in model_metrics:
        model_name = m["model"]
        col = f"prob_{model_name}"
        y_p = preds_df[col].values.astype(float)
        valid = ~np.isnan(y_p)
        if valid.sum() == 0:
            continue
        mean_pred = float(np.mean(y_p[valid]))
        mean_actual = float(np.mean(y_true[valid]))
        bias = mean_pred - mean_actual
        cal_status = "well-calibrated" if abs(bias) < 0.02 else ("over-confident" if bias > 0 else "under-confident")
        lines.append(f"- **{model_name}**: {cal_status} (bias: {bias:+.4f})")

    # Calibration curve for the ensemble (or best model)
    ensemble_col = "prob_ensemble" if "prob_ensemble" in preds_df.columns else prob_cols[0]
    y_ens = preds_df[ensemble_col].values.astype(float)
    valid_ens = ~np.isnan(y_ens) & ~np.isnan(y_true)
    if valid_ens.sum() > 0:
        mean_pred, mean_actual, bin_counts = compute_calibration_curve(
            y_true[valid_ens], y_ens[valid_ens], n_bins=10,
        )
        ens_name = ensemble_col.replace("prob_", "", 1)
        lines.append(f"\n### Calibration Curve (`{ens_name}`)\n")
        lines.append("| Predicted | Actual | Count |")
        lines.append("|-----------|--------|-------|")
        for p, a, c in zip(mean_pred, mean_actual, bin_counts):
            lines.append(f"| {p:.3f} | {a:.3f} | {c} |")

    # Feature importance (if we can load the config and data)
    try:
        from easyml.core.runner.data_utils import get_feature_columns, get_features_df, load_data_config
        config = load_data_config(project_dir)
        df = get_features_df(project_dir, config)
        _pi_data = _load_yaml(_get_config_dir(project_dir) / "pipeline.yaml")
        _pi_fold_col = _pi_data.get("backtest", {}).get("fold_column")
        feature_cols = get_feature_columns(df, config, fold_column=_pi_fold_col)
        if feature_cols and config.target_column in df.columns:
            from easyml.core.runner.feature_discovery import compute_feature_importance
            importance_df = compute_feature_importance(
                df, feature_columns=feature_cols, top_n=15,
            )
            if len(importance_df) > 0:
                lines.append(f"\n### Feature Importance (top {len(importance_df)})\n")
                lines.append("| Feature | Importance |")
                lines.append("|---------|------------|")
                for _, row in importance_df.iterrows():
                    lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
    except Exception as fi_exc:
        logger.debug("Could not compute feature importance: %s", fi_exc)

    # Meta-learner coefficients (from pooled_metrics.json or run artifacts)
    meta_path = run_dir / "diagnostics" / "pooled_metrics.json"
    if meta_path.exists():
        try:
            pooled = json.loads(meta_path.read_text())
            meta_coeff = pooled.get("meta_coefficients")
            if meta_coeff and isinstance(meta_coeff, dict):
                lines.append("\n### Meta-Learner Weights\n")
                lines.append("| Model | Weight |")
                lines.append("|-------|--------|")
                for name, weight in sorted(meta_coeff.items(), key=lambda x: -abs(x[1])):
                    lines.append(f"| {name} | {weight:+.4f} |")
        except Exception:
            pass

    return "\n".join(lines)


def explain_model(project_dir: Path, *, name: str | None = None, run_id: str | None = None, top_n: int = 10) -> str:
    """Run SHAP explainability on a trained model from a backtest run."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return "**Error**: `shap` package is not installed. Install with `pip install shap`."

    from easyml.core.runner.explainability import compute_shap_summary, format_shap_report
    from easyml.core.runner.data_utils import get_features_df, load_data_config, get_feature_columns

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)
    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "**Error**: No outputs_dir configured."

    # Find the run directory
    from easyml.core.runner.run_manager import RunManager
    mgr = RunManager(project_dir / outputs_dir)

    if run_id:
        run_path = project_dir / outputs_dir / run_id
    else:
        runs = mgr.list_runs()
        if not runs:
            return "**Error**: No runs found. Run a backtest first."
        run_path = Path(runs[0]["path"])

    if not run_path.exists():
        return f"**Error**: Run directory not found: {run_path}"

    # Find model artifacts
    models_dir = run_path / "models"
    if not models_dir.exists():
        return f"**Error**: No models directory in run {run_path.name}."

    # Pick a model (specific or first available)
    import pickle
    model_files = sorted(models_dir.glob("*.pkl"))
    if not model_files:
        return "**Error**: No model artifacts (.pkl) found."

    if name:
        target_file = models_dir / f"{name}.pkl"
        if not target_file.exists():
            available = [f.stem for f in model_files]
            return f"**Error**: Model `{name}` not found. Available: {', '.join(available)}"
        model_file = target_file
    else:
        model_file = model_files[0]

    with open(model_file, "rb") as f:
        model = pickle.load(f)

    # Load feature data
    config = load_data_config(project_dir)
    df = get_features_df(project_dir, config)
    fold_column = pipeline_data.get("backtest", {}).get("fold_column")
    feature_cols = get_feature_columns(df, config, fold_column=fold_column)
    X = df[feature_cols].values

    # Compute SHAP
    try:
        results = compute_shap_summary(model, X, feature_cols, top_n=top_n)
    except Exception as e:
        return f"**Error** computing SHAP values: {e}"

    return format_shap_report(results, model_name=model_file.stem)


def promote_experiment(
    project_dir: Path,
    experiment_id: str,
    *,
    primary_metric: str = "brier_score",
) -> str:
    """Promote a successful experiment's config changes to production.

    Returns markdown confirmation or rejection reason.
    """
    project_dir = Path(project_dir)

    try:
        from easyml.core.runner.experiment import promote_experiment as _promote

        result = _promote(
            experiment_id=experiment_id,
            experiments_dir=project_dir / "experiments",
            config_dir=project_dir / "config",
            primary_metric=primary_metric,
        )

        if result.get("promoted"):
            lines = [f"## Promoted: `{experiment_id}`\n"]
            improvement = result.get("improvement", {})
            for metric, val in improvement.items():
                lines.append(f"- **{metric}**: improved by {val:+.4f}")
            changes = result.get("changes", [])
            if changes:
                lines.append("\n### Changes Applied\n")
                for c in changes:
                    lines.append(f"- {c}")
            warning = result.get("warning")
            if warning:
                lines.append(f"\n**Warning**: {warning}")
            return "\n".join(lines)
        else:
            reason = result.get("reason", "Unknown reason")
            return f"**Not promoted**: {reason}"

    except Exception as exc:
        return f"**Promotion failed**: {exc}"


# -----------------------------------------------------------------------
# Exploration (Bayesian search)
# -----------------------------------------------------------------------


def run_exploration(
    project_dir: Path,
    search_space: dict,
    on_progress=None,
) -> str:
    """Run a Bayesian exploration over a search space.

    Delegates to :func:`easyml.runner.exploration.run_exploration` and
    returns the markdown report.
    """
    from easyml.core.runner.exploration import run_exploration as _run_exploration

    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    try:
        result = _run_exploration(
            project_dir=project_dir,
            search_space=search_space,
            config_dir=config_dir,
            on_progress=on_progress,
        )
        return result["report"]
    except ImportError as exc:
        return f"**Error**: {exc}"
    except Exception as exc:
        return f"**Exploration failed**: {exc}"


def promote_exploration_trial(
    project_dir: Path,
    exploration_id: str,
    *,
    trial: int | None = None,
    primary_metric: str = "brier",
    hypothesis: str = "",
) -> str:
    """Promote a specific trial (or best trial) from an exploration run.

    Reads the trial overlay, creates a new exp-NNN experiment, runs it,
    and returns the results. No re-running of the backtest from scratch —
    the trial's overlay is simply applied as a new experiment.

    Parameters
    ----------
    exploration_id : str
        e.g. 'expl-002'
    trial : int | None
        Trial number to promote. If None, uses best_overlay.yaml (the
        best trial according to the exploration's primary metric).
    """
    project_dir = Path(project_dir)
    expl_dir = project_dir / "experiments" / exploration_id

    if not expl_dir.exists():
        return f"**Error**: Exploration '{exploration_id}' not found."

    if trial is not None:
        overlay_path = expl_dir / "trials" / f"trial-{trial:03d}" / "overlay.yaml"
        trial_label = f"trial {trial}"
    else:
        overlay_path = expl_dir / "best_overlay.yaml"
        trial_label = "best trial"

    if not overlay_path.exists():
        return f"**Error**: Overlay not found for {trial_label} in {exploration_id} ({overlay_path})."

    overlay = _load_yaml(overlay_path)
    if not overlay:
        return f"**Error**: Empty overlay for {trial_label} in {exploration_id}."

    description = f"Promote {trial_label} from {exploration_id}"
    if not hypothesis:
        hypothesis = f"Applying overlay from {exploration_id} {trial_label}."

    create_result = experiment_create(project_dir, description, hypothesis=hypothesis)
    if "Error" in create_result:
        return create_result

    import re
    id_match = re.search(r"(exp-\d+)", create_result, re.IGNORECASE)
    if not id_match:
        return f"**Error**: Could not extract experiment ID.\n\n{create_result}"
    experiment_id = id_match.group(1)

    overlay_result = write_overlay(project_dir, experiment_id, overlay)
    if "Error" in overlay_result:
        return overlay_result

    run_result = run_experiment(project_dir, experiment_id, primary_metric=primary_metric)

    return "\n".join([
        f"## Promote: `{exploration_id}` {trial_label} → `{experiment_id}`\n",
        f"### Overlay\n\n{overlay_result}",
        f"\n### Results\n\n{run_result}",
    ])


# -----------------------------------------------------------------------
# Experiment journal
# -----------------------------------------------------------------------


def log_experiment_result(
    project_dir: Path,
    experiment_id: str,
    *,
    description: str = "",
    hypothesis: str = "",
    metrics: dict | None = None,
    overlay: dict | None = None,
    verdict: str = "",
) -> str:
    """Append an experiment result to the journal (JSONL format)."""
    from datetime import datetime

    project_dir = Path(project_dir)
    journal_path = project_dir / "experiments" / "journal.jsonl"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_id": experiment_id,
        "description": description,
        "hypothesis": hypothesis,
        "metrics": metrics or {},
        "verdict": verdict,
    }
    if overlay:
        entry["overlay_summary"] = str(overlay)[:200]

    with open(journal_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return f"Logged experiment `{experiment_id}` to journal."


def show_journal(project_dir: Path, *, last_n: int = 20) -> str:
    """Show the experiment journal as a markdown table."""
    project_dir = Path(project_dir)
    journal_path = project_dir / "experiments" / "journal.jsonl"

    if not journal_path.exists():
        return "No experiment journal found. Run experiments to build history."

    entries = []
    for line in journal_path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))

    if not entries:
        return "Journal is empty."

    entries = entries[-last_n:]

    # Detect all metric keys across entries
    metric_keys = []
    for e in entries:
        for k in e.get("metrics", {}):
            if k not in metric_keys:
                metric_keys.append(k)

    lines = ["## Experiment Journal\n"]
    header = "| # | ID | Description | " + " | ".join(metric_keys) + " | Verdict |"
    sep = "|---|----|-----------| " + " | ".join("------" for _ in metric_keys) + " |---------|"
    lines.extend([header, sep])

    for i, e in enumerate(entries, 1):
        metrics = e.get("metrics", {})
        vals = " | ".join(
            f"{metrics.get(k, '—'):.4f}" if isinstance(metrics.get(k), (int, float)) else str(metrics.get(k, "—"))
            for k in metric_keys
        )
        verdict = e.get("verdict", "")
        desc = e.get("description", "")[:40]
        lines.append(f"| {i} | {e['experiment_id']} | {desc} | {vals} | {verdict} |")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# View / source management tools
# -----------------------------------------------------------------------


def add_source(
    project_dir: Path,
    name: str,
    path: str,
    format: str = "auto",
) -> str:
    """Register a raw data source in pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}
    if "sources" not in data["data"]:
        data["data"]["sources"] = {}

    if name in data["data"]["sources"]:
        return f"**Error**: Source `{name}` already exists."

    # Verify file exists
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = Path(project_dir) / path
    if not source_path.exists():
        return f"**Error**: File not found: {source_path}"

    # Read to get row/column info
    import pandas as pd
    if path.endswith(".csv"):
        df = pd.read_csv(source_path, nrows=5)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(source_path).head(5)
    else:
        df = pd.read_csv(source_path, nrows=5)  # try csv

    data["data"]["sources"][name] = {
        "name": name,
        "path": path,
        "format": format,
    }
    _save_yaml(pipeline_path, data)

    return (
        f"**Added source**: `{name}`\n"
        f"- Path: {path}\n"
        f"- Columns ({len(df.columns)}): {', '.join(df.columns[:15])}"
        f"{'...' if len(df.columns) > 15 else ''}\n"
        f"- Format: {format}"
    )


def add_view(
    project_dir: Path,
    name: str,
    source: str,
    steps: list[dict],
    description: str = "",
) -> str:
    """Declare a view in pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}
    if "views" not in data["data"]:
        data["data"]["views"] = {}

    if name in data["data"]["views"]:
        return f"**Error**: View `{name}` already exists. Remove it first or use a different name."

    # Validate source exists
    sources = data["data"].get("sources", {})
    views = data["data"].get("views", {})
    all_names = set(sources.keys()) | set(views.keys())
    if source not in all_names:
        return (
            f"**Error**: Source `{source}` not found. "
            f"Available: {sorted(all_names)}"
        )

    # Validate steps parse as TransformStep
    from pydantic import TypeAdapter, ValidationError
    from easyml.core.runner.schema import TransformStep
    adapter = TypeAdapter(list[TransformStep])
    try:
        adapter.validate_python(steps)
    except ValidationError as e:
        return f"**Error**: Invalid steps:\n```\n{e}\n```"

    view_def = {"source": source, "steps": steps}
    if description:
        view_def["description"] = description

    data["data"]["views"][name] = view_def
    _save_yaml(pipeline_path, data)

    confirmation = (
        f"**Added view**: `{name}`\n"
        f"- Source: {source}\n"
        f"- Steps: {len(steps)}\n"
        f"- Description: {description or '(none)'}"
    )

    try:
        preview = preview_view(project_dir, name, n_rows=3)
        return confirmation + "\n\n" + preview
    except Exception:
        return confirmation


def remove_view(project_dir: Path, name: str) -> str:
    """Remove a view from pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    if name not in views:
        return f"**Error**: View `{name}` not found."

    del data["data"]["views"][name]

    # Clear features_view if it pointed to this view
    if data.get("data", {}).get("features_view") == name:
        data["data"]["features_view"] = None

    _save_yaml(pipeline_path, data)
    _invalidate_view_cache(project_dir, name)
    return f"**Removed view**: `{name}`"


def update_view(
    project_dir: Path,
    name: str,
    source: str | None = None,
    steps: list[dict] | None = None,
    description: str | None = None,
) -> str:
    """Update an existing view in pipeline.yaml.

    Only provided fields are merged — None values keep the existing value.
    Returns updated confirmation with an inline preview.
    """
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    if name not in views:
        return f"**Error**: View `{name}` not found. Available: {sorted(views.keys())}"

    view_def = views[name]

    # Merge provided fields
    if source is not None:
        # Validate source exists
        sources = data["data"].get("sources", {})
        all_names = set(sources.keys()) | set(views.keys()) - {name}
        if source not in all_names:
            return (
                f"**Error**: Source `{source}` not found. "
                f"Available: {sorted(all_names)}"
            )
        view_def["source"] = source

    if steps is not None:
        from pydantic import TypeAdapter, ValidationError
        from easyml.core.runner.schema import TransformStep
        adapter = TypeAdapter(list[TransformStep])
        try:
            adapter.validate_python(steps)
        except ValidationError as e:
            return f"**Error**: Invalid steps:\n```\n{e}\n```"
        view_def["steps"] = steps

    if description is not None:
        if description:
            view_def["description"] = description
        else:
            view_def.pop("description", None)

    data["data"]["views"][name] = view_def
    _save_yaml(pipeline_path, data)
    _invalidate_view_cache(project_dir, name)

    confirmation = (
        f"**Updated view**: `{name}`\n"
        f"- Source: {view_def.get('source', '?')}\n"
        f"- Steps: {len(view_def.get('steps', []))}\n"
        f"- Description: {view_def.get('description', '(none)')}"
    )

    try:
        preview = preview_view(project_dir, name, n_rows=3)
        return confirmation + "\n\n" + preview
    except Exception:
        return confirmation


def list_views(project_dir: Path) -> str:
    """List all views with descriptions and dependency info."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    features_view = data.get("data", {}).get("features_view")

    if not views:
        return "No views defined. Use `add_view` to create one."

    lines = [f"## Views ({len(views)})\n"]
    for name, view_def in views.items():
        marker = " **[prediction table]**" if name == features_view else ""
        source = view_def.get("source", "?")
        steps = view_def.get("steps", [])
        desc = view_def.get("description", "")
        lines.append(f"### `{name}`{marker}")
        lines.append(f"- Source: `{source}`")
        lines.append(f"- Steps: {len(steps)}")
        if desc:
            lines.append(f"- Description: {desc}")

        # Show step summary
        for i, step in enumerate(steps):
            op = step.get("op", "?")
            lines.append(f"  {i+1}. `{op}`")
        lines.append("")

    return "\n".join(lines)


def preview_view(project_dir: Path, name: str, n_rows: int = 5) -> str:
    """Materialize a view and show schema + first N rows."""
    from easyml.core.runner.data_utils import load_data_config

    config = load_data_config(Path(project_dir))

    all_names = set(config.sources.keys()) | set(config.views.keys())
    if name not in all_names:
        return f"**Error**: `{name}` not found. Available: {sorted(all_names)}"

    try:
        from easyml.core.runner.view_resolver import ViewResolver
        resolver = ViewResolver(project_dir, config)
        df = resolver.resolve(name)
    except Exception as e:
        return f"**Error resolving view `{name}`**: {e}"

    lines = [f"## Preview: `{name}`\n"]
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append("")

    # Schema
    lines.append("### Schema\n")
    lines.append("| Column | Type | Non-null | Sample |")
    lines.append("|--------|------|----------|--------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        sample = str(df[col].iloc[0]) if len(df) > 0 else ""
        if len(sample) > 30:
            sample = sample[:27] + "..."
        lines.append(f"| {col} | {dtype} | {non_null:,} | {sample} |")
    lines.append("")

    # Sample rows
    lines.append(f"### First {min(n_rows, len(df))} rows\n")
    lines.append(df.head(n_rows).to_markdown(index=False))

    return "\n".join(lines)


def set_features_view(project_dir: Path, name: str) -> str:
    """Set which view becomes the prediction table."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    if name not in views:
        return f"**Error**: View `{name}` not found. Available: {sorted(views.keys())}"

    if "data" not in data:
        data["data"] = {}
    data["data"]["features_view"] = name
    _save_yaml(pipeline_path, data)

    return f"**Set features_view**: `{name}` is now the prediction table."


def view_dag(project_dir: Path) -> str:
    """Show the full view dependency graph."""
    from easyml.core.runner.data_utils import load_data_config

    config = load_data_config(Path(project_dir))

    if not config.views:
        return "No views defined."

    lines = ["## View Dependency Graph\n"]
    lines.append("```")

    for name, view_def in config.views.items():
        deps = [view_def.source]
        for step in view_def.steps:
            step_dict = step if isinstance(step, dict) else step.model_dump()
            if "other" in step_dict:
                deps.append(step_dict["other"])

        marker = " [prediction table]" if name == config.features_view else ""
        lines.append(f"{name}{marker} <- {', '.join(deps)}")

    lines.append("```")

    # Show sources (leaf nodes)
    lines.append("\n### Sources (raw data)\n")
    for name, source in config.sources.items():
        path = source.path or "(no path)"
        lines.append(f"- `{name}`: {path}")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Source management (registry, freshness, validation, adapters)
# -----------------------------------------------------------------------

def _get_source_registry(project_dir: Path):
    """Lazily import and return a SourceRegistry for the project."""
    from easyml.core.runner.sources.registry import SourceRegistry

    config_dir = _get_config_dir(Path(project_dir))
    return SourceRegistry(config_dir)


def _get_freshness_tracker(project_dir: Path):
    """Lazily import and return a FreshnessTracker for the project."""
    from easyml.core.runner.sources.freshness import FreshnessTracker

    config_dir = _get_config_dir(Path(project_dir))
    state_file = config_dir / "sources_state.json"
    return FreshnessTracker(state_file)


def check_freshness(project_dir: Path) -> str:
    """Check freshness of all registered sources.

    Returns a markdown table of stale sources, or a confirmation that
    everything is fresh.
    """
    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)
    tracker = _get_freshness_tracker(project_dir)

    sources = registry.list_all()
    if not sources:
        return "**No sources registered.** Use `manage_data(action='add_source')` to register sources first."

    stale = tracker.check_all(sources)
    if not stale:
        return f"All **{len(sources)}** source(s) are fresh."

    lines = [f"## Stale Sources ({len(stale)} of {len(sources)})\n"]
    lines.append("| Source | Frequency | Last Fetched |")
    lines.append("|--------|-----------|--------------|")
    for s in stale:
        lines.append(f"| {s['name']} | {s['refresh_frequency']} | {s['last_fetched']} |")
    lines.append(
        "\nUse `manage_data(action='refresh', name='<source>')` to update a stale source."
    )
    return "\n".join(lines)


def refresh_source(project_dir: Path, name: str) -> str:
    """Fetch a single source using its adapter, validate, and update freshness.

    Returns a markdown summary of what was loaded and any validation issues.
    """
    from easyml.core.runner.sources.adapters import ADAPTERS
    from easyml.core.runner.sources.validation import validate_source

    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)
    tracker = _get_freshness_tracker(project_dir)

    src = registry.get(name)
    if src is None:
        return f"**Error**: Source '{name}' not found in registry."

    adapter_cls = ADAPTERS.get(src.source_type)
    if adapter_cls is None:
        return f"**Error**: No adapter for source type '{src.source_type}'."

    try:
        if src.source_type == "file":
            df = adapter_cls.load(src.path_pattern)
        elif src.source_type == "url":
            fmt = src.schema.get("format", "csv")
            auth_headers = src.auth.get("headers") if src.auth else None
            df = adapter_cls.load(src.path_pattern, format=fmt, auth_headers=auth_headers)
        elif src.source_type == "api":
            auth_headers = src.auth.get("headers") if src.auth else None
            pagination = src.schema.get("pagination")
            df = adapter_cls.load(
                src.path_pattern,
                rate_limit=src.rate_limit,
                auth_headers=auth_headers,
                pagination=pagination,
            )
        elif src.source_type == "computed":
            return "**Error**: Computed sources must be refreshed programmatically."
        else:
            return f"**Error**: Unknown source type '{src.source_type}'."
    except Exception as e:
        return f"**Error** refreshing '{name}': {e}"

    # Validate
    violations = validate_source(src, df)
    tracker.record_fetch(name, row_count=len(df))

    lines = [
        f"Refreshed **{name}**",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
    ]
    if violations:
        lines.append(f"\n### Validation Issues ({len(violations)})")
        for v in violations:
            lines.append(f"- [{v.severity}] {v.column}: {v.message}")
    else:
        lines.append("- Validation: passed")

    return "\n".join(lines)


def refresh_all_sources(project_dir: Path) -> str:
    """Fetch all stale sources in topological order.

    Returns a markdown summary with per-source results.
    """
    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)
    tracker = _get_freshness_tracker(project_dir)

    sources = registry.list_all()
    if not sources:
        return "**No sources registered.**"

    stale_set = {s["name"] for s in tracker.check_all(sources)}
    if not stale_set:
        return f"All **{len(sources)}** source(s) are fresh. Nothing to refresh."

    # Refresh in dependency order, but only stale ones
    order = registry.topological_order()
    results = []
    for name in order:
        if name not in stale_set:
            continue
        result = refresh_source(project_dir, name)
        results.append(f"### {name}\n{result}")

    return f"## Refresh Summary ({len(results)} source(s))\n\n" + "\n\n".join(results)


def validate_source_data(project_dir: Path, name: str) -> str:
    """Load a source and validate against its schema definition.

    Returns a markdown report of validation results.
    """
    from easyml.core.runner.sources.adapters import ADAPTERS
    from easyml.core.runner.sources.validation import validate_source

    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)

    src = registry.get(name)
    if src is None:
        return f"**Error**: Source '{name}' not found in registry."

    if src.source_type != "file":
        return f"**Error**: Validation preview only supported for file sources (got '{src.source_type}')."

    adapter_cls = ADAPTERS.get(src.source_type)
    if adapter_cls is None:
        return f"**Error**: No adapter for source type '{src.source_type}'."

    try:
        df = adapter_cls.load(src.path_pattern)
    except Exception as e:
        return f"**Error** loading '{name}': {e}"

    violations = validate_source(src, df)

    lines = [
        f"## Validation: {name}",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
    ]
    if not violations:
        lines.append("- Result: **all checks passed**")
    else:
        lines.append(f"\n### Issues ({len(violations)})")
        for v in violations:
            lines.append(f"- [{v.severity}] {v.column}: {v.message}")

    return "\n".join(lines)


def format_target_comparison(results: dict[str, dict]) -> str:
    """Format multi-target comparison as markdown table."""
    if not results:
        return "No results to compare."

    # Collect all metric keys across targets
    metric_keys: list[str] = []
    for metrics in results.values():
        for k in metrics:
            if k not in metric_keys:
                metric_keys.append(k)

    lines = ["## Target Comparison\n"]
    header = "| Target | " + " | ".join(metric_keys) + " |"
    sep = "|--------|" + "|".join("------" for _ in metric_keys) + "|"
    lines.extend([header, sep])

    for target_name, metrics in results.items():
        vals = " | ".join(
            f"{metrics.get(k, '—'):.4f}" if isinstance(metrics.get(k), (int, float)) else str(metrics.get(k, "—"))
            for k in metric_keys
        )
        lines.append(f"| {target_name} | {vals} |")

    # Highlight best per metric
    lines.append("\n**Best per metric:**")
    for k in metric_keys:
        vals = {name: m.get(k) for name, m in results.items() if k in m and isinstance(m.get(k), (int, float))}
        if not vals:
            continue
        lower_better = k in ("brier", "ece", "log_loss", "rmse", "mae")
        best_name = min(vals, key=vals.get) if lower_better else max(vals, key=vals.get)
        lines.append(f"- {k}: **{best_name}** ({vals[best_name]:.4f})")

    return "\n".join(lines)
