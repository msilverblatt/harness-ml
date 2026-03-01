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
        from easyml.runner.presets import apply_preset
        model_def = apply_preset(preset, overrides=params or {})
    elif model_type:
        model_def["type"] = model_type
        if params:
            model_def["params"] = params
    else:
        return "**Error**: Either `model_type` or `preset` must be specified."

    if features:
        model_def["features"] = features
    model_def["active"] = active
    model_def["include_in_ensemble"] = include_in_ensemble

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


def remove_model(project_dir: Path, name: str) -> str:
    """Remove a model from models.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        return f"**Error**: Model `{name}` not found."

    del models[name]
    _save_yaml(models_path, data)
    return f"**Removed model**: `{name}`"


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


def show_presets() -> str:
    """List available model presets."""
    from easyml.runner.presets import get_preset, list_presets

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
    **kwargs,
) -> str:
    """Update ensemble.yaml configuration."""
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

    for key, val in kwargs.items():
        ens[key] = val

    _save_yaml(ensemble_path, data)

    return (
        f"**Updated ensemble config**\n"
        f"- Method: {ens.get('method', 'average')}\n"
        f"- Temperature: {ens.get('temperature', 1.0)}"
    )


# -----------------------------------------------------------------------
# Backtest tools
# -----------------------------------------------------------------------

def configure_backtest(
    project_dir: Path,
    *,
    cv_strategy: str | None = None,
    seasons: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
) -> str:
    """Update backtest section of pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "backtest" not in data:
        data["backtest"] = {}

    bt = data["backtest"]

    if cv_strategy is not None:
        bt["cv_strategy"] = cv_strategy
    if seasons is not None:
        bt["seasons"] = seasons
    if metrics is not None:
        bt["metrics"] = metrics
    if min_train_folds is not None:
        bt["min_train_folds"] = min_train_folds

    _save_yaml(pipeline_path, data)

    return (
        f"**Updated backtest config**\n"
        f"- CV strategy: {bt.get('cv_strategy', 'N/A')}\n"
        f"- Seasons: {bt.get('seasons', [])}\n"
        f"- Metrics: {bt.get('metrics', [])}"
    )


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
) -> str:
    """Add a new dataset by merging into matchup_features.parquet."""
    from easyml.runner.data_ingest import ingest_dataset

    result = ingest_dataset(
        project_dir=Path(project_dir),
        data_path=data_path,
        join_on=join_on,
        prefix=prefix,
        features_dir=features_dir,
    )
    return result.format_summary()


def profile_data(project_dir: Path, category: str | None = None) -> str:
    """Profile the matchup features dataset."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    pipeline_data = _load_yaml(pipeline_path)

    features_dir = pipeline_data.get("data", {}).get("features_dir")
    if not features_dir:
        return "**Error**: No features_dir configured in pipeline.yaml."

    parquet_path = Path(features_dir) / "matchup_features.parquet"
    if not parquet_path.exists():
        return f"**Error**: Data file not found: {parquet_path}"

    from easyml.runner.data_profiler import profile_dataset

    profile = profile_dataset(parquet_path)

    if category:
        return profile.format_columns(category=category)
    return profile.format_summary()


def available_features(project_dir: Path, prefix: str | None = None) -> str:
    """List available feature columns from the dataset."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    pipeline_data = _load_yaml(pipeline_path)

    features_dir = pipeline_data.get("data", {}).get("features_dir")
    if not features_dir:
        return "**Error**: No features_dir configured."

    import pandas as pd

    parquet_path = Path(features_dir) / "matchup_features.parquet"
    if not parquet_path.exists():
        return f"**Error**: Data file not found: {parquet_path}"

    df = pd.read_parquet(parquet_path)
    cols = sorted(df.columns)
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]

    if not cols:
        return "No features found."

    lines = [f"## Available Features ({len(cols)} columns)\n"]
    for col in cols:
        lines.append(f"- `{col}`")
    return "\n".join(lines)


# -----------------------------------------------------------------------
# Feature tools
# -----------------------------------------------------------------------

def add_feature(
    project_dir: Path,
    name: str,
    formula: str,
    *,
    description: str = "",
) -> str:
    """Create a new feature from a formula expression."""
    from easyml.runner.feature_engine import create_feature

    result = create_feature(
        project_dir=Path(project_dir),
        name=name,
        formula=formula,
        description=description,
    )
    return result.format_summary()


def add_features_batch(
    project_dir: Path,
    features: list[dict],
) -> str:
    """Create multiple features from formula expressions."""
    from easyml.runner.feature_engine import create_features_batch

    results = create_features_batch(Path(project_dir), features)
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
    from easyml.runner.transformation_tester import run_transformation_tests

    report = run_transformation_tests(
        project_dir=Path(project_dir),
        features=features,
        test_interactions=test_interactions,
    )
    return report.format_summary()


def discover_features(
    project_dir: Path,
    *,
    top_n: int = 20,
    method: str = "xgboost",
) -> str:
    """Run feature discovery analysis."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    pipeline_data = _load_yaml(pipeline_path)

    features_dir = pipeline_data.get("data", {}).get("features_dir")
    if not features_dir:
        return "**Error**: No features_dir configured."

    import pandas as pd

    parquet_path = Path(features_dir) / "matchup_features.parquet"
    if not parquet_path.exists():
        return f"**Error**: Data file not found: {parquet_path}"

    df = pd.read_parquet(parquet_path)

    from easyml.runner.feature_discovery import (
        compute_feature_correlations,
        compute_feature_importance,
        detect_redundant_features,
        format_discovery_report,
        suggest_feature_groups,
    )

    correlations = compute_feature_correlations(df, top_n=top_n)
    importance = compute_feature_importance(df, method=method, top_n=top_n)
    redundant = detect_redundant_features(df)
    groups = suggest_feature_groups(df)

    return format_discovery_report(correlations, importance, redundant, groups)


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

    from easyml.runner.experiment import auto_next_id

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


def write_overlay(
    project_dir: Path,
    experiment_id: str,
    overlay: dict,
) -> str:
    """Write an overlay YAML to an experiment directory."""
    experiments_dir = Path(project_dir) / "experiments"
    exp_dir = experiments_dir / experiment_id

    if not exp_dir.exists():
        return f"**Error**: Experiment directory not found: {exp_dir}"

    overlay_path = exp_dir / "overlay.yaml"
    overlay_path.write_text(
        yaml.dump(overlay, default_flow_style=False, sort_keys=False)
    )

    return (
        f"**Overlay written**: `{overlay_path}`\n"
        f"- Keys: {list(overlay.keys())}"
    )


def show_config(project_dir: Path) -> str:
    """Show the resolved project configuration."""
    config_dir = _get_config_dir(Path(project_dir))
    from easyml.runner.validator import validate_project

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
    lines.append(f"- Seasons: {config.backtest.seasons}")
    lines.append(f"- Metrics: {config.backtest.metrics}")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Run tools
# -----------------------------------------------------------------------

def list_runs(project_dir: Path) -> str:
    """List all pipeline runs."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    pipeline_data = _load_yaml(pipeline_path)

    outputs_dir = pipeline_data.get("data", {}).get("outputs_dir")
    if not outputs_dir:
        return "No outputs_dir configured."

    from easyml.runner.run_manager import RunManager

    mgr = RunManager(Path(outputs_dir))
    runs = mgr.list_runs()

    if not runs:
        return "No runs found."

    lines = ["## Pipeline Runs\n"]
    for r in runs:
        marker = " **(current)**" if r["is_current"] else ""
        lines.append(f"- `{r['run_id']}`{marker}")
    return "\n".join(lines)
