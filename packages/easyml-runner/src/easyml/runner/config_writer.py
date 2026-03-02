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
    auto_clean: bool = True,
) -> str:
    """Add a new dataset by merging into the features parquet."""
    from easyml.runner.data_ingest import ingest_dataset

    result = ingest_dataset(
        project_dir=Path(project_dir),
        data_path=data_path,
        join_on=join_on,
        prefix=prefix,
        features_dir=features_dir,
        auto_clean=auto_clean,
    )
    return result.format_summary()


def profile_data(project_dir: Path, category: str | None = None) -> str:
    """Profile the features dataset."""
    from easyml.runner.data_utils import get_features_path, load_data_config
    from easyml.runner.schema import DataConfig

    project_dir = Path(project_dir)
    config = None
    try:
        config = load_data_config(project_dir)
        parquet_path = get_features_path(project_dir, config)
    except Exception:
        parquet_path = get_features_path(project_dir, DataConfig())

    if not parquet_path.exists():
        return f"**Error**: Data file not found: {parquet_path}"

    from easyml.runner.data_profiler import profile_dataset

    profile = profile_dataset(parquet_path, config=config)

    if category:
        return profile.format_columns(category=category)
    return profile.format_summary()


def available_features(project_dir: Path, prefix: str | None = None) -> str:
    """List available feature columns from the dataset."""
    from easyml.runner.data_utils import get_features_path, load_data_config
    from easyml.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
        parquet_path = get_features_path(project_dir, config)
    except Exception:
        parquet_path = get_features_path(project_dir, DataConfig())

    import pandas as pd

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


def generate_pairwise_features(
    project_dir: Path,
    entity_col: str,
    group_col: str,
    feature_cols: list[str],
    *,
    filter_path: str | None = None,
) -> str:
    """Generate pairwise diff features from entity-level data.

    For each pair of entities within each group, computes diff features
    (entity_A value - entity_B value). Writes result back to the
    feature store, replacing the entity-level data.

    Parameters
    ----------
    entity_col : str
        Column identifying entities (e.g. product_id, candidate_id, team_id).
    group_col : str
        Column for grouping (e.g. quarter, cohort, season). Pairs are
        generated within each group independently.
    feature_cols : list[str]
        Columns to compute diffs for.
    filter_path : str | None
        Optional CSV/parquet with entity_col and group_col columns.
        If provided, only entities present in this file are included.

    Returns markdown summary.
    """
    from easyml.runner.data_utils import get_features_path, load_data_config

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
        parquet_path = get_features_path(project_dir, config)
    except Exception:
        parquet_path = project_dir / "data" / "features" / "features.parquet"

    import pandas as pd
    import numpy as np
    from itertools import combinations

    if not parquet_path.exists():
        return f"**Error**: Features file not found at {parquet_path}"

    entity_df = pd.read_parquet(parquet_path)

    if entity_col not in entity_df.columns:
        return f"**Error**: Entity column '{entity_col}' not found in features."
    if group_col not in entity_df.columns:
        return f"**Error**: Group column '{group_col}' not found in features."

    missing = [f for f in feature_cols if f not in entity_df.columns]
    if missing:
        return f"**Error**: Feature columns not found: {missing}"

    # Load filter if provided
    filter_df = None
    if filter_path:
        filter_path_obj = Path(filter_path)
        if filter_path_obj.suffix == ".csv":
            filter_df = pd.read_csv(filter_path_obj)
        else:
            filter_df = pd.read_parquet(filter_path_obj)

    groups = sorted(entity_df[group_col].unique())
    all_rows = []

    for group_val in groups:
        group_data = entity_df[entity_df[group_col] == group_val].copy()

        # Apply filter if provided
        if filter_df is not None:
            group_filter = filter_df[filter_df[group_col] == group_val]
            qualified_ids = set(group_filter[entity_col].unique())
            group_data = group_data[group_data[entity_col].isin(qualified_ids)]

        if len(group_data) < 2:
            continue

        indexed = group_data.set_index(entity_col)
        entity_ids = sorted(indexed.index.unique())

        for id_a, id_b in combinations(entity_ids, 2):
            row_a = indexed.loc[id_a]
            row_b = indexed.loc[id_b]

            # Handle duplicate entity IDs in same group
            if isinstance(row_a, pd.DataFrame):
                row_a = row_a.iloc[0]
            if isinstance(row_b, pd.DataFrame):
                row_b = row_b.iloc[0]

            pair = {
                "entity_a": id_a,
                "entity_b": id_b,
                group_col: group_val,
            }

            for feat in feature_cols:
                val_a = row_a[feat] if feat in row_a.index else np.nan
                val_b = row_b[feat] if feat in row_b.index else np.nan
                try:
                    pair[f"diff_{feat}"] = float(val_a) - float(val_b)
                except (TypeError, ValueError):
                    pair[f"diff_{feat}"] = 0.0

            all_rows.append(pair)

    if not all_rows:
        return "**Error**: No pairs could be generated. Check entity/group data."

    result_df = pd.DataFrame(all_rows)

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(parquet_path, index=False)

    n_groups = result_df[group_col].nunique()
    n_diff_cols = len([c for c in result_df.columns if c.startswith("diff_")])

    return (
        f"**Generated pairwise features**\n"
        f"- Pairs: {len(result_df)}\n"
        f"- Groups: {n_groups}\n"
        f"- Diff features: {n_diff_cols}\n"
        f"- Columns: {sorted(result_df.columns.tolist())}"
    )


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
    from easyml.runner.data_utils import get_feature_columns, get_features_path, load_data_config
    from easyml.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
        parquet_path = get_features_path(project_dir, config)
    except Exception:
        parquet_path = get_features_path(project_dir, DataConfig())
        config = None

    import pandas as pd

    if not parquet_path.exists():
        return f"**Error**: Data file not found: {parquet_path}"

    df = pd.read_parquet(parquet_path)

    # Get feature columns from config if available
    feature_cols = None
    if config is not None:
        feature_cols = get_feature_columns(df, config)

    from easyml.runner.feature_discovery import (
        compute_feature_correlations,
        compute_feature_importance,
        detect_redundant_features,
        format_discovery_report,
        suggest_feature_groups,
    )

    correlations = compute_feature_correlations(df, top_n=top_n, feature_columns=feature_cols)
    importance = compute_feature_importance(df, method=method, top_n=top_n, feature_columns=feature_cols)
    redundant = detect_redundant_features(df, feature_columns=feature_cols)
    groups = suggest_feature_groups(df, feature_columns=feature_cols)

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
        from easyml.runner.scaffold import scaffold_project

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

    # Meta-learner coefficients
    meta_coeff = result.get("meta_coefficients")
    if meta_coeff:
        lines.append("\n### Meta-Learner Weights\n")
        lines.append("| Model | Weight |")
        lines.append("|-------|--------|")
        for name, weight in sorted(meta_coeff.items(), key=lambda x: -abs(x[1])):
            lines.append(f"| {name} | {weight:+.4f} |")

    # Per-season breakdown
    per_fold = result.get("per_fold", {})
    if per_fold:
        lines.append(f"\n### Per-Season Breakdown ({len(per_fold)} folds)\n")
        lines.append("| Season | Brier | Accuracy |")
        lines.append("|--------|-------|----------|")
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
        from easyml.runner.run_manager import RunManager
        mgr = RunManager(project_dir / outputs_dir)
        run_dir = mgr.new_run()
        run_id = run_dir.name

    try:
        from easyml.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
            overlay=overlay,
            run_dir=run_dir,
        )
        runner.load()
        result = runner.backtest()

        return _format_backtest_result(result, run_id=run_id)
    except Exception as exc:
        return f"**Backtest failed**: {exc}"


def run_predict(
    project_dir: Path,
    season: int,
    *,
    run_id: str | None = None,
    variant: str | None = None,
) -> str:
    """Generate predictions for a target season.

    Trains on all historical data before the target season,
    then predicts every row in the target season.

    Returns markdown-formatted prediction summary.
    """
    project_dir = Path(project_dir)
    config_dir = _get_config_dir(project_dir)

    try:
        from easyml.runner.pipeline import PipelineRunner

        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
        )
        runner.load()
        preds_df = runner.predict(season, run_id=run_id)

        if preds_df is None or len(preds_df) == 0:
            return f"**No predictions**: No data found for season {season}."

        prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
        lines = [f"## Predictions for Season {season}\n"]
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
        from easyml.runner.pipeline import PipelineRunner

        # Run experiment backtest (with overlay)
        runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
            overlay=overlay,
        )
        runner.load()
        exp_result = runner.backtest()

        # Run baseline backtest (without overlay)
        baseline_runner = PipelineRunner(
            project_dir=project_dir,
            config_dir=config_dir,
            variant=variant,
        )
        baseline_runner.load()
        baseline_result = baseline_runner.backtest()

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
            from easyml.runner.experiment import auto_log_result
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
        from easyml.runner.experiment import promote_experiment as _promote

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
