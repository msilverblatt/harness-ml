"""EasyML MCP server — exposes config_writer functions as MCP tools.

Each tool maps 1:1 to an easyml.runner.config_writer function.
All tools accept project_dir (defaults to cwd) and return markdown.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("easyml")


def _resolve_project_dir(project_dir: str | None) -> Path:
    """Resolve project directory from param or cwd."""
    if project_dir:
        p = Path(project_dir).resolve()
    else:
        p = Path.cwd()

    config_dir = p / "config"
    if not config_dir.exists():
        raise ValueError(
            f"No config/ directory found at {p}. "
            f"Is this an easyml project? Run scaffold_project() first."
        )
    return p


# -----------------------------------------------------------------------
# Model tools
# -----------------------------------------------------------------------


@mcp.tool()
def add_model(
    name: str,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    params: str | None = None,
    active: bool = True,
    include_in_ensemble: bool = True,
    project_dir: str | None = None,
) -> str:
    """Add a model to the project.

    Either model_type or preset must be specified. Presets provide
    sensible defaults (e.g., 'xgboost_classifier', 'catboost_classifier').
    Pass features as a list of column names. Params is a JSON string of
    model hyperparameters.
    """
    from easyml.runner.config_writer import add_model as _add_model

    parsed_params = json.loads(params) if params else None
    return _add_model(
        _resolve_project_dir(project_dir),
        name,
        model_type=model_type,
        preset=preset,
        features=features,
        params=parsed_params,
        active=active,
        include_in_ensemble=include_in_ensemble,
    )


@mcp.tool()
def remove_model(name: str, project_dir: str | None = None) -> str:
    """Remove a model from the project."""
    from easyml.runner.config_writer import remove_model as _remove_model

    return _remove_model(_resolve_project_dir(project_dir), name)


@mcp.tool()
def show_models(project_dir: str | None = None) -> str:
    """List all models with type, status, feature count, and ensemble membership."""
    from easyml.runner.config_writer import show_models as _show_models

    return _show_models(_resolve_project_dir(project_dir))


@mcp.tool()
def show_presets() -> str:
    """List available model presets (pre-configured model templates)."""
    from easyml.runner.config_writer import show_presets as _show_presets

    return _show_presets()


# -----------------------------------------------------------------------
# Ensemble tools
# -----------------------------------------------------------------------


@mcp.tool()
def configure_ensemble(
    method: str | None = None,
    temperature: float | None = None,
    exclude_models: list[str] | None = None,
    project_dir: str | None = None,
) -> str:
    """Update ensemble configuration (method, temperature, excluded models)."""
    from easyml.runner.config_writer import configure_ensemble as _configure_ensemble

    return _configure_ensemble(
        _resolve_project_dir(project_dir),
        method=method,
        temperature=temperature,
        exclude_models=exclude_models,
    )


# -----------------------------------------------------------------------
# Backtest tools
# -----------------------------------------------------------------------


@mcp.tool()
def configure_backtest(
    cv_strategy: str | None = None,
    seasons: list[int] | None = None,
    metrics: list[str] | None = None,
    min_train_folds: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Update backtest configuration (CV strategy, seasons, metrics)."""
    from easyml.runner.config_writer import configure_backtest as _configure_backtest

    return _configure_backtest(
        _resolve_project_dir(project_dir),
        cv_strategy=cv_strategy,
        seasons=seasons,
        metrics=metrics,
        min_train_folds=min_train_folds,
    )


# -----------------------------------------------------------------------
# Data tools
# -----------------------------------------------------------------------


@mcp.tool()
def add_dataset(
    data_path: str,
    join_on: list[str] | None = None,
    prefix: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Add a new dataset by merging into the project's feature store.

    Reads CSV/parquet/Excel, auto-detects join keys, merges with
    existing features, and reports columns added + correlation preview.
    """
    from easyml.runner.config_writer import add_dataset as _add_dataset

    return _add_dataset(
        _resolve_project_dir(project_dir),
        data_path,
        join_on=join_on,
        prefix=prefix,
    )


@mcp.tool()
def profile_data(
    category: str | None = None,
    project_dir: str | None = None,
) -> str:
    """Profile the project's feature dataset — column stats, types, null rates."""
    from easyml.runner.config_writer import profile_data as _profile_data

    return _profile_data(_resolve_project_dir(project_dir), category=category)


@mcp.tool()
def available_features(
    prefix: str | None = None,
    project_dir: str | None = None,
) -> str:
    """List all available feature columns in the dataset."""
    from easyml.runner.config_writer import available_features as _available_features

    return _available_features(_resolve_project_dir(project_dir), prefix=prefix)


# -----------------------------------------------------------------------
# Feature tools
# -----------------------------------------------------------------------


@mcp.tool()
def add_feature(
    name: str,
    formula: str,
    description: str = "",
    project_dir: str | None = None,
) -> str:
    """Create a new feature from a formula expression.

    Formulas can reference existing columns (diff_adj_em) or other
    created features (@my_feature). Supports math ops (+, -, *, /, **),
    functions (log, sqrt, cbrt, abs), and parentheses.

    Examples:
      "diff_adj_em * diff_barthag"
      "log(abs(diff_adj_em) + 1)"
      "@efficiency_ratio * diff_seed_num"
    """
    from easyml.runner.config_writer import add_feature as _add_feature

    return _add_feature(
        _resolve_project_dir(project_dir),
        name,
        formula,
        description=description,
    )


@mcp.tool()
def add_features_batch(
    features: str,
    project_dir: str | None = None,
) -> str:
    """Create multiple features from formula expressions.

    features is a JSON array of objects, each with 'name' and 'formula'
    keys (and optional 'description'). Handles @-references between
    features in the batch via topological ordering.

    Example: [{"name": "em_sq", "formula": "diff_adj_em ** 2"},
              {"name": "em_sq_tempo", "formula": "@em_sq * diff_tempo"}]
    """
    from easyml.runner.config_writer import add_features_batch as _add_features_batch

    parsed = json.loads(features)
    return _add_features_batch(_resolve_project_dir(project_dir), parsed)


@mcp.tool()
def test_transformations(
    features: list[str],
    test_interactions: bool = True,
    project_dir: str | None = None,
) -> str:
    """Test mathematical transformations of features automatically.

    Tests log, sqrt, cbrt, squared, reciprocal, rank, and z-score
    transformations. If test_interactions=True, also tests pairwise
    interactions (multiply, divide, subtract) with top correlated features.

    Returns ranked results showing which transformation improves
    correlation with the target variable.
    """
    from easyml.runner.config_writer import test_feature_transformations as _test

    return _test(
        _resolve_project_dir(project_dir),
        features,
        test_interactions=test_interactions,
    )


@mcp.tool()
def discover_features(
    top_n: int = 20,
    method: str = "xgboost",
    project_dir: str | None = None,
) -> str:
    """Run feature discovery analysis.

    Computes correlations, feature importance (via XGBoost or mutual info),
    detects redundant feature pairs, and suggests feature groupings.
    Returns a comprehensive markdown report.
    """
    from easyml.runner.config_writer import discover_features as _discover_features

    return _discover_features(
        _resolve_project_dir(project_dir),
        top_n=top_n,
        method=method,
    )


# -----------------------------------------------------------------------
# Experiment tools
# -----------------------------------------------------------------------


@mcp.tool()
def experiment_create(
    description: str,
    hypothesis: str = "",
    project_dir: str | None = None,
) -> str:
    """Create a new experiment with auto-generated ID.

    Creates an experiment directory with an empty overlay YAML
    and optional hypothesis file. Returns the experiment ID and path.
    """
    from easyml.runner.config_writer import experiment_create as _experiment_create

    return _experiment_create(
        _resolve_project_dir(project_dir),
        description,
        hypothesis=hypothesis,
    )


@mcp.tool()
def write_overlay(
    experiment_id: str,
    overlay: str,
    project_dir: str | None = None,
) -> str:
    """Write an overlay YAML to an experiment directory.

    overlay is a JSON string representing the overlay config to write.
    Example: {"models": {"xgb_v1": {"params": {"learning_rate": 0.2}}}}
    """
    from easyml.runner.config_writer import write_overlay as _write_overlay

    parsed = json.loads(overlay)
    return _write_overlay(
        _resolve_project_dir(project_dir),
        experiment_id,
        parsed,
    )


# -----------------------------------------------------------------------
# Config / Run tools
# -----------------------------------------------------------------------


@mcp.tool()
def show_config(project_dir: str | None = None) -> str:
    """Show the full resolved project configuration (models, ensemble, backtest)."""
    from easyml.runner.config_writer import show_config as _show_config

    return _show_config(_resolve_project_dir(project_dir))


@mcp.tool()
def list_runs(project_dir: str | None = None) -> str:
    """List all pipeline runs with their status."""
    from easyml.runner.config_writer import list_runs as _list_runs

    return _list_runs(_resolve_project_dir(project_dir))


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
