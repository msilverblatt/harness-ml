"""Project scaffolding for harnessml init.

Generates a new project directory with config files, data directories,
and starter templates that pass validate_project().
"""
from __future__ import annotations

from pathlib import Path

import yaml


# -----------------------------------------------------------------------
# YAML templates
# -----------------------------------------------------------------------

def _pipeline_yaml(
    *,
    task: str = "classification",
    target_column: str = "result",
    key_columns: list[str] | None = None,
    time_column: str | None = None,
) -> dict:
    """Generate pipeline.yaml template with DataConfig + BacktestConfig."""
    data_config: dict = {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "features_dir": "data/features",
        "features_file": "features.parquet",
        "outputs_dir": "outputs",
        "task": task,
        "target_column": target_column,
        "views": {},
        "features_view": None,
    }
    if key_columns:
        data_config["key_columns"] = key_columns
    if time_column:
        data_config["time_column"] = time_column

    return {
        "data": data_config,
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_values": [],
            "metrics": ["brier", "accuracy", "ece", "log_loss"],
            "min_train_folds": 1,
        },
    }


def _models_yaml() -> dict:
    """Generate models.yaml template — empty models dict, ready for user to add."""
    return {
        "models": {},
    }


def _ensemble_yaml() -> dict:
    """Generate ensemble.yaml template."""
    return {
        "ensemble": {
            "method": "stacked",
            "meta_learner": {},
            "temperature": 1.0,
            "clip_floor": 0.0,
            "logit_adjustments": [],
            "exclude_models": [],
        },
    }


def _sources_yaml() -> dict:
    """Generate sources.yaml template with temporal safety guidance."""
    return {
        "sources": {
            "example_source": {
                "module": "features.my_features",
                "function": "compute_features",
                "category": "strength",
                "temporal_safety": "pre_event",
                "outputs": ["feature_a", "feature_b"],
                "leakage_notes": "Uses only pre-event data, safe for prediction",
            },
        },
        "guardrails": {
            "feature_leakage_denylist": [],
        },
    }


def _sources_yaml_comment() -> str:
    """Generate commented sources.yaml with explanation."""
    return """\
# Data Source Declarations — Temporal Safety & Leakage Tracking
#
# Every data source should declare its temporal_safety level:
#   pre_event   — Uses only data available before the prediction target
#   post_event  — Contains post-event outcomes (LEAKAGE if used for prediction)
#   mixed       — Contains both pre and post data (needs careful filtering)
#   unknown     — Not yet audited (treat as potential leakage)
#
# The guardrails.feature_leakage_denylist lists column names that should NEVER
# be used as model features (e.g., statistics that include target-period outcomes).
#
# Example leakage patterns to watch for:
#   - Final statistics that include target-period outcomes
#   - Aggregate stats computed from future data
#   - Rankings/ratings published after the prediction cutoff
#   - Features derived from outcome data

sources:
  example_source:
    module: features.my_features
    function: compute_features
    category: strength
    temporal_safety: pre_event
    outputs:
    - feature_a
    - feature_b
    leakage_notes: Uses only pre-event data, safe for prediction

guardrails:
  feature_leakage_denylist: []
  # Add column names that must never be used as features, e.g.:
  # - final_period_wins  (includes target-period outcomes)
  # - career_total_score (includes future outcomes)
"""


def _server_yaml(project_name: str) -> dict:
    """Generate server.yaml template with train/backtest/pipeline tools + inspection."""
    return {
        "server": {
            "name": project_name,
            "tools": {
                "train": {
                    "command": "uv run python -m harnessml.runner train",
                    "description": "Train all active models",
                    "timeout": 300,
                },
                "backtest": {
                    "command": "uv run python -m harnessml.runner backtest",
                    "description": "Run backtesting across configured folds",
                    "timeout": 600,
                },
                "pipeline": {
                    "command": "uv run python -m harnessml.runner pipeline",
                    "description": "Run the full pipeline",
                    "timeout": 900,
                },
            },
            "inspection": [
                "show_config",
                "list_models",
                "list_features",
                "list_experiments",
                "profile_data",
                "available_features",
                "list_runs",
                "list_presets",
                "discover_features",
            ],
        },
    }


def _claude_md(project_name: str) -> str:
    """Generate CLAUDE.md with AI agent operation instructions."""
    return f"""\
# {project_name}

## Quick Start

- Validate config: `harnessml validate`
- Inspect models: `harnessml inspect models`
- Run training: `harnessml run train`
- Run backtest: `harnessml run backtest`

## Project Structure

- `config/` — YAML configuration (pipeline, models, ensemble, server)
- `data/` — Raw, processed, and feature data
- `features/` — Feature computation modules
- `experiments/` — Experiment overlays and results
- `models/` — Trained model artifacts

## Rules

- ALWAYS use `uv run` — never bare `python`
- Run `harnessml validate` after any config change
- One variable per experiment
- Log every experiment in EXPERIMENT_LOG.md
"""


def _experiment_log_md() -> str:
    """Generate empty EXPERIMENT_LOG.md."""
    return """\
# Experiment Log

| ID | Hypothesis | Changes | Verdict | Notes |
|----|-----------|---------|---------|-------|
"""


# -----------------------------------------------------------------------
# Main scaffold function
# -----------------------------------------------------------------------

def scaffold_project(
    project_dir: Path,
    project_name: str | None = None,
    *,
    task: str = "classification",
    target_column: str = "result",
    key_columns: list[str] | None = None,
    time_column: str | None = None,
) -> None:
    """Generate a new harnessml project directory structure.

    Parameters
    ----------
    project_dir:
        Path where the project will be created.
    project_name:
        Human-readable project name. Defaults to the directory name.
    task:
        ML task type (classification, regression, ranking).
    target_column:
        Name of the target column.
    key_columns:
        Row identifier columns (e.g. game_id, customer_id).
    time_column:
        Column for temporal CV splits (e.g. season, date).

    Raises
    ------
    FileExistsError
        If project_dir already exists and is non-empty.
    """
    project_dir = Path(project_dir)

    if project_dir.exists() and any(project_dir.iterdir()):
        raise FileExistsError(
            f"Directory {project_dir} already exists and is not empty."
        )

    if project_name is None:
        project_name = project_dir.name

    # Create directory structure
    directories = [
        project_dir / "config",
        project_dir / "data" / "raw",
        project_dir / "data" / "processed",
        project_dir / "data" / "features",
        project_dir / "outputs",
        project_dir / "features",
        project_dir / "experiments",
        project_dir / "models",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    # Write YAML config files
    _write_yaml(project_dir / "config" / "pipeline.yaml", _pipeline_yaml(
        task=task,
        target_column=target_column,
        key_columns=key_columns,
        time_column=time_column,
    ))
    _write_yaml(project_dir / "config" / "models.yaml", _models_yaml())
    _write_yaml(project_dir / "config" / "ensemble.yaml", _ensemble_yaml())
    _write_yaml(project_dir / "config" / "server.yaml", _server_yaml(project_name))

    # sources.yaml gets the commented version with leakage guidance
    (project_dir / "config" / "sources.yaml").write_text(_sources_yaml_comment())

    # Write text files
    (project_dir / "CLAUDE.md").write_text(_claude_md(project_name))
    (project_dir / "EXPERIMENT_LOG.md").write_text(_experiment_log_md())


def _write_yaml(path: Path, data: dict) -> None:
    """Write a dict as YAML to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
