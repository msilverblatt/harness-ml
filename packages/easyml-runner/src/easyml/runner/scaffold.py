"""Project scaffolding for easyml init.

Generates a new project directory with config files, data directories,
and starter templates that pass validate_project().
"""
from __future__ import annotations

from pathlib import Path

import yaml


# -----------------------------------------------------------------------
# YAML templates
# -----------------------------------------------------------------------

def _pipeline_yaml() -> dict:
    """Generate pipeline.yaml template with DataConfig + BacktestConfig."""
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "features_dir": "data/features",
            "gender": "M",
        },
        "backtest": {
            "cv_strategy": "leave_one_season_out",
            "seasons": [],
            "metrics": ["brier", "accuracy", "ece", "log_loss"],
            "min_train_folds": 1,
        },
    }


def _models_yaml() -> dict:
    """Generate models.yaml template with a starter logreg_baseline model."""
    return {
        "models": {
            "logreg_baseline": {
                "type": "logistic_regression",
                "features": ["feature_a", "feature_b"],
                "params": {
                    "C": 1.0,
                },
                "active": True,
                "mode": "classifier",
                "n_seeds": 1,
            },
        },
    }


def _ensemble_yaml() -> dict:
    """Generate ensemble.yaml template."""
    return {
        "ensemble": {
            "method": "stacked",
            "meta_learner": {},
            "temperature": 1.0,
            "clip_floor": 0.0,
            "availability_adjustment": 0.1,
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
                "temporal_safety": "pre_tournament",
                "outputs": ["feature_a", "feature_b"],
                "leakage_notes": "Uses regular-season data only, safe for tournament prediction",
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
#   pre_tournament  — Uses only data available before the tournament
#   post_tournament — Contains post-tournament outcomes (LEAKAGE if used for prediction)
#   mixed           — Contains both pre and post data (needs careful filtering)
#   unknown         — Not yet audited (treat as potential leakage)
#
# The guardrails.feature_leakage_denylist lists column names that should NEVER
# be used as model features (e.g., end-of-season stats that include tournament results).
#
# Example leakage patterns to watch for:
#   - Season-final statistics that include tournament game results
#   - Aggregate career stats computed from future data
#   - Rankings/ratings published after tournament selection
#   - Features derived from the tournament bracket itself (seed-based features ARE safe)

sources:
  example_source:
    module: features.my_features
    function: compute_features
    category: strength
    temporal_safety: pre_tournament
    outputs:
    - feature_a
    - feature_b
    leakage_notes: Uses regular-season data only, safe for tournament prediction

guardrails:
  feature_leakage_denylist: []
  # Add column names that must never be used as features, e.g.:
  # - final_season_wins  (includes tournament games)
  # - career_total_pake  (includes future tournament results)
"""


def _server_yaml(project_name: str) -> dict:
    """Generate server.yaml template with train/backtest/pipeline tools + inspection."""
    return {
        "server": {
            "name": project_name,
            "tools": {
                "train": {
                    "command": "uv run python -m easyml.runner train",
                    "description": "Train all active models",
                    "timeout": 300,
                },
                "backtest": {
                    "command": "uv run python -m easyml.runner backtest",
                    "description": "Run backtesting across configured seasons",
                    "timeout": 600,
                },
                "pipeline": {
                    "command": "uv run python -m easyml.runner pipeline",
                    "description": "Run the full pipeline",
                    "timeout": 900,
                },
            },
            "inspection": [
                "show_config",
                "list_models",
                "list_features",
                "list_experiments",
            ],
        },
    }


def _claude_md(project_name: str) -> str:
    """Generate CLAUDE.md with AI agent operation instructions."""
    return f"""\
# {project_name}

## Quick Start

- Validate config: `easyml validate`
- Inspect models: `easyml inspect models`
- Run training: `easyml run train`
- Run backtest: `easyml run backtest`

## Project Structure

- `config/` — YAML configuration (pipeline, models, ensemble, server)
- `data/` — Raw, processed, and feature data
- `features/` — Feature computation modules
- `experiments/` — Experiment overlays and results
- `models/` — Trained model artifacts

## Rules

- ALWAYS use `uv run` — never bare `python`
- Run `easyml validate` after any config change
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
) -> None:
    """Generate a new easyml project directory structure.

    Parameters
    ----------
    project_dir:
        Path where the project will be created.
    project_name:
        Human-readable project name. Defaults to the directory name.

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
        project_dir / "features",
        project_dir / "experiments",
        project_dir / "models",
    ]
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    # Write YAML config files
    _write_yaml(project_dir / "config" / "pipeline.yaml", _pipeline_yaml())
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
