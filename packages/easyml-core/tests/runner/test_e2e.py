"""End-to-end YAML-driven integration tests.

Tests the full workflow: scaffold -> validate -> inspect -> pipeline -> experiment.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from easyml.core.runner.cli import main
from easyml.core.runner.pipeline import PipelineRunner
from easyml.core.runner.scaffold import scaffold_project
from easyml.core.runner.validator import validate_project


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def scaffolded_project(tmp_path):
    """Scaffold a project into tmp_path and write mock parquet data.

    Creates 200 rows across 3 seasons (2022, 2023, 2024) with
    diff_prior feature and result column.  Updates pipeline.yaml
    to point at the absolute data path and set seasons.
    """
    project_dir = tmp_path / "test_project"
    scaffold_project(project_dir, project_name="test_project")

    # Write mock matchup data
    features_dir = project_dir / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_rows = 200
    seasons = rng.choice([2022, 2023, 2024], size=n_rows)

    # diff_prior: higher seed advantage -> more likely to win
    diff_prior = rng.standard_normal(n_rows)
    # result correlated with diff_prior so models can learn something
    prob = 1 / (1 + np.exp(-diff_prior))
    result = (rng.random(n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "season": seasons,
        "diff_prior": diff_prior,
        "result": result,
    })
    df.to_parquet(features_dir / "features.parquet", index=False)

    # Update pipeline.yaml to use absolute path and set seasons
    pipeline_path = project_dir / "config" / "pipeline.yaml"
    pipeline_data = {
        "data": {
            "raw_dir": str(project_dir / "data" / "raw"),
            "processed_dir": str(project_dir / "data" / "processed"),
            "features_dir": str(features_dir),
            "gender": "M",
        },
        "backtest": {
            "cv_strategy": "leave_one_season_out",
            "seasons": [2022, 2023, 2024],
            "metrics": ["brier", "accuracy", "ece", "log_loss"],
            "min_train_folds": 1,
        },
    }
    pipeline_path.write_text(yaml.dump(pipeline_data, default_flow_style=False, sort_keys=False))

    # Update models.yaml to use diff_prior feature
    models_path = project_dir / "config" / "models.yaml"
    models_data = {
        "models": {
            "logreg_baseline": {
                "type": "logistic_regression",
                "features": ["diff_prior"],
                "params": {"C": 1.0, "max_iter": 200},
                "active": True,
                "mode": "classifier",
                "n_seeds": 1,
            },
        },
    }
    models_path.write_text(yaml.dump(models_data, default_flow_style=False, sort_keys=False))

    # Add experiments section to config
    experiments_yaml_path = project_dir / "config" / "experiments.yaml"
    experiments_data = {
        "experiments": {
            "experiments_dir": str(project_dir / "experiments"),
            "log_path": str(project_dir / "EXPERIMENT_LOG.md"),
        },
    }
    experiments_yaml_path.write_text(
        yaml.dump(experiments_data, default_flow_style=False, sort_keys=False)
    )

    return project_dir


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestE2EValidate:
    """Scaffold a project and verify validate passes."""

    def test_e2e_validate(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed: {result.format()}"
        assert result.config is not None
        assert "logreg_baseline" in result.config.models

    def test_e2e_validate_cli(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(config_dir), "validate"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "ok" in result.output.lower()


class TestE2EInspect:
    """Inspect config and models on scaffolded project."""

    def test_e2e_inspect_config(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(config_dir), "inspect", "config"])
        assert result.exit_code == 0
        assert "logreg_baseline" in result.output
        assert "logistic_regression" in result.output

    def test_e2e_inspect_models(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(config_dir), "inspect", "models"])
        assert result.exit_code == 0
        assert "logreg_baseline" in result.output
        assert "active" in result.output

    def test_e2e_inspect_config_section(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config-dir", str(config_dir), "inspect", "config", "--section", "backtest"],
        )
        assert result.exit_code == 0
        assert "leave_one_season_out" in result.output


class TestE2EFullPipeline:
    """Scaffold project, run PipelineRunner.run_full(), verify results."""

    def test_e2e_full_pipeline(self, scaffolded_project):
        config_dir = scaffolded_project / "config"

        runner = PipelineRunner(
            project_dir=str(scaffolded_project),
            config_dir=str(config_dir),
        )
        result = runner.run_full()

        assert result["status"] == "success"
        assert "metrics" in result
        assert "brier" in result["metrics"]
        # With correlated data, brier should be meaningfully below 0.5
        assert result["metrics"]["brier"] < 0.5
        assert "logreg_baseline" in result["models_trained"]

    def test_e2e_full_pipeline_model_artifacts(self, scaffolded_project):
        config_dir = scaffolded_project / "config"

        runner = PipelineRunner(
            project_dir=str(scaffolded_project),
            config_dir=str(config_dir),
        )
        runner.run_full()

        # Verify model directory was created
        models_dir = scaffolded_project / "models"
        assert models_dir.is_dir()


class TestE2EExperimentCreate:
    """Create an experiment and verify overlay.yaml exists."""

    def test_e2e_experiment_create(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config-dir", str(config_dir), "experiment", "create", "test-exp-001"],
        )
        assert result.exit_code == 0
        exp_dir = scaffolded_project / "experiments" / "test-exp-001"
        assert exp_dir.is_dir()
        assert (exp_dir / "overlay.yaml").is_file()


class TestE2EExperimentList:
    """List experiments after creating one."""

    def test_e2e_experiment_list(self, scaffolded_project):
        config_dir = scaffolded_project / "config"
        runner = CliRunner()

        # Create an experiment first
        result = runner.invoke(
            main,
            ["--config-dir", str(config_dir), "experiment", "create", "test-exp-001"],
        )
        assert result.exit_code == 0

        # Log it so we can create another (mandatory logging enforcement)
        result = runner.invoke(
            main,
            [
                "--config-dir", str(config_dir),
                "experiment", "log", "test-exp-001",
                "--hypothesis", "testing list",
                "--changes", "none",
                "--verdict", "revert",
            ],
        )
        assert result.exit_code == 0

        # Create a second experiment
        result = runner.invoke(
            main,
            ["--config-dir", str(config_dir), "experiment", "create", "test-exp-002"],
        )
        assert result.exit_code == 0

        # List experiments
        result = runner.invoke(
            main,
            ["--config-dir", str(config_dir), "experiment", "list"],
        )
        assert result.exit_code == 0
        assert "test-exp-001" in result.output
        assert "test-exp-002" in result.output
