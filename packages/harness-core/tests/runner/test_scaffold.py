"""Tests for project scaffolding (harnessml init)."""
from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from harnessml.core.runner.cli import main
from harnessml.core.runner.scaffold import scaffold_project
from harnessml.core.runner.validator import validate_project


# -----------------------------------------------------------------------
# Tests: scaffold_project
# -----------------------------------------------------------------------

class TestScaffoldDirectories:
    """scaffold_project creates all expected directories and files."""

    def test_creates_config_dir(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "config").is_dir()

    def test_creates_data_dirs(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "data" / "raw").is_dir()
        assert (project_dir / "data" / "processed").is_dir()
        assert (project_dir / "data" / "features").is_dir()

    def test_creates_features_dir(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "features").is_dir()

    def test_creates_experiments_dir(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "experiments").is_dir()

    def test_creates_models_dir(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "models").is_dir()

    def test_creates_all_config_files(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "config" / "pipeline.yaml").is_file()
        assert (project_dir / "config" / "models.yaml").is_file()
        assert (project_dir / "config" / "ensemble.yaml").is_file()
        assert (project_dir / "config" / "server.yaml").is_file()

    def test_creates_claude_md(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "CLAUDE.md").is_file()
        content = (project_dir / "CLAUDE.md").read_text()
        assert "my_project" in content

    def test_creates_experiment_log(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)
        assert (project_dir / "EXPERIMENT_LOG.md").is_file()
        content = (project_dir / "EXPERIMENT_LOG.md").read_text()
        assert "Experiment Log" in content

    def test_all_expected_structure(self, tmp_path):
        """Comprehensive check for all directories and files."""
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        expected_dirs = [
            "config",
            "data/raw",
            "data/processed",
            "data/features",
            "outputs",
            "features",
            "experiments",
            "models",
        ]
        for d in expected_dirs:
            assert (project_dir / d).is_dir(), f"Missing directory: {d}"

        expected_files = [
            "config/pipeline.yaml",
            "config/models.yaml",
            "config/ensemble.yaml",
            "config/server.yaml",
            "CLAUDE.md",
            "EXPERIMENT_LOG.md",
        ]
        for f in expected_files:
            assert (project_dir / f).is_file(), f"Missing file: {f}"


# -----------------------------------------------------------------------
# Tests: scaffolded config validates
# -----------------------------------------------------------------------

class TestScaffoldValidation:
    """Scaffolded config passes validate_project()."""

    def test_scaffolded_config_is_valid(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.valid, f"Validation failed: {result.format()}"
        assert result.config is not None

    def test_scaffolded_config_has_data(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.config.data.raw_dir == "data/raw"
        assert result.config.data.features_dir == "data/features"

    def test_scaffolded_config_has_empty_models(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert len(result.config.models) == 0

    def test_scaffolded_config_has_ensemble(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.config.ensemble.method == "stacked"

    def test_scaffolded_config_has_server(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.config.server is not None
        assert result.config.server.name == "my_project"

    def test_scaffolded_config_has_backtest(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.config.backtest.cv_strategy == "leave_one_out"

    def test_scaffolded_config_has_empty_views(self, tmp_path):
        project_dir = tmp_path / "my_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.config.data.views == {}
        assert result.config.data.features_view is None


# -----------------------------------------------------------------------
# Tests: error handling
# -----------------------------------------------------------------------

class TestScaffoldErrors:
    """scaffold_project refuses non-empty directories."""

    def test_refuses_non_empty_directory(self, tmp_path):
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        (project_dir / "existing_file.txt").write_text("something")

        with pytest.raises(FileExistsError):
            scaffold_project(project_dir)

    def test_allows_nonexistent_directory(self, tmp_path):
        project_dir = tmp_path / "brand_new"
        scaffold_project(project_dir)
        assert project_dir.is_dir()

    def test_allows_empty_directory(self, tmp_path):
        project_dir = tmp_path / "empty_dir"
        project_dir.mkdir()
        scaffold_project(project_dir)
        assert (project_dir / "config" / "pipeline.yaml").is_file()


# -----------------------------------------------------------------------
# Tests: project name
# -----------------------------------------------------------------------

class TestScaffoldProjectName:
    """Project name defaults to directory name or uses provided name."""

    def test_uses_provided_name(self, tmp_path):
        project_dir = tmp_path / "my_dir"
        scaffold_project(project_dir, project_name="Custom Name")

        result = validate_project(project_dir / "config")
        assert result.config.server.name == "Custom Name"

        content = (project_dir / "CLAUDE.md").read_text()
        assert "Custom Name" in content

    def test_defaults_to_dir_name(self, tmp_path):
        project_dir = tmp_path / "cool_project"
        scaffold_project(project_dir)

        result = validate_project(project_dir / "config")
        assert result.config.server.name == "cool_project"


# -----------------------------------------------------------------------
# Tests: CLI init command
# -----------------------------------------------------------------------

class TestCLIInit:
    """CLI init command works."""

    def test_cli_init_creates_project(self, tmp_path):
        project_dir = tmp_path / "cli_project"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(project_dir)])
        assert result.exit_code == 0
        assert "initialized" in result.output.lower()
        assert (project_dir / "config" / "pipeline.yaml").is_file()

    def test_cli_init_with_name(self, tmp_path):
        project_dir = tmp_path / "cli_project"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(project_dir), "--name", "My ML Project"])
        assert result.exit_code == 0
        assert (project_dir / "config" / "pipeline.yaml").is_file()

        # Verify the name was used
        vr = validate_project(project_dir / "config")
        assert vr.config.server.name == "My ML Project"

    def test_cli_init_shows_next_steps(self, tmp_path):
        project_dir = tmp_path / "cli_project"
        runner = CliRunner()
        result = runner.invoke(main, ["init", str(project_dir)])
        assert "Next steps" in result.output
        assert "harnessml validate" in result.output

    def test_cli_init_refuses_non_empty(self, tmp_path):
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("content")

        runner = CliRunner()
        result = runner.invoke(main, ["init", str(project_dir)])
        assert result.exit_code != 0

    def test_cli_init_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "PROJECT_DIR" in result.output
