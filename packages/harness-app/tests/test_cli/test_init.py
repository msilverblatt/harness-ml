from pathlib import Path

from click.testing import CliRunner

from harness.app.cli.main import cli


def test_init_creates_workspace(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init", "myproject", "--task-type", "binary", "--target", "target"])
        assert result.exit_code == 0, result.output
        assert "Initialized" in result.output
        assert (Path("myproject") / "harness.yaml").exists()


def test_init_default_directory(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init", "--task-type", "regression", "--target", "price"])
        assert result.exit_code == 0, result.output
        assert "Initialized" in result.output
        assert Path("harness.yaml").exists()


def test_init_echoes_task_type_and_target(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init", "proj", "--task-type", "multiclass", "--target", "label"])
        assert result.exit_code == 0, result.output
        assert "multiclass" in result.output
        assert "label" in result.output
