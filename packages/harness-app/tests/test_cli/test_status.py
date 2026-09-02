from pathlib import Path

from click.testing import CliRunner

from harness.app.cli.main import cli


def test_status_no_workspace(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 1
        assert "No harness workspace" in result.output


def test_status_with_workspace(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # First init a workspace
        init_result = runner.invoke(cli, ["init", "--task-type", "binary", "--target", "target"])
        assert init_result.exit_code == 0, init_result.output
        # Then check status
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0, result.output
        assert "Workspace:" in result.output
        assert "Current version:" in result.output


def test_doctor_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Python" in result.output
