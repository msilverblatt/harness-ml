"""Tests for Click CLI commands."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from harnessml.core.runner.cli import main


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _minimal_pipeline() -> dict:
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "features_dir": "data/features",
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_values": [2023, 2024],
        },
    }


def _minimal_models() -> dict:
    return {
        "models": {
            "xgb_core": {
                "type": "xgboost",
                "features": ["feat_a", "feat_b"],
                "params": {"max_depth": 3},
            }
        }
    }


def _minimal_ensemble() -> dict:
    return {"ensemble": {"method": "stacked"}}


def _setup_minimal(tmp_path: Path) -> None:
    """Write pipeline.yaml, models.yaml, ensemble.yaml for a minimal valid config."""
    _write_yaml(tmp_path / "pipeline.yaml", _minimal_pipeline())
    _write_yaml(tmp_path / "models.yaml", _minimal_models())
    _write_yaml(tmp_path / "ensemble.yaml", _minimal_ensemble())


# -----------------------------------------------------------------------
# Tests: --help
# -----------------------------------------------------------------------

class TestHelp:
    """Top-level --help prints usage."""

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output


# -----------------------------------------------------------------------
# Tests: validate
# -----------------------------------------------------------------------

class TestValidate:
    """validate command calls validate_project and prints result."""

    def test_validate_success(self, tmp_path):
        _setup_minimal(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "validate"])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "ok" in result.output.lower()

    def test_validate_fail_missing_pipeline(self, tmp_path):
        # No pipeline.yaml — validation should fail
        _write_yaml(tmp_path / "models.yaml", _minimal_models())
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "validate"])
        assert result.exit_code != 0 or "pipeline.yaml" in result.output


# -----------------------------------------------------------------------
# Tests: inspect
# -----------------------------------------------------------------------

class TestInspectConfig:
    """inspect config dumps resolved config as JSON."""

    def test_inspect_config_json(self, tmp_path):
        _setup_minimal(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "inspect", "config"])
        assert result.exit_code == 0
        assert "xgb_core" in result.output
        assert "data" in result.output

    def test_inspect_config_section(self, tmp_path):
        _setup_minimal(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main, ["--config-dir", str(tmp_path), "inspect", "config", "--section", "data"]
        )
        assert result.exit_code == 0
        assert "raw_dir" in result.output


class TestInspectModels:
    """inspect models lists model info."""

    def test_inspect_models(self, tmp_path):
        _setup_minimal(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "inspect", "models"])
        assert result.exit_code == 0
        assert "xgb_core" in result.output
        assert "xgboost" in result.output


class TestInspectFeatures:
    """inspect features lists declared features."""

    def test_inspect_features_none(self, tmp_path):
        _setup_minimal(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "inspect", "features"])
        assert result.exit_code == 0
        assert "no features" in result.output.lower() or result.output.strip() != ""

    def test_inspect_features_with_features(self, tmp_path):
        _setup_minimal(tmp_path)
        _write_yaml(
            tmp_path / "features.yaml",
            {
                "features": {
                    "eff": {
                        "module": "proj.feat",
                        "function": "compute_eff",
                        "category": "efficiency",
                        "level": "team",
                        "columns": ["adj_oe", "adj_de"],
                    }
                }
            },
        )
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "inspect", "features"])
        assert result.exit_code == 0
        assert "eff" in result.output


# -----------------------------------------------------------------------
# Tests: experiment
# -----------------------------------------------------------------------

class TestExperimentCreate:
    """experiment create creates experiment directory."""

    def test_experiment_create(self, tmp_path):
        _setup_minimal(tmp_path)
        _write_yaml(
            tmp_path / "experiments.yaml",
            {"experiments": {"experiments_dir": str(tmp_path / "experiments")}},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config-dir", str(tmp_path), "experiment", "create", "test-exp-001"],
        )
        assert result.exit_code == 0
        assert (tmp_path / "experiments" / "test-exp-001").is_dir()


class TestExperimentList:
    """experiment list lists experiment directories."""

    def test_experiment_list(self, tmp_path):
        _setup_minimal(tmp_path)
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        (exp_dir / "exp-001").mkdir()
        (exp_dir / "exp-002").mkdir()
        _write_yaml(
            tmp_path / "experiments.yaml",
            {"experiments": {"experiments_dir": str(exp_dir)}},
        )
        runner = CliRunner()
        result = runner.invoke(
            main, ["--config-dir", str(tmp_path), "experiment", "list"]
        )
        assert result.exit_code == 0
        assert "exp-001" in result.output
        assert "exp-002" in result.output


class TestExperimentLog:
    """experiment log logs experiment with hypothesis/changes/verdict."""

    def test_experiment_log(self, tmp_path):
        _setup_minimal(tmp_path)
        exp_dir = tmp_path / "experiments"
        exp_dir.mkdir()
        (exp_dir / "exp-001").mkdir()
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        _write_yaml(
            tmp_path / "experiments.yaml",
            {
                "experiments": {
                    "experiments_dir": str(exp_dir),
                    "log_path": str(log_path),
                }
            },
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config-dir", str(tmp_path),
                "experiment", "log", "exp-001",
                "--hypothesis", "test hypothesis",
                "--changes", "changed X",
                "--verdict", "keep",
            ],
        )
        assert result.exit_code == 0
        assert log_path.exists()
        content = log_path.read_text()
        assert "exp-001" in content
        assert "test hypothesis" in content


# -----------------------------------------------------------------------
# Tests: run (stubs)
# -----------------------------------------------------------------------

class TestRunStubs:
    """run subcommands exist and are callable."""

    def test_run_train_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "train", "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_run_backtest_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "backtest", "--help"])
        assert result.exit_code == 0

    def test_run_pipeline_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "pipeline", "--help"])
        assert result.exit_code == 0


# -----------------------------------------------------------------------
# Tests: serve / init stubs
# -----------------------------------------------------------------------

class TestServe:
    """serve command tests."""

    def test_serve_no_config(self, tmp_path):
        _setup_minimal(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["--config-dir", str(tmp_path), "serve"])
        assert result.exit_code == 1
        # Should print an error message about no server config
        assert "server" in result.output.lower() or "no" in result.output.lower()


class TestServeWithConfig:
    """serve command starts server when config exists."""

    def test_serve_with_server_config(self, tmp_path, monkeypatch):
        """Serve command calls generate_server and runs the MCP server."""
        _setup_minimal(tmp_path)
        _write_yaml(
            tmp_path / "server.yaml",
            {
                "server": {
                    "name": "test-server",
                    "tools": {},
                    "inspection": ["show_config"],
                }
            },
        )

        # Mock fastmcp to avoid actual server startup
        class MockFastMCP:
            def __init__(self, name):
                self.name = name
                self._tools = {}

            def tool(self, **kwargs):
                def decorator(fn):
                    self._tools[kwargs.get("name", fn.__name__)] = fn
                    return fn
                return decorator

            def run(self):
                pass  # Don't actually start server

        import harnessml.core.runner.server_gen as sg
        monkeypatch.setattr(sg, "FastMCP", MockFastMCP, raising=False)
        # Patch the import in to_fastmcp
        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"fastmcp": mock.MagicMock()}):
            # Need to mock at the import level inside to_fastmcp
            original_to_fastmcp = sg.GeneratedServer.to_fastmcp

            def patched_to_fastmcp(self):
                mcp = MockFastMCP(self.name)
                for name, spec in self.tools.items():
                    mcp.tool(name=name, description=spec.description)(spec.fn)
                return mcp

            monkeypatch.setattr(sg.GeneratedServer, "to_fastmcp", patched_to_fastmcp)

            runner = CliRunner()
            result = runner.invoke(main, ["--config-dir", str(tmp_path), "serve"])
            # Should succeed (server.run() is mocked as no-op)
            assert result.exit_code == 0


# -----------------------------------------------------------------------
# Tests: run predict
# -----------------------------------------------------------------------

class TestRunPredict:
    """run predict command tests."""

    def test_predict_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "predict", "--help"])
        assert result.exit_code == 0
        assert "--fold-value" in result.output

    def test_predict_invocation(self, tmp_path, monkeypatch):
        """Predict command creates PipelineRunner and calls predict."""
        _setup_minimal(tmp_path)

        import harnessml.core.runner.pipeline as pipeline_mod

        calls = []

        class MockRunner:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def load(self):
                calls.append(("load",))

            def predict(self, fold_value, run_id=None):
                calls.append(("predict", fold_value, run_id))
                import pandas as pd
                return pd.DataFrame({"x": [1, 2, 3]})

        monkeypatch.setattr(pipeline_mod, "PipelineRunner", MockRunner)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config-dir", str(tmp_path), "run", "predict", "--fold-value", "2024"],
        )
        assert result.exit_code == 0
        assert "3 matchups" in result.output
        # Verify predict was called with correct fold value
        predict_calls = [c for c in calls if c[0] == "predict"]
        assert len(predict_calls) == 1
        assert predict_calls[0][1] == 2024


# -----------------------------------------------------------------------
# Tests: experiment run
# -----------------------------------------------------------------------

class TestExperimentRun:
    """experiment run command tests."""

    def test_experiment_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["experiment", "run", "--help"])
        assert result.exit_code == 0

    def test_experiment_run_missing_dir(self, tmp_path):
        """experiment run with nonexistent experiment dir fails."""
        _setup_minimal(tmp_path)
        _write_yaml(
            tmp_path / "experiments.yaml",
            {"experiments": {"experiments_dir": str(tmp_path / "experiments")}},
        )
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config-dir", str(tmp_path), "experiment", "run", "nonexistent-exp"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_experiment_run_creates_overlay_runner(self, tmp_path, monkeypatch):
        """experiment run loads overlay and creates PipelineRunner."""
        _setup_minimal(tmp_path)
        exp_dir = tmp_path / "experiments" / "test-exp-001"
        exp_dir.mkdir(parents=True)
        _write_yaml(
            tmp_path / "experiments.yaml",
            {"experiments": {"experiments_dir": str(tmp_path / "experiments")}},
        )
        # Write overlay
        _write_yaml(
            exp_dir / "overlay.yaml",
            {"ensemble": {"temperature": 1.5}},
        )

        import harnessml.core.runner.pipeline as pipeline_mod

        calls = []

        class MockRunner:
            def __init__(self, **kwargs):
                calls.append(("init", kwargs))

            def load(self):
                calls.append(("load",))

            def backtest(self):
                calls.append(("backtest",))
                return {"status": "success", "metrics": {}}

        monkeypatch.setattr(pipeline_mod, "PipelineRunner", MockRunner)

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config-dir", str(tmp_path), "experiment", "run", "test-exp-001"],
        )
        assert result.exit_code == 0

        # Verify overlay was passed
        init_calls = [c for c in calls if c[0] == "init"]
        assert len(init_calls) == 1
        overlay = init_calls[0][1].get("overlay", {})
        assert overlay.get("ensemble", {}).get("temperature") == 1.5


class TestInitStub:
    """init command exists."""

    def test_init_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
