"""Tests for MCP server generator."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from easyml.core.runner.schema import GuardrailDef, ServerDef, ServerToolDef
from easyml.core.runner.server_gen import GeneratedServer, ToolSpec, generate_server


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
            "cv_strategy": "leave_one_season_out",
            "seasons": [2023, 2024],
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


def _make_server_def(
    tools: dict[str, ServerToolDef] | None = None,
    inspection: list[str] | None = None,
) -> ServerDef:
    """Build a ServerDef for testing."""
    return ServerDef(
        name="test-server",
        tools=tools or {},
        inspection=inspection or [],
    )


# -----------------------------------------------------------------------
# Tests: execution tools
# -----------------------------------------------------------------------

class TestExecutionTools:
    """generate_server creates execution tools from config."""

    def test_creates_execution_tools(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo hello",
            args=["--verbose"],
            guardrails=["sanity_check"],
            description="Run training",
            timeout=30,
        )
        server_def = _make_server_def(tools={"train": tool_def})
        server = generate_server(server_def, tmp_path)

        assert "train" in server.tools
        assert isinstance(server.tools["train"], ToolSpec)

    def test_execution_tool_is_callable(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(command="echo test", description="Test tool")
        server_def = _make_server_def(tools={"test_tool": tool_def})
        server = generate_server(server_def, tmp_path)

        assert callable(server.tools["test_tool"].fn)

    def test_multiple_execution_tools(self, tmp_path):
        _setup_minimal(tmp_path)
        tools = {
            "train": ServerToolDef(command="echo train", description="Train"),
            "backtest": ServerToolDef(command="echo backtest", description="Backtest"),
            "predict": ServerToolDef(command="echo predict", description="Predict"),
        }
        server_def = _make_server_def(tools=tools)
        server = generate_server(server_def, tmp_path)

        assert len(server.tools) == 3
        assert "train" in server.tools
        assert "backtest" in server.tools
        assert "predict" in server.tools


# -----------------------------------------------------------------------
# Tests: inspection tools
# -----------------------------------------------------------------------

class TestInspectionTools:
    """generate_server creates inspection tools from config.inspection."""

    def test_creates_inspection_tools(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def(
            inspection=["show_config", "list_models"],
        )
        server = generate_server(server_def, tmp_path)

        assert "show_config" in server.tools
        assert "list_models" in server.tools

    def test_all_inspection_tools(self, tmp_path):
        _setup_minimal(tmp_path)
        all_inspections = ["show_config", "list_models", "list_features", "list_experiments"]
        server_def = _make_server_def(inspection=all_inspections)
        server = generate_server(server_def, tmp_path)

        for name in all_inspections:
            assert name in server.tools
            assert callable(server.tools[name].fn)

    def test_inspection_tools_match_config_list(self, tmp_path):
        _setup_minimal(tmp_path)
        requested = ["show_config", "list_models"]
        server_def = _make_server_def(inspection=requested)
        server = generate_server(server_def, tmp_path)

        inspection_tools = [
            name for name in server.tools
            if name in {"show_config", "list_models", "list_features", "list_experiments"}
        ]
        assert sorted(inspection_tools) == sorted(requested)

    def test_unknown_inspection_tool_skipped(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def(
            inspection=["show_config", "nonexistent_tool"],
        )
        server = generate_server(server_def, tmp_path)

        assert "show_config" in server.tools
        assert "nonexistent_tool" not in server.tools


# -----------------------------------------------------------------------
# Tests: tool metadata
# -----------------------------------------------------------------------

class TestToolMetadata:
    """Tool descriptions and guardrails are preserved."""

    def test_description_preserved(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo train",
            description="Train all active models using current config",
        )
        server_def = _make_server_def(tools={"train": tool_def})
        server = generate_server(server_def, tmp_path)

        assert server.tools["train"].description == "Train all active models using current config"

    def test_guardrails_preserved(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo train",
            guardrails=["sanity_check", "leakage_check"],
            description="Train",
        )
        server_def = _make_server_def(tools={"train": tool_def})
        server = generate_server(server_def, tmp_path)

        assert server.tools["train"].guardrails == ["sanity_check", "leakage_check"]

    def test_args_preserved(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo train",
            args=["--verbose", "--force"],
            description="Train",
        )
        server_def = _make_server_def(tools={"train": tool_def})
        server = generate_server(server_def, tmp_path)

        assert server.tools["train"].args == ["--verbose", "--force"]

    def test_default_description(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(command="echo train")
        server_def = _make_server_def(tools={"train": tool_def})
        server = generate_server(server_def, tmp_path)

        # Should have a default description when none is provided
        assert server.tools["train"].description is not None
        assert len(server.tools["train"].description) > 0

    def test_inspection_tool_has_description(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def(inspection=["show_config"])
        server = generate_server(server_def, tmp_path)

        assert server.tools["show_config"].description is not None
        assert len(server.tools["show_config"].description) > 0


# -----------------------------------------------------------------------
# Tests: GeneratedServer
# -----------------------------------------------------------------------

class TestGeneratedServer:
    """GeneratedServer data class works correctly."""

    def test_server_name(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def()
        server = generate_server(server_def, tmp_path)
        assert server.name == "test-server"

    def test_empty_server(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def()
        server = generate_server(server_def, tmp_path)
        assert server.tools == {}

    def test_mixed_execution_and_inspection(self, tmp_path):
        _setup_minimal(tmp_path)
        tools = {
            "train": ServerToolDef(command="echo train", description="Train"),
        }
        server_def = _make_server_def(
            tools=tools,
            inspection=["show_config", "list_models"],
        )
        server = generate_server(server_def, tmp_path)

        assert len(server.tools) == 3
        assert "train" in server.tools
        assert "show_config" in server.tools
        assert "list_models" in server.tools


# -----------------------------------------------------------------------
# Tests: execution tool async behavior
# -----------------------------------------------------------------------

class TestExecutionToolAsync:
    """Execution tools run commands via subprocess."""

    def test_execution_tool_runs_command(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo hello_world",
            description="Test echo",
            timeout=10,
        )
        server_def = _make_server_def(tools={"echo_test": tool_def})
        server = generate_server(server_def, tmp_path)

        import json
        result_str = asyncio.run(server.tools["echo_test"].fn())
        result = json.loads(result_str)
        assert result["status"] == "success"
        assert "hello_world" in result["stdout"]

    def test_execution_tool_returns_error_on_bad_command(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="/nonexistent/command",
            description="Bad command",
            timeout=5,
        )
        server_def = _make_server_def(tools={"bad": tool_def})
        server = generate_server(server_def, tmp_path)

        import json
        result_str = asyncio.run(server.tools["bad"].fn())
        result = json.loads(result_str)
        assert result["status"] == "error"


# -----------------------------------------------------------------------
# Tests: inspection tool async behavior
# -----------------------------------------------------------------------

class TestInspectionToolAsync:
    """Inspection tools return config data."""

    def test_show_config_returns_json(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def(inspection=["show_config"])
        server = generate_server(server_def, tmp_path)

        import json
        result_str = asyncio.run(server.tools["show_config"].fn())
        result = json.loads(result_str)
        assert "data" in result
        assert "models" in result

    def test_list_models_returns_model_info(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def(inspection=["list_models"])
        server = generate_server(server_def, tmp_path)

        result_str = asyncio.run(server.tools["list_models"].fn())
        assert "xgb_core" in result_str
        assert "xgboost" in result_str

    def test_list_features_no_features(self, tmp_path):
        _setup_minimal(tmp_path)
        server_def = _make_server_def(inspection=["list_features"])
        server = generate_server(server_def, tmp_path)

        result_str = asyncio.run(server.tools["list_features"].fn())
        assert "no features" in result_str.lower()

    def test_list_features_with_features(self, tmp_path):
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
        server_def = _make_server_def(inspection=["list_features"])
        server = generate_server(server_def, tmp_path)

        result_str = asyncio.run(server.tools["list_features"].fn())
        assert "eff" in result_str
        assert "efficiency" in result_str


# -----------------------------------------------------------------------
# Tests: guardrail enforcement
# -----------------------------------------------------------------------

class TestGuardrailEnforcement:
    """Execution tools with guardrails reject invalid inputs."""

    def test_naming_guardrail_rejects_bad_name(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo create",
            guardrails=["naming_check"],
            description="Create experiment",
            timeout=10,
        )
        guardrail_config = GuardrailDef(
            naming_pattern=r"^exp-\d{3}-.*$",
        )
        server_def = _make_server_def(tools={"create": tool_def})
        server = generate_server(server_def, tmp_path, guardrails=guardrail_config)

        import json
        result_str = asyncio.run(
            server.tools["create"].fn(experiment_id="bad_name")
        )
        result = json.loads(result_str)
        assert result["status"] == "guardrail_failed"
        assert "naming" in result["stderr"].lower()

    def test_naming_guardrail_accepts_valid_name(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo create",
            guardrails=["naming_check"],
            description="Create experiment",
            timeout=10,
        )
        guardrail_config = GuardrailDef(
            naming_pattern=r"^exp-\d{3}-.*$",
        )
        server_def = _make_server_def(tools={"create": tool_def})
        server = generate_server(server_def, tmp_path, guardrails=guardrail_config)

        import json
        result_str = asyncio.run(
            server.tools["create"].fn(experiment_id="exp-001-test")
        )
        result = json.loads(result_str)
        assert result["status"] == "success"

    def test_leakage_guardrail_rejects_denied_feature(self, tmp_path):
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo train",
            guardrails=["leakage_check"],
            description="Train model",
            timeout=10,
        )
        guardrail_config = GuardrailDef(
            feature_leakage_denylist=["kp_adj_o", "kp_adj_d"],
        )
        server_def = _make_server_def(tools={"train": tool_def})
        server = generate_server(server_def, tmp_path, guardrails=guardrail_config)

        import json
        result_str = asyncio.run(
            server.tools["train"].fn(features="kp_adj_o,diff_x")
        )
        result = json.loads(result_str)
        assert result["status"] == "guardrail_failed"
        assert "leakage" in result["stderr"].lower()

    def test_no_guardrails_passes(self, tmp_path):
        """Tool with no guardrails executes normally."""
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo hello",
            guardrails=[],
            description="Test",
            timeout=10,
        )
        server_def = _make_server_def(tools={"test": tool_def})
        server = generate_server(server_def, tmp_path)

        import json
        result_str = asyncio.run(server.tools["test"].fn())
        result = json.loads(result_str)
        assert result["status"] == "success"

    def test_no_guardrail_config_passes(self, tmp_path):
        """Tool with guardrails but no guardrail config passes (no config to enforce)."""
        _setup_minimal(tmp_path)
        tool_def = ServerToolDef(
            command="echo hello",
            guardrails=["naming_check"],
            description="Test",
            timeout=10,
        )
        server_def = _make_server_def(tools={"test": tool_def})
        # No guardrails config passed to generate_server
        server = generate_server(server_def, tmp_path)

        import json
        result_str = asyncio.run(
            server.tools["test"].fn(experiment_id="anything")
        )
        result = json.loads(result_str)
        assert result["status"] == "success"
