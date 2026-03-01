"""Tests for PipelineServer — tool registry, guardrails, and inspection tools."""

import pytest

from easyml.guardrails.base import GuardrailError
from easyml.guardrails.inventory import NamingConventionGuardrail
from easyml.guardrails.server import PipelineServer


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_server_registers_tools():
    server = PipelineServer(name="test")
    server.register_tool("my_tool", lambda: "ok", description="test tool")
    assert "my_tool" in server._tools
    assert server._tools["my_tool"].description == "test tool"


def test_server_registers_multiple_tools():
    server = PipelineServer(name="test")
    server.register_tool("tool_a", lambda: "a")
    server.register_tool("tool_b", lambda: "b")
    assert "tool_a" in server._tools
    assert "tool_b" in server._tools


# ---------------------------------------------------------------------------
# Guardrail execution
# ---------------------------------------------------------------------------


def test_server_runs_guardrails_passing():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    server = PipelineServer(name="test", guardrails=[g])
    server.register_tool("create_exp", lambda: None, guardrails=["naming_convention"])
    results = server.run_guardrails(
        "create_exp",
        context={"experiment_id": "exp-001-test"},
    )
    assert all(r["passed"] for r in results)


def test_server_guardrail_blocks():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    server = PipelineServer(name="test", guardrails=[g])
    server.register_tool("create_exp", lambda: None, guardrails=["naming_convention"])
    with pytest.raises(GuardrailError):
        server.run_guardrails("create_exp", context={"experiment_id": "bad"})


def test_server_guardrail_override():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    server = PipelineServer(name="test", guardrails=[g])
    server.register_tool("create_exp", lambda: None, guardrails=["naming_convention"])
    results = server.run_guardrails(
        "create_exp",
        context={"experiment_id": "bad"},
        human_override=True,
    )
    assert all(r["passed"] for r in results)


def test_server_no_guardrails_for_tool():
    g = NamingConventionGuardrail(pattern=r"exp-\d{3}-[a-z]+$")
    server = PipelineServer(name="test", guardrails=[g])
    server.register_tool("other_tool", lambda: None)
    # No guardrails configured for this tool, should return empty list
    results = server.run_guardrails("other_tool", context={})
    assert results == []


def test_server_unknown_tool():
    server = PipelineServer(name="test")
    results = server.run_guardrails("nonexistent", context={})
    assert results == []


def test_server_unknown_guardrail_name():
    server = PipelineServer(name="test")
    # Tool references a guardrail that doesn't exist in the server
    server.register_tool("tool", lambda: None, guardrails=["nonexistent_guardrail"])
    results = server.run_guardrails("tool", context={})
    assert results == []


# ---------------------------------------------------------------------------
# Inspection tools
# ---------------------------------------------------------------------------


def test_add_inspection_tools_config():
    server = PipelineServer(name="test")
    server.add_inspection_tools(config_resolver=lambda: {"key": "value"})
    assert "show_config" in server._tools
    # Actually call the generated tool
    result = server._tools["show_config"].fn()
    assert result == {"key": "value"}


def test_add_inspection_tools_features():
    class MockRegistry:
        def list(self):
            return ["feat_a", "feat_b"]

    server = PipelineServer(name="test")
    server.add_inspection_tools(feature_registry=MockRegistry())
    assert "list_features" in server._tools
    result = server._tools["list_features"].fn()
    assert result == ["feat_a", "feat_b"]


def test_add_inspection_tools_models():
    class MockRegistry:
        def list(self):
            return ["xgb_core", "logreg_seed"]

    server = PipelineServer(name="test")
    server.add_inspection_tools(model_registry=MockRegistry())
    assert "list_models" in server._tools


def test_add_inspection_tools_experiments():
    class MockManager:
        def list(self):
            return ["exp-001-test"]

    server = PipelineServer(name="test")
    server.add_inspection_tools(experiment_manager=MockManager())
    assert "list_experiments" in server._tools


def test_add_inspection_tools_all():
    class MockObj:
        def list(self):
            return []

    server = PipelineServer(name="test")
    server.add_inspection_tools(
        config_resolver=lambda: {},
        feature_registry=MockObj(),
        model_registry=MockObj(),
        experiment_manager=MockObj(),
    )
    assert "show_config" in server._tools
    assert "list_features" in server._tools
    assert "list_models" in server._tools
    assert "list_experiments" in server._tools


def test_add_inspection_tools_none():
    server = PipelineServer(name="test")
    server.add_inspection_tools()  # all None — should add nothing
    assert len(server._tools) == 0


# ---------------------------------------------------------------------------
# FastMCP integration
# ---------------------------------------------------------------------------


def test_to_fastmcp():
    server = PipelineServer(name="test-mcp")
    server.register_tool("ping", lambda: "pong", description="Health check")
    mcp = server.to_fastmcp()
    # Just verify it returned a FastMCP instance
    from fastmcp import FastMCP
    assert isinstance(mcp, FastMCP)
