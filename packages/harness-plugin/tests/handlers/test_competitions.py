"""Tests for the competitions MCP handler."""
from __future__ import annotations

import json

import pytest
from harnessml.plugin.handlers.competitions import _REGISTRY, ACTIONS, dispatch


@pytest.fixture(autouse=True)
def _clear_registry():
    """Clear the in-memory competition registry between tests."""
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


class TestDispatchValidation:
    """Test dispatch routing and validation."""

    def test_invalid_action_returns_error(self):
        result = dispatch("nonexistent_action")
        assert "**Error**" in result
        assert "Invalid" in result
        assert "nonexistent_action" in result

    def test_invalid_action_suggests_close_match(self):
        result = dispatch("creat")
        assert "**Error**" in result
        assert "create" in result

    def test_all_actions_registered(self):
        expected = {
            "create", "list_formats", "simulate", "standings",
            "round_probs", "generate_brackets", "score_bracket",
            "adjust", "explain", "profiles", "confidence",
            "export", "list_strategies",
        }
        assert set(ACTIONS.keys()) == expected


class TestListFormats:
    """Test list_formats action."""

    def test_returns_all_five_formats(self):
        result = dispatch("list_formats")
        assert "single_elimination" in result
        assert "double_elimination" in result
        assert "round_robin" in result
        assert "swiss" in result
        assert "group_knockout" in result

    def test_returns_markdown_table(self):
        result = dispatch("list_formats")
        assert "| Format |" in result
        assert "|--------|" in result


class TestListStrategies:
    """Test list_strategies action."""

    def test_returns_builtin_strategies(self):
        result = dispatch("list_strategies")
        assert "chalk" in result
        assert "near_chalk" in result
        assert "random_sim" in result
        assert "contrarian" in result
        assert "late_contrarian" in result
        assert "champion_anchor" in result

    def test_returns_markdown_table(self):
        result = dispatch("list_strategies")
        assert "| Strategy |" in result


class TestCreate:
    """Test create action."""

    def test_create_requires_config(self):
        result = dispatch("create", config=None)
        assert "**Error**" in result
        assert "config" in result.lower()

    def test_create_with_valid_config(self):
        config = json.dumps({
            "format": "single_elimination",
            "n_participants": 8,
        })
        result = dispatch("create", config=config)
        assert "Competition Created" in result
        assert "single_elimination" in result
        assert "8" in result

    def test_create_stores_in_registry(self):
        config = json.dumps({
            "format": "round_robin",
            "n_participants": 4,
        })
        dispatch("create", config=config, name="my_comp")
        assert "my_comp" in _REGISTRY
        assert _REGISTRY["my_comp"]["config"].format.value == "round_robin"

    def test_create_with_invalid_config(self):
        config = json.dumps({"format": "invalid_format"})
        result = dispatch("create", config=config)
        assert "**Error**" in result

    def test_create_config_must_be_object(self):
        result = dispatch("create", config="[1, 2, 3]")
        assert "**Error**" in result
        assert "JSON object" in result


class TestSimulate:
    """Test simulate action validation."""

    def test_simulate_requires_existing_competition(self):
        result = dispatch("simulate", name="nonexistent")
        assert "**Error**" in result
        assert "not found" in result

    def test_simulate_requires_simulator(self):
        config = json.dumps({
            "format": "single_elimination",
            "n_participants": 4,
        })
        dispatch("create", config=config, name="test")
        result = dispatch("simulate", name="test")
        assert "**Error**" in result
        assert "simulator" in result.lower()


class TestScoreBracket:
    """Test score_bracket action validation."""

    def test_score_bracket_requires_picks(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        result = dispatch("score_bracket", name="test", picks=None, actuals="{}")
        assert "**Error**" in result
        assert "picks" in result.lower()

    def test_score_bracket_requires_actuals(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        result = dispatch("score_bracket", name="test", picks="{}", actuals=None)
        assert "**Error**" in result
        assert "actuals" in result.lower()


class TestGenerateBrackets:
    """Test generate_brackets action validation."""

    def test_generate_brackets_requires_pool_size(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        result = dispatch("generate_brackets", name="test", pool_size=None)
        assert "**Error**" in result

    def test_generate_brackets_requires_simulator(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        result = dispatch("generate_brackets", name="test", pool_size=100)
        assert "**Error**" in result
        assert "simulator" in result.lower()


class TestAdjust:
    """Test adjust action validation."""

    def test_adjust_requires_adjustments(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        result = dispatch("adjust", name="test", adjustments=None)
        assert "**Error**" in result
        assert "adjustments" in result.lower()

    def test_adjust_with_valid_input(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        adj = json.dumps({"entity_multipliers": {"A": 1.1}, "probability_overrides": {}})
        result = dispatch("adjust", name="test", adjustments=adj)
        assert "Adjustments Registered" in result
        assert "1" in result  # 1 entity multiplier


class TestExport:
    """Test export action validation."""

    def test_export_requires_output_dir(self):
        config = json.dumps({"format": "single_elimination", "n_participants": 4})
        dispatch("create", config=config, name="test")
        result = dispatch("export", name="test", output_dir=None)
        assert "**Error**" in result
        assert "output_dir" in result.lower()
