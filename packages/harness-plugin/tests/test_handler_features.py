"""Tests for features handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Uses the new protomcp tool_group dispatch.
"""
from __future__ import annotations

import harnessml.plugin.handlers.features  # noqa: F401 — triggers registration
from protomcp.group import _dispatch_group_action, get_registered_groups


def _call(action_name, **kwargs):
    """Dispatch to the features tool group."""
    for group in get_registered_groups():
        if group.name == "features":
            return _dispatch_group_action(group, action=action_name, **kwargs)
    raise ValueError("Tool 'features' not found")


class TestFeaturesDispatch:
    """Test features handler dispatch routing and validation."""

    # ---- unknown action ----

    def test_dispatch_unknown_action(self):
        result = _call("nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_dispatch_unknown_action_suggests_close_match(self):
        result = _call("ad")
        assert result.is_error
        assert "add" in result.result

    # ---- required params validated ----

    def test_add_missing_name(self):
        result = _call("add", name=None, formula="a + b")
        assert result.is_error

    def test_add_missing_all_definition_fields(self):
        """add requires at least one of: formula, type, source, condition."""
        from harnessml.plugin.handlers.features import _handle_add
        result = _handle_add(
            name="feat1",
            formula=None, type=None, source=None, condition=None,
            column=None, pairwise_mode=None, category=None, description=None,
            project_dir=None,
        )
        assert "**Error**" in result

    def test_add_batch_missing_features(self):
        result = _call("add_batch", features=None)
        assert result.is_error

    def test_test_transformations_missing_features(self):
        result = _call("test_transformations", features=None)
        assert result.is_error

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "features"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "add", "add_batch", "test_transformations",
            "discover", "diversity", "auto_search", "prune",
        }
        assert action_names == expected
