"""Tests for models handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Uses the new protomcp tool_group dispatch.
"""
from __future__ import annotations

import harnessml.plugin.handlers.models  # noqa: F401 — triggers registration
from protomcp.group import _dispatch_group_action, get_registered_groups


def _call(action_name, **kwargs):
    """Dispatch to the models tool group."""
    for group in get_registered_groups():
        if group.name == "models":
            return _dispatch_group_action(group, action=action_name, **kwargs)
    raise ValueError("Tool 'models' not found")


class TestModelsDispatch:
    """Test models handler dispatch routing and validation."""

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
        result = _call("add", name=None, model_type="xgboost")
        assert result.is_error

    def test_update_missing_name(self):
        result = _call("update", name=None)
        assert result.is_error

    def test_remove_missing_name(self):
        result = _call("remove", name=None, purge=False)
        assert result.is_error

    def test_show_missing_name(self):
        result = _call("show", name=None)
        assert result.is_error

    def test_clone_missing_name(self):
        result = _call("clone", name=None, new_name="x")
        assert result.is_error

    def test_clone_missing_new_name(self):
        from harnessml.plugin.handlers.models import _handle_clone
        result = _handle_clone(name="xgb_1", new_name=None, project_dir=None)
        assert "**Error**" in result
        assert "new_name" in result.lower()

    def test_add_batch_missing_items(self):
        from harnessml.plugin.handlers.models import _handle_add_batch
        result = _handle_add_batch(items=None, project_dir=None)
        assert "**Error**" in result
        assert "items" in result.lower()

    def test_update_batch_missing_items(self):
        from harnessml.plugin.handlers.models import _handle_update_batch
        result = _handle_update_batch(items=None, project_dir=None)
        assert "**Error**" in result
        assert "items" in result.lower()

    def test_remove_batch_missing_items(self):
        from harnessml.plugin.handlers.models import _handle_remove_batch
        result = _handle_remove_batch(items=None, project_dir=None)
        assert "**Error**" in result
        assert "items" in result.lower()

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "models"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "add", "update", "remove", "list", "show", "presets",
            "add_batch", "update_batch", "remove_batch", "clone",
        }
        assert action_names == expected
