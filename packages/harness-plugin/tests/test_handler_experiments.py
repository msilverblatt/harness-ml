"""Tests for experiments handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Uses the new protomcp tool_group dispatch.
"""
from __future__ import annotations

import harnessml.plugin.handlers.experiments  # noqa: F401 — triggers registration
from protomcp.group import _dispatch_group_action, get_registered_groups


def _call(action_name, **kwargs):
    """Dispatch to the experiments tool group."""
    for group in get_registered_groups():
        if group.name == "experiments":
            return _dispatch_group_action(group, action=action_name, **kwargs)
    raise ValueError("Tool 'experiments' not found")


class TestExperimentsDispatch:
    """Test experiments handler dispatch routing and validation."""

    # ---- unknown action ----

    def test_dispatch_unknown_action(self):
        result = _call("nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_dispatch_unknown_action_suggests_close_match(self):
        result = _call("compre")
        assert result.is_error
        assert "compare" in result.result

    # ---- required params validated ----

    def test_quick_run_missing_description(self):
        result = _call("quick_run", description=None, overlay='{"models": {}}')
        assert result.is_error

    def test_quick_run_missing_overlay(self):
        result = _call("quick_run", description="test", overlay=None)
        assert result.is_error

    def test_explore_missing_search_space(self):
        result = _call("explore", search_space=None)
        assert result.is_error

    def test_promote_trial_missing_experiment_id(self):
        result = _call("promote_trial", experiment_id=None)
        assert result.is_error

    def test_compare_missing_experiment_ids(self):
        from harnessml.plugin.handlers.experiments import _handle_compare
        result = _handle_compare(experiment_ids=None, project_dir=None)
        assert "**Error**" in result
        assert "experiment_ids" in result.lower()

    def test_compare_insufficient_experiment_ids(self):
        from harnessml.plugin.handlers.experiments import _handle_compare
        result = _handle_compare(experiment_ids=["only_one"], project_dir=None)
        assert "**Error**" in result
        assert "experiment_ids" in result.lower()

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "experiments"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "quick_run", "explore", "promote_trial", "compare", "journal",
        }
        assert action_names == expected
