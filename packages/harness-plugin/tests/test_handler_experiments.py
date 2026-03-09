"""Tests for experiments handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Experiments handler has an async dispatch, so tests use asyncio.run().
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


class TestExperimentsDispatch:
    """Test experiments handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.experiments import dispatch

        defaults = dict(
            action="create", description=None, hypothesis=None,
            experiment_id=None, overlay=None, primary_metric=None,
            variant=None, search_space=None, detail=None,
            trial=None, experiment_ids=None, last_n=None,
            conclusion=None, verdict=None, ctx=None, project_dir=None,
        )
        defaults.update(kwargs)
        return asyncio.run(dispatch(**defaults))

    # ---- unknown action ----

    def test_dispatch_unknown_action(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_dispatch_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="creat")
        assert "**Error**" in result
        assert "create" in result

    # ---- valid action routing ----

    def test_dispatch_valid_action_create(self):
        from harnessml.plugin.handlers import experiments

        mock_fn = MagicMock(return_value="Experiment exp_001 created")
        original = experiments.ACTIONS["create"]
        experiments.ACTIONS["create"] = mock_fn
        try:
            result = self._dispatch(
                action="create",
                description="Test baseline",
                hypothesis="XGBoost will outperform LightGBM",
            )
            mock_fn.assert_called_once()
            assert "Experiment exp_001 created" in result
        finally:
            experiments.ACTIONS["create"] = original

    def test_dispatch_valid_action_journal(self):
        from harnessml.plugin.handlers import experiments

        mock_fn = MagicMock(return_value="## Experiment Journal\n| ID | Description |")
        original = experiments.ACTIONS["journal"]
        experiments.ACTIONS["journal"] = mock_fn
        try:
            result = self._dispatch(action="journal")
            mock_fn.assert_called_once()
            assert "Experiment Journal" in result
        finally:
            experiments.ACTIONS["journal"] = original

    def test_dispatch_valid_action_promote(self):
        from harnessml.plugin.handlers import experiments

        mock_fn = MagicMock(return_value="Promoted experiment exp_001 to baseline")
        original = experiments.ACTIONS["promote"]
        experiments.ACTIONS["promote"] = mock_fn
        try:
            result = self._dispatch(action="promote", experiment_id="exp_001")
            mock_fn.assert_called_once()
            assert "Promoted" in result
        finally:
            experiments.ACTIONS["promote"] = original

    def test_dispatch_valid_action_log_result(self):
        from harnessml.plugin.handlers import experiments

        mock_fn = MagicMock(return_value="Logged result for exp_001")
        original = experiments.ACTIONS["log_result"]
        experiments.ACTIONS["log_result"] = mock_fn
        try:
            result = self._dispatch(
                action="log_result", experiment_id="exp_001",
                conclusion="Confirmed hypothesis",
            )
            mock_fn.assert_called_once()
            assert "Logged result" in result
        finally:
            experiments.ACTIONS["log_result"] = original

    # ---- required params validated ----

    def test_create_missing_description(self):
        result = self._dispatch(action="create", description=None)
        assert "**Error**" in result
        assert "description" in result

    def test_write_overlay_missing_experiment_id(self):
        result = self._dispatch(
            action="write_overlay", experiment_id=None, overlay='{"models": {}}',
        )
        assert "**Error**" in result
        assert "experiment_id" in result

    def test_write_overlay_missing_overlay(self):
        result = self._dispatch(
            action="write_overlay", experiment_id="exp_001", overlay=None,
        )
        assert "**Error**" in result
        assert "overlay" in result

    def test_run_missing_experiment_id(self):
        result = self._dispatch(action="run", experiment_id=None)
        assert "**Error**" in result
        assert "experiment_id" in result

    def test_promote_missing_experiment_id(self):
        result = self._dispatch(action="promote", experiment_id=None)
        assert "**Error**" in result
        assert "experiment_id" in result

    def test_quick_run_missing_description(self):
        result = self._dispatch(
            action="quick_run", description=None, overlay='{"models": {}}',
        )
        assert "**Error**" in result
        assert "description" in result

    def test_quick_run_missing_overlay(self):
        result = self._dispatch(
            action="quick_run", description="test", overlay=None,
        )
        assert "**Error**" in result
        assert "overlay" in result

    def test_explore_missing_search_space(self):
        result = self._dispatch(action="explore", search_space=None)
        assert "**Error**" in result
        assert "search_space" in result

    def test_promote_trial_missing_experiment_id(self):
        result = self._dispatch(action="promote_trial", experiment_id=None)
        assert "**Error**" in result
        assert "experiment_id" in result

    def test_compare_missing_experiment_ids(self):
        result = self._dispatch(action="compare", experiment_ids=None)
        assert "**Error**" in result
        assert "experiment_ids" in result.lower()

    def test_compare_insufficient_experiment_ids(self):
        result = self._dispatch(action="compare", experiment_ids=["only_one"])
        assert "**Error**" in result
        assert "experiment_ids" in result.lower()

    def test_log_result_missing_experiment_id(self):
        result = self._dispatch(action="log_result", experiment_id=None)
        assert "**Error**" in result
        assert "experiment_id" in result

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.experiments import ACTIONS

        expected = {
            "create", "write_overlay", "run", "promote", "quick_run",
            "explore", "promote_trial", "compare", "journal", "log_result",
        }
        assert set(ACTIONS.keys()) == expected
