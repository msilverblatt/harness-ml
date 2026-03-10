"""Tests for models handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Uses mocking to avoid real project/config_writer calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock


class TestModelsDispatch:
    """Test models handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.models import dispatch

        defaults = dict(
            action="add", name=None, new_name=None, model_type=None, preset=None,
            features=None, params=None, active=None, include_in_ensemble=None,
            mode=None, prediction_type=None, cdf_scale=None,
            zero_fill_features=None, items=None, replace_params=None,
            purge=None, project_dir=None,
        )
        defaults.update(kwargs)
        return dispatch(**defaults)

    # ---- unknown action ----

    def test_dispatch_unknown_action(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_dispatch_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="ad")
        assert "**Error**" in result
        assert "add" in result

    # ---- valid action routing ----

    def test_dispatch_valid_action_add(self):
        from harnessml.plugin.handlers import models

        mock_fn = MagicMock(return_value="Added model xgb_1")
        original = models.ACTIONS["add"]
        models.ACTIONS["add"] = mock_fn
        try:
            result = self._dispatch(action="add", name="xgb_1", model_type="xgboost")
            mock_fn.assert_called_once()
            assert "Added model xgb_1" in result
        finally:
            models.ACTIONS["add"] = original

    def test_dispatch_valid_action_list(self):
        from harnessml.plugin.handlers import models

        mock_fn = MagicMock(return_value="Models: xgb_1, lgb_1")
        original = models.ACTIONS["list"]
        models.ACTIONS["list"] = mock_fn
        try:
            result = self._dispatch(action="list")
            mock_fn.assert_called_once()
            assert "Models: xgb_1, lgb_1" in result
        finally:
            models.ACTIONS["list"] = original

    def test_dispatch_valid_action_presets(self):
        from harnessml.plugin.handlers import models

        mock_fn = MagicMock(return_value="Available presets: default, fast")
        original = models.ACTIONS["presets"]
        models.ACTIONS["presets"] = mock_fn
        try:
            result = self._dispatch(action="presets")
            mock_fn.assert_called_once()
            assert "Available presets" in result
        finally:
            models.ACTIONS["presets"] = original

    def test_dispatch_valid_action_show(self):
        from harnessml.plugin.handlers import models

        mock_fn = MagicMock(return_value="xgb_1: type=xgboost, active=true")
        original = models.ACTIONS["show"]
        models.ACTIONS["show"] = mock_fn
        try:
            result = self._dispatch(action="show", name="xgb_1")
            mock_fn.assert_called_once()
            assert "xgb_1" in result
        finally:
            models.ACTIONS["show"] = original

    def test_dispatch_valid_action_clone(self):
        from harnessml.plugin.handlers import models

        mock_fn = MagicMock(return_value="Cloned model xgb_1 as xgb_2")
        original = models.ACTIONS["clone"]
        models.ACTIONS["clone"] = mock_fn
        try:
            result = self._dispatch(
                action="clone", name="xgb_1",
                new_name="xgb_2",
            )
            mock_fn.assert_called_once()
            assert "Cloned" in result
        finally:
            models.ACTIONS["clone"] = original

    # ---- required params validated ----

    def test_add_missing_name(self):
        result = self._dispatch(action="add", name=None, model_type="xgboost")
        assert "**Error**" in result
        assert "name" in result

    def test_update_missing_name(self):
        result = self._dispatch(action="update", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_remove_missing_name(self):
        result = self._dispatch(action="remove", name=None, purge=False)
        assert "**Error**" in result
        assert "name" in result

    def test_show_missing_name(self):
        result = self._dispatch(action="show", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_clone_missing_name(self):
        result = self._dispatch(action="clone", name=None, new_name="x")
        assert "**Error**" in result
        assert "name" in result

    def test_clone_missing_new_name(self):
        result = self._dispatch(action="clone", name="xgb_1", new_name=None)
        assert "**Error**" in result
        assert "new_name" in result.lower()

    def test_add_batch_missing_items(self):
        result = self._dispatch(action="add_batch", items=None)
        assert "**Error**" in result
        assert "items" in result.lower()

    def test_update_batch_missing_items(self):
        result = self._dispatch(action="update_batch", items=None)
        assert "**Error**" in result
        assert "items" in result.lower()

    def test_remove_batch_missing_items(self):
        result = self._dispatch(action="remove_batch", items=None)
        assert "**Error**" in result
        assert "items" in result.lower()

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.models import ACTIONS

        expected = {
            "add", "update", "remove", "list", "show", "presets",
            "add_batch", "update_batch", "remove_batch", "clone",
        }
        assert set(ACTIONS.keys()) == expected
