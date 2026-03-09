"""Tests for features handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Features handler has an async dispatch, so tests use asyncio.run().
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock


class TestFeaturesDispatch:
    """Test features handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.features import dispatch

        defaults = dict(
            action="add", name=None, formula=None, type=None,
            source=None, column=None, condition=None,
            pairwise_mode=None, category=None, description=None,
            features=None, test_interactions=None, top_n=None,
            method=None, search_types=None, ctx=None, project_dir=None,
        )
        defaults.update(kwargs)
        return asyncio.run(dispatch(**defaults))

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
        from harnessml.plugin.handlers import features

        mock_fn = MagicMock(return_value="Feature win_pct added")
        original = features.ACTIONS["add"]
        features.ACTIONS["add"] = mock_fn
        try:
            result = self._dispatch(
                action="add", name="win_pct", formula="wins / games",
            )
            mock_fn.assert_called_once()
            assert "Feature win_pct added" in result
        finally:
            features.ACTIONS["add"] = original

    def test_dispatch_valid_action_add_batch(self):
        from harnessml.plugin.handlers import features

        mock_fn = MagicMock(return_value="Added 3 features")
        original = features.ACTIONS["add_batch"]
        features.ACTIONS["add_batch"] = mock_fn
        try:
            result = self._dispatch(
                action="add_batch",
                features='[{"name": "f1", "formula": "a+b"}]',
            )
            mock_fn.assert_called_once()
            assert "Added 3 features" in result
        finally:
            features.ACTIONS["add_batch"] = original

    def test_dispatch_valid_action_diversity(self):
        from harnessml.plugin.handlers import features

        mock_fn = MagicMock(return_value="Diversity report: 5 unique feature sets")
        original = features.ACTIONS["diversity"]
        features.ACTIONS["diversity"] = mock_fn
        try:
            result = self._dispatch(action="diversity")
            mock_fn.assert_called_once()
            assert "Diversity report" in result
        finally:
            features.ACTIONS["diversity"] = original

    def test_dispatch_valid_action_auto_search(self):
        from harnessml.plugin.handlers import features

        mock_fn = MagicMock(return_value="Found 10 candidate features")
        original = features.ACTIONS["auto_search"]
        features.ACTIONS["auto_search"] = mock_fn
        try:
            result = self._dispatch(action="auto_search")
            mock_fn.assert_called_once()
            assert "Found 10 candidate features" in result
        finally:
            features.ACTIONS["auto_search"] = original

    # ---- required params validated ----

    def test_add_missing_name(self):
        result = self._dispatch(action="add", name=None, formula="a + b")
        assert "**Error**" in result
        assert "name" in result

    def test_add_missing_all_definition_fields(self):
        """add requires at least one of: formula, type, source, condition."""
        result = self._dispatch(
            action="add", name="feat1",
            formula=None, type=None, source=None, condition=None,
        )
        assert "**Error**" in result

    def test_add_batch_missing_features(self):
        result = self._dispatch(action="add_batch", features=None)
        assert "**Error**" in result
        assert "features" in result

    def test_test_transformations_missing_features(self):
        result = self._dispatch(action="test_transformations", features=None)
        assert "**Error**" in result
        assert "features" in result

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.features import ACTIONS

        expected = {
            "add", "add_batch", "test_transformations",
            "discover", "diversity", "auto_search",
        }
        assert set(ACTIONS.keys()) == expected
