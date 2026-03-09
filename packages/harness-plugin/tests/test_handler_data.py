"""Tests for data handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Uses mocking to avoid real project/config_writer calls.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestDataDispatch:
    """Test data handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.data import dispatch

        defaults = dict(
            action="add", data_path=None, join_on=None, prefix=None,
            auto_clean=None, column=None, strategy=None, value=None,
            columns=None, mapping=None, condition=None, name=None,
            expression=None, group_by=None, dtype=None, category=None,
            source=None, steps=None, description=None, n_rows=None,
            format=None, sources=None, views=None, fraction=None,
            stratify_column=None, seed=None, project_dir=None,
            files=None, folder_id=None, folder_name=None,
            dataset_slug=None, title=None,
        )
        defaults.update(kwargs)
        return dispatch(**defaults)

    # ---- unknown action ----

    def test_dispatch_unknown_action(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_dispatch_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="profil")
        assert "**Error**" in result
        assert "profile" in result

    # ---- valid action routing ----

    def test_dispatch_valid_action_add(self):
        from harnessml.plugin.handlers import data

        mock_fn = MagicMock(return_value="Dataset added successfully")
        original = data.ACTIONS["add"]
        data.ACTIONS["add"] = mock_fn
        try:
            result = self._dispatch(action="add", data_path="/tmp/data.csv")
            mock_fn.assert_called_once()
            assert "Dataset added successfully" in result
        finally:
            data.ACTIONS["add"] = original

    def test_dispatch_valid_action_inspect(self):
        from harnessml.plugin.handlers import data

        mock_fn = MagicMock(return_value="Column stats: min=0, max=100")
        original = data.ACTIONS["inspect"]
        data.ACTIONS["inspect"] = mock_fn
        try:
            result = self._dispatch(action="inspect", column="score")
            mock_fn.assert_called_once()
            assert "Column stats" in result
        finally:
            data.ACTIONS["inspect"] = original

    def test_dispatch_valid_action_profile(self):
        from harnessml.plugin.handlers import data

        mock_fn = MagicMock(return_value="Profile: 10 columns, 500 rows")
        original = data.ACTIONS["profile"]
        data.ACTIONS["profile"] = mock_fn
        try:
            result = self._dispatch(action="profile")
            mock_fn.assert_called_once()
            assert "Profile" in result
        finally:
            data.ACTIONS["profile"] = original

    def test_dispatch_valid_action_status(self):
        from harnessml.plugin.handlers import data

        mock_fn = MagicMock(return_value="Feature store: 3 sources, 50 features")
        original = data.ACTIONS["status"]
        data.ACTIONS["status"] = mock_fn
        try:
            result = self._dispatch(action="status")
            mock_fn.assert_called_once()
            assert "Feature store" in result
        finally:
            data.ACTIONS["status"] = original

    def test_dispatch_valid_action_drop_duplicates(self, tmp_path):
        """drop_duplicates has no required params, routes through resolve_project_dir."""
        from harnessml.plugin.handlers import data

        mock_fn = MagicMock(return_value="Dropped 5 duplicate rows")
        original = data.ACTIONS["drop_duplicates"]
        data.ACTIONS["drop_duplicates"] = mock_fn
        try:
            result = self._dispatch(action="drop_duplicates")
            mock_fn.assert_called_once()
            assert "Dropped 5 duplicate rows" in result
        finally:
            data.ACTIONS["drop_duplicates"] = original

    # ---- required params validated ----

    def test_add_missing_data_path(self):
        result = self._dispatch(action="add", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result

    def test_validate_missing_data_path(self):
        result = self._dispatch(action="validate", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result

    def test_fill_nulls_missing_column(self):
        result = self._dispatch(action="fill_nulls", column=None)
        assert "**Error**" in result
        assert "column" in result

    def test_rename_missing_mapping(self):
        result = self._dispatch(action="rename", mapping=None)
        assert "**Error**" in result
        assert "mapping" in result

    def test_derive_column_missing_name(self):
        result = self._dispatch(action="derive_column", name=None, expression="a + b")
        assert "**Error**" in result
        assert "name" in result

    def test_derive_column_missing_expression(self):
        result = self._dispatch(action="derive_column", name="col_x", expression=None)
        assert "**Error**" in result
        assert "expression" in result

    def test_add_source_missing_name(self):
        result = self._dispatch(action="add_source", name=None, data_path="/tmp/x.csv")
        assert "**Error**" in result
        assert "name" in result

    def test_add_source_missing_data_path(self):
        result = self._dispatch(action="add_source", name="src1", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result

    def test_add_view_missing_name(self):
        result = self._dispatch(action="add_view", name=None, source="raw")
        assert "**Error**" in result
        assert "name" in result

    def test_add_view_missing_source(self):
        result = self._dispatch(action="add_view", name="v1", source=None)
        assert "**Error**" in result
        assert "source" in result

    def test_update_view_missing_name(self):
        result = self._dispatch(action="update_view", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_remove_view_missing_name(self):
        result = self._dispatch(action="remove_view", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_preview_view_missing_name(self):
        result = self._dispatch(action="preview_view", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_set_features_view_missing_name(self):
        result = self._dispatch(action="set_features_view", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_sample_missing_fraction(self):
        result = self._dispatch(action="sample", fraction=None)
        assert "**Error**" in result
        assert "fraction" in result

    def test_refresh_missing_name(self):
        result = self._dispatch(action="refresh", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_validate_source_missing_name(self):
        result = self._dispatch(action="validate_source", name=None)
        assert "**Error**" in result
        assert "name" in result

    def test_fetch_url_missing_data_path(self):
        result = self._dispatch(action="fetch_url", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result

    def test_add_sources_batch_missing_sources(self):
        result = self._dispatch(action="add_sources_batch", sources=None)
        assert "**Error**" in result
        assert "sources" in result.lower()

    def test_fill_nulls_batch_missing_columns(self):
        result = self._dispatch(action="fill_nulls_batch", columns=None)
        assert "**Error**" in result
        assert "columns" in result.lower()

    def test_add_views_batch_missing_views(self):
        result = self._dispatch(action="add_views_batch", views=None)
        assert "**Error**" in result
        assert "views" in result.lower()

    def test_upload_drive_missing_files(self):
        result = self._dispatch(action="upload_drive", files=None)
        assert "**Error**" in result
        assert "files" in result.lower()

    def test_upload_kaggle_missing_dataset_slug(self):
        result = self._dispatch(action="upload_kaggle", dataset_slug=None, files=["x.csv"])
        assert "**Error**" in result
        assert "dataset_slug" in result

    def test_upload_kaggle_missing_files(self):
        result = self._dispatch(action="upload_kaggle", dataset_slug="user/ds", files=None)
        assert "**Error**" in result
        assert "files" in result.lower()
