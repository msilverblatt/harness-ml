"""Tests for data handler dispatch logic.

Tests dispatch routing, unknown action errors, and required param validation.
Uses the new protomcp tool_group dispatch.
"""
from __future__ import annotations

import harnessml.plugin.handlers.data  # noqa: F401 — triggers registration
from protomcp.group import _dispatch_group_action, get_registered_groups


def _call(action_name, **kwargs):
    """Dispatch to the data tool group."""
    for group in get_registered_groups():
        if group.name == "data":
            return _dispatch_group_action(group, action=action_name, **kwargs)
    raise ValueError("Tool 'data' not found")


class TestDataDispatch:
    """Test data handler dispatch routing and validation."""

    # ---- unknown action ----

    def test_dispatch_unknown_action(self):
        result = _call("nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_dispatch_unknown_action_suggests_close_match(self):
        result = _call("profil")
        assert result.is_error
        assert "profile" in result.result

    # ---- required params validated ----

    def test_add_missing_data_path(self):
        result = _call("add", data_path=None)
        assert result.is_error

    def test_validate_missing_data_path(self):
        from harnessml.plugin.handlers.data import _handle_validate
        result = _handle_validate(data_path=None, project_dir=None)
        assert "**Error**" in result
        assert "data_path" in result

    def test_fill_nulls_missing_column(self):
        result = _call("fill_nulls", column=None)
        assert result.is_error

    def test_rename_missing_mapping(self):
        result = _call("rename", mapping=None)
        assert result.is_error

    def test_derive_column_missing_name(self):
        result = _call("derive_column", name=None, expression="a + b")
        assert result.is_error

    def test_derive_column_missing_expression(self):
        result = _call("derive_column", name="col_x", expression=None)
        assert result.is_error

    def test_add_source_missing_name(self):
        result = _call("add_source", name=None, data_path="/tmp/x.csv")
        assert result.is_error

    def test_add_source_missing_data_path(self):
        result = _call("add_source", name="src1", data_path=None)
        assert result.is_error

    def test_add_view_missing_name(self):
        result = _call("add_view", name=None, source="raw")
        assert result.is_error

    def test_add_view_missing_source(self):
        result = _call("add_view", name="v1", source=None)
        assert result.is_error

    def test_update_view_missing_name(self):
        result = _call("update_view", name=None)
        assert result.is_error

    def test_remove_view_missing_name(self):
        result = _call("remove_view", name=None)
        assert result.is_error

    def test_preview_view_missing_name(self):
        result = _call("preview_view", name=None)
        assert result.is_error

    def test_set_features_view_missing_name(self):
        result = _call("set_features_view", name=None)
        assert result.is_error

    def test_sample_missing_fraction(self):
        from harnessml.plugin.handlers.data import _handle_sample
        result = _handle_sample(fraction=None, stratify_column=None, seed=None, project_dir=None)
        assert "**Error**" in result
        assert "fraction" in result

    def test_refresh_missing_name(self):
        result = _call("refresh", name=None)
        assert result.is_error

    def test_validate_source_missing_name(self):
        result = _call("validate_source", name=None)
        assert result.is_error

    def test_fetch_url_missing_data_path(self):
        result = _call("fetch_url", data_path=None)
        assert result.is_error

    def test_add_sources_batch_missing_sources(self):
        from harnessml.plugin.handlers.data import _handle_add_sources_batch
        result = _handle_add_sources_batch(sources=None, project_dir=None)
        assert "**Error**" in result
        assert "sources" in result.lower()

    def test_fill_nulls_batch_missing_columns(self):
        from harnessml.plugin.handlers.data import _handle_fill_nulls_batch
        result = _handle_fill_nulls_batch(columns=None, project_dir=None)
        assert "**Error**" in result
        assert "columns" in result.lower()

    def test_add_views_batch_missing_views(self):
        from harnessml.plugin.handlers.data import _handle_add_views_batch
        result = _handle_add_views_batch(views=None, project_dir=None)
        assert "**Error**" in result
        assert "views" in result.lower()

    def test_upload_drive_missing_files(self):
        from harnessml.plugin.handlers.data import _handle_upload_drive
        result = _handle_upload_drive(files=None, folder_id=None, folder_name=None, name=None, project_dir=None)
        assert "**Error**" in result
        assert "files" in result.lower()

    def test_upload_kaggle_missing_dataset_slug(self):
        from harnessml.plugin.handlers.data import _handle_upload_kaggle
        result = _handle_upload_kaggle(dataset_slug=None, files=["x.csv"], title=None, name=None, project_dir=None)
        assert "**Error**" in result
        assert "dataset_slug" in result

    def test_upload_kaggle_missing_files(self):
        from harnessml.plugin.handlers.data import _handle_upload_kaggle
        result = _handle_upload_kaggle(dataset_slug="user/ds", files=None, title=None, name=None, project_dir=None)
        assert "**Error**" in result
        assert "files" in result.lower()

    # ---- all actions registered ----

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "data"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "add", "validate", "fill_nulls", "drop_duplicates", "drop_rows",
            "detect_outliers",
            "rename", "derive_column", "inspect", "profile", "list_features",
            "status", "list_sources", "add_source", "add_view", "update_view",
            "remove_view", "list_views", "preview_view", "set_features_view",
            "view_dag", "add_sources_batch", "fill_nulls_batch",
            "add_views_batch", "sample", "restore", "check_freshness",
            "refresh", "refresh_all", "validate_source", "fetch_url",
            "upload_drive", "upload_kaggle",
            "snapshot", "restore_snapshot",
        }
        assert action_names == expected
