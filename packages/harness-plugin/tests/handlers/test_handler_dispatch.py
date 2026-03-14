"""Tests for MCP handler dispatch functions (models, data, config, pipeline).

Tests the dispatch layer: action routing, unknown action errors, and required
param validation. Uses the new protomcp tool_group dispatch.
"""
from __future__ import annotations

import harnessml.plugin.handlers.config  # noqa: F401
import harnessml.plugin.handlers.data  # noqa: F401

# Trigger all registrations
import harnessml.plugin.handlers.models  # noqa: F401
import harnessml.plugin.handlers.pipeline  # noqa: F401
from protomcp.group import _dispatch_group_action, get_registered_groups


def _call(tool_name, action_name, **kwargs):
    """Dispatch to a tool group by name."""
    for group in get_registered_groups():
        if group.name == tool_name:
            return _dispatch_group_action(group, action=action_name, **kwargs)
    raise ValueError(f"Tool {tool_name} not found")


# ---------------------------------------------------------------------------
# Models handler
# ---------------------------------------------------------------------------

class TestModelsDispatch:
    """Test models handler dispatch routing and validation."""

    def test_unknown_action_returns_error(self):
        result = _call("models", "nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_unknown_action_suggests_close_match(self):
        result = _call("models", "ad")
        assert result.is_error
        assert "add" in result.result

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "models"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "add", "update", "remove", "list", "show", "presets",
            "add_batch", "update_batch", "remove_batch", "clone",
        }
        assert action_names == expected

    def test_add_missing_name_returns_error(self):
        result = _call("models", "add", name=None, model_type="xgboost")
        # protomcp requires= validation returns ToolResult with is_error
        assert result.is_error

    def test_update_missing_name_returns_error(self):
        result = _call("models", "update", name=None)
        assert result.is_error

    def test_remove_missing_name_returns_error(self):
        result = _call("models", "remove", name=None, purge=False)
        assert result.is_error

    def test_show_missing_name_returns_error(self):
        result = _call("models", "show", name=None)
        assert result.is_error

    def test_add_batch_missing_items_returns_error(self):
        from harnessml.plugin.handlers.models import _handle_add_batch
        result = _handle_add_batch(items=None, project_dir=None)
        assert "**Error**" in result
        assert "items" in result.lower()


# ---------------------------------------------------------------------------
# Data handler
# ---------------------------------------------------------------------------

class TestDataDispatch:
    """Test data handler dispatch routing and validation."""

    def test_unknown_action_returns_error(self):
        result = _call("data", "nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_unknown_action_suggests_close_match(self):
        result = _call("data", "profil")
        assert result.is_error
        assert "profile" in result.result

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

    def test_add_missing_data_path_returns_error(self):
        result = _call("data", "add", data_path=None)
        assert result.is_error

    def test_derive_column_missing_name_returns_error(self):
        result = _call("data", "derive_column", name=None, expression="a + b")
        assert result.is_error

    def test_derive_column_missing_expression_returns_error(self):
        result = _call("data", "derive_column", name="col_x", expression=None)
        assert result.is_error

    def test_add_source_missing_name_returns_error(self):
        result = _call("data", "add_source", name=None, data_path="/tmp/x.csv")
        assert result.is_error

    def test_add_source_missing_data_path_returns_error(self):
        result = _call("data", "add_source", name="src1", data_path=None)
        assert result.is_error

    def test_sample_missing_fraction_returns_error(self):
        from harnessml.plugin.handlers.data import _handle_sample
        result = _handle_sample(fraction=None, stratify_column=None, seed=None, project_dir=None)
        assert "**Error**" in result
        assert "fraction" in result.lower()

    def test_add_sources_batch_missing_sources_returns_error(self):
        from harnessml.plugin.handlers.data import _handle_add_sources_batch
        result = _handle_add_sources_batch(sources=None, project_dir=None)
        assert "**Error**" in result
        assert "sources" in result.lower()

    def test_add_views_batch_missing_views_returns_error(self):
        from harnessml.plugin.handlers.data import _handle_add_views_batch
        result = _handle_add_views_batch(views=None, project_dir=None)
        assert "**Error**" in result
        assert "views" in result.lower()

    def test_refresh_missing_name_returns_error(self):
        result = _call("data", "refresh", name=None)
        assert result.is_error

    def test_fetch_url_missing_data_path_returns_error(self):
        result = _call("data", "fetch_url", data_path=None)
        assert result.is_error


# ---------------------------------------------------------------------------
# Config handler
# ---------------------------------------------------------------------------

class TestConfigDispatch:
    """Test config handler dispatch routing and validation."""

    def test_unknown_action_returns_error(self):
        result = _call("configure", "nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_unknown_action_suggests_close_match(self):
        result = _call("configure", "sho")
        assert result.is_error
        assert "show" in result.result

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "configure"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "init", "update_data", "ensemble", "backtest", "show",
            "check_guardrails", "exclude_columns", "set_denylist",
            "add_target", "list_targets", "set_target", "studio",
            "suggest_cv",
        }
        assert action_names == expected

    def test_add_target_missing_name_returns_error(self):
        result = _call("configure", "add_target", name=None, target_column="result")
        assert result.is_error

    def test_add_target_missing_target_column_returns_error(self):
        result = _call("configure", "add_target", name="win", target_column=None)
        assert result.is_error

    def test_set_target_missing_name_returns_error(self):
        result = _call("configure", "set_target", name=None)
        assert result.is_error


# ---------------------------------------------------------------------------
# Pipeline handler
# ---------------------------------------------------------------------------

class TestPipelineDispatch:
    """Test pipeline handler dispatch routing and validation."""

    def test_unknown_action_returns_error(self):
        result = _call("pipeline", "nonexistent")
        assert result.is_error
        assert "nonexistent" in result.result

    def test_unknown_action_suggests_close_match(self):
        result = _call("pipeline", "diagnotics")
        assert result.is_error
        assert "diagnostics" in result.result

    def test_all_actions_registered(self):
        groups = [g for g in get_registered_groups() if g.name == "pipeline"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "progress", "run_backtest", "predict", "diagnostics", "list_runs",
            "show_run", "compare_runs", "compare_latest", "compare_targets",
            "explain", "inspect_predictions", "export_notebook", "clear_cache",
            "model_correlation", "residual_analysis",
        }
        assert action_names == expected

    def test_predict_missing_fold_value_returns_error(self):
        result = _call("pipeline", "predict", fold_value=None)
        assert result.is_error

    def test_compare_runs_missing_run_ids_returns_error(self):
        from harnessml.plugin.handlers.pipeline import _handle_compare_runs
        result = _handle_compare_runs(run_ids=None, project_dir=None)
        assert "**Error**" in result
        assert "run_ids" in result.lower()

    def test_compare_runs_insufficient_run_ids_returns_error(self):
        from harnessml.plugin.handlers.pipeline import _handle_compare_runs
        result = _handle_compare_runs(run_ids=["only_one"], project_dir=None)
        assert "**Error**" in result
        assert "run_ids" in result.lower()

    def test_export_notebook_missing_destination_returns_error(self):
        result = _call("pipeline", "export_notebook", destination=None)
        assert result.is_error

    def test_export_notebook_invalid_destination_returns_error(self):
        from harnessml.plugin.handlers.pipeline import _handle_export_notebook
        result = _handle_export_notebook(destination="invalid_dest", output_path=None, project_dir=None)
        assert "**Error**" in result
        assert "invalid_dest" in result
