"""Tests for MCP handler dispatch functions (models, data, config, pipeline).

Tests the dispatch layer: action routing, unknown action errors, and required
param validation. Uses mocking to avoid real project/config_writer calls.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Models handler
# ---------------------------------------------------------------------------

class TestModelsDispatch:
    """Test models handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.models import dispatch
        defaults = dict(
            action="add", name=None, model_type=None, preset=None,
            features=None, params=None, active=None, include_in_ensemble=None,
            mode=None, prediction_type=None, cdf_scale=None,
            zero_fill_features=None, items=None, replace_params=None,
            project_dir=None,
        )
        defaults.update(kwargs)
        return dispatch(**defaults)

    def test_unknown_action_returns_error(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="ad")
        assert "**Error**" in result
        assert "add" in result

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.models import ACTIONS
        expected = {
            "add", "update", "remove", "list", "show", "presets",
            "add_batch", "update_batch", "remove_batch", "clone",
        }
        assert set(ACTIONS.keys()) == expected

    def test_add_missing_name_returns_error(self):
        result = self._dispatch(action="add", name=None, model_type="xgboost")
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_update_missing_name_returns_error(self):
        result = self._dispatch(action="update", name=None)
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_remove_missing_name_returns_error(self):
        result = self._dispatch(action="remove", name=None, purge=False)
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_show_missing_name_returns_error(self):
        result = self._dispatch(action="show", name=None)
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_dispatch_routes_to_add(self):
        from harnessml.plugin.handlers import models
        mock_add = MagicMock(return_value="Added model xgb_1")
        original = models.ACTIONS["add"]
        models.ACTIONS["add"] = mock_add
        try:
            result = self._dispatch(action="add", name="xgb_1", model_type="xgboost")
            mock_add.assert_called_once()
            assert "Added model xgb_1" in result
        finally:
            models.ACTIONS["add"] = original

    def test_dispatch_routes_to_list(self):
        from harnessml.plugin.handlers import models
        mock_list = MagicMock(return_value="Models: xgb_1, lgb_1")
        original = models.ACTIONS["list"]
        models.ACTIONS["list"] = mock_list
        try:
            result = self._dispatch(action="list")
            mock_list.assert_called_once()
            assert "Models: xgb_1, lgb_1" in result
        finally:
            models.ACTIONS["list"] = original

    def test_dispatch_routes_to_presets(self):
        from harnessml.plugin.handlers import models
        mock_presets = MagicMock(return_value="Available presets: default, fast")
        original = models.ACTIONS["presets"]
        models.ACTIONS["presets"] = mock_presets
        try:
            result = self._dispatch(action="presets")
            mock_presets.assert_called_once()
            assert "Available presets" in result
        finally:
            models.ACTIONS["presets"] = original

    def test_add_batch_missing_items_returns_error(self):
        result = self._dispatch(action="add_batch", items=None)
        assert "**Error**" in result
        assert "items" in result.lower()


# ---------------------------------------------------------------------------
# Data handler
# ---------------------------------------------------------------------------

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

    def test_unknown_action_returns_error(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="profil")
        assert "**Error**" in result
        assert "profile" in result

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.data import ACTIONS
        expected = {
            "add", "validate", "fill_nulls", "drop_duplicates", "drop_rows",
            "rename", "derive_column", "inspect", "profile", "list_features",
            "status", "list_sources", "add_source", "add_view", "update_view",
            "remove_view", "list_views", "preview_view", "set_features_view",
            "view_dag", "add_sources_batch", "fill_nulls_batch",
            "add_views_batch", "sample", "restore", "check_freshness",
            "refresh", "refresh_all", "validate_source", "fetch_url",
            "upload_drive", "upload_kaggle",
        }
        assert set(ACTIONS.keys()) == expected

    def test_add_missing_data_path_returns_error(self):
        result = self._dispatch(action="add", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result.lower()

    def test_derive_column_missing_name_returns_error(self):
        result = self._dispatch(action="derive_column", name=None, expression="a + b")
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_derive_column_missing_expression_returns_error(self):
        result = self._dispatch(action="derive_column", name="col_x", expression=None)
        assert "**Error**" in result
        assert "expression" in result.lower()

    def test_add_source_missing_name_returns_error(self):
        result = self._dispatch(action="add_source", name=None, data_path="/tmp/x.csv")
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_add_source_missing_data_path_returns_error(self):
        result = self._dispatch(action="add_source", name="src1", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result.lower()

    def test_sample_missing_fraction_returns_error(self):
        result = self._dispatch(action="sample", fraction=None)
        assert "**Error**" in result
        assert "fraction" in result.lower()

    def test_dispatch_routes_to_add(self):
        from harnessml.plugin.handlers import data
        mock_add = MagicMock(return_value="Dataset added")
        original = data.ACTIONS["add"]
        data.ACTIONS["add"] = mock_add
        try:
            result = self._dispatch(action="add", data_path="/tmp/data.csv")
            mock_add.assert_called_once()
            assert "Dataset added" in result
        finally:
            data.ACTIONS["add"] = original

    def test_dispatch_routes_to_inspect(self):
        from harnessml.plugin.handlers import data
        mock_inspect = MagicMock(return_value="Column stats: ...")
        original = data.ACTIONS["inspect"]
        data.ACTIONS["inspect"] = mock_inspect
        try:
            result = self._dispatch(action="inspect")
            mock_inspect.assert_called_once()
            assert "Column stats" in result
        finally:
            data.ACTIONS["inspect"] = original

    def test_add_sources_batch_missing_sources_returns_error(self):
        result = self._dispatch(action="add_sources_batch", sources=None)
        assert "**Error**" in result
        assert "sources" in result.lower()

    def test_add_views_batch_missing_views_returns_error(self):
        result = self._dispatch(action="add_views_batch", views=None)
        assert "**Error**" in result
        assert "views" in result.lower()

    def test_refresh_missing_name_returns_error(self):
        result = self._dispatch(action="refresh", name=None)
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_fetch_url_missing_data_path_returns_error(self):
        result = self._dispatch(action="fetch_url", data_path=None)
        assert "**Error**" in result
        assert "data_path" in result.lower()


# ---------------------------------------------------------------------------
# Config handler
# ---------------------------------------------------------------------------

class TestConfigDispatch:
    """Test config handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.config import dispatch
        defaults = dict(
            action="show", project_name=None, task=None, target_column=None,
            key_columns=None, time_column=None, method=None, temperature=None,
            exclude_models=None, calibration=None, pre_calibration=None,
            prior_feature=None, spline_prob_max=None, spline_n_bins=None,
            cv_strategy=None, fold_values=None, metrics=None,
            min_train_folds=None, fold_column=None, detail=None,
            section=None, add_columns=None, remove_columns=None,
            name=None, project_dir=None,
        )
        defaults.update(kwargs)
        return dispatch(**defaults)

    def test_unknown_action_returns_error(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="sho")
        assert "**Error**" in result
        assert "show" in result

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.config import ACTIONS
        expected = {
            "init", "update_data", "ensemble", "backtest", "show",
            "check_guardrails", "exclude_columns", "set_denylist",
            "add_target", "list_targets", "set_target",
        }
        assert set(ACTIONS.keys()) == expected

    def test_add_target_missing_name_returns_error(self):
        result = self._dispatch(action="add_target", name=None, target_column="result")
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_add_target_missing_target_column_returns_error(self):
        result = self._dispatch(action="add_target", name="win", target_column=None)
        assert "**Error**" in result
        assert "target_column" in result.lower()

    def test_set_target_missing_name_returns_error(self):
        result = self._dispatch(action="set_target", name=None)
        assert "**Error**" in result
        assert "name" in result.lower()

    def test_dispatch_routes_to_show(self):
        from harnessml.plugin.handlers import config
        mock_show = MagicMock(return_value="project:\n  name: test")
        original = config.ACTIONS["show"]
        config.ACTIONS["show"] = mock_show
        try:
            result = self._dispatch(action="show")
            mock_show.assert_called_once()
            assert "project:" in result
        finally:
            config.ACTIONS["show"] = original

    def test_dispatch_routes_to_init(self):
        from harnessml.plugin.handlers import config
        mock_init = MagicMock(return_value="Project initialized at /tmp/test")
        original = config.ACTIONS["init"]
        config.ACTIONS["init"] = mock_init
        try:
            result = self._dispatch(action="init", project_name="test")
            mock_init.assert_called_once()
            assert "Project initialized" in result
        finally:
            config.ACTIONS["init"] = original

    def test_dispatch_routes_to_ensemble(self):
        from harnessml.plugin.handlers import config
        mock_ens = MagicMock(return_value="Ensemble configured")
        original = config.ACTIONS["ensemble"]
        config.ACTIONS["ensemble"] = mock_ens
        try:
            result = self._dispatch(action="ensemble", method="stacking")
            mock_ens.assert_called_once()
            assert "Ensemble configured" in result
        finally:
            config.ACTIONS["ensemble"] = original


# ---------------------------------------------------------------------------
# Pipeline handler (async dispatch)
# ---------------------------------------------------------------------------

class TestPipelineDispatch:
    """Test pipeline handler dispatch routing and validation."""

    def _dispatch(self, **kwargs):
        from harnessml.plugin.handlers.pipeline import dispatch
        defaults = dict(
            action="list_runs", fold_value=None, run_id=None, run_ids=None,
            variant=None, experiment_id=None, detail=None, name=None,
            top_n=None, destination=None, output_path=None, mode=None,
            ctx=None, project_dir=None,
        )
        defaults.update(kwargs)
        return asyncio.run(dispatch(**defaults))

    def test_unknown_action_returns_error(self):
        result = self._dispatch(action="nonexistent")
        assert "**Error**" in result
        assert "nonexistent" in result

    def test_unknown_action_suggests_close_match(self):
        result = self._dispatch(action="diagnotics")
        assert "**Error**" in result
        assert "diagnostics" in result

    def test_all_actions_registered(self):
        from harnessml.plugin.handlers.pipeline import ACTIONS
        expected = {
            "progress", "run_backtest", "predict", "diagnostics", "list_runs",
            "show_run", "compare_runs", "compare_latest", "compare_targets",
            "explain", "inspect_predictions", "export_notebook",
        }
        assert set(ACTIONS.keys()) == expected

    def test_predict_missing_fold_value_returns_error(self):
        result = self._dispatch(action="predict", fold_value=None)
        assert "**Error**" in result
        assert "fold_value" in result.lower()

    def test_compare_runs_missing_run_ids_returns_error(self):
        result = self._dispatch(action="compare_runs", run_ids=None)
        assert "**Error**" in result
        assert "run_ids" in result.lower()

    def test_compare_runs_insufficient_run_ids_returns_error(self):
        result = self._dispatch(action="compare_runs", run_ids=["only_one"])
        assert "**Error**" in result
        assert "run_ids" in result.lower()

    def test_export_notebook_missing_destination_returns_error(self):
        result = self._dispatch(action="export_notebook", destination=None)
        assert "**Error**" in result
        assert "destination" in result.lower()

    def test_export_notebook_invalid_destination_returns_error(self):
        result = self._dispatch(action="export_notebook", destination="invalid_dest")
        assert "**Error**" in result
        assert "invalid_dest" in result

    def test_dispatch_routes_to_list_runs(self):
        from harnessml.plugin.handlers import pipeline
        mock_list = MagicMock(return_value="No runs found.")
        original = pipeline.ACTIONS["list_runs"]
        pipeline.ACTIONS["list_runs"] = mock_list
        try:
            result = self._dispatch(action="list_runs")
            mock_list.assert_called_once()
            assert "No runs found" in result
        finally:
            pipeline.ACTIONS["list_runs"] = original

    def test_dispatch_routes_to_diagnostics(self):
        from harnessml.plugin.handlers import pipeline
        mock_diag = MagicMock(return_value="## Diagnostics\nBrier: 0.14")
        original = pipeline.ACTIONS["diagnostics"]
        pipeline.ACTIONS["diagnostics"] = mock_diag
        try:
            result = self._dispatch(action="diagnostics")
            mock_diag.assert_called_once()
            assert "Diagnostics" in result
        finally:
            pipeline.ACTIONS["diagnostics"] = original
