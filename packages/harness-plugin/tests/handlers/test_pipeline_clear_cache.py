"""Tests for pipeline(action='clear_cache') handler."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def project_with_cache(tmp_path):
    """Set up a project dir with a populated prediction cache."""
    from harnessml.core.runner.training.prediction_cache import PredictionCache

    project_dir = tmp_path / "project"
    (project_dir / "config").mkdir(parents=True)
    cache_dir = project_dir / ".cache" / "predictions"

    cache = PredictionCache(cache_dir)
    preds = pd.DataFrame({"prediction": [0.1, 0.2, 0.3]})
    cache.store("model_a", 2024, "fp1", preds)
    cache.store("model_b", 2024, "fp2", preds)

    return project_dir


# -----------------------------------------------------------------------
# Handler dispatch
# -----------------------------------------------------------------------

class TestClearCacheAction:

    def test_clear_cache_registered_in_actions(self):
        import harnessml.plugin.handlers.pipeline  # noqa: F401
        from protomcp.group import get_registered_groups

        groups = [g for g in get_registered_groups() if g.name == "pipeline"]
        action_names = {a.name for a in groups[0].actions}
        assert "clear_cache" in action_names

    def test_clear_cache_removes_entries(self, project_with_cache):
        from harnessml.core.runner.training.prediction_cache import PredictionCache
        from harnessml.plugin.handlers.pipeline import _handle_clear_cache

        result = _handle_clear_cache(project_dir=str(project_with_cache))

        assert "2" in result  # 2 entries removed
        assert "cleared" in result.lower()

        # Verify cache is actually empty
        cache_dir = project_with_cache / ".cache" / "predictions"
        cache = PredictionCache(cache_dir)
        assert cache.lookup("model_a", 2024, "fp1") is None
        assert cache.lookup("model_b", 2024, "fp2") is None

    def test_clear_cache_no_cache_dir(self, tmp_path):
        """No error when cache dir doesn't exist."""
        from harnessml.plugin.handlers.pipeline import _handle_clear_cache

        project_dir = tmp_path / "empty_project"
        project_dir.mkdir()
        (project_dir / "config").mkdir()

        result = _handle_clear_cache(project_dir=str(project_dir))

        assert "nothing to clear" in result.lower() or "no prediction cache" in result.lower()

    def test_all_pipeline_actions_registered(self):
        """Updated action registry includes clear_cache."""
        import harnessml.plugin.handlers.pipeline  # noqa: F401
        from protomcp.group import get_registered_groups

        groups = [g for g in get_registered_groups() if g.name == "pipeline"]
        action_names = {a.name for a in groups[0].actions}
        expected = {
            "progress", "run_backtest", "predict", "diagnostics", "list_runs",
            "show_run", "compare_runs", "compare_latest", "compare_targets",
            "explain", "inspect_predictions", "export_notebook", "clear_cache",
            "model_correlation", "residual_analysis",
        }
        assert action_names == expected
