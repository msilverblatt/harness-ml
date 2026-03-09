"""Tests for WorkflowTracker and WorkflowStatus."""
from __future__ import annotations

import json

import pytest
import yaml
from harnessml.core.runner.workflow.tracker import WorkflowGateError, WorkflowStatus, WorkflowTracker

# ---------------------------------------------------------------------------
# WorkflowStatus unit tests
# ---------------------------------------------------------------------------

class TestWorkflowStatus:
    """Tests for WorkflowStatus dataclass."""

    def test_defaults_not_ready(self):
        status = WorkflowStatus()
        assert not status.phase1_complete
        assert not status.phase2_complete
        assert not status.ready_for_tuning

    def test_phase1_complete_with_discovery(self):
        status = WorkflowStatus(feature_discovery_run=True)
        assert status.phase1_complete

    def test_phase1_not_required(self):
        status = WorkflowStatus(require_feature_discovery=False)
        assert status.phase1_complete

    def test_phase2_complete(self):
        status = WorkflowStatus(
            model_categories_tried=["boosted_tree", "linear", "bagging", "neural"],
            baseline_established=True,
        )
        assert status.phase2_complete

    def test_phase2_incomplete_few_types(self):
        status = WorkflowStatus(
            model_categories_tried=["boosted_tree"],
            baseline_established=True,
        )
        assert not status.phase2_complete

    def test_ready_for_tuning(self):
        status = WorkflowStatus(
            feature_discovery_run=True,
            model_categories_tried=["boosted_tree", "linear", "bagging", "neural"],
            baseline_established=True,
        )
        assert status.ready_for_tuning

    def test_not_ready_missing_phase1(self):
        status = WorkflowStatus(
            feature_discovery_run=False,
            model_categories_tried=["boosted_tree", "linear", "bagging", "neural"],
            baseline_established=True,
        )
        assert not status.ready_for_tuning

    def test_warnings_all_missing(self):
        status = WorkflowStatus()
        warns = status.warnings()
        assert len(warns) >= 3  # discovery, auto_search, baseline, model types

    def test_warnings_empty_when_ready(self):
        status = WorkflowStatus(
            feature_discovery_run=True,
            auto_search_run=True,
            model_categories_tried=["boosted_tree", "linear", "bagging", "neural"],
            baseline_established=True,
            diversity_analysis_run=True,
        )
        assert len(status.warnings()) == 0

    def test_format_markdown(self):
        status = WorkflowStatus(
            feature_discovery_run=True,
            model_types_tried=["xgboost", "logistic_regression"],
            model_categories_tried=["boosted_tree", "linear"],
            baseline_established=True,
            active_model_count=2,
            total_experiments_run=5,
        )
        md = status.format_markdown()
        assert "## Workflow Progress" in md
        assert "[x] Feature discovery" in md
        assert "[ ] Auto-search" in md
        assert "boosted_tree" in md
        assert "2/4 required" in md

    def test_custom_min_model_types(self):
        status = WorkflowStatus(
            model_categories_tried=["boosted_tree", "linear"],
            baseline_established=True,
            min_model_types=2,
        )
        assert status.phase2_complete


# ---------------------------------------------------------------------------
# WorkflowTracker integration tests
# ---------------------------------------------------------------------------

class TestWorkflowTracker:
    """Tests for WorkflowTracker analysis."""

    def _setup_project(self, tmp_path, models=None, journal_entries=None, has_outputs=False):
        """Create a minimal project structure."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)

        models = models or {
            "logreg": {"type": "logistic_regression", "features": ["x1"]},
        }
        (config_dir / "models.yaml").write_text(
            yaml.dump({"models": models}, default_flow_style=False)
        )
        (config_dir / "pipeline.yaml").write_text(
            yaml.dump({"data": {"features_dir": "data/features"}}, default_flow_style=False)
        )

        if journal_entries:
            experiments_dir = tmp_path / "experiments"
            experiments_dir.mkdir(parents=True)
            journal_path = experiments_dir / "journal.jsonl"
            with open(journal_path, "w") as f:
                for entry in journal_entries:
                    f.write(json.dumps(entry) + "\n")

        if has_outputs:
            outputs_dir = tmp_path / "outputs" / "20260101_000000"
            outputs_dir.mkdir(parents=True)
            (outputs_dir / "metrics.json").write_text("{}")

        return tmp_path

    def test_empty_project(self, tmp_path):
        proj = self._setup_project(tmp_path)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert not status.feature_discovery_run
        assert not status.baseline_established
        assert len(status.model_categories_tried) == 1  # linear from logreg

    def test_detects_model_types(self, tmp_path):
        models = {
            "xgb": {"type": "xgboost", "features": ["x1"]},
            "lgb": {"type": "lightgbm", "features": ["x1"]},
            "lr": {"type": "logistic_regression", "features": ["x1"]},
            "rf": {"type": "random_forest", "features": ["x1"]},
        }
        proj = self._setup_project(tmp_path, models=models)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert "xgboost" in status.model_types_tried
        assert "lightgbm" in status.model_types_tried
        assert "logistic_regression" in status.model_types_tried
        assert "random_forest" in status.model_types_tried
        assert "boosted_tree" in status.model_categories_tried
        assert "linear" in status.model_categories_tried
        assert "bagging" in status.model_categories_tried

    def test_detects_baseline(self, tmp_path):
        proj = self._setup_project(tmp_path, has_outputs=True)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert status.baseline_established

    def test_detects_feature_discovery_from_journal(self, tmp_path):
        journal = [
            {"description": "Run feature discovery", "hypothesis": "correlation analysis"},
        ]
        proj = self._setup_project(tmp_path, journal_entries=journal)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert status.feature_discovery_run

    def test_detects_auto_search_from_journal(self, tmp_path):
        journal = [
            {"description": "auto_search for interactions", "hypothesis": ""},
        ]
        proj = self._setup_project(tmp_path, journal_entries=journal)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert status.auto_search_run

    def test_counts_experiments(self, tmp_path):
        journal = [
            {"description": "Add feature X", "hypothesis": "feature test"},
            {"description": "Add feature Y", "hypothesis": "feature signal"},
            {"description": "Tune XGBoost", "hypothesis": "hyperparameter sweep"},
        ]
        proj = self._setup_project(tmp_path, journal_entries=journal)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert status.total_experiments_run == 3
        assert status.feature_experiments_run == 2
        assert status.tuning_experiments_run == 1

    def test_check_ready_soft_warning(self, tmp_path):
        proj = self._setup_project(tmp_path)
        tracker = WorkflowTracker(proj)
        result = tracker.check_ready_for_tuning(enforce=False)
        assert result is not None
        assert "Warning" in result

    def test_check_ready_hard_gate(self, tmp_path):
        proj = self._setup_project(tmp_path)
        tracker = WorkflowTracker(proj)
        with pytest.raises(WorkflowGateError):
            tracker.check_ready_for_tuning(enforce=True)

    def test_check_ready_passes_when_complete(self, tmp_path):
        models = {
            "xgb": {"type": "xgboost", "features": ["x1"]},
            "lgb": {"type": "lightgbm", "features": ["x1"]},
            "lr": {"type": "logistic_regression", "features": ["x1"]},
            "rf": {"type": "random_forest", "features": ["x1"]},
            "mlp": {"type": "mlp", "features": ["x1"]},
        }
        journal = [
            {"description": "feature discovery correlations", "hypothesis": ""},
        ]
        proj = self._setup_project(tmp_path, models=models, journal_entries=journal, has_outputs=True)
        tracker = WorkflowTracker(proj)
        result = tracker.check_ready_for_tuning(enforce=True)
        assert result is None

    def test_custom_config_thresholds(self, tmp_path):
        models = {
            "xgb": {"type": "xgboost", "features": ["x1"]},
            "lr": {"type": "logistic_regression", "features": ["x1"]},
        }
        proj = self._setup_project(tmp_path, models=models, has_outputs=True)
        journal = [
            {"description": "feature discovery", "hypothesis": "correlation check"},
        ]
        # Add journal
        experiments_dir = proj / "experiments"
        experiments_dir.mkdir(parents=True, exist_ok=True)
        with open(experiments_dir / "journal.jsonl", "w") as f:
            for entry in journal:
                f.write(json.dumps(entry) + "\n")

        tracker = WorkflowTracker(
            proj,
            workflow_config={"min_model_types": 2, "require_feature_discovery": True},
        )
        result = tracker.check_ready_for_tuning(enforce=False)
        assert result is None  # 2 categories (boosted_tree, linear) meets min_model_types=2

    def test_inactive_models_not_counted(self, tmp_path):
        models = {
            "xgb": {"type": "xgboost", "features": ["x1"], "active": False},
            "lr": {"type": "logistic_regression", "features": ["x1"]},
        }
        proj = self._setup_project(tmp_path, models=models)
        tracker = WorkflowTracker(proj)
        status = tracker.get_status()
        assert status.active_model_count == 1
        # But types should still be counted (they were tried)
        assert "xgboost" in status.model_types_tried
