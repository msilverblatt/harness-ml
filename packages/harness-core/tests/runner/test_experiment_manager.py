"""Tests for ExperimentManager — creation, change detection, logging, do-not-retry, promotion."""

from __future__ import annotations

import json

import pytest
import yaml
from harnessml.core.runner.experiment_manager import (
    ChangeReport,
    ExperimentError,
    ExperimentManager,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path, *, log_path=None, do_not_retry_path=None, naming_pattern=None):
    """Build an ExperimentManager rooted in tmp_path."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    return ExperimentManager(
        experiments_dir=experiments_dir,
        log_path=log_path,
        do_not_retry_path=do_not_retry_path,
        naming_pattern=naming_pattern,
    )


def _write_pipeline_yaml(path, config):
    """Write a minimal pipeline YAML file."""
    path.write_text(yaml.dump(config, default_flow_style=False))


# ---------------------------------------------------------------------------
# 1. test_create_experiment
# ---------------------------------------------------------------------------


class TestCreateExperiment:
    def test_creates_directory_and_overlay(self, tmp_path):
        mgr = _make_manager(tmp_path)
        exp_dir = mgr.create("exp-001")

        assert exp_dir.exists()
        assert exp_dir.is_dir()
        assert (exp_dir / "overlay.yaml").exists()

    def test_naming_pattern_accepted(self, tmp_path):
        mgr = _make_manager(tmp_path, naming_pattern=r"^exp-\d{3}$")
        exp_dir = mgr.create("exp-001")
        assert exp_dir.exists()

    def test_naming_pattern_rejected(self, tmp_path):
        mgr = _make_manager(tmp_path, naming_pattern=r"^exp-\d{3}$")
        with pytest.raises(ExperimentError, match="naming pattern"):
            mgr.create("bad_name")

    def test_duplicate_raises(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.create("exp-001")
        with pytest.raises(ExperimentError, match="already exists"):
            mgr.create("exp-001")


# ---------------------------------------------------------------------------
# 2. test_create_experiment_requires_hypothesis (mandatory logging)
# ---------------------------------------------------------------------------


class TestCreateRequiresLogging:
    def test_blocks_when_unlogged_experiment_exists(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        log_path.touch()
        mgr = _make_manager(tmp_path, log_path=log_path)

        # Create first experiment (no log entry yet)
        mgr.create("exp-001")

        # Second create should be blocked because exp-001 is unlogged
        with pytest.raises(ExperimentError, match="not been logged"):
            mgr.create("exp-002")

    def test_allows_when_all_logged(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        log_path.touch()
        mgr = _make_manager(tmp_path, log_path=log_path)

        mgr.create("exp-001")
        # Log it
        mgr.log("exp-001", hypothesis="test", changes="none", verdict="keep")

        # Now creating another should succeed
        exp_dir = mgr.create("exp-002")
        assert exp_dir.exists()


# ---------------------------------------------------------------------------
# 3. test_list_experiments (via has_unlogged / internal helpers)
# ---------------------------------------------------------------------------


class TestListExperiments:
    def test_list_created_experiments(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.create("exp-001")
        mgr.create("exp-002")
        mgr.create("exp-003")

        created = mgr._list_created_experiments()
        assert created == {"exp-001", "exp-002", "exp-003"}

    def test_list_created_empty(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr._list_created_experiments() == set()

    def test_list_logged_experiments(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        log_path.touch()
        mgr = _make_manager(tmp_path, log_path=log_path)

        mgr.create("exp-001")
        mgr.log("exp-001", hypothesis="h1", changes="c1", verdict="keep")

        logged = mgr._list_logged_experiments()
        assert "exp-001" in logged

    def test_has_unlogged_true(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        log_path.touch()
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")
        assert mgr.has_unlogged() is True

    def test_has_unlogged_false(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        log_path.touch()
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")
        mgr.log("exp-001", hypothesis="h", changes="c", verdict="revert")
        assert mgr.has_unlogged() is False


# ---------------------------------------------------------------------------
# 4. test_detect_changes
# ---------------------------------------------------------------------------


class TestDetectChanges:
    def test_no_changes(self, tmp_path):
        mgr = _make_manager(tmp_path)
        prod = {"models": {"xgb": {"lr": 0.1}}}
        overlay = {}
        report = mgr.detect_changes(prod, overlay)
        assert report.total_changes == 0

    def test_changed_model(self, tmp_path):
        mgr = _make_manager(tmp_path)
        prod = {"models": {"xgb": {"lr": 0.1}}}
        overlay = {"models": {"xgb": {"lr": 0.05}}}
        report = mgr.detect_changes(prod, overlay)
        assert "xgb" in report.changed_models

    def test_new_model(self, tmp_path):
        mgr = _make_manager(tmp_path)
        prod = {"models": {"xgb": {"lr": 0.1}}}
        overlay = {"models": {"xgb": {"lr": 0.1}, "lgb": {"lr": 0.2}}}
        report = mgr.detect_changes(prod, overlay)
        assert "lgb" in report.new_models

    def test_removed_model(self, tmp_path):
        mgr = _make_manager(tmp_path)
        prod = {"models": {"xgb": {"lr": 0.1}, "lgb": {"lr": 0.2}}}
        overlay = {"models": {"xgb": {"lr": 0.1}}}
        report = mgr.detect_changes(prod, overlay)
        assert "lgb" in report.removed_models

    def test_ensemble_change(self, tmp_path):
        mgr = _make_manager(tmp_path)
        prod = {"ensemble": {"method": "stack"}}
        overlay = {"ensemble": {"method": "blend"}}
        report = mgr.detect_changes(prod, overlay)
        assert "method" in report.ensemble_changes

    def test_total_changes_counts_all(self, tmp_path):
        mgr = _make_manager(tmp_path)
        prod = {"models": {"xgb": {"lr": 0.1}, "rf": {"n": 100}}}
        overlay = {
            "models": {"xgb": {"lr": 0.05}, "new_model": {"lr": 0.3}},
            "ensemble": {"method": "blend"},
        }
        report = mgr.detect_changes(prod, overlay)
        # xgb changed, new_model added, rf removed, ensemble method added
        assert report.total_changes == 4


# ---------------------------------------------------------------------------
# 5. test_log_result
# ---------------------------------------------------------------------------


class TestLogResult:
    def test_log_writes_entry(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")

        mgr.log(
            "exp-001",
            hypothesis="Adding lr decay improves Brier",
            changes="lr: 0.1 -> 0.05",
            verdict="keep",
            conclusion="Brier improved by 0.002",
        )

        content = log_path.read_text()
        assert "## exp-001" in content
        assert "Adding lr decay improves Brier" in content
        assert "keep" in content
        assert "Brier improved by 0.002" in content

    def test_log_without_conclusion(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")

        mgr.log("exp-001", hypothesis="h", changes="c", verdict="revert")

        content = log_path.read_text()
        assert "## exp-001" in content
        assert "**Conclusion:**" not in content

    def test_log_raises_without_log_path(self, tmp_path):
        mgr = _make_manager(tmp_path)  # no log_path
        with pytest.raises(ExperimentError, match="No log_path"):
            mgr.log("exp-001", hypothesis="h", changes="c", verdict="keep")

    def test_log_multiple_entries(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")
        mgr.log("exp-001", hypothesis="h1", changes="c1", verdict="keep")

        mgr.create("exp-002")
        mgr.log("exp-002", hypothesis="h2", changes="c2", verdict="revert")

        content = log_path.read_text()
        assert "## exp-001" in content
        assert "## exp-002" in content


# ---------------------------------------------------------------------------
# 6. test_do_not_retry
# ---------------------------------------------------------------------------


class TestDoNotRetry:
    def test_check_passes_when_no_patterns(self, tmp_path):
        mgr = _make_manager(tmp_path)
        # Should not raise
        mgr.check_do_not_retry("anything goes")

    def test_add_and_check_blocks(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_do_not_retry(
            pattern="remove all features",
            reference="exp-002",
            reason="Caused catastrophic Brier regression",
        )

        with pytest.raises(ExperimentError, match="do-not-retry"):
            mgr.check_do_not_retry("Let's remove all features and see what happens")

    def test_case_insensitive_match(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.add_do_not_retry(pattern="Drop Momentum", reference="exp-003", reason="bad")

        with pytest.raises(ExperimentError, match="do-not-retry"):
            mgr.check_do_not_retry("try to drop momentum feature")

    def test_persists_to_disk(self, tmp_path):
        dnr_path = tmp_path / "do_not_retry.json"
        mgr = _make_manager(tmp_path, do_not_retry_path=dnr_path)
        mgr.add_do_not_retry(pattern="bad idea", reference="exp-001", reason="terrible")

        assert dnr_path.exists()
        data = json.loads(dnr_path.read_text())
        assert len(data) == 1
        assert data[0]["pattern"] == "bad idea"

    def test_loads_from_disk(self, tmp_path):
        dnr_path = tmp_path / "do_not_retry.json"
        dnr_path.write_text(json.dumps([
            {"pattern": "loaded pattern", "reference": "exp-010", "reason": "from disk"}
        ]))

        mgr = _make_manager(tmp_path, do_not_retry_path=dnr_path)

        with pytest.raises(ExperimentError, match="do-not-retry"):
            mgr.check_do_not_retry("this has loaded pattern in it")


# ---------------------------------------------------------------------------
# 7. test_promote_experiment
# ---------------------------------------------------------------------------


class TestPromoteExperiment:
    def test_promote_merges_overlay(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        mgr = _make_manager(tmp_path, log_path=log_path)

        # Create experiment with overlay
        exp_dir = mgr.create("exp-001")
        overlay = {"models": {"xgb": {"lr": 0.05}}}
        _write_pipeline_yaml(exp_dir / "overlay.yaml", overlay)

        # Log with keep verdict
        mgr.log("exp-001", hypothesis="h", changes="c", verdict="keep")

        # Write production config
        prod_path = tmp_path / "pipeline.yaml"
        prod_config = {"models": {"xgb": {"lr": 0.1, "depth": 6}}}
        _write_pipeline_yaml(prod_path, prod_config)

        # Promote
        backup_path = mgr.promote("exp-001", prod_path)

        # Verify backup was created
        assert backup_path.exists()

        # Verify merged config
        merged = yaml.safe_load(prod_path.read_text())
        assert merged["models"]["xgb"]["lr"] == 0.05
        assert merged["models"]["xgb"]["depth"] == 6

    def test_promote_requires_logged_verdict(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        log_path.touch()
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")

        prod_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(prod_path, {"models": {}})

        with pytest.raises(ExperimentError, match="no logged verdict"):
            mgr.promote("exp-001", prod_path)

    def test_promote_rejects_revert_verdict(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        mgr = _make_manager(tmp_path, log_path=log_path)
        mgr.create("exp-001")
        mgr.log("exp-001", hypothesis="h", changes="c", verdict="revert")

        prod_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(prod_path, {"models": {}})

        with pytest.raises(ExperimentError, match="verdict 'revert'"):
            mgr.promote("exp-001", prod_path)

    def test_promote_partial_verdict_allowed(self, tmp_path):
        log_path = tmp_path / "experiment_log.md"
        mgr = _make_manager(tmp_path, log_path=log_path)

        exp_dir = mgr.create("exp-001")
        overlay = {"models": {"lgb": {"lr": 0.2}}}
        _write_pipeline_yaml(exp_dir / "overlay.yaml", overlay)

        mgr.log("exp-001", hypothesis="h", changes="c", verdict="partial")

        prod_path = tmp_path / "pipeline.yaml"
        _write_pipeline_yaml(prod_path, {"models": {"xgb": {"lr": 0.1}}})

        backup_path = mgr.promote("exp-001", prod_path)
        assert backup_path.exists()

        merged = yaml.safe_load(prod_path.read_text())
        assert "lgb" in merged["models"]
