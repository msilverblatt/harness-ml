"""Tests for experiment creation, naming validation, and change detection."""

import pytest
from harnessml.core.runner.experiments.manager import ChangeReport, ExperimentManager

# ---------------------------------------------------------------------------
# 6.1  Experiment creation + naming validation
# ---------------------------------------------------------------------------


def test_create_experiment(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    exp = mgr.create("exp-001-test")
    assert (tmp_path / "exp-001-test").exists()
    assert (tmp_path / "exp-001-test" / "overlay.yaml").exists()


def test_create_experiment_bad_name(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    with pytest.raises(Exception, match="naming"):
        mgr.create("bad_name")


def test_create_experiment_duplicate(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    mgr.create("exp-001-test")
    with pytest.raises(Exception, match="exists"):
        mgr.create("exp-001-test")


def test_create_experiment_no_naming_pattern(tmp_path):
    """When no naming pattern is set, any ID is accepted."""
    mgr = ExperimentManager(experiments_dir=tmp_path)
    exp = mgr.create("anything-goes")
    assert (tmp_path / "anything-goes").exists()


def test_create_returns_path(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    result = mgr.create("exp-001-test")
    assert result == tmp_path / "exp-001-test"


# ---------------------------------------------------------------------------
# 6.2  Change detection + single-variable enforcement
# ---------------------------------------------------------------------------


def test_detect_changes_single(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    production = {"models": {"xgb": {"params": {"depth": 3}}}}
    overlay = {"models": {"xgb": {"params": {"depth": 5}}}}
    changes = mgr.detect_changes(production, overlay)
    assert changes.changed_models == ["xgb"]
    assert len(changes.new_models) == 0
    assert changes.total_changes == 1


def test_detect_changes_new_model(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    production = {"models": {"xgb": {"params": {"depth": 3}}}}
    overlay = {
        "models": {
            "xgb": {"params": {"depth": 3}},
            "new_model": {"type": "catboost"},
        }
    }
    changes = mgr.detect_changes(production, overlay)
    assert "new_model" in changes.new_models
    assert changes.total_changes == 1


def test_detect_changes_multi_warns(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    production = {
        "models": {
            "xgb": {"params": {"depth": 3}},
            "cat": {"params": {"lr": 0.05}},
        }
    }
    overlay = {
        "models": {
            "xgb": {"params": {"depth": 5}},
            "cat": {"params": {"lr": 0.01}},
        }
    }
    changes = mgr.detect_changes(production, overlay)
    assert changes.total_changes > 1


def test_detect_changes_ensemble(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    production = {"ensemble": {"method": "stacked", "temperature": 1.0}}
    overlay = {"ensemble": {"method": "stacked", "temperature": 1.5}}
    changes = mgr.detect_changes(production, overlay)
    assert "temperature" in changes.ensemble_changes
    assert changes.total_changes == 1


def test_detect_changes_no_changes(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    production = {"models": {"xgb": {"params": {"depth": 3}}}}
    overlay = {"models": {"xgb": {"params": {"depth": 3}}}}
    changes = mgr.detect_changes(production, overlay)
    assert changes.total_changes == 0


def test_detect_changes_removed_model(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    production = {
        "models": {
            "xgb": {"params": {"depth": 3}},
            "cat": {"params": {"lr": 0.05}},
        }
    }
    overlay = {"models": {"xgb": {"params": {"depth": 3}}}}
    changes = mgr.detect_changes(production, overlay)
    assert "cat" in changes.removed_models


def test_change_report_total():
    report = ChangeReport(
        changed_models=["a"],
        new_models=["b"],
        removed_models=["c"],
        ensemble_changes=["d"],
    )
    assert report.total_changes == 4
