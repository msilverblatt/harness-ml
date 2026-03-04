"""Tests for atomic promote with backup and rollback."""

import pytest
import yaml

from easyml.core.runner.experiment_manager import ExperimentManager


def test_promote_requires_logged_verdict(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    prod_config = tmp_path / "production.yaml"
    prod_config.write_text("models:\n  xgb:\n    type: xgboost")

    with pytest.raises(Exception, match="verdict"):
        mgr.promote("exp-001-test", production_config_path=prod_config)


def test_promote_requires_keep_verdict(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="revert",
    )
    prod_config = tmp_path / "production.yaml"
    prod_config.write_text("models:\n  xgb:\n    type: xgboost")

    with pytest.raises(Exception, match="keep"):
        mgr.promote("exp-001-test", production_config_path=prod_config)


def test_promote_creates_backup(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="keep",
    )

    # Write overlay
    overlay_path = tmp_path / "exp-001-test" / "overlay.yaml"
    overlay_path.write_text("models:\n  xgb:\n    params:\n      depth: 5")

    prod_config = tmp_path / "production.yaml"
    prod_config.write_text("models:\n  xgb:\n    params:\n      depth: 3")

    mgr.promote("exp-001-test", production_config_path=prod_config)

    # Backup should exist
    backups = list(tmp_path.glob("production.yaml.bak.*"))
    assert len(backups) == 1

    # Production config should be updated
    updated = yaml.safe_load(prod_config.read_text())
    assert updated["models"]["xgb"]["params"]["depth"] == 5


def test_promote_with_partial_verdict(tmp_path):
    """partial verdict should also allow promotion."""
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="partial",
    )

    overlay_path = tmp_path / "exp-001-test" / "overlay.yaml"
    overlay_path.write_text("models:\n  xgb:\n    params:\n      depth: 7")

    prod_config = tmp_path / "production.yaml"
    prod_config.write_text("models:\n  xgb:\n    params:\n      depth: 3")

    mgr.promote("exp-001-test", production_config_path=prod_config)

    updated = yaml.safe_load(prod_config.read_text())
    assert updated["models"]["xgb"]["params"]["depth"] == 7


def test_promote_deep_merges(tmp_path):
    """Promote should deep merge overlay into production, not replace."""
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="keep",
    )

    overlay_path = tmp_path / "exp-001-test" / "overlay.yaml"
    overlay_path.write_text("models:\n  xgb:\n    params:\n      depth: 5")

    prod_config = tmp_path / "production.yaml"
    prod_config.write_text(
        "models:\n  xgb:\n    params:\n      depth: 3\n      lr: 0.01\n  cat:\n    type: catboost"
    )

    mgr.promote("exp-001-test", production_config_path=prod_config)

    updated = yaml.safe_load(prod_config.read_text())
    # xgb depth changed
    assert updated["models"]["xgb"]["params"]["depth"] == 5
    # xgb lr preserved
    assert updated["models"]["xgb"]["params"]["lr"] == 0.01
    # cat model preserved
    assert updated["models"]["cat"]["type"] == "catboost"


def test_promote_backup_content(tmp_path):
    """Backup should contain the original production config."""
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="keep",
    )

    overlay_path = tmp_path / "exp-001-test" / "overlay.yaml"
    overlay_path.write_text("models:\n  xgb:\n    params:\n      depth: 5")

    original_content = "models:\n  xgb:\n    params:\n      depth: 3"
    prod_config = tmp_path / "production.yaml"
    prod_config.write_text(original_content)

    mgr.promote("exp-001-test", production_config_path=prod_config)

    backups = list(tmp_path.glob("production.yaml.bak.*"))
    backup_content = yaml.safe_load(backups[0].read_text())
    assert backup_content["models"]["xgb"]["params"]["depth"] == 3
