"""Tests for ExperimentManager.rollback()."""
import pytest
import yaml
from harnessml.core.runner.experiment_manager import ExperimentError, ExperimentManager


@pytest.fixture
def manager(tmp_path):
    return ExperimentManager(
        experiments_dir=tmp_path / "experiments",
        journal_path=tmp_path / "journal.jsonl",
        log_path=tmp_path / "EXPERIMENT_LOG.md",
    )


def test_rollback_reverts_to_backup(manager, tmp_path):
    # Set up production config
    prod_config = tmp_path / "config" / "pipeline.yaml"
    prod_config.parent.mkdir(parents=True)
    prod_config.write_text("models:\n  xgb:\n    learning_rate: 0.1\n")

    # Create and promote an experiment
    manager.create("exp-001", hypothesis="Test")
    overlay = manager.experiments_dir / "exp-001" / "overlay.yaml"
    overlay.write_text("models:\n  xgb:\n    learning_rate: 0.05\n")
    manager.log("exp-001", hypothesis="Test", changes="LR", verdict="keep")
    backup = manager.promote("exp-001", prod_config)

    # Verify promotion happened
    promoted = yaml.safe_load(prod_config.read_text())
    assert promoted["models"]["xgb"]["learning_rate"] == 0.05

    # Rollback
    manager.rollback(prod_config, backup)
    rolled_back = yaml.safe_load(prod_config.read_text())
    assert rolled_back["models"]["xgb"]["learning_rate"] == 0.1


def test_rollback_missing_backup_raises(manager, tmp_path):
    prod_config = tmp_path / "config" / "pipeline.yaml"
    prod_config.parent.mkdir(parents=True)
    prod_config.write_text("models: {}\n")

    with pytest.raises(ExperimentError, match="Backup file not found"):
        manager.rollback(prod_config, tmp_path / "nonexistent.bak")


def test_rollback_preserves_backup_file(manager, tmp_path):
    prod_config = tmp_path / "config" / "pipeline.yaml"
    prod_config.parent.mkdir(parents=True)
    prod_config.write_text("models:\n  xgb:\n    learning_rate: 0.1\n")

    manager.create("exp-001", hypothesis="Test")
    overlay = manager.experiments_dir / "exp-001" / "overlay.yaml"
    overlay.write_text("models:\n  xgb:\n    learning_rate: 0.05\n")
    manager.log("exp-001", hypothesis="Test", changes="LR", verdict="keep")
    backup = manager.promote("exp-001", prod_config)

    manager.rollback(prod_config, backup)
    # Backup file should still exist after rollback
    assert backup.exists()
