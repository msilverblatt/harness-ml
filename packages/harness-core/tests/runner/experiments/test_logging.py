"""Tests for experiment logging and mandatory logging enforcement."""

import pytest

from harnessml.core.runner.experiment_manager import ExperimentManager


def test_log_experiment(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="test hypothesis",
        changes="test change",
        verdict="revert",
        notes="test notes",
    )
    log_content = (tmp_path / "LOG.md").read_text()
    assert "exp-001-test" in log_content
    assert "test hypothesis" in log_content
    assert "revert" in log_content


def test_mandatory_logging_blocks_next(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    mgr.create("exp-001-test")
    with pytest.raises(Exception, match="not been logged"):
        mgr.create("exp-002-next")


def test_mandatory_logging_allows_after_log(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    mgr.create("exp-001-test")
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="revert",
    )
    mgr.create("exp-002-next")  # should succeed


def test_has_unlogged(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    assert mgr.has_unlogged() is False
    mgr.create("exp-001-test")
    assert mgr.has_unlogged() is True
    mgr.log(
        experiment_id="exp-001-test",
        hypothesis="h",
        changes="c",
        verdict="revert",
    )
    assert mgr.has_unlogged() is False


def test_log_no_log_path(tmp_path):
    """Logging without a configured log_path raises."""
    mgr = ExperimentManager(experiments_dir=tmp_path)
    with pytest.raises(Exception, match="log_path"):
        mgr.log(
            experiment_id="exp-001-test",
            hypothesis="h",
            changes="c",
            verdict="revert",
        )


def test_log_appends_multiple(tmp_path):
    """Multiple log entries are appended, not overwritten."""
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        log_path=tmp_path / "LOG.md",
    )
    mgr.create("exp-001-first")
    mgr.log(
        experiment_id="exp-001-first",
        hypothesis="h1",
        changes="c1",
        verdict="revert",
    )
    mgr.create("exp-002-second")
    mgr.log(
        experiment_id="exp-002-second",
        hypothesis="h2",
        changes="c2",
        verdict="keep",
    )
    log_content = (tmp_path / "LOG.md").read_text()
    assert "exp-001-first" in log_content
    assert "exp-002-second" in log_content
    assert "h1" in log_content
    assert "h2" in log_content


def test_has_unlogged_no_log_path(tmp_path):
    """has_unlogged returns False when no log_path is configured."""
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.create("exp-001-test")
    assert mgr.has_unlogged() is False
