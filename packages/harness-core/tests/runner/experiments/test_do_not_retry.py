"""Tests for do-not-retry registry."""

import pytest
from harnessml.core.runner.experiment_manager import ExperimentManager


def test_do_not_retry_blocks(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.add_do_not_retry(
        pattern="temperature scaling",
        reference="EXP-002",
        reason="hurts monotonically",
    )

    with pytest.raises(Exception, match="temperature scaling"):
        mgr.check_do_not_retry("Let's try temperature scaling T=1.5")


def test_do_not_retry_allows_unmatched(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.add_do_not_retry(pattern="temperature", reference="EXP-002", reason="hurts")
    mgr.check_do_not_retry("Adding a new XGBoost model")  # should not raise


def test_do_not_retry_save_load(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        do_not_retry_path=tmp_path / "dnr.json",
    )
    mgr.add_do_not_retry(
        pattern="test pattern", reference="EXP-001", reason="failed"
    )

    mgr2 = ExperimentManager(
        experiments_dir=tmp_path,
        do_not_retry_path=tmp_path / "dnr.json",
    )
    with pytest.raises(Exception, match="test pattern"):
        mgr2.check_do_not_retry("test pattern here")


def test_do_not_retry_case_insensitive(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.add_do_not_retry(
        pattern="Temperature Scaling",
        reference="EXP-002",
        reason="hurts",
    )
    with pytest.raises(Exception, match="Temperature Scaling"):
        mgr.check_do_not_retry("trying temperature scaling again")


def test_do_not_retry_multiple_patterns(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.add_do_not_retry(pattern="temperature", reference="EXP-002", reason="hurts")
    mgr.add_do_not_retry(pattern="pruning", reference="EXP-006", reason="kills diversity")

    with pytest.raises(Exception):
        mgr.check_do_not_retry("let's try pruning models")

    mgr.check_do_not_retry("adding a brand new feature")  # should not raise


def test_do_not_retry_empty_list(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    # No patterns added — should never raise
    mgr.check_do_not_retry("temperature scaling")
