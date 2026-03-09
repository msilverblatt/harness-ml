from pathlib import Path

import pytest
from harnessml.core.runner.experiments.manager import ExperimentManager
from harnessml.core.runner.experiments.schema import ExperimentStatus, ExperimentVerdict


@pytest.fixture
def manager(tmp_path):
    return ExperimentManager(
        experiments_dir=tmp_path / "experiments",
        journal_path=tmp_path / "journal.jsonl",
        log_path=tmp_path / "EXPERIMENT_LOG.md",
    )


def test_create_experiment_writes_to_journal(manager):
    manager.create("exp-001", hypothesis="Test hypothesis")
    record = manager.journal.get("exp-001")
    assert record is not None
    assert record.hypothesis == "Test hypothesis"
    assert record.status == ExperimentStatus.CREATED


def test_create_requires_hypothesis(manager):
    """When journal is configured, create without hypothesis still works
    (backward compat) but does not write to journal."""
    exp_dir = manager.create("exp-001")
    assert exp_dir.exists()
    # No journal entry since no hypothesis
    record = manager.journal.get("exp-001")
    assert record is None


def test_log_experiment_updates_journal(manager):
    manager.create("exp-001", hypothesis="Test H")
    manager.log(
        experiment_id="exp-001",
        hypothesis="Test H",
        changes="Changed learning rate",
        verdict="keep",
        conclusion="It worked",
        metrics={"brier": 0.13, "accuracy": 0.82},
        baseline_metrics={"brier": 0.14},
    )
    record = manager.journal.get("exp-001")
    assert record.status == ExperimentStatus.COMPLETED
    assert record.conclusion is not None
    assert record.conclusion.verdict == ExperimentVerdict.KEEP


def test_log_auto_generates_markdown(manager):
    manager.create("exp-001", hypothesis="Test H")
    manager.log(
        experiment_id="exp-001",
        hypothesis="Test H",
        changes="Changed LR",
        verdict="keep",
        conclusion="Good result",
    )
    assert manager.log_path.exists()
    content = manager.log_path.read_text()
    assert "exp-001" in content
    assert "Test H" in content


def test_create_with_parent(manager):
    manager.create("exp-parent", hypothesis="Parent experiment")
    manager.log(
        experiment_id="exp-parent",
        hypothesis="Parent experiment",
        changes="Baseline",
        verdict="keep",
    )
    manager.create(
        "exp-child",
        hypothesis="Child experiment",
        parent_id="exp-parent",
        branching_reason="Exploring variant of parent",
    )
    record = manager.journal.get("exp-child")
    assert record.parent_id == "exp-parent"


def test_has_unlogged_uses_journal(manager):
    manager.create("exp-001", hypothesis="Test H")
    assert manager.has_unlogged()
    manager.log(
        experiment_id="exp-001",
        hypothesis="Test H",
        changes="Test",
        verdict="keep",
    )
    assert not manager.has_unlogged()
