from pathlib import Path

import pytest
from harnessml.core.runner.experiment_journal import ExperimentJournal
from harnessml.core.runner.experiment_schema import (
    ExperimentRecord,
    ExperimentStatus,
    ExperimentVerdict,
    StructuredConclusion,
)


@pytest.fixture
def journal(tmp_path):
    return ExperimentJournal(tmp_path / "journal.jsonl")


def test_append_and_read(journal):
    r = ExperimentRecord(
        experiment_id="exp-001",
        hypothesis="Test hypothesis",
        status=ExperimentStatus.CREATED,
    )
    journal.append(r)
    records = journal.read_all()
    assert len(records) == 1
    assert records[0].experiment_id == "exp-001"


def test_append_multiple(journal):
    for i in range(3):
        journal.append(ExperimentRecord(
            experiment_id=f"exp-{i:03d}",
            hypothesis=f"Hypothesis {i}",
            status=ExperimentStatus.CREATED,
        ))
    assert len(journal.read_all()) == 3


def test_get_by_id(journal):
    journal.append(ExperimentRecord(
        experiment_id="exp-target",
        hypothesis="Find me",
        status=ExperimentStatus.CREATED,
    ))
    journal.append(ExperimentRecord(
        experiment_id="exp-other",
        hypothesis="Not me",
        status=ExperimentStatus.CREATED,
    ))
    record = journal.get("exp-target")
    assert record is not None
    assert record.hypothesis == "Find me"


def test_get_missing_returns_none(journal):
    assert journal.get("nonexistent") is None


def test_update_record(journal):
    journal.append(ExperimentRecord(
        experiment_id="exp-001",
        hypothesis="Initial",
        status=ExperimentStatus.CREATED,
    ))
    journal.update("exp-001", status=ExperimentStatus.COMPLETED)
    record = journal.get("exp-001")
    assert record.status == ExperimentStatus.COMPLETED


def test_get_latest_by_id(journal):
    """When multiple records exist for same ID, get() returns the latest."""
    journal.append(ExperimentRecord(
        experiment_id="exp-001",
        hypothesis="Initial",
        status=ExperimentStatus.CREATED,
    ))
    journal.update("exp-001", status=ExperimentStatus.RUNNING)
    journal.update("exp-001", status=ExperimentStatus.COMPLETED)
    record = journal.get("exp-001")
    assert record.status == ExperimentStatus.COMPLETED


def test_list_by_status(journal):
    journal.append(ExperimentRecord(
        experiment_id="exp-001", hypothesis="H1",
        status=ExperimentStatus.COMPLETED,
    ))
    journal.append(ExperimentRecord(
        experiment_id="exp-002", hypothesis="H2",
        status=ExperimentStatus.CREATED,
    ))
    completed = journal.list_by_status(ExperimentStatus.COMPLETED)
    assert len(completed) == 1
    assert completed[0].experiment_id == "exp-001"


def test_get_children(journal):
    journal.append(ExperimentRecord(
        experiment_id="exp-parent", hypothesis="Parent",
        status=ExperimentStatus.COMPLETED,
    ))
    journal.append(ExperimentRecord(
        experiment_id="exp-child-1", hypothesis="Child 1",
        status=ExperimentStatus.COMPLETED,
        parent_id="exp-parent",
    ))
    journal.append(ExperimentRecord(
        experiment_id="exp-child-2", hypothesis="Child 2",
        status=ExperimentStatus.CREATED,
        parent_id="exp-parent",
    ))
    children = journal.get_children("exp-parent")
    assert len(children) == 2


def test_generate_markdown(journal):
    journal.append(ExperimentRecord(
        experiment_id="exp-001", hypothesis="Test H",
        status=ExperimentStatus.COMPLETED,
        conclusion=StructuredConclusion(
            verdict=ExperimentVerdict.KEEP,
            primary_metric="brier",
            baseline_value=0.14,
            result_value=0.13,
            improvement=0.01,
            improvement_pct=7.14,
            learnings="It worked",
        ),
    ))
    md = journal.generate_markdown()
    assert "exp-001" in md
    assert "Test H" in md
    assert "keep" in md.lower()
    assert "brier" in md
    assert "0.14" in md


def test_empty_journal_reads_empty(journal):
    assert journal.read_all() == []
