"""Tests for markdown-to-JSONL migration."""

from harnessml.core.runner.experiments.journal import ExperimentJournal
from harnessml.core.runner.experiments.manager import migrate_markdown_to_jsonl


def test_migrate_markdown_to_jsonl(tmp_path):
    """Existing EXPERIMENT_LOG.md can be migrated to journal.jsonl."""
    log_path = tmp_path / "EXPERIMENT_LOG.md"
    log_path.write_text(
        "## exp-001\n"
        "**Date:** 2026-03-01 10:00:00 UTC\n"
        "**Hypothesis:** Test hypothesis\n"
        "**Changes:** Changed LR\n"
        "**Verdict:** keep\n"
        "**Conclusion:** It worked\n\n"
    )
    journal_path = tmp_path / "journal.jsonl"
    migrate_markdown_to_jsonl(log_path, journal_path)

    journal = ExperimentJournal(journal_path)
    records = journal.read_all()
    assert len(records) == 1
    assert records[0].experiment_id == "exp-001"
    assert records[0].hypothesis == "Test hypothesis"
    assert records[0].conclusion.verdict.value == "keep"


def test_migrate_multiple_experiments(tmp_path):
    """Multiple experiments in one markdown file."""
    log_path = tmp_path / "EXPERIMENT_LOG.md"
    log_path.write_text(
        "## exp-001\n"
        "**Hypothesis:** First experiment\n"
        "**Verdict:** keep\n\n"
        "## exp-002\n"
        "**Hypothesis:** Second experiment\n"
        "**Verdict:** revert\n"
        "**Conclusion:** Did not work\n\n"
    )
    journal_path = tmp_path / "journal.jsonl"
    count = migrate_markdown_to_jsonl(log_path, journal_path)
    assert count == 2

    journal = ExperimentJournal(journal_path)
    records = journal.read_all()
    assert len(records) == 2
    assert records[0].experiment_id == "exp-001"
    assert records[1].experiment_id == "exp-002"
    assert records[1].conclusion.verdict.value == "revert"
    assert records[1].conclusion.learnings == "Did not work"


def test_migrate_nonexistent_log(tmp_path):
    """Migration of nonexistent file returns 0."""
    log_path = tmp_path / "nonexistent.md"
    journal_path = tmp_path / "journal.jsonl"
    count = migrate_markdown_to_jsonl(log_path, journal_path)
    assert count == 0


def test_migrate_empty_log(tmp_path):
    """Migration of empty file returns 0."""
    log_path = tmp_path / "empty.md"
    log_path.write_text("")
    journal_path = tmp_path / "journal.jsonl"
    count = migrate_markdown_to_jsonl(log_path, journal_path)
    assert count == 0


def test_migrate_preserves_conclusion(tmp_path):
    """Conclusion text becomes learnings in structured conclusion."""
    log_path = tmp_path / "EXPERIMENT_LOG.md"
    log_path.write_text(
        "## exp-001\n"
        "**Hypothesis:** Test\n"
        "**Verdict:** partial\n"
        "**Conclusion:** Some folds improved\n\n"
    )
    journal_path = tmp_path / "journal.jsonl"
    migrate_markdown_to_jsonl(log_path, journal_path)

    journal = ExperimentJournal(journal_path)
    record = journal.get("exp-001")
    assert record.conclusion.verdict.value == "partial"
    assert record.conclusion.learnings == "Some folds improved"
