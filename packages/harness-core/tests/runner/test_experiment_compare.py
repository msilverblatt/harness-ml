"""Tests for ExperimentManager.compare()."""
import pytest
from harnessml.core.runner.experiments.manager import ExperimentManager


@pytest.fixture
def manager(tmp_path):
    return ExperimentManager(
        experiments_dir=tmp_path / "experiments",
        journal_path=tmp_path / "journal.jsonl",
        log_path=tmp_path / "EXPERIMENT_LOG.md",
    )


def test_compare_experiments(manager):
    # Create and log two experiments
    manager.create("exp-a", hypothesis="Approach A")
    manager.log("exp-a", hypothesis="Approach A", changes="A changes",
                verdict="keep", metrics={"brier": 0.13, "accuracy": 0.82},
                baseline_metrics={"brier": 0.14})

    manager.create("exp-b", hypothesis="Approach B")
    manager.log("exp-b", hypothesis="Approach B", changes="B changes",
                verdict="keep", metrics={"brier": 0.12, "accuracy": 0.83},
                baseline_metrics={"brier": 0.14})

    comparison = manager.compare(["exp-a", "exp-b"])
    assert "exp-a" in comparison
    assert "exp-b" in comparison
    assert "brier" in comparison  # Metric comparison included


def test_compare_generates_table(manager):
    manager.create("exp-a", hypothesis="Approach A")
    manager.log("exp-a", hypothesis="Approach A", changes="A changes",
                verdict="keep", metrics={"brier": 0.13},
                baseline_metrics={"brier": 0.14})

    manager.create("exp-b", hypothesis="Approach B")
    manager.log("exp-b", hypothesis="Approach B", changes="B changes",
                verdict="revert", metrics={"brier": 0.15},
                baseline_metrics={"brier": 0.14})

    comparison = manager.compare(["exp-a", "exp-b"])
    assert "|" in comparison  # Markdown table


def test_compare_missing_experiment(manager):
    manager.create("exp-a", hypothesis="Approach A")
    manager.log("exp-a", hypothesis="A", changes="A", verdict="keep")
    result = manager.compare(["exp-a", "exp-nonexistent"])
    assert "not found" in result


def test_compare_without_journal(tmp_path):
    manager = ExperimentManager(
        experiments_dir=tmp_path / "experiments",
        log_path=tmp_path / "EXPERIMENT_LOG.md",
    )
    result = manager.compare(["exp-a", "exp-b"])
    assert "not configured" in result.lower() or "Journal" in result


def test_compare_includes_hypothesis_and_verdict(manager):
    manager.create("exp-a", hypothesis="Lower learning rate")
    manager.log("exp-a", hypothesis="Lower learning rate", changes="LR",
                verdict="keep", metrics={"brier": 0.13},
                baseline_metrics={"brier": 0.14})

    manager.create("exp-b", hypothesis="Higher depth")
    manager.log("exp-b", hypothesis="Higher depth", changes="depth",
                verdict="revert", metrics={"brier": 0.15},
                baseline_metrics={"brier": 0.14})

    comparison = manager.compare(["exp-a", "exp-b"])
    assert "Hypothesis" in comparison
    assert "Verdict" in comparison
    assert "keep" in comparison
    assert "revert" in comparison
