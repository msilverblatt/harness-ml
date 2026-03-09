"""Tests for ExperimentManager._build_conclusion()."""
import pytest
from harnessml.core.runner.experiment_manager import ExperimentManager
from harnessml.core.runner.experiment_schema import ExperimentVerdict


@pytest.fixture
def manager(tmp_path):
    return ExperimentManager(
        experiments_dir=tmp_path / "exp",
        journal_path=tmp_path / "j.jsonl",
        baseline_metrics={"brier": 0.14, "accuracy": 0.81},
    )


def test_build_conclusion_keep(manager):
    conclusion = manager._build_conclusion(
        verdict="keep",
        learnings="Lower LR helped",
        metrics={"brier": 0.13, "accuracy": 0.82},
        baseline_metrics={"brier": 0.14, "accuracy": 0.81},
    )
    assert conclusion.verdict == ExperimentVerdict.KEEP
    assert conclusion.primary_metric == "brier"
    assert conclusion.improvement == pytest.approx(0.01)  # 0.14 - 0.13
    assert conclusion.improvement_pct == pytest.approx(7.14, abs=0.1)


def test_build_conclusion_revert(tmp_path):
    manager = ExperimentManager(
        experiments_dir=tmp_path / "exp",
        journal_path=tmp_path / "j.jsonl",
    )
    conclusion = manager._build_conclusion(
        verdict="revert",
        learnings="Made things worse",
        metrics={"brier": 0.15},
        baseline_metrics={"brier": 0.14},
    )
    assert conclusion.improvement == pytest.approx(-0.01)  # Negative = worse
    assert conclusion.verdict == ExperimentVerdict.REVERT


def test_build_conclusion_uses_instance_baseline(manager):
    """When baseline_metrics param is None, uses manager's baseline_metrics."""
    conclusion = manager._build_conclusion(
        verdict="keep",
        learnings="Test",
        metrics={"brier": 0.13},
    )
    assert conclusion.primary_metric == "brier"
    assert conclusion.baseline_value == 0.14
    assert conclusion.result_value == 0.13
    assert conclusion.improvement == pytest.approx(0.01)


def test_build_conclusion_no_baseline(tmp_path):
    """When no baseline is available, improvement fields are None."""
    manager = ExperimentManager(
        experiments_dir=tmp_path / "exp",
        journal_path=tmp_path / "j.jsonl",
    )
    conclusion = manager._build_conclusion(
        verdict="inconclusive",
        learnings="No baseline to compare",
        metrics={"brier": 0.13},
    )
    assert conclusion.verdict == ExperimentVerdict.INCONCLUSIVE
    assert conclusion.primary_metric == ""
    assert conclusion.improvement is None


def test_build_conclusion_higher_is_better(tmp_path):
    """For accuracy, higher is better: improvement = result - baseline."""
    manager = ExperimentManager(
        experiments_dir=tmp_path / "exp",
        journal_path=tmp_path / "j.jsonl",
    )
    conclusion = manager._build_conclusion(
        verdict="keep",
        learnings="Accuracy improved",
        metrics={"accuracy": 0.85},
        baseline_metrics={"accuracy": 0.81},
    )
    assert conclusion.improvement == pytest.approx(0.04)
    assert conclusion.improvement_pct == pytest.approx(4.94, abs=0.1)


def test_build_conclusion_secondary_metrics(manager):
    """Metrics other than primary are stored in secondary_metrics."""
    conclusion = manager._build_conclusion(
        verdict="keep",
        learnings="Test",
        metrics={"brier": 0.13, "accuracy": 0.82, "ece": 0.03},
        baseline_metrics={"brier": 0.14},
    )
    assert "accuracy" in conclusion.secondary_metrics
    assert "ece" in conclusion.secondary_metrics
    assert conclusion.secondary_metrics["accuracy"] == 0.82


def test_build_conclusion_learnings_preserved(manager):
    conclusion = manager._build_conclusion(
        verdict="partial",
        learnings="Partial improvement on some folds",
        metrics={"brier": 0.135},
        baseline_metrics={"brier": 0.14},
    )
    assert conclusion.learnings == "Partial improvement on some folds"
    assert conclusion.verdict == ExperimentVerdict.PARTIAL
