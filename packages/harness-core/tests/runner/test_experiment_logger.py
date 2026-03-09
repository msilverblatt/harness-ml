"""Tests for experiment logging: ExperimentManager.log() and auto_log_result()."""

from __future__ import annotations

from pathlib import Path

import pytest
from harnessml.core.runner.experiment import auto_log_result
from harnessml.core.runner.experiment_logger import ExperimentManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager(tmp_path: Path) -> ExperimentManager:
    """Create an ExperimentManager wired to tmp_path."""
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()
    log_path = tmp_path / "EXPERIMENT_LOG.md"
    return ExperimentManager(
        experiments_dir=experiments_dir,
        log_path=log_path,
    )


# ---------------------------------------------------------------------------
# ExperimentManager.log() tests
# ---------------------------------------------------------------------------

class TestExperimentManagerLog:
    def test_log_creates_markdown_file(self, tmp_path: Path) -> None:
        """Logging an experiment creates EXPERIMENT_LOG.md with a heading."""
        mgr = _make_manager(tmp_path)
        mgr.log(
            experiment_id="exp-001",
            hypothesis="Adding feature X improves accuracy",
            changes="Added feature X",
            verdict="keep",
        )
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        assert log_path.exists()
        content = log_path.read_text()
        assert "## exp-001" in content

    def test_log_appends_row(self, tmp_path: Path) -> None:
        """Successive log calls append separate entries."""
        mgr = _make_manager(tmp_path)
        mgr.log(
            experiment_id="exp-001",
            hypothesis="H1",
            changes="C1",
            verdict="keep",
        )
        mgr.log(
            experiment_id="exp-002",
            hypothesis="H2",
            changes="C2",
            verdict="revert",
        )
        content = (tmp_path / "EXPERIMENT_LOG.md").read_text()
        assert "## exp-001" in content
        assert "## exp-002" in content

    def test_log_includes_hypothesis(self, tmp_path: Path) -> None:
        """Hypothesis text is written to the log entry."""
        mgr = _make_manager(tmp_path)
        mgr.log(
            experiment_id="exp-003",
            hypothesis="Rolling features reduce Brier score",
            changes="Added rolling mean",
            verdict="keep",
        )
        content = (tmp_path / "EXPERIMENT_LOG.md").read_text()
        assert "Rolling features reduce Brier score" in content
        assert "**Hypothesis:**" in content

    def test_log_includes_conclusion(self, tmp_path: Path) -> None:
        """Conclusion text appears when provided."""
        mgr = _make_manager(tmp_path)
        mgr.log(
            experiment_id="exp-004",
            hypothesis="H",
            changes="C",
            verdict="keep",
            conclusion="Rolling mean improved Brier by 0.002",
        )
        content = (tmp_path / "EXPERIMENT_LOG.md").read_text()
        assert "Rolling mean improved Brier by 0.002" in content
        assert "**Conclusion:**" in content

    def test_log_omits_conclusion_when_empty(self, tmp_path: Path) -> None:
        """Conclusion line is absent when not supplied."""
        mgr = _make_manager(tmp_path)
        mgr.log(
            experiment_id="exp-005",
            hypothesis="H",
            changes="C",
            verdict="revert",
        )
        content = (tmp_path / "EXPERIMENT_LOG.md").read_text()
        assert "**Conclusion:**" not in content


# ---------------------------------------------------------------------------
# auto_log_result() tests
# ---------------------------------------------------------------------------

class TestAutoLogResult:
    def test_auto_log_creates_markdown_table(self, tmp_path: Path) -> None:
        """First call creates EXPERIMENT_LOG.md with table header."""
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path=log_path,
            experiment_id="exp-010",
            hypothesis="Baseline run",
            changes="none",
            metrics={"accuracy": 0.81, "brier_score": 0.14},
            baseline_metrics={},
            verdict="keep",
        )
        content = log_path.read_text()
        assert "# Experiment Log" in content
        assert "| ID |" in content
        assert "exp-010" in content

    def test_auto_log_appends_row(self, tmp_path: Path) -> None:
        """Successive calls append rows to the same table."""
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        for i in range(3):
            auto_log_result(
                log_path=log_path,
                experiment_id=f"exp-{i:03d}",
                hypothesis=f"H{i}",
                changes=f"C{i}",
                metrics={"accuracy": 0.80 + i * 0.01},
                baseline_metrics={"accuracy": 0.80},
                verdict="keep",
            )
        content = log_path.read_text()
        assert "exp-000" in content
        assert "exp-001" in content
        assert "exp-002" in content

    def test_auto_log_includes_hypothesis(self, tmp_path: Path) -> None:
        """Hypothesis column is present in the table row."""
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path=log_path,
            experiment_id="exp-020",
            hypothesis="Test hypothesis text",
            changes="some change",
            metrics={},
            baseline_metrics={},
            verdict="keep",
        )
        content = log_path.read_text()
        assert "Hypothesis" in content  # header
        assert "Test hypothesis text" in content  # row value

    def test_auto_log_includes_conclusion(self, tmp_path: Path) -> None:
        """Conclusion column is present in the table row."""
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path=log_path,
            experiment_id="exp-030",
            hypothesis="H",
            changes="C",
            metrics={},
            baseline_metrics={},
            verdict="keep",
            conclusion="Learned something useful",
        )
        content = log_path.read_text()
        assert "Conclusion" in content  # header
        assert "Learned something useful" in content  # row value

    def test_auto_log_result_with_metrics(self, tmp_path: Path) -> None:
        """Metrics and deltas are formatted in the table row."""
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path=log_path,
            experiment_id="exp-040",
            hypothesis="H",
            changes="C",
            metrics={"accuracy": 0.8200, "brier_score": 0.1350},
            baseline_metrics={"accuracy": 0.8100, "brier_score": 0.1400},
            verdict="improved",
        )
        content = log_path.read_text()
        # Should contain formatted metric values
        assert "0.8200" in content
        assert "0.1350" in content
        # Should contain deltas
        assert "+0.0100" in content   # accuracy delta
        assert "-0.0050" in content   # brier delta

    def test_auto_log_missing_metric_shows_dash(self, tmp_path: Path) -> None:
        """Missing metrics render as '-' in the table."""
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path=log_path,
            experiment_id="exp-050",
            hypothesis="H",
            changes="C",
            metrics={},  # no metrics at all
            baseline_metrics={},
            verdict="neutral",
        )
        content = log_path.read_text()
        # The row should have dashes for missing metrics
        row_lines = [l for l in content.splitlines() if "exp-050" in l]
        assert len(row_lines) == 1
        assert row_lines[0].count("-") >= 4  # at least 4 metric columns as dashes
