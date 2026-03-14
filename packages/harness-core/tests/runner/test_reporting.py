"""Tests for backtest reporting module."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.analysis.diagnostics import compute_model_agreement
from harnessml.core.runner.analysis.reporting import (
    build_diagnostics_report,
    export_backtest_artifacts,
    generate_markdown_report,
)

# -----------------------------------------------------------------------
# Tests: compute_model_agreement
# -----------------------------------------------------------------------


class TestComputeModelAgreement:
    """Test model agreement computation."""

    def test_all_agree(self):
        """All models agree with ensemble -> 1.0."""
        df = pd.DataFrame({
            "prob_ensemble": [0.8, 0.2, 0.7],
            "prob_model_a": [0.9, 0.1, 0.6],
            "prob_model_b": [0.7, 0.3, 0.8],
        })
        agreement = compute_model_agreement(df)
        np.testing.assert_array_almost_equal(agreement, [1.0, 1.0, 1.0])

    def test_split_agreement(self):
        """Half agree -> 0.5."""
        df = pd.DataFrame({
            "prob_ensemble": [0.7],
            "prob_model_a": [0.8],  # agrees (both > 0.5)
            "prob_model_b": [0.3],  # disagrees (model <= 0.5, ensemble > 0.5)
        })
        agreement = compute_model_agreement(df)
        assert agreement[0] == pytest.approx(0.5)

    def test_none_agree(self):
        """No models agree with ensemble -> 0.0."""
        df = pd.DataFrame({
            "prob_ensemble": [0.8, 0.2],
            "prob_model_a": [0.3, 0.7],
            "prob_model_b": [0.4, 0.6],
        })
        agreement = compute_model_agreement(df)
        np.testing.assert_array_almost_equal(agreement, [0.0, 0.0])

    def test_excludes_logreg_seed(self):
        """prob_logreg_seed is excluded from agreement calculation."""
        df = pd.DataFrame({
            "prob_ensemble": [0.8],
            "prob_logreg_seed": [0.3],  # would disagree, but excluded
            "prob_model_a": [0.9],  # agrees
        })
        agreement = compute_model_agreement(df)
        assert agreement[0] == pytest.approx(1.0)

    def test_no_ensemble_column(self):
        """Returns all 1.0 when no prob_ensemble column."""
        df = pd.DataFrame({
            "prob_model_a": [0.7, 0.3],
        })
        agreement = compute_model_agreement(df)
        np.testing.assert_array_equal(agreement, [1.0, 1.0])

    def test_no_model_columns(self):
        """Returns all 1.0 when no individual model columns."""
        df = pd.DataFrame({
            "prob_ensemble": [0.7, 0.3],
        })
        agreement = compute_model_agreement(df)
        np.testing.assert_array_equal(agreement, [1.0, 1.0])


# -----------------------------------------------------------------------
# Tests: build_diagnostics_report
# -----------------------------------------------------------------------


class TestBuildDiagnosticsReport:
    """Test diagnostics report construction."""

    def test_basic_report(self):
        """Builds per-fold metrics table."""
        fold_data = {
            2023: pd.DataFrame({
                "result": [1, 0, 1, 0, 1] * 10,
                "prob_ensemble": [0.7, 0.3, 0.8, 0.2, 0.6] * 10,
            }),
            2024: pd.DataFrame({
                "result": [1, 0, 1, 0, 1] * 10,
                "prob_ensemble": [0.65, 0.35, 0.75, 0.25, 0.55] * 10,
            }),
        }
        report = build_diagnostics_report(fold_data, fold_column="season")
        assert len(report) == 2
        assert "season" in report.columns
        assert "brier_score" in report.columns
        assert "accuracy" in report.columns
        assert "ece" in report.columns
        assert "log_loss" in report.columns
        assert "n_samples" in report.columns
        assert list(report["season"]) == [2023, 2024]

    def test_single_fold(self):
        """Works with a single fold."""
        fold_data = {
            2024: pd.DataFrame({
                "result": [1, 0, 1, 0],
                "prob_ensemble": [0.7, 0.3, 0.8, 0.2],
            }),
        }
        report = build_diagnostics_report(fold_data, fold_column="season")
        assert len(report) == 1
        assert report["n_samples"].iloc[0] == 4

    def test_empty_input(self):
        """Empty input returns empty DataFrame."""
        report = build_diagnostics_report({})
        assert len(report) == 0


# -----------------------------------------------------------------------
# Tests: generate_markdown_report
# -----------------------------------------------------------------------


class TestGenerateMarkdownReport:
    """Test markdown report generation."""

    def test_includes_top_line_metrics(self):
        """Report contains metrics table."""
        pooled = {
            "ensemble": {
                "brier_score": 0.1751,
                "accuracy": 0.7564,
                "ece": 0.0340,
                "log_loss": 0.5300,
                "n_samples": 670,
            },
        }
        report = generate_markdown_report(pooled)
        assert "# Backtest Report" in report
        assert "Top-Line Metrics" in report
        assert "0.1751" in report
        assert "0.7564" in report

    def test_includes_per_fold(self):
        """Report includes fold breakdown when provided."""
        pooled = {
            "ensemble": {
                "brier_score": 0.18,
                "accuracy": 0.75,
                "ece": 0.03,
                "log_loss": 0.53,
                "n_samples": 100,
            },
        }
        diag_df = pd.DataFrame({
            "season": [2023, 2024],
            "brier_score": [0.17, 0.19],
            "accuracy": [0.76, 0.74],
            "ece": [0.03, 0.04],
            "log_loss": [0.52, 0.54],
            "n_samples": [50, 50],
        })
        report = generate_markdown_report(pooled, diagnostics_df=diag_df, fold_column="season")
        assert "Per-Fold Breakdown" in report
        assert "2023" in report
        assert "2024" in report

    def test_includes_meta_coefficients(self):
        """Report includes meta-learner coefficients when provided."""
        pooled = {
            "ensemble": {
                "brier_score": 0.18,
                "accuracy": 0.75,
                "ece": 0.03,
                "log_loss": 0.53,
                "n_samples": 100,
            },
        }
        coeffs = {"xgb_core": 1.5, "logreg_seed": 0.3}
        report = generate_markdown_report(pooled, meta_coefficients=coeffs)
        assert "Meta-Learner Coefficients" in report
        assert "xgb_core" in report

    def test_includes_pick_analysis(self):
        """Report includes pick analysis when pick_log provided."""
        pooled = {
            "ensemble": {
                "brier_score": 0.18,
                "accuracy": 0.75,
                "ece": 0.03,
                "log_loss": 0.53,
                "n_samples": 100,
            },
        }
        pick_log = pd.DataFrame({
            "correct": [True, True, False],
            "confidence": [0.3, 0.2, 0.1],
            "model_agreement_pct": [1.0, 0.8, 0.5],
        })
        report = generate_markdown_report(pooled, pick_log=pick_log)
        assert "Pick Analysis" in report
        assert "Total Picks" in report
        assert "3" in report

    def test_no_ensemble_in_pooled(self):
        """Handles missing ensemble key gracefully."""
        pooled = {
            "model_a": {
                "brier_score": 0.20,
                "accuracy": 0.70,
                "ece": 0.05,
                "log_loss": 0.60,
                "n_samples": 50,
            },
        }
        report = generate_markdown_report(pooled)
        assert "No ensemble metrics available" in report

    def test_minimal_report(self):
        """Generates report with only pooled metrics (no optional sections)."""
        pooled = {
            "ensemble": {
                "brier_score": 0.18,
                "accuracy": 0.75,
                "ece": 0.03,
                "log_loss": 0.53,
                "n_samples": 100,
            },
        }
        report = generate_markdown_report(pooled)
        assert "Per-Fold Breakdown" not in report
        assert "Meta-Learner Coefficients" not in report
        assert "Pick Analysis" not in report


# -----------------------------------------------------------------------
# Tests: export_backtest_artifacts
# -----------------------------------------------------------------------


class TestExportBacktestArtifacts:
    """Test artifact export."""

    def test_exports_all_files(self, tmp_path):
        """Creates all expected files in run_dir."""
        fold_data = {
            2023: pd.DataFrame({
                "result": [1, 0, 1],
                "prob_ensemble": [0.7, 0.3, 0.8],
            }),
            2024: pd.DataFrame({
                "result": [1, 0],
                "prob_ensemble": [0.6, 0.4],
            }),
        }
        pooled_metrics = {
            "ensemble": {"brier_score": 0.18, "accuracy": 0.75},
        }
        diagnostics_df = pd.DataFrame({
            "season": [2023, 2024],
            "brier_score": [0.17, 0.19],
            "accuracy": [0.76, 0.74],
            "ece": [0.03, 0.04],
            "log_loss": [0.52, 0.54],
            "n_samples": [3, 2],
        })
        pick_log = pd.DataFrame({
            "game_id": [0, 1, 2],
            "correct": [True, True, False],
            "confidence": [0.2, 0.3, 0.1],
            "model_agreement_pct": [1.0, 0.8, 0.5],
        })
        report_md = "# Test Report\n\nSome content."

        run_dir = tmp_path / "test_run"
        export_backtest_artifacts(
            run_dir=run_dir,
            fold_data=fold_data,
            pooled_metrics=pooled_metrics,
            diagnostics_df=diagnostics_df,
            pick_log=pick_log,
            report_md=report_md,
        )

        # Check predictions
        assert (run_dir / "predictions" / "2023_probabilities.parquet").exists()
        assert (run_dir / "predictions" / "2024_probabilities.parquet").exists()

        # Check diagnostics
        assert (run_dir / "diagnostics" / "diagnostics.parquet").exists()
        assert (run_dir / "diagnostics" / "pooled_metrics.json").exists()
        assert (run_dir / "diagnostics" / "pick_log.parquet").exists()
        assert (run_dir / "diagnostics" / "report.md").exists()

        # Verify JSON content
        with open(run_dir / "diagnostics" / "pooled_metrics.json") as f:
            loaded = json.load(f)
        assert loaded["ensemble"]["brier_score"] == 0.18

        # Verify markdown content
        with open(run_dir / "diagnostics" / "report.md") as f:
            content = f.read()
        assert "Test Report" in content

        # Verify parquet readability
        loaded_diag = pd.read_parquet(
            run_dir / "diagnostics" / "diagnostics.parquet"
        )
        assert len(loaded_diag) == 2

        loaded_preds = pd.read_parquet(
            run_dir / "predictions" / "2023_probabilities.parquet"
        )
        assert len(loaded_preds) == 3

    def test_creates_directories(self, tmp_path):
        """Creates prediction and diagnostics subdirectories."""
        run_dir = tmp_path / "new_run"
        fold_data = {
            2023: pd.DataFrame({
                "result": [1],
                "prob_ensemble": [0.7],
            }),
        }
        export_backtest_artifacts(
            run_dir=run_dir,
            fold_data=fold_data,
            pooled_metrics={},
            diagnostics_df=pd.DataFrame(),
            pick_log=pd.DataFrame(
                columns=["game_id", "correct", "confidence", "model_agreement_pct"]
            ),
            report_md="",
        )
        assert (run_dir / "predictions").is_dir()
        assert (run_dir / "diagnostics").is_dir()

    def test_numpy_serialization(self, tmp_path):
        """JSON export handles numpy types."""
        run_dir = tmp_path / "np_run"
        pooled_metrics = {
            "ensemble": {
                "brier_score": np.float64(0.18),
                "n_samples": np.int64(100),
            },
        }
        export_backtest_artifacts(
            run_dir=run_dir,
            fold_data={},
            pooled_metrics=pooled_metrics,
            diagnostics_df=pd.DataFrame(),
            pick_log=pd.DataFrame(
                columns=["game_id", "correct", "confidence", "model_agreement_pct"]
            ),
            report_md="",
        )
        with open(run_dir / "diagnostics" / "pooled_metrics.json") as f:
            loaded = json.load(f)
        assert loaded["ensemble"]["brier_score"] == pytest.approx(0.18)
        assert loaded["ensemble"]["n_samples"] == 100
