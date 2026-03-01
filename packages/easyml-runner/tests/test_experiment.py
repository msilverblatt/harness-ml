"""Tests for experiment change detection and smart execution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyml.runner.experiment import (
    ChangeSet,
    ExperimentResult,
    compute_deltas,
    detect_experiment_changes,
    format_change_summary,
    format_delta_table,
    load_baseline_metrics,
)


# -----------------------------------------------------------------------
# ChangeSet dataclass
# -----------------------------------------------------------------------

class TestChangeSet:
    """ChangeSet property tests."""

    def test_empty_change_set(self):
        cs = ChangeSet()
        assert cs.total_changes == 0
        assert not cs.ensemble_only

    def test_ensemble_only_true(self):
        cs = ChangeSet(ensemble_changed=True)
        assert cs.ensemble_only is True
        assert cs.total_changes == 1

    def test_ensemble_only_false_with_model_changes(self):
        cs = ChangeSet(changed_models=["m1"], ensemble_changed=True)
        assert cs.ensemble_only is False

    def test_ensemble_only_false_with_feature_config(self):
        cs = ChangeSet(ensemble_changed=True, feature_config_changed=True)
        assert cs.ensemble_only is False

    def test_total_changes_counts_all(self):
        cs = ChangeSet(
            changed_models=["m1"],
            new_models=["m2", "m3"],
            removed_models=["m4"],
            ensemble_changed=True,
            feature_config_changed=True,
        )
        assert cs.total_changes == 6


# -----------------------------------------------------------------------
# ExperimentResult dataclass
# -----------------------------------------------------------------------

class TestExperimentResult:
    """ExperimentResult instantiation."""

    def test_defaults(self):
        er = ExperimentResult(experiment_id="exp-001")
        assert er.experiment_id == "exp-001"
        assert er.metrics == {}
        assert er.baseline_metrics == {}
        assert er.deltas == {}
        assert er.models_trained == []
        assert er.change_set.total_changes == 0


# -----------------------------------------------------------------------
# detect_experiment_changes
# -----------------------------------------------------------------------

class TestDetectExperimentChanges:
    """Tests for detect_experiment_changes function."""

    def test_no_changes(self):
        """Empty overlay produces no changes."""
        prod = {"models": {"m1": {"type": "xgboost"}}}
        change_set = detect_experiment_changes(prod, {})
        assert change_set.total_changes == 0

    def test_changed_model(self):
        """Modified model detected."""
        prod = {"models": {"m1": {"type": "xgboost", "params": {"depth": 3}}}}
        overlay = {"models": {"m1": {"type": "xgboost", "params": {"depth": 5}}}}
        change_set = detect_experiment_changes(prod, overlay)
        assert "m1" in change_set.changed_models

    def test_new_model(self):
        """New model in overlay detected."""
        prod = {"models": {"m1": {"type": "xgboost"}}}
        overlay = {"models": {"m1": {"type": "xgboost"}, "m2": {"type": "catboost"}}}
        change_set = detect_experiment_changes(prod, overlay)
        assert "m2" in change_set.new_models

    def test_ensemble_only(self):
        """Only ensemble changes produces ensemble_only True."""
        prod = {"models": {"m1": {"type": "xgboost"}}, "ensemble": {"method": "stacked"}}
        overlay = {"ensemble": {"method": "average"}}
        change_set = detect_experiment_changes(prod, overlay)
        assert change_set.ensemble_only

    def test_feature_config_changed(self):
        """Feature config change detected."""
        prod = {"models": {}, "feature_config": {"first_season": 2003}}
        overlay = {"feature_config": {"first_season": 2005}}
        change_set = detect_experiment_changes(prod, overlay)
        assert change_set.feature_config_changed

    def test_feature_config_same_not_flagged(self):
        """Same feature config not flagged as changed."""
        prod = {"models": {}, "feature_config": {"first_season": 2003}}
        overlay = {"feature_config": {"first_season": 2003}}
        change_set = detect_experiment_changes(prod, overlay)
        assert not change_set.feature_config_changed

    def test_removed_model(self):
        """Model in production but not in overlay models section is removed."""
        prod = {"models": {"m1": {"type": "xgboost"}, "m2": {"type": "catboost"}}}
        overlay = {"models": {"m1": {"type": "xgboost"}}}
        change_set = detect_experiment_changes(prod, overlay)
        assert "m2" in change_set.removed_models

    def test_multiple_changes(self):
        """Multiple change types detected simultaneously."""
        prod = {
            "models": {"m1": {"type": "xgboost", "params": {"depth": 3}}},
            "ensemble": {"method": "stacked"},
            "feature_config": {"first_season": 2003},
        }
        overlay = {
            "models": {"m1": {"type": "xgboost", "params": {"depth": 5}}, "m2": {"type": "catboost"}},
            "ensemble": {"method": "average"},
            "feature_config": {"first_season": 2005},
        }
        change_set = detect_experiment_changes(prod, overlay)
        assert "m1" in change_set.changed_models
        assert "m2" in change_set.new_models
        assert change_set.ensemble_changed
        assert change_set.feature_config_changed


# -----------------------------------------------------------------------
# load_baseline_metrics
# -----------------------------------------------------------------------

class TestLoadBaselineMetrics:
    """Tests for load_baseline_metrics function."""

    def test_loads_from_file(self, tmp_path):
        """Loads pooled_metrics.json from diagnostics dir."""
        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        metrics = {"brier_score": 0.18, "accuracy": 0.75}
        (diag_dir / "pooled_metrics.json").write_text(json.dumps(metrics))
        result = load_baseline_metrics(tmp_path)
        assert result["brier_score"] == 0.18
        assert result["accuracy"] == 0.75

    def test_returns_empty_when_missing(self, tmp_path):
        """Returns empty dict when file doesn't exist."""
        result = load_baseline_metrics(tmp_path)
        assert result == {}

    def test_returns_empty_on_invalid_json(self, tmp_path):
        """Returns empty dict when file contains invalid JSON."""
        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        (diag_dir / "pooled_metrics.json").write_text("not valid json{{{")
        result = load_baseline_metrics(tmp_path)
        assert result == {}

    def test_returns_empty_when_diagnostics_dir_missing(self, tmp_path):
        """Returns empty dict when diagnostics subdir doesn't exist."""
        result = load_baseline_metrics(tmp_path)
        assert result == {}


# -----------------------------------------------------------------------
# compute_deltas
# -----------------------------------------------------------------------

class TestComputeDeltas:
    """Tests for compute_deltas function."""

    def test_basic_deltas(self):
        """Computes experiment - baseline."""
        exp = {"brier_score": 0.17, "accuracy": 0.76}
        base = {"brier_score": 0.18, "accuracy": 0.75}
        deltas = compute_deltas(exp, base)
        assert deltas["brier_score"] == pytest.approx(-0.01)
        assert deltas["accuracy"] == pytest.approx(0.01)

    def test_only_common_keys(self):
        """Only computes deltas for keys present in both dicts."""
        exp = {"brier_score": 0.17, "extra_metric": 0.5}
        base = {"brier_score": 0.18, "other_metric": 0.3}
        deltas = compute_deltas(exp, base)
        assert "brier_score" in deltas
        assert "extra_metric" not in deltas
        assert "other_metric" not in deltas

    def test_empty_inputs(self):
        """Empty dicts produce empty deltas."""
        deltas = compute_deltas({}, {})
        assert deltas == {}

    def test_non_numeric_values_skipped(self):
        """Non-numeric values are skipped gracefully."""
        exp = {"brier_score": 0.17, "label": "test"}
        base = {"brier_score": 0.18, "label": "baseline"}
        deltas = compute_deltas(exp, base)
        assert "brier_score" in deltas
        assert "label" not in deltas


# -----------------------------------------------------------------------
# format_change_summary
# -----------------------------------------------------------------------

class TestFormatChangeSummary:
    """Tests for format_change_summary function."""

    def test_no_changes(self):
        """No changes produces appropriate message."""
        cs = ChangeSet()
        summary = format_change_summary(cs)
        assert "No changes" in summary

    def test_formats_changed_models(self):
        """Summary includes changed model names."""
        cs = ChangeSet(changed_models=["m1", "m2"])
        summary = format_change_summary(cs)
        assert "m1" in summary
        assert "m2" in summary
        assert "Changed models" in summary

    def test_formats_new_models(self):
        """Summary includes new model names."""
        cs = ChangeSet(new_models=["m_new"])
        summary = format_change_summary(cs)
        assert "m_new" in summary
        assert "New models" in summary

    def test_formats_removed_models(self):
        """Summary includes removed model names."""
        cs = ChangeSet(removed_models=["m_old"])
        summary = format_change_summary(cs)
        assert "m_old" in summary
        assert "Removed models" in summary

    def test_formats_ensemble_changed(self):
        """Summary mentions ensemble config change."""
        cs = ChangeSet(ensemble_changed=True)
        summary = format_change_summary(cs)
        assert "Ensemble" in summary

    def test_formats_feature_config_changed(self):
        """Summary mentions feature config change."""
        cs = ChangeSet(feature_config_changed=True)
        summary = format_change_summary(cs)
        assert "Feature config" in summary

    def test_total_count_in_header(self):
        """Summary header includes total change count."""
        cs = ChangeSet(changed_models=["m1"], ensemble_changed=True)
        summary = format_change_summary(cs)
        assert "2 change(s)" in summary


# -----------------------------------------------------------------------
# format_delta_table
# -----------------------------------------------------------------------

class TestFormatDeltaTable:
    """Tests for format_delta_table function."""

    def test_formats_table(self):
        """Delta table includes all metrics."""
        exp = {"brier_score": 0.17, "accuracy": 0.76}
        base = {"brier_score": 0.18, "accuracy": 0.75}
        deltas = {"brier_score": -0.01, "accuracy": 0.01}
        table = format_delta_table(exp, base, deltas)
        assert "Metric" in table
        assert "Baseline" in table
        assert "Experiment" in table
        assert "Delta" in table
        assert "brier_score" in table
        assert "accuracy" in table

    def test_delta_sign_formatting(self):
        """Positive deltas get + prefix, negative get -."""
        exp = {"brier_score": 0.17}
        base = {"brier_score": 0.18}
        deltas = {"brier_score": -0.01}
        table = format_delta_table(exp, base, deltas)
        assert "-0.0100" in table

    def test_positive_delta_formatting(self):
        """Positive delta shows + sign."""
        exp = {"accuracy": 0.76}
        base = {"accuracy": 0.75}
        deltas = {"accuracy": 0.01}
        table = format_delta_table(exp, base, deltas)
        assert "+0.0100" in table

    def test_empty_deltas(self):
        """Empty deltas produce header-only table."""
        table = format_delta_table({}, {}, {})
        assert "Metric" in table
        # Should only have header rows
        lines = table.strip().split("\n")
        assert len(lines) == 2
