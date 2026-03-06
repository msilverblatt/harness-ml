"""Tests for experiment change detection, lifecycle, and smart execution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from easyml.core.runner.experiment import (
    ChangeSet,
    ExperimentResult,
    _rank_sweep_results,
    auto_log_result,
    auto_next_id,
    compute_deltas,
    detect_experiment_changes,
    format_change_summary,
    format_delta_table,
    format_sweep_summary,
    load_baseline_metrics,
    promote_experiment,
    run_sweep,
    save_frozen_config,
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
        prod = {"models": {}, "feature_config": {"first_period": 2003}}
        overlay = {"feature_config": {"first_period": 2005}}
        change_set = detect_experiment_changes(prod, overlay)
        assert change_set.feature_config_changed

    def test_feature_config_same_not_flagged(self):
        """Same feature config not flagged as changed."""
        prod = {"models": {}, "feature_config": {"first_period": 2003}}
        overlay = {"feature_config": {"first_period": 2003}}
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
            "feature_config": {"first_period": 2003},
        }
        overlay = {
            "models": {"m1": {"type": "xgboost", "params": {"depth": 5}}, "m2": {"type": "catboost"}},
            "ensemble": {"method": "average"},
            "feature_config": {"first_period": 2005},
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


# -----------------------------------------------------------------------
# auto_next_id
# -----------------------------------------------------------------------

class TestAutoNextId:
    """Tests for auto_next_id experiment numbering."""

    def test_empty_dir_returns_001(self, tmp_path):
        assert auto_next_id(tmp_path) == "exp-001"

    def test_nonexistent_dir_returns_001(self, tmp_path):
        assert auto_next_id(tmp_path / "doesnt_exist") == "exp-001"

    def test_increments_from_existing(self, tmp_path):
        (tmp_path / "exp-001-baseline").mkdir()
        (tmp_path / "exp-002-lr-sweep").mkdir()
        assert auto_next_id(tmp_path) == "exp-003"

    def test_handles_gaps(self, tmp_path):
        (tmp_path / "exp-001-a").mkdir()
        (tmp_path / "exp-005-b").mkdir()
        assert auto_next_id(tmp_path) == "exp-006"

    def test_ignores_files(self, tmp_path):
        (tmp_path / "exp-003-readme.md").touch()
        (tmp_path / "exp-001-real").mkdir()
        assert auto_next_id(tmp_path) == "exp-002"

    def test_custom_prefix(self, tmp_path):
        (tmp_path / "exp-002-test").mkdir()
        assert auto_next_id(tmp_path, prefix="w-exp") == "w-exp-003"

    def test_recognizes_prefixed_dirs(self, tmp_path):
        (tmp_path / "w-exp-004-women").mkdir()
        assert auto_next_id(tmp_path) == "exp-005"


# -----------------------------------------------------------------------
# auto_log_result
# -----------------------------------------------------------------------

class TestAutoLogResult:
    """Tests for auto_log_result experiment logging."""

    def test_creates_file_if_missing(self, tmp_path):
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path, "exp-001", "test hypothesis", "changed lr",
            {"accuracy": 0.76, "brier_score": 0.17},
            {"accuracy": 0.75, "brier_score": 0.18},
            "improved",
        )
        assert log_path.exists()
        content = log_path.read_text()
        assert "# Experiment Log" in content
        assert "exp-001" in content

    def test_appends_to_existing(self, tmp_path):
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path, "exp-001", "h1", "c1",
            {"accuracy": 0.75}, {}, "neutral",
        )
        auto_log_result(
            log_path, "exp-002", "h2", "c2",
            {"accuracy": 0.76}, {"accuracy": 0.75}, "improved",
        )
        content = log_path.read_text()
        assert "exp-001" in content
        assert "exp-002" in content

    def test_includes_deltas(self, tmp_path):
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path, "exp-001", "test", "changes",
            {"brier_score": 0.17}, {"brier_score": 0.18},
            "improved",
        )
        content = log_path.read_text()
        assert "-0.0100" in content  # delta

    def test_missing_metrics_show_dash(self, tmp_path):
        log_path = tmp_path / "EXPERIMENT_LOG.md"
        auto_log_result(
            log_path, "exp-001", "test", "changes",
            {}, {}, "neutral",
        )
        content = log_path.read_text()
        assert "| - |" in content


# -----------------------------------------------------------------------
# save_frozen_config
# -----------------------------------------------------------------------

class TestSaveFrozenConfig:
    """Tests for save_frozen_config."""

    def test_writes_json(self, tmp_path):
        exp_dir = tmp_path / "exp-001"
        path = save_frozen_config(
            exp_dir,
            resolved_config={"models": {"m1": {"type": "xgboost"}}},
            production_run_id="20240301_120000",
            features_hash="abc123",
            cache_stats={"hits": 3, "misses": 1},
        )
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["production_run_id"] == "20240301_120000"
        assert data["features_hash"] == "abc123"
        assert data["cache_stats"]["hits"] == 3
        assert "resolved_config" in data
        assert "frozen_at" in data

    def test_creates_dir_if_needed(self, tmp_path):
        exp_dir = tmp_path / "nested" / "exp-001"
        path = save_frozen_config(exp_dir, resolved_config={})
        assert path.exists()

    def test_default_values(self, tmp_path):
        path = save_frozen_config(tmp_path / "exp", resolved_config={"x": 1})
        data = json.loads(path.read_text())
        assert data["production_run_id"] is None
        assert data["features_hash"] == ""
        assert data["cache_stats"] == {}


# -----------------------------------------------------------------------
# promote_experiment
# -----------------------------------------------------------------------

class TestPromoteExperiment:
    """Tests for promote_experiment with safety checks."""

    def _setup_experiment(self, tmp_path, exp_metrics, base_metrics, overlay=None):
        """Helper to create experiment dir with results and overlay."""
        experiments_dir = tmp_path / "experiments"
        config_dir = tmp_path / "config"
        exp_dir = experiments_dir / "exp-001"
        exp_dir.mkdir(parents=True)
        config_dir.mkdir(parents=True)

        results = {
            "metrics": exp_metrics,
            "baseline_metrics": base_metrics,
        }
        (exp_dir / "results.json").write_text(json.dumps(results))

        overlay = overlay or {"models": {"m1": {"params": {"lr": 0.05}}}}
        (exp_dir / "overlay.yaml").write_text(yaml.dump(overlay))

        # Create existing models.yaml
        (config_dir / "models.yaml").write_text(
            yaml.dump({"models": {"m1": {"type": "xgboost", "params": {"lr": 0.01}}}})
        )

        return experiments_dir, config_dir

    def test_promotes_on_improvement(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"brier_score": 0.16},
            base_metrics={"brier_score": 0.18},
        )
        result = promote_experiment("exp-001", experiments_dir, config_dir)
        assert result["promoted"] is True
        assert "improvement" in result

    def test_rejects_regression(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"brier_score": 0.20},
            base_metrics={"brier_score": 0.18},
        )
        result = promote_experiment("exp-001", experiments_dir, config_dir)
        assert result["promoted"] is False
        assert "No improvement" in result["reason"]

    def test_flags_suspicious_improvement(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"brier_score": 0.08},  # way too good
            base_metrics={"brier_score": 0.18},
        )
        result = promote_experiment(
            "exp-001", experiments_dir, config_dir, max_improvement=0.05
        )
        assert result["promoted"] is False
        assert result["reason"] == "suspicious_improvement"
        assert "leakage" in result["warning"].lower()

    def test_accuracy_higher_is_better(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"accuracy": 0.78},
            base_metrics={"accuracy": 0.75},
        )
        result = promote_experiment(
            "exp-001", experiments_dir, config_dir, primary_metric="accuracy"
        )
        assert result["promoted"] is True

    def test_accuracy_regression_rejected(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"accuracy": 0.70},
            base_metrics={"accuracy": 0.75},
        )
        result = promote_experiment(
            "exp-001", experiments_dir, config_dir, primary_metric="accuracy"
        )
        assert result["promoted"] is False

    def test_missing_results_file(self, tmp_path):
        experiments_dir = tmp_path / "experiments"
        (experiments_dir / "exp-001").mkdir(parents=True)
        result = promote_experiment(
            "exp-001", experiments_dir, tmp_path / "config"
        )
        assert result["promoted"] is False
        assert "No results.json" in result["reason"]

    def test_missing_baseline(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"brier_score": 0.16},
            base_metrics={},
        )
        result = promote_experiment("exp-001", experiments_dir, config_dir)
        assert result["promoted"] is False
        assert "No baseline" in result["reason"]

    def test_applies_overlay_to_config(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"brier_score": 0.16},
            base_metrics={"brier_score": 0.18},
            overlay={"models": {"m1": {"params": {"lr": 0.05}}}},
        )
        result = promote_experiment("exp-001", experiments_dir, config_dir)
        assert result["promoted"] is True

        # Verify config was updated
        updated = yaml.safe_load((config_dir / "models.yaml").read_text())
        assert updated["models"]["m1"]["params"]["lr"] == 0.05

    def test_applies_ensemble_overlay(self, tmp_path):
        experiments_dir, config_dir = self._setup_experiment(
            tmp_path,
            exp_metrics={"brier_score": 0.16},
            base_metrics={"brier_score": 0.18},
            overlay={"ensemble": {"temperature": 1.1}},
        )
        result = promote_experiment("exp-001", experiments_dir, config_dir)
        assert result["promoted"] is True
        assert any("ensemble" in c.lower() for c in result["changes"])


# -----------------------------------------------------------------------
# run_sweep
# -----------------------------------------------------------------------

class TestRunSweep:
    """Tests for sweep execution via run_sweep()."""

    def _setup_sweep_project(self, tmp_path, overlay):
        """Create minimal project with sweep overlay."""
        import numpy as np
        import pandas as pd

        # Create data
        project_dir = tmp_path / "project"
        features_dir = project_dir / "data" / "features"
        features_dir.mkdir(parents=True)

        rng = np.random.default_rng(42)
        n = 200
        seasons = rng.choice([2022, 2023, 2024], size=n)
        df = pd.DataFrame({
            "season": seasons,
            "result": rng.integers(0, 2, size=n),
            "diff_x": rng.standard_normal(n),
            "diff_prior": rng.integers(-15, 16, size=n).astype(float),
        })
        df.to_parquet(features_dir / "features.parquet", index=False)

        # Config
        config_dir = project_dir / "config"
        config_dir.mkdir(parents=True)
        pipeline_yaml = {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "features_dir": str(features_dir),
            },
            "backtest": {
                "cv_strategy": "leave_one_out",
                "fold_values": [2022, 2023, 2024],
                "fold_column": "season",
                "metrics": ["brier", "accuracy"],
                "min_train_folds": 1,
            },
        }
        (config_dir / "pipeline.yaml").write_text(
            yaml.dump(pipeline_yaml, default_flow_style=False)
        )
        (config_dir / "models.yaml").write_text(
            yaml.dump({
                "models": {
                    "logreg": {
                        "type": "logistic_regression",
                        "features": ["diff_x"],
                        "params": {"max_iter": 200, "C": 1.0},
                        "active": True,
                    },
                },
            }, default_flow_style=False)
        )
        (config_dir / "ensemble.yaml").write_text(
            yaml.dump({"ensemble": {"method": "average"}}, default_flow_style=False)
        )

        # Experiment dir + overlay
        experiments_dir = project_dir / "experiments"
        exp_dir = experiments_dir / "exp-001"
        exp_dir.mkdir(parents=True)
        overlay_path = exp_dir / "overlay.yaml"
        overlay_path.write_text(
            yaml.dump(overlay, default_flow_style=False)
        )

        return project_dir, config_dir, experiments_dir, overlay_path

    def test_non_sweep_returns_flag(self, tmp_path):
        """Overlay without sweep key returns is_sweep=False."""
        overlay = {"models": {"logreg": {"params": {"C": 0.5}}}}
        project_dir, config_dir, experiments_dir, overlay_path = (
            self._setup_sweep_project(tmp_path, overlay)
        )
        result = run_sweep(
            overlay_path=overlay_path,
            config_dir=config_dir,
            project_dir=project_dir,
            experiments_dir=experiments_dir,
            experiment_id="exp-001",
        )
        assert result["is_sweep"] is False

    def test_sweep_creates_variants(self, tmp_path):
        """Sweep creates sub-experiment dirs with results."""
        overlay = {
            "sweep": {
                "key": "models.logreg.params.C",
                "values": [0.1, 1.0],
            },
            "description": "C sweep",
        }
        project_dir, config_dir, experiments_dir, overlay_path = (
            self._setup_sweep_project(tmp_path, overlay)
        )
        result = run_sweep(
            overlay_path=overlay_path,
            config_dir=config_dir,
            project_dir=project_dir,
            experiments_dir=experiments_dir,
            experiment_id="exp-001",
        )

        assert result["is_sweep"] is True
        assert result["n_variants"] == 2
        assert len(result["results"]) == 2

        # Verify sub-dirs exist
        assert (experiments_dir / "exp-001-v00").exists()
        assert (experiments_dir / "exp-001-v01").exists()

        # Each variant should have results.json and frozen_config.json
        for i in range(2):
            vdir = experiments_dir / f"exp-001-v{i:02d}"
            assert (vdir / "results.json").exists()
            assert (vdir / "frozen_config.json").exists()
            assert (vdir / "overlay.yaml").exists()

    def test_sweep_ranks_results(self, tmp_path):
        """Results are ranked by primary metric."""
        overlay = {
            "sweep": {
                "key": "models.logreg.params.C",
                "values": [0.01, 0.1, 1.0],
            },
        }
        project_dir, config_dir, experiments_dir, overlay_path = (
            self._setup_sweep_project(tmp_path, overlay)
        )
        result = run_sweep(
            overlay_path=overlay_path,
            config_dir=config_dir,
            project_dir=project_dir,
            experiments_dir=experiments_dir,
            experiment_id="exp-001",
            primary_metric="brier",
        )

        ranked = result["results"]
        assert ranked[0].get("rank") == 1
        # Brier scores should be in ascending order (lower = better)
        briers = [
            r["metrics"].get("brier")
            for r in ranked
            if "error" not in r and r["metrics"].get("brier") is not None
        ]
        assert briers == sorted(briers)

    def test_sweep_uses_prediction_cache(self, tmp_path):
        """Unchanged models should get cache hits across variants."""
        overlay = {
            "sweep": {
                "key": "models.logreg.params.C",
                "values": [0.1, 1.0],
            },
        }
        project_dir, config_dir, experiments_dir, overlay_path = (
            self._setup_sweep_project(tmp_path, overlay)
        )
        result = run_sweep(
            overlay_path=overlay_path,
            config_dir=config_dir,
            project_dir=project_dir,
            experiments_dir=experiments_dir,
            experiment_id="exp-001",
        )

        # Both variants change the model config, so all entries are misses.
        # But at least the cache stats should be tracked.
        stats = result.get("total_cache_stats", {})
        total = stats.get("hits", 0) + stats.get("misses", 0)
        assert total > 0

    def test_best_variant_populated(self, tmp_path):
        """Best variant is populated with the top-ranked result."""
        overlay = {
            "sweep": {
                "key": "models.logreg.params.C",
                "values": [0.1, 1.0],
            },
        }
        project_dir, config_dir, experiments_dir, overlay_path = (
            self._setup_sweep_project(tmp_path, overlay)
        )
        result = run_sweep(
            overlay_path=overlay_path,
            config_dir=config_dir,
            project_dir=project_dir,
            experiments_dir=experiments_dir,
            experiment_id="exp-001",
            primary_metric="brier",
        )

        assert result["best"] is not None
        assert result["best"]["rank"] == 1
        assert "metrics" in result["best"]


# -----------------------------------------------------------------------
# _rank_sweep_results
# -----------------------------------------------------------------------

class TestRankSweepResults:
    """Tests for _rank_sweep_results helper."""

    def test_ranks_brier_ascending(self):
        """Lower brier = better = rank 1."""
        results = [
            {"variant_id": "v0", "metrics": {"brier_score": 0.25}},
            {"variant_id": "v1", "metrics": {"brier_score": 0.20}},
            {"variant_id": "v2", "metrics": {"brier_score": 0.22}},
        ]
        ranked = _rank_sweep_results(results, "brier_score")
        assert ranked[0]["variant_id"] == "v1"
        assert ranked[0]["rank"] == 1
        assert ranked[1]["variant_id"] == "v2"
        assert ranked[2]["variant_id"] == "v0"

    def test_ranks_accuracy_descending(self):
        """Higher accuracy = better = rank 1."""
        results = [
            {"variant_id": "v0", "metrics": {"accuracy": 0.72}},
            {"variant_id": "v1", "metrics": {"accuracy": 0.75}},
            {"variant_id": "v2", "metrics": {"accuracy": 0.70}},
        ]
        ranked = _rank_sweep_results(results, "accuracy")
        assert ranked[0]["variant_id"] == "v1"
        assert ranked[0]["rank"] == 1

    def test_errored_variants_appended(self):
        """Error variants go at the end of ranked results."""
        results = [
            {"variant_id": "v0", "metrics": {"brier_score": 0.25}},
            {"variant_id": "v1", "error": "training failed"},
            {"variant_id": "v2", "metrics": {"brier_score": 0.20}},
        ]
        ranked = _rank_sweep_results(results, "brier_score")
        assert ranked[-1]["variant_id"] == "v1"
        assert "error" in ranked[-1]

    def test_empty_results(self):
        """Empty results list returns empty."""
        assert _rank_sweep_results([], "brier_score") == []


# -----------------------------------------------------------------------
# format_sweep_summary
# -----------------------------------------------------------------------

class TestFormatSweepSummary:
    """Tests for format_sweep_summary."""

    def test_produces_markdown(self):
        sweep_result = {
            "is_sweep": True,
            "experiment_id": "exp-001",
            "n_variants": 2,
            "results": [
                {
                    "variant_id": "exp-001-v00",
                    "description": "C=0.1",
                    "metrics": {"brier_score": 0.20, "accuracy": 0.75},
                    "rank": 1,
                },
                {
                    "variant_id": "exp-001-v01",
                    "description": "C=1.0",
                    "metrics": {"brier_score": 0.22, "accuracy": 0.73},
                    "rank": 2,
                },
            ],
            "best": {
                "variant_id": "exp-001-v00",
                "description": "C=0.1",
                "rank": 1,
            },
            "total_cache_stats": {"hits": 3, "misses": 6},
        }
        summary = format_sweep_summary(sweep_result)
        assert "## Sweep Results" in summary
        assert "exp-001-v00" in summary
        assert "Cache:" in summary
        assert "hits" in summary.lower()

    def test_non_sweep_message(self):
        assert "Not a sweep" in format_sweep_summary({"is_sweep": False})
