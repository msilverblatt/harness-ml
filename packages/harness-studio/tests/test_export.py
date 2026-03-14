"""Tests for static dashboard HTML export."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from harnessml.studio.export import export_html, gather_export_data
from harnessml.studio.server import app


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with config, experiments, and outputs."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text(
        "name: test-project\n"
        "data:\n"
        "  target_column: Survived\n"
        "  task: binary\n"
        "  features_file: features.parquet\n"
        "ensemble:\n"
        "  method: stacking\n"
        "  calibration:\n"
        "    type: isotonic\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n"
        "  xgb_1:\n"
        "    type: xgboost\n"
        "    features: [a, b, c]\n"
        "    active: true\n"
        "  lgb_1:\n"
        "    type: lightgbm\n"
        "    features: [a, d]\n"
        "    active: true\n"
    )

    # Create a run with metrics
    run_dir = tmp_path / "outputs" / "run_001" / "diagnostics"
    run_dir.mkdir(parents=True)
    (run_dir / "pooled_metrics.json").write_text(json.dumps({
        "ensemble": {"brier": 0.14, "accuracy": 0.81, "ece": 0.03},
        "meta_coefficients": {"xgb_1": 1.5, "lgb_1": 0.8},
    }))

    # Create experiments
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    exp1_dir = exp_dir / "exp_001"
    exp1_dir.mkdir()
    (exp1_dir / "hypothesis.txt").write_text("Adding feature interactions will improve brier score")
    (exp1_dir / "conclusion.txt").write_text("Feature interactions improved brier by 0.005")
    (exp1_dir / "results.json").write_text(json.dumps({
        "metrics": {"brier": 0.145, "accuracy": 0.80},
        "primary_metric": "brier",
        "primary_delta": -0.005,
        "verdict": "improved",
    }))

    exp2_dir = exp_dir / "exp_002"
    exp2_dir.mkdir()
    (exp2_dir / "hypothesis.txt").write_text("Tuning learning rate will help")
    (exp2_dir / "results.json").write_text(json.dumps({
        "metrics": {"brier": 0.14, "accuracy": 0.81},
        "primary_metric": "brier",
        "primary_delta": -0.005,
        "verdict": "improved",
    }))

    # Journal
    journal_lines = [
        json.dumps({"experiment_id": "exp_001", "status": "completed"}),
        json.dumps({"experiment_id": "exp_002", "status": "completed"}),
    ]
    (exp_dir / "journal.jsonl").write_text("\n".join(journal_lines))

    return tmp_path


@pytest.fixture
def minimal_project_dir(tmp_path):
    """Create a project dir with only config, no runs or experiments."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text(
        "data:\n  target_column: y\n  task: regression\n"
    )
    (config_dir / "models.yaml").write_text("models: {}\n")
    return tmp_path


@pytest.fixture
def client(project_dir):
    app.state.project_dir = str(project_dir)
    app.state.event_store = None
    return TestClient(app)


class TestGatherExportData:
    def test_gathers_project_info(self, project_dir):
        data = gather_export_data(project_dir)
        assert data["project_name"] == "test-project"
        assert data["task"] == "binary"
        assert data["target_column"] == "Survived"

    def test_gathers_latest_metrics(self, project_dir):
        data = gather_export_data(project_dir)
        assert data["latest_metrics"]["brier"] == 0.14
        assert data["latest_metrics"]["accuracy"] == 0.81
        assert data["primary_metric_name"] == "brier"
        assert data["primary_metric_value"] == 0.14

    def test_gathers_meta_coefficients(self, project_dir):
        data = gather_export_data(project_dir)
        assert data["meta_coefficients"]["xgb_1"] == 1.5
        assert data["meta_coefficients"]["lgb_1"] == 0.8

    def test_gathers_experiments(self, project_dir):
        data = gather_export_data(project_dir)
        assert len(data["experiments"]) == 2
        # Newest first
        assert data["experiments"][0]["experiment_id"] == "exp_002"
        assert data["experiments"][1]["experiment_id"] == "exp_001"

    def test_gathers_metric_trend(self, project_dir):
        data = gather_export_data(project_dir)
        assert len(data["metric_trend"]) == 2
        # Chronological order
        assert data["metric_trend"][0]["experiment_id"] == "exp_001"
        assert data["metric_trend"][0]["value"] == 0.145

    def test_gathers_active_models(self, project_dir):
        data = gather_export_data(project_dir)
        assert len(data["active_models"]) == 2
        names = {m["name"] for m in data["active_models"]}
        assert names == {"xgb_1", "lgb_1"}

    def test_has_export_date(self, project_dir):
        data = gather_export_data(project_dir)
        assert "UTC" in data["export_date"]

    def test_scoped_to_run_id(self, project_dir):
        data = gather_export_data(project_dir, run_id="run_001")
        assert data["latest_run_id"] == "run_001"

    def test_nonexistent_run_id_gives_empty(self, project_dir):
        data = gather_export_data(project_dir, run_id="nonexistent")
        assert data["latest_metrics"] == {}
        assert data["latest_run_id"] is None


class TestGatherMinimalData:
    def test_minimal_project_no_errors(self, minimal_project_dir):
        data = gather_export_data(minimal_project_dir)
        assert data["task"] == "regression"
        assert data["latest_metrics"] == {}
        assert data["experiments"] == []
        assert data["metric_trend"] == []
        assert data["folds"] == []
        assert data["calibration"] == []

    def test_missing_config_dir(self, tmp_path):
        """Completely empty directory should not crash."""
        data = gather_export_data(tmp_path)
        assert data["project_name"] == tmp_path.name
        assert data["latest_metrics"] == {}


class TestExportHtml:
    def test_generates_html_file(self, project_dir, tmp_path):
        out = tmp_path / "report.html"
        result = export_html(project_dir, out)
        assert result == out
        assert out.exists()
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "test-project" in html

    def test_contains_expected_sections(self, project_dir, tmp_path):
        out = tmp_path / "report.html"
        export_html(project_dir, out)
        html = out.read_text()
        assert "Metrics Summary" in html
        assert "Experiments" in html
        assert "Metric Trend" in html
        assert "Model Coefficients" in html
        assert "Per-Fold Breakdown" in html
        assert "Calibration Curve" in html

    def test_contains_metric_values(self, project_dir, tmp_path):
        out = tmp_path / "report.html"
        export_html(project_dir, out)
        html = out.read_text()
        assert "0.14" in html  # brier
        assert "0.81" in html  # accuracy

    def test_contains_experiment_data(self, project_dir, tmp_path):
        out = tmp_path / "report.html"
        export_html(project_dir, out)
        html = out.read_text()
        assert "exp_001" in html
        assert "exp_002" in html

    def test_contains_chart_js(self, project_dir, tmp_path):
        out = tmp_path / "report.html"
        export_html(project_dir, out)
        html = out.read_text()
        assert "chart.js" in html

    def test_minimal_project_export(self, minimal_project_dir, tmp_path):
        """Export with no runs/experiments should still produce valid HTML."""
        out = tmp_path / "report.html"
        export_html(minimal_project_dir, out)
        html = out.read_text()
        assert "<!DOCTYPE html>" in html
        assert "No run metrics available" in html
        assert "No experiments recorded" in html

    def test_creates_parent_dirs(self, project_dir, tmp_path):
        out = tmp_path / "nested" / "deep" / "report.html"
        export_html(project_dir, out)
        assert out.exists()


class TestExportApiEndpoint:
    def test_returns_html_content_type(self, client):
        r = client.get("/api/export")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_returns_valid_html(self, client):
        r = client.get("/api/export")
        assert "<!DOCTYPE html>" in r.text
        assert "test-project" in r.text

    def test_contains_sections(self, client):
        r = client.get("/api/export")
        assert "Metrics Summary" in r.text
        assert "Experiments" in r.text

    def test_with_run_id_param(self, client):
        r = client.get("/api/export?run_id=run_001")
        assert r.status_code == 200
        assert "<!DOCTYPE html>" in r.text
