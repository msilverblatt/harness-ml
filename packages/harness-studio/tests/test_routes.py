"""Tests for Studio REST endpoints."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from harnessml.studio.server import app


@pytest.fixture
def project_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text(
        "data:\n  target_column: Survived\n  features_file: features.parquet\nensemble:\n  method: stacking\n  calibration:\n    type: isotonic\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n  xgb_1:\n    type: xgboost\n    features: [a, b, c]\n    active: true\n  lgb_1:\n    type: lightgbm\n    features: [a, d]\n    active: true\n"
    )
    return tmp_path


@pytest.fixture
def client(project_dir):
    app.state.project_dir = str(project_dir)
    app.state.event_store = None
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestProjectEndpoints:
    def test_config(self, client):
        r = client.get("/api/project/config")
        assert r.status_code == 200
        data = r.json()
        assert "pipeline" in data
        assert "models" in data
        assert "xgb_1" in data["models"]["models"]

    def test_dag(self, client):
        r = client.get("/api/project/dag")
        assert r.status_code == 200
        dag = r.json()
        assert "nodes" in dag
        assert "edges" in dag
        node_ids = [n["id"] for n in dag["nodes"]]
        assert "model_xgb_1" in node_ids
        assert "model_lgb_1" in node_ids
        assert "ensemble" in node_ids
        assert "output" in node_ids
        # Should have edges from models to ensemble
        ensemble_edges = [e for e in dag["edges"] if e["target"] == "ensemble"]
        assert len(ensemble_edges) == 2


class TestExperimentEndpoints:
    def test_list_empty(self, client):
        r = client.get("/api/experiments")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_with_journal(self, client, project_dir):
        journal_dir = project_dir / "experiments"
        journal_dir.mkdir()
        journal = journal_dir / "journal.jsonl"
        entries = [
            {"experiment_id": "exp-001", "verdict": "keep", "metrics": {"brier": 0.14}},
            {"experiment_id": "exp-002", "verdict": "revert", "metrics": {"brier": 0.16}},
        ]
        journal.write_text("\n".join(json.dumps(e) for e in entries))
        r = client.get("/api/experiments")
        data = r.json()
        assert len(data) == 2
        assert data[0]["experiment_id"] == "exp-002"  # newest first

    def test_get_experiment(self, client, project_dir):
        exp_dir = project_dir / "experiments" / "exp-001"
        exp_dir.mkdir(parents=True)
        (exp_dir / "hypothesis.txt").write_text("More trees = better")
        (exp_dir / "conclusion.txt").write_text("Confirmed, Brier improved")
        r = client.get("/api/experiments/exp-001")
        data = r.json()
        assert data["hypothesis"] == "More trees = better"
        assert data["conclusion"] == "Confirmed, Brier improved"


class TestRunEndpoints:
    def test_list_empty(self, client):
        r = client.get("/api/runs")
        assert r.status_code == 200
        assert r.json() == []

    def test_list_with_runs(self, client, project_dir):
        run_dir = project_dir / "outputs" / "20260307_120000"
        run_dir.mkdir(parents=True)
        (run_dir / "pooled_metrics.json").write_text(json.dumps({"brier": 0.14, "accuracy": 0.82}))
        r = client.get("/api/runs")
        data = r.json()
        assert len(data) == 1
        assert data[0]["id"] == "20260307_120000"
        assert data[0]["metrics"]["brier"] == 0.14

    def test_run_metrics(self, client, project_dir):
        run_dir = project_dir / "outputs" / "run1"
        run_dir.mkdir(parents=True)
        (run_dir / "pooled_metrics.json").write_text(json.dumps({"brier": 0.13}))
        r = client.get("/api/runs/run1/metrics")
        assert r.json()["brier"] == 0.13
