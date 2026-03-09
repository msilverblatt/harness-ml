"""Tests for Studio REST endpoints."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from harnessml.studio.event_store import EventStore
from harnessml.studio.server import app


@pytest.fixture
def project_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text(
        "data:\n  target_column: Survived\n  task: binary\n  features_file: features.parquet\nensemble:\n  method: stacking\n  calibration:\n    type: isotonic\n"
    )
    (config_dir / "models.yaml").write_text(
        "models:\n  xgb_1:\n    type: xgboost\n    features: [a, b, c]\n    active: true\n  lgb_1:\n    type: lightgbm\n    features: [a, d]\n    active: true\n"
    )
    return tmp_path


@pytest.fixture
def event_store(tmp_path):
    store = EventStore(tmp_path / "test_events.db")
    store.init()
    return store


@pytest.fixture
def client(project_dir):
    app.state.project_dir = str(project_dir)
    app.state.event_store = None
    return TestClient(app)


@pytest.fixture
def client_with_store(project_dir, event_store):
    app.state.project_dir = str(project_dir)
    app.state.event_store = event_store
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_health_returns_ok_status(self, client):
        """GET /api/health returns 200 with status ok."""
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert body["status"] == "ok"


class TestEventsEndpoint:
    def test_events_endpoint_empty_no_store(self, client):
        """Returns empty list when event_store is None."""
        r = client.get("/api/events")
        assert r.status_code == 200
        assert r.json() == []

    def test_events_endpoint_empty_with_store(self, client_with_store):
        """Returns empty list when store has no events."""
        r = client_with_store.get("/api/events")
        assert r.status_code == 200
        assert r.json() == []

    def test_events_endpoint_with_data(self, client_with_store, event_store):
        """Returns events after recording them in the store."""
        event_store.record(
            tool="models",
            action="list",
            params={"project": "test"},
            result="listed 3 models",
            duration_ms=42,
            status="success",
            project="test-project",
        )
        event_store.record(
            tool="features",
            action="add",
            params={"name": "feat_1"},
            result="added feature feat_1",
            duration_ms=15,
            status="success",
            project="test-project",
        )
        r = client_with_store.get("/api/events")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 2
        # newest first
        assert data[0]["tool"] == "features"
        assert data[1]["tool"] == "models"
        # check fields present
        for event in data:
            assert "id" in event
            assert "timestamp" in event
            assert "action" in event
            assert "status" in event

    def test_events_endpoint_filter_by_tool(self, client_with_store, event_store):
        """Filtering by tool parameter returns only matching events."""
        event_store.record(
            tool="models", action="list", params={},
            result="ok", duration_ms=10, status="success",
        )
        event_store.record(
            tool="features", action="add", params={},
            result="ok", duration_ms=10, status="success",
        )
        r = client_with_store.get("/api/events?tool=models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 1
        assert data[0]["tool"] == "models"

    def test_events_stats_empty(self, client_with_store):
        """Stats endpoint returns zeroes when no events exist."""
        r = client_with_store.get("/api/events/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_calls"] == 0
        assert data["errors"] == 0

    def test_events_stats_with_data(self, client_with_store, event_store):
        """Stats endpoint reflects recorded events."""
        event_store.record(
            tool="models", action="list", params={},
            result="ok", duration_ms=10, status="success",
        )
        event_store.record(
            tool="models", action="create", params={},
            result="error", duration_ms=5, status="error",
        )
        r = client_with_store.get("/api/events/stats")
        assert r.status_code == 200
        data = r.json()
        assert data["total_calls"] == 2
        assert data["errors"] == 1
        assert data["by_tool"]["models"] == 2


class TestProjectStatus:
    def test_project_status(self, client, project_dir):
        """GET /api/project/status returns project snapshot."""
        r = client.get("/api/project/status")
        assert r.status_code == 200
        data = r.json()
        assert data["target_column"] == "Survived"
        assert data["task"] == "binary"
        assert data["active_models"] == 2
        assert data["model_types_tried"] == 2
        assert data["experiments_run"] == 0
        assert data["run_count"] == 0
        assert isinstance(data["feature_count"], int)
        assert data["feature_count"] > 0
        assert isinstance(data["latest_metrics"], dict)

    def test_project_status_with_experiments(self, client, project_dir):
        """Status reflects experiment and run counts."""
        # Add experiments
        journal_dir = project_dir / "experiments"
        journal_dir.mkdir()
        entries = [
            {"experiment_id": "exp-001", "verdict": "keep", "metrics": {"brier": 0.14}},
            {"experiment_id": "exp-002", "verdict": "revert", "metrics": {"brier": 0.16}},
        ]
        (journal_dir / "journal.jsonl").write_text(
            "\n".join(json.dumps(e) for e in entries)
        )
        # Add a run
        run_dir = project_dir / "outputs" / "20260307_120000"
        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(parents=True)
        (diag_dir / "pooled_metrics.json").write_text(
            json.dumps({"brier": 0.14, "accuracy": 0.82})
        )

        r = client.get("/api/project/status")
        data = r.json()
        assert data["experiments_run"] == 2
        assert data["run_count"] == 1
        assert data["latest_metrics"]["brier"] == 0.14


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
        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(parents=True)
        (diag_dir / "pooled_metrics.json").write_text(json.dumps({"brier": 0.14, "accuracy": 0.82}))
        r = client.get("/api/runs")
        data = r.json()
        assert len(data) == 1
        assert data[0]["id"] == "20260307_120000"
        assert data[0]["metrics"]["brier"] == 0.14

    def test_run_metrics(self, client, project_dir):
        run_dir = project_dir / "outputs" / "run1"
        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(parents=True)
        (diag_dir / "pooled_metrics.json").write_text(json.dumps({"brier": 0.13}))
        r = client.get("/api/runs/run1/metrics")
        assert r.json()["metrics"]["brier"] == 0.13
