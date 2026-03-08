"""Tests for SQLite event store."""
from __future__ import annotations

import json
import time

import pytest
from harnessml.studio.event_store import EventStore


@pytest.fixture
def store(tmp_path):
    return EventStore(tmp_path / "events.db")


class TestEventStore:
    def test_creates_db_file(self, store, tmp_path):
        store.init()
        assert (tmp_path / "events.db").exists()

    def test_record_event(self, store):
        store.init()
        store.record(tool="pipeline", action="run_backtest", params={"variant": None}, result="## Results\nBrier: 0.14", duration_ms=1234, status="success")
        events = store.query(limit=10)
        assert len(events) == 1
        assert events[0]["tool"] == "pipeline"
        assert events[0]["action"] == "run_backtest"
        assert events[0]["duration_ms"] == 1234

    def test_query_filters_by_tool(self, store):
        store.init()
        store.record(tool="models", action="add", params={}, result="ok", duration_ms=10, status="success")
        store.record(tool="pipeline", action="run_backtest", params={}, result="ok", duration_ms=20, status="success")
        events = store.query(tool="models", limit=10)
        assert len(events) == 1
        assert events[0]["tool"] == "models"

    def test_query_orders_by_timestamp_desc(self, store):
        store.init()
        store.record(tool="a", action="first", params={}, result="", duration_ms=0, status="success")
        time.sleep(0.01)
        store.record(tool="b", action="second", params={}, result="", duration_ms=0, status="success")
        events = store.query(limit=10)
        assert events[0]["action"] == "second"

    def test_query_respects_limit(self, store):
        store.init()
        for i in range(5):
            store.record(tool="t", action=f"a{i}", params={}, result="", duration_ms=0, status="success")
        events = store.query(limit=3)
        assert len(events) == 3

    def test_session_stats(self, store):
        store.init()
        store.record(tool="pipeline", action="run_backtest", params={}, result="", duration_ms=100, status="success")
        store.record(tool="models", action="add", params={}, result="", duration_ms=50, status="success")
        store.record(tool="pipeline", action="run_backtest", params={}, result="", duration_ms=200, status="error")
        stats = store.session_stats()
        assert stats["total_calls"] == 3
        assert stats["by_tool"]["pipeline"] == 2
        assert stats["errors"] == 1

    def test_params_stored_as_json(self, store):
        store.init()
        store.record(tool="models", action="add", params={"name": "xgb_1", "features": ["a", "b"]}, result="ok", duration_ms=5, status="success")
        events = store.query(limit=1)
        params = events[0]["params"]
        assert params["name"] == "xgb_1"
        assert params["features"] == ["a", "b"]
