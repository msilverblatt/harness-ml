"""Tests for MCP event emitter."""
from __future__ import annotations

from unittest.mock import MagicMock

from harnessml.plugin.event_emitter import EventEmitter


class TestEventEmitter:
    def test_disabled_by_default(self):
        emitter = EventEmitter()
        assert not emitter.enabled

    def test_enabled_with_store(self):
        mock_store = MagicMock()
        emitter = EventEmitter(store=mock_store)
        assert emitter.enabled

    def test_emit_calls_store_record(self):
        mock_store = MagicMock()
        emitter = EventEmitter(store=mock_store)
        emitter.emit(tool="models", action="add", params={"name": "xgb"}, result="Added model xgb", duration_ms=42, status="success")
        mock_store.record.assert_called_once_with(tool="models", action="add", params={"name": "xgb"}, result="Added model xgb", duration_ms=42, status="success")

    def test_emit_noop_when_disabled(self):
        emitter = EventEmitter()
        emitter.emit(tool="models", action="add", params={}, result="ok", duration_ms=0, status="success")

    def test_emit_swallows_exceptions(self):
        mock_store = MagicMock()
        mock_store.record.side_effect = RuntimeError("db locked")
        emitter = EventEmitter(store=mock_store)
        emitter.emit(tool="models", action="add", params={}, result="ok", duration_ms=0, status="success")
