"""SQLite-backed event store for MCP tool call logging."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


class EventStore:
    """Append-only event log backed by SQLite. Thread-safe."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def init(self) -> None:
        """Create the database and events table if needed."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                tool TEXT NOT NULL,
                action TEXT NOT NULL,
                params TEXT NOT NULL DEFAULT '{}',
                result TEXT NOT NULL DEFAULT '',
                duration_ms INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'success'
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
        self._conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.init()
        return self._conn  # type: ignore[return-value]

    def record(self, *, tool: str, action: str, params: dict, result: str, duration_ms: int, status: str) -> int:
        """Record a tool call event. Returns the event ID."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "INSERT INTO events (timestamp, tool, action, params, result, duration_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (time.time(), tool, action, json.dumps(params, default=str), result, duration_ms, status),
            )
            conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def query(self, *, tool: str | None = None, limit: int = 500, before_id: int | None = None, exclude_transient: bool = False) -> list[dict]:
        """Query events, newest first."""
        with self._lock:
            conn = self._get_conn()
            clauses = []
            values: list = []
            if exclude_transient:
                clauses.append("status NOT IN ('running', 'progress')")
            if tool:
                clauses.append("tool = ?")
                values.append(tool)
            if before_id is not None:
                clauses.append("id < ?")
                values.append(before_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"SELECT id, timestamp, tool, action, params, result, duration_ms, status FROM events {where} ORDER BY timestamp DESC LIMIT ?",
                [*values, limit],
            ).fetchall()

        results = []
        for r in rows:
            ts = r[1]
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            params = r[4]
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            results.append({
                "id": r[0], "timestamp": ts, "tool": r[2], "action": r[3],
                "params": params, "result": r[5], "duration_ms": r[6], "status": r[7],
            })
        return results

    def session_stats(self) -> dict:
        """Aggregate stats for the current session."""
        with self._lock:
            conn = self._get_conn()
            total = conn.execute("SELECT COUNT(*) FROM events WHERE status NOT IN ('running', 'progress')").fetchone()[0]
            errors = conn.execute("SELECT COUNT(*) FROM events WHERE status = 'error'").fetchone()[0]
            by_tool_rows = conn.execute("SELECT tool, COUNT(*) FROM events WHERE status NOT IN ('running', 'progress') GROUP BY tool").fetchall()
        return {"total_calls": total, "errors": errors, "by_tool": {r[0]: r[1] for r in by_tool_rows}}
