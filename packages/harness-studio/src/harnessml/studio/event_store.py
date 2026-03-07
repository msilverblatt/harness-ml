"""SQLite-backed event store for MCP tool call logging."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class EventStore:
    """Append-only event log backed by SQLite."""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    def init(self) -> None:
        """Create the database and events table if needed."""
        self._conn = sqlite3.connect(self._db_path)
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
        return self._conn

    def record(self, *, tool: str, action: str, params: dict, result: str, duration_ms: int, status: str) -> int:
        """Record a tool call event. Returns the event ID."""
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO events (timestamp, tool, action, params, result, duration_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), tool, action, json.dumps(params, default=str), result, duration_ms, status),
        )
        conn.commit()
        return cur.lastrowid

    def query(self, *, tool: str | None = None, limit: int = 50, before_id: int | None = None) -> list[dict]:
        """Query events, newest first."""
        conn = self._get_conn()
        clauses = []
        values: list = []
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
        return [
            {"id": r[0], "timestamp": r[1], "tool": r[2], "action": r[3], "params": r[4], "result": r[5], "duration_ms": r[6], "status": r[7]}
            for r in rows
        ]

    def session_stats(self) -> dict:
        """Aggregate stats for the current session."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        errors = conn.execute("SELECT COUNT(*) FROM events WHERE status = 'error'").fetchone()[0]
        by_tool_rows = conn.execute("SELECT tool, COUNT(*) FROM events GROUP BY tool").fetchall()
        return {"total_calls": total, "errors": errors, "by_tool": {r[0]: r[1] for r in by_tool_rows}}
