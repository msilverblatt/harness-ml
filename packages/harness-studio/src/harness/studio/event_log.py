"""SQLite-based event log for MCP monitoring (append-only)."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class EventLog:
    """Append-only SQLite event log."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    tool TEXT NOT NULL,
                    action TEXT NOT NULL,
                    params TEXT,
                    result TEXT,
                    duration_ms REAL,
                    status TEXT NOT NULL DEFAULT 'ok'
                )
            """)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def log(
        self,
        tool: str,
        action: str,
        params: dict[str, Any] | None = None,
        result: Any = None,
        duration_ms: float | None = None,
        status: str = "ok",
    ) -> int:
        """Insert an event and return its id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO events (timestamp, tool, action, params, result, duration_ms, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    tool,
                    action,
                    json.dumps(params) if params else None,
                    json.dumps(result) if result is not None else None,
                    duration_ms,
                    status,
                ),
            )
            return cursor.lastrowid

    def query(
        self,
        limit: int = 50,
        offset: int = 0,
        tool: str | None = None,
    ) -> list[dict[str, Any]]:
        """Paginated query of events, newest first."""
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if tool:
                rows = conn.execute(
                    "SELECT * FROM events WHERE tool = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (tool, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        """Return total event count and error count."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            errors = conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'error'"
            ).fetchone()[0]
            return {"total": total, "errors": errors}
