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
        self._conn.row_factory = sqlite3.Row
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
                status TEXT NOT NULL DEFAULT 'success',
                project TEXT NOT NULL DEFAULT ''
            )
        """)
        # Add columns to existing tables that lack them
        for col, default in [("project", "''"), ("caller", "''"), ("project_dir", "''")]:
            try:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT NOT NULL DEFAULT {default}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_project ON events(project)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_tool_project ON events(tool, project)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_project_dir ON events(project_dir)")
        # Project registry table — keyed by project_dir (full path) for uniqueness
        self._migrate_projects_table()
        self._conn.commit()

    def _migrate_projects_table(self) -> None:
        """Ensure projects table uses project_dir as PK with display_name."""
        assert self._conn is not None
        # Check if old schema (name as PK) exists
        rows = self._conn.execute("PRAGMA table_info(projects)").fetchall()
        col_names = [r["name"] for r in rows]
        if not rows:
            # Table doesn't exist yet
            self._conn.execute("""
                CREATE TABLE projects (
                    project_dir TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    last_seen REAL NOT NULL
                )
            """)
        elif "display_name" not in col_names:
            # Old schema — migrate to new PK
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS projects_new (
                    project_dir TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    last_seen REAL NOT NULL
                )
            """)
            self._conn.execute("""
                INSERT OR IGNORE INTO projects_new (project_dir, name, display_name, last_seen)
                SELECT project_dir, name, '', last_seen FROM projects
            """)
            self._conn.execute("DROP TABLE projects")
            self._conn.execute("ALTER TABLE projects_new RENAME TO projects")
        # else: already migrated

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.init()
        assert self._conn is not None
        return self._conn

    def record(self, *, tool: str, action: str, params: dict, result: str,
               duration_ms: int, status: str, project: str = "",
               project_dir: str = "", caller: str = "") -> int:
        """Record a tool call event. Returns the event ID."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "INSERT INTO events (timestamp, tool, action, params, result, duration_ms, status, project, project_dir, caller) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (time.time(), tool, action, json.dumps(params, default=str),
                 result, duration_ms, status, project, project_dir, caller),
            )
            conn.commit()
            assert cur.lastrowid is not None
            return cur.lastrowid

    def query(self, *, tool: str | None = None, project: str | None = None,
              project_dir: str | None = None, limit: int = 500,
              before_id: int | None = None,
              exclude_transient: bool = False) -> list[dict]:
        """Query events, newest first.

        When ``project_dir`` is provided it takes precedence over ``project``
        for filtering, ensuring two projects with the same basename at
        different paths don't contaminate each other.
        """
        with self._lock:
            conn = self._get_conn()
            clauses = []
            values: list = []
            if exclude_transient:
                clauses.append("status NOT IN ('running', 'progress')")
            if tool:
                clauses.append("tool = ?")
                values.append(tool)
            if project_dir:
                clauses.append("project_dir = ?")
                values.append(project_dir)
            elif project:
                clauses.append("project = ?")
                values.append(project)
            if before_id is not None:
                clauses.append("id < ?")
                values.append(before_id)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = conn.execute(
                f"SELECT id, timestamp, tool, action, params, result, duration_ms, status, project, project_dir, caller "
                f"FROM events {where} ORDER BY timestamp DESC LIMIT ?",
                [*values, limit],
            ).fetchall()

        results = []
        for r in rows:
            ts = r["timestamp"]
            if isinstance(ts, (int, float)):
                ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            params = r["params"]
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, TypeError):
                    params = {}
            results.append({
                "id": r["id"], "timestamp": ts, "tool": r["tool"], "action": r["action"],
                "params": params, "result": r["result"], "duration_ms": r["duration_ms"],
                "status": r["status"], "project": r["project"],
                "project_dir": r["project_dir"] if "project_dir" in r.keys() else "",
                "caller": r["caller"] if "caller" in r.keys() else "",
            })
        return results

    def register_project(self, name: str, project_dir: str,
                         display_name: str = "") -> None:
        """Upsert a project in the registry (keyed by project_dir)."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT INTO projects (project_dir, name, display_name, last_seen) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(project_dir) DO UPDATE SET name = excluded.name, last_seen = excluded.last_seen",
                (project_dir, name, display_name, time.time()),
            )
            conn.commit()

    def rename_project(self, project_dir: str, display_name: str) -> None:
        """Set a custom display name for a project."""
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE projects SET display_name = ? WHERE project_dir = ?",
                (display_name, project_dir),
            )
            conn.commit()

    def get_project_dir(self, name: str) -> str | None:
        """Look up a project's directory by name."""
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT project_dir FROM projects WHERE name = ?", (name,)
            ).fetchone()
        return row["project_dir"] if row else None

    def list_projects(self) -> list[str]:
        """Return distinct project names that have events."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT DISTINCT project FROM events WHERE project != '' ORDER BY project"
            ).fetchall()
        return [r["project"] for r in rows]

    def list_projects_with_dirs(self) -> list[dict]:
        """Return all registered projects with name, dir, display_name, and last_seen."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT project_dir, name, display_name, last_seen FROM projects ORDER BY last_seen DESC"
            ).fetchall()
        return [
            {
                "name": r["display_name"] if r["display_name"] else r["name"],
                "project_dir": r["project_dir"],
                "display_name": r["display_name"],
                "last_seen": r["last_seen"],
            }
            for r in rows
        ]

    def session_stats(self, *, project: str | None = None,
                      project_dir: str | None = None) -> dict:
        """Aggregate stats for the current session."""
        with self._lock:
            conn = self._get_conn()
            conditions = ["status NOT IN ('running', 'progress')"]
            params: list = []
            if project_dir:
                conditions.append("project_dir = ?")
                params.append(project_dir)
            elif project:
                conditions.append("project = ?")
                params.append(project)
            where = " AND ".join(conditions)

            total = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM events WHERE {where}",
                params,
            ).fetchone()["cnt"]

            err_conditions = ["status = 'error'"]
            err_params: list = list(params)  # copy filter params
            if project_dir:
                err_conditions.append("project_dir = ?")
            elif project:
                err_conditions.append("project = ?")
            err_where = " AND ".join(err_conditions)

            errors = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM events WHERE {err_where}",
                err_params,
            ).fetchone()["cnt"]

            by_tool_rows = conn.execute(
                f"SELECT tool, COUNT(*) AS cnt FROM events WHERE {where} GROUP BY tool",
                params,
            ).fetchall()
        return {"total_calls": total, "errors": errors, "by_tool": {r["tool"]: r["cnt"] for r in by_tool_rows}}
