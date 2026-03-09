"""Event emitter for MCP tool calls — bridges plugin to Studio event store."""
from __future__ import annotations

import os
from pathlib import Path

from harnessml.core.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_DB_DIR = Path.home() / ".harnessml"


class EventEmitter:
    """Emits tool call events to the Studio event store.

    Designed to be fail-safe: if the store is unavailable or errors,
    tool calls proceed unaffected.
    """

    def __init__(self, store=None):
        self._store = store
        self._current_tool: str | None = None
        self._current_action: str | None = None
        self._current_project: str = ""
        self._caller: str = os.environ.get("HARNESS_CALLER", "Claude Opus 4.6")

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def set_project(self, project_dir: str) -> None:
        """Set the project name for subsequent events (derived from project_dir)."""
        resolved = Path(project_dir).resolve()
        self._current_project = resolved.name
        if self._store is not None:
            try:
                self._store.register_project(resolved.name, str(resolved))
            except Exception:
                logger.debug("Project registration failed (non-fatal)", exc_info=True)

    def emit(self, *, tool: str, action: str, params: dict, result: str,
             duration_ms: int, status: str) -> None:
        if self._store is None:
            return
        try:
            self._store.record(
                tool=tool, action=action, params=params, result=result,
                duration_ms=duration_ms, status=status,
                project=self._current_project, caller=self._caller,
            )
        except Exception:
            logger.debug("Event emission failed (non-fatal)", exc_info=True)

    def progress(self, *, current: int, total: int, message: str,
                  tool_override: str | None = None, action_override: str | None = None) -> None:
        """Emit a progress event for the currently running tool call."""
        if self._store is None:
            return
        tool = tool_override or self._current_tool or "unknown"
        action = action_override or self._current_action or "unknown"
        try:
            self._store.record(
                tool=tool, action=action,
                params={"current": current, "total": total},
                result=message, duration_ms=0, status="progress",
                project=self._current_project, caller=self._caller,
            )
        except Exception:
            logger.debug("Progress emission failed (non-fatal)", exc_info=True)

    def set_current(self, tool: str, action: str) -> None:
        """Set the current tool/action context for progress messages."""
        self._current_tool = tool
        self._current_action = action

    def clear_current(self) -> None:
        self._current_tool = None
        self._current_action = None


def create_emitter() -> EventEmitter:
    """Create an EventEmitter.

    Events are stored in ~/.harnessml/events.db so all projects and
    MCP sessions share a single event stream. Override with
    HARNESS_STUDIO_DB env var if needed.
    """
    db_path = os.environ.get("HARNESS_STUDIO_DB")
    if not db_path:
        db_path = str(_DEFAULT_DB_DIR / "events.db")

    try:
        from harnessml.studio.event_store import EventStore
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        store = EventStore(db_path)
        store.init()
        return EventEmitter(store=store)
    except Exception:
        # Stale WAL/SHM files from crashed daemon threads can corrupt the DB.
        # Delete and retry once before giving up.
        logger.debug("Event store init failed, retrying with fresh DB", exc_info=True)
        try:
            for suffix in ("", "-wal", "-shm"):
                p = Path(db_path + suffix)
                if p.exists():
                    p.unlink()
            from harnessml.studio.event_store import EventStore
            store = EventStore(db_path)
            store.init()
            return EventEmitter(store=store)
        except Exception:
            logger.debug("Could not initialize event store (non-fatal)", exc_info=True)
            return EventEmitter()
