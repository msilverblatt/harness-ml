"""Event emitter for MCP tool calls — bridges plugin to Studio event store."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class EventEmitter:
    """Emits tool call events to the Studio event store.

    Designed to be fail-safe: if the store is unavailable or errors,
    tool calls proceed unaffected.
    """

    def __init__(self, store=None):
        self._store = store
        # Track the current running tool call for progress messages
        self._current_tool: str | None = None
        self._current_action: str | None = None

    @property
    def enabled(self) -> bool:
        return self._store is not None

    def emit(self, *, tool: str, action: str, params: dict, result: str, duration_ms: int, status: str) -> None:
        if self._store is None:
            return
        try:
            self._store.record(tool=tool, action=action, params=params, result=result, duration_ms=duration_ms, status=status)
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


def create_emitter(project_dir: str | Path | None = None) -> EventEmitter:
    """Create an EventEmitter.

    Resolution order for the database path:
      1. HARNESS_STUDIO_DB env var (explicit, set by user or .mcp.json)
      2. <project_dir>/.studio/events.db (per-project fallback)

    To ensure MCP events appear in Studio, set HARNESS_STUDIO_DB in
    your .mcp.json env block to match Studio's --db path.
    """
    db_path = os.environ.get("HARNESS_STUDIO_DB")
    if not db_path and project_dir:
        db_path = str(Path(project_dir) / ".studio" / "events.db")

    if not db_path:
        return EventEmitter()

    try:
        from harnessml.studio.event_store import EventStore
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        store = EventStore(db_path)
        store.init()
        return EventEmitter(store=store)
    except Exception:
        logger.debug("Could not initialize event store (non-fatal)", exc_info=True)
        return EventEmitter()
