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


def create_emitter(project_dir: str | Path | None = None) -> EventEmitter:
    """Create an EventEmitter, enabled if HARNESS_STUDIO_DB is set or project_dir given."""
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
