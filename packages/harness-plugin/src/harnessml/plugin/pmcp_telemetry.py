"""Telemetry sink: forward tool call events to Studio's SQLite event store."""
from __future__ import annotations

import os
import threading

from protomcp import ToolCallEvent, telemetry_sink

_emitter = None
_init_lock = threading.Lock()


def _get_emitter():
    global _emitter
    if _emitter is not None:
        return _emitter
    with _init_lock:
        if _emitter is not None:
            return _emitter
        from harnessml.plugin.event_emitter import create_emitter
        _emitter = create_emitter()
    return _emitter


@telemetry_sink
def studio_telemetry(event: ToolCallEvent):
    """Forward tool call events to Studio's SQLite event store."""
    emitter = _get_emitter()
    if not emitter.enabled:
        return

    if event.phase == "start":
        emitter.set_project(str(event.args.get("project_dir", os.getcwd())))
        emitter.set_current(event.tool_name, event.action)
        emitter.emit(
            tool=event.tool_name, action=event.action,
            params={k: v for k, v in event.args.items() if k != "ctx"},
            result="", duration_ms=0, status="running",
        )
    elif event.phase == "success":
        emitter.clear_current()
        emitter.emit(
            tool=event.tool_name, action=event.action,
            params={}, result=event.result[:20000],
            duration_ms=event.duration_ms, status="success",
        )
    elif event.phase == "error":
        emitter.clear_current()
        emitter.emit(
            tool=event.tool_name, action=event.action,
            params={}, result=str(event.error)[:20000],
            duration_ms=event.duration_ms, status="error",
        )
    elif event.phase == "progress":
        emitter.progress(
            current=event.progress, total=event.total,
            message=event.message,
            tool_override=event.tool_name, action_override=event.action,
        )
