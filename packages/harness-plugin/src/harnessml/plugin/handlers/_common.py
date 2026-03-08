"""Shared helpers for MCP handlers."""
from __future__ import annotations

import json
from pathlib import Path

# Module-level emitter reference, set by _safe_tool before each tool call.
# This avoids circular imports between handlers and mcp_server.
_active_emitter = None


def set_active_emitter(emitter):
    """Set the active emitter for the current tool call."""
    global _active_emitter
    _active_emitter = emitter


def get_active_emitter():
    """Get the active emitter set by _safe_tool."""
    return _active_emitter


def resolve_project_dir(project_dir: str | None, *, allow_missing: bool = False) -> Path:
    """Resolve project directory from param or cwd."""
    if project_dir:
        p = Path(project_dir).resolve()
    else:
        p = Path.cwd()
    if not allow_missing:
        config_dir = p / "config"
        if not config_dir.exists():
            raise ValueError(
                f"No config/ directory found at {p}. "
                f"Is this an harnessml project? Run configure(action='init') first."
            )
    return p


def make_progress_callback(ctx, loop):
    """Create a progress callback that reports to both MCP client and Studio event store."""
    import asyncio

    # Capture the emitter AND tool/action at callback creation time
    # so they're available from worker threads.
    emitter = get_active_emitter()
    tool = emitter._current_tool if emitter else None
    action = emitter._current_action if emitter else None

    def _progress_callback(current, total, message):
        # Report to MCP client
        if ctx is not None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=current, total=total, message=message),
                loop,
            )
        # Report to Studio event store
        if emitter is not None:
            emitter.progress(current=current, total=total, message=message,
                             tool_override=tool, action_override=action)

    return _progress_callback


def parse_json_param(value: str | dict | list | None) -> dict | list | None:
    """Parse a JSON string parameter, or return as-is if already parsed."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value
