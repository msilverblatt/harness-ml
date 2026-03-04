"""Shared helpers for MCP handlers."""
from __future__ import annotations

import json
from pathlib import Path


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
                f"Is this an easyml project? Run configure(action='init') first."
            )
    return p


def parse_json_param(value: str | dict | list | None) -> dict | list | None:
    """Parse a JSON string parameter, or return as-is if already parsed."""
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value
