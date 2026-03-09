"""Notebook entry endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from harnessml.studio.routes.project import resolve_project_dir_from_request

router = APIRouter(tags=["notebook"])


def _list_entries(project_dir: Path) -> list[dict]:
    """Read notebook/entries.jsonl, dedup by ID (latest wins), return all including struck.

    Returns newest first by timestamp.
    """
    jsonl_path = project_dir / "notebook" / "entries.jsonl"
    if not jsonl_path.exists():
        return []

    snapshots: dict[str, dict] = {}
    for line in jsonl_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        entry_id = entry.get("id")
        if entry_id:
            snapshots[entry_id] = entry

    entries = list(snapshots.values())
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


@router.get("/notebook")
async def list_notebook_entries(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    return _list_entries(project_dir)


@router.get("/notebook/{entry_id}")
async def get_notebook_entry(request: Request, entry_id: str, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    entries = _list_entries(project_dir)
    for entry in entries:
        if entry.get("id") == entry_id:
            return entry
    return {"error": f"Notebook entry '{entry_id}' not found"}
