"""Event log endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["events"])


def _resolve_project_dir(request: Request, project_dir: str | None) -> str | None:
    """In single-project mode, default to the --project-dir full path."""
    if project_dir:
        return project_dir
    if getattr(request.app.state, "single_project", False):
        return getattr(request.app.state, "project_dir", None)
    return None


@router.get("/events")
async def list_events(
    request: Request,
    tool: str | None = None,
    project: str | None = None,
    project_dir: str | None = None,
    limit: int = 500,
    before_id: int | None = None,
):
    store = request.app.state.event_store
    if store is None:
        return []
    resolved_dir = _resolve_project_dir(request, project_dir)
    return store.query(tool=tool, project=project, project_dir=resolved_dir,
                       limit=limit, before_id=before_id, exclude_transient=True)


@router.get("/events/stats")
async def event_stats(request: Request, project: str | None = None,
                      project_dir: str | None = None):
    store = request.app.state.event_store
    if store is None:
        return {"total_calls": 0, "errors": 0, "by_tool": {}}
    resolved_dir = _resolve_project_dir(request, project_dir)
    return store.session_stats(project=project, project_dir=resolved_dir)


@router.get("/events/projects")
async def list_projects(request: Request):
    store = request.app.state.event_store
    if store is None:
        return []
    if getattr(request.app.state, "single_project", False):
        pd = getattr(request.app.state, "project_dir", None)
        if pd:
            from pathlib import Path
            return [{"name": Path(pd).name, "project_dir": pd}]
    return store.list_projects_with_dirs()
