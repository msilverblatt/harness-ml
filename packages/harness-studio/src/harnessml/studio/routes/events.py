"""Event log endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["events"])


@router.get("/events")
async def list_events(request: Request, tool: str | None = None, limit: int = 500, before_id: int | None = None):
    store = request.app.state.event_store
    if store is None:
        return []
    return store.query(tool=tool, limit=limit, before_id=before_id, exclude_transient=True)


@router.get("/events/stats")
async def event_stats(request: Request):
    store = request.app.state.event_store
    if store is None:
        return {"total_calls": 0, "errors": 0, "by_tool": {}}
    return store.session_stats()
