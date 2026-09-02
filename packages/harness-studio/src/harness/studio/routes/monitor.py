"""Monitor routes — query the SQLite event log."""
from __future__ import annotations

from fastapi import APIRouter, Query, Request

from harness.studio.event_log import EventLog

router = APIRouter()


def _event_log(request: Request) -> EventLog:
    return request.app.state.event_log


@router.get("/events")
def monitor_events(
    request: Request,
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    tool: str | None = None,
):
    """Query the event log with pagination."""
    log = _event_log(request)
    events = log.query(limit=limit, offset=offset, tool=tool)
    return {"events": events, "limit": limit, "offset": offset}


@router.get("/stats")
def monitor_stats(request: Request):
    """Event count and error count."""
    log = _event_log(request)
    return log.stats()
