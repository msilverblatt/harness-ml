"""Harness Studio — companion dashboard server."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from harnessml.studio.broadcaster import EventBroadcaster
from harnessml.studio.routes import (
    config,
    data,
    ensemble,
    events,
    experiments,
    features,
    models,
    predictions,
    project,
    runs,
    ws,
)


async def _poll_events(app: FastAPI, interval: float = 1.0):
    """Background task: poll SQLite for new events and broadcast via WebSocket."""
    import asyncio
    import json as _json
    from datetime import datetime, timezone

    last_id = 0
    store = app.state.event_store
    broadcaster = app.state.broadcaster
    if store is None:
        return

    # Get current max ID so we only stream new events
    try:
        conn = store._get_conn()
        row = conn.execute("SELECT MAX(id) FROM events").fetchone()
        if row and row[0]:
            last_id = row[0]
    except Exception:
        pass

    while True:
        await asyncio.sleep(interval)
        try:
            conn = store._get_conn()
            rows = conn.execute(
                "SELECT id, timestamp, tool, action, params, result, duration_ms, status, project, caller "
                "FROM events WHERE id > ? ORDER BY id ASC",
                (last_id,),
            ).fetchall()
            for r in rows:
                ts = r[1]
                if isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                event = {
                    "id": r[0],
                    "timestamp": ts,
                    "tool": r[2],
                    "action": r[3],
                    "params": r[4] if isinstance(r[4], dict) else _json.loads(r[4]) if isinstance(r[4], str) else {},
                    "result": r[5],
                    "duration_ms": r[6],
                    "status": r[7],
                    "project": r[8] if len(r) > 8 else "",
                    "caller": r[9] if len(r) > 9 else "",
                }
                broadcaster.notify(event)
                last_id = r[0]
        except Exception:
            pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    import asyncio

    db_path_str = (
        getattr(application.state, "db_path", None)
        or os.environ.get("HARNESS_STUDIO_DB")
        or str(Path.home() / ".harnessml" / "events.db")
    )
    db_path = Path(db_path_str)
    try:
        from harnessml.studio.event_store import EventStore
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = EventStore(db_path)
        store.init()
        application.state.event_store = store
    except Exception:
        application.state.event_store = None
    application.state.broadcaster = EventBroadcaster()

    # Start background poller for live event streaming
    poll_task = asyncio.create_task(_poll_events(application))
    try:
        yield
    finally:
        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Harness Studio", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(events.router, prefix="/api")
app.include_router(project.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(data.router, prefix="/api")
app.include_router(features.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(ensemble.router, prefix="/api")
app.include_router(predictions.router, prefix="/api")
app.include_router(config.router, prefix="/api")
app.include_router(ws.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve pre-built frontend static files (must be AFTER API routes)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    # Serve static assets (JS, CSS, images) at /assets/*
    _assets_dir = _static_dir / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    # SPA catch-all: any non-API route returns index.html for client-side routing
    @app.get("/{full_path:path}")
    async def spa_fallback(full_path: str):
        file_path = _static_dir / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_static_dir / "index.html"))
