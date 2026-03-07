"""Harness Studio — companion dashboard server."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from harnessml.studio.broadcaster import EventBroadcaster
from harnessml.studio.routes import events, experiments, project, runs, ws


@asynccontextmanager
async def lifespan(application: FastAPI):
    project_dir = getattr(application.state, "project_dir", ".")
    db_path = Path(project_dir) / ".studio" / "events.db"
    try:
        from harnessml.studio.event_store import EventStore
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store = EventStore(db_path)
        store.init()
        application.state.event_store = store
    except Exception:
        application.state.event_store = None
    application.state.broadcaster = EventBroadcaster()
    yield


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
app.include_router(ws.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# Serve pre-built frontend static files (must be AFTER API routes)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")
