"""FastAPI application factory for harness-studio."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from harness.studio.event_log import EventLog
from harness.studio.routes.versions import router as versions_router
from harness.studio.routes.pipeline import router as pipeline_router
from harness.studio.routes.diagnostics import router as diagnostics_router
from harness.studio.routes.predictions import router as predictions_router
from harness.studio.routes.data import router as data_router
from harness.studio.routes.monitor import router as monitor_router


def create_app(workspace_dir: str | Path) -> FastAPI:
    """Create and configure the FastAPI application."""
    workspace_dir = Path(workspace_dir)

    app = FastAPI(title="Harness Studio", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store workspace dir and event log in app state
    app.state.workspace_dir = workspace_dir
    app.state.event_log = EventLog(workspace_dir / ".harness" / "events.db")

    # Mount routers
    app.include_router(versions_router, prefix="/api/versions", tags=["versions"])
    app.include_router(pipeline_router, prefix="/api/pipeline", tags=["pipeline"])
    app.include_router(diagnostics_router, prefix="/api/diagnostics", tags=["diagnostics"])
    app.include_router(predictions_router, prefix="/api/predictions", tags=["predictions"])
    app.include_router(data_router, prefix="/api/data", tags=["data"])
    app.include_router(monitor_router, prefix="/api/monitor", tags=["monitor"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "workspace": str(workspace_dir)}

    return app
