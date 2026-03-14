"""Static dashboard export endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from harnessml.studio.export import export_html
from harnessml.studio.routes.project import resolve_project_dir_from_request

router = APIRouter(tags=["export"])


@router.get("/export", response_class=HTMLResponse)
async def export_dashboard(
    request: Request,
    project: str | None = None,
    run_id: str | None = None,
):
    """Export a self-contained HTML dashboard report.

    Returns the HTML directly as the response body.
    """
    import tempfile

    project_dir = resolve_project_dir_from_request(request, project)
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        export_html(project_dir, tmp_path, run_id=run_id)
        html = tmp_path.read_text()
    finally:
        tmp_path.unlink(missing_ok=True)

    return HTMLResponse(content=html)
