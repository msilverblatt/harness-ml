"""Diagnostics routes — read metrics and diagnostics from version run dirs."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


def _run_dir(request: Request, version_id: str) -> Path:
    return request.app.state.workspace_dir / "versions" / version_id / "run"


@router.get("/{version_id}")
def version_diagnostics(version_id: str, request: Request):
    """Metrics + diagnostics from a version's run directory."""
    run_dir = _run_dir(request, version_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Version run not found: {version_id}")

    metrics_path = run_dir / "metrics.json"
    diag_path = run_dir / "diagnostics.json"
    eval_path = run_dir / "eval_report.json"

    metrics = {}
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())

    diagnostics = {}
    if diag_path.exists():
        diagnostics = json.loads(diag_path.read_text())

    eval_report = {}
    if eval_path.exists():
        eval_report = json.loads(eval_path.read_text())

    return {
        "version_id": version_id,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "eval_report": eval_report,
    }
