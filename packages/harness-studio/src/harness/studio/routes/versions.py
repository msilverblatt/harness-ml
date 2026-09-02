"""Version tree routes — read workspace version files directly."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from harness.app.workspace.versions import VersionTree

router = APIRouter()


def _tree(request: Request) -> VersionTree:
    return VersionTree(request.app.state.workspace_dir)


def _current(workspace_dir: Path) -> str | None:
    pointer = workspace_dir / "current"
    if not pointer.exists():
        return None
    return pointer.read_text().strip()


@router.get("/tree")
def version_tree(request: Request):
    """List all versions and the current pointer."""
    tree = _tree(request)
    versions = tree.list_versions()
    current = _current(request.app.state.workspace_dir)
    return {
        "current": current,
        "versions": [asdict(v) for v in versions],
    }


@router.get("/compare/{v1}/{v2}")
def compare_versions(v1: str, v2: str, request: Request):
    """Metric comparison between two versions."""
    tree = _tree(request)
    try:
        deltas = tree.compare(v1, v2)
    except ValueError:
        raise HTTPException(status_code=404, detail="One or both versions not found")
    return {"v1": v1, "v2": v2, "deltas": deltas}


@router.get("/{version_id}")
def version_detail(version_id: str, request: Request):
    """Version detail from meta.yaml."""
    tree = _tree(request)
    meta = tree.get_version(version_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")
    return asdict(meta)


@router.get("/{version_id}/ancestry")
def version_ancestry(version_id: str, request: Request):
    """Path from root to this version."""
    tree = _tree(request)
    meta = tree.get_version(version_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Version not found: {version_id}")
    chain = tree.ancestry(version_id)
    return {"version_id": version_id, "ancestry": [asdict(v) for v in chain]}
