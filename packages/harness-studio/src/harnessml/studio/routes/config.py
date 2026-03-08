"""Config viewer endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from harnessml.studio.routes.project import resolve_project_dir_from_request

router = APIRouter(tags=["config"])

ALLOWED_EXTENSIONS = {".yaml", ".yml", ".json"}


@router.get("/config/files")
async def list_config_files(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    config_dir = project_dir / "config"
    if not config_dir.exists():
        return []

    files = []
    for f in sorted(config_dir.rglob("*")):
        if f.is_file() and f.suffix in ALLOWED_EXTENSIONS:
            rel = f.relative_to(config_dir)
            files.append({
                "name": str(rel),
                "size": f.stat().st_size,
            })
    return files


@router.get("/config/files/{file_path:path}")
async def get_config_file(request: Request, file_path: str, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    full_path = (project_dir / "config" / file_path).resolve()

    # Security: ensure path stays within config dir
    config_dir = (project_dir / "config").resolve()
    if not str(full_path).startswith(str(config_dir)):
        raise HTTPException(403, "Path traversal not allowed")

    if not full_path.exists():
        raise HTTPException(404, f"Config file not found: {file_path}")

    if full_path.suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {full_path.suffix}")

    return {
        "name": file_path,
        "content": full_path.read_text(),
    }
