"""Data source endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request
from harnessml.studio.routes.project import _load_yaml, resolve_project_dir_from_request

router = APIRouter(tags=["data"])


@router.get("/data/sources")
async def list_sources(request: Request, project: str | None = None):
    """List all registered data sources with metadata."""
    project_dir = resolve_project_dir_from_request(request, project)

    # Source registry
    registry_path = project_dir / "data" / "source_registry.json"
    sources = []
    seen_names: set[str] = set()
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        raw_sources = registry.get("sources", registry if isinstance(registry, list) else [])
        if isinstance(raw_sources, list):
            for src in raw_sources:
                name = src.get("name", "unknown")
                if name in seen_names:
                    continue
                seen_names.add(name)
                # Prefer paths relative to project_dir that actually resolve
                raw_path = src.get("path", "")
                sources.append({
                    "name": name,
                    "path": raw_path,
                    "columns": src.get("columns_added", []),
                    "rows": src.get("rows", 0),
                    "adapter": src.get("adapter", "file"),
                    "is_bootstrap": src.get("is_bootstrap", False),
                })

    # Fallback to pipeline config
    if not sources:
        pipeline = _load_yaml(project_dir / "config" / "pipeline.yaml")
        pipe_sources = pipeline.get("data", {}).get("sources", {})
        if isinstance(pipe_sources, dict):
            for name, src in pipe_sources.items():
                sources.append({
                    "name": name,
                    "path": src.get("path", str(src)) if isinstance(src, dict) else str(src),
                    "columns": [],
                    "rows": 0,
                    "adapter": src.get("adapter", "file") if isinstance(src, dict) else "file",
                    "is_bootstrap": False,
                })

    return sources


def _resolve_source_file(project_dir: Path, source_name: str) -> Path | None:
    """Find the on-disk file for a named source, trying all registry entries."""
    registry_path = project_dir / "data" / "source_registry.json"
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text())
    raw_sources = registry.get("sources", registry if isinstance(registry, list) else [])
    if not isinstance(raw_sources, list):
        return None
    for src in raw_sources:
        if src.get("name") != source_name:
            continue
        raw_path = src.get("path", "")
        full = Path(raw_path)
        if not full.is_absolute():
            full = project_dir / full
        if full.exists():
            return full
    return None


@router.get("/data/sources/{source_name}/preview")
async def source_preview(request: Request, source_name: str, rows: int = 20, project: str | None = None):
    """Preview first N rows of a data source."""
    project_dir = resolve_project_dir_from_request(request, project)
    full_path = _resolve_source_file(project_dir, source_name)

    if not full_path:
        return {"error": f"Source '{source_name}' not found or file missing"}

    try:
        if full_path.suffix == ".parquet":
            df = pd.read_parquet(full_path).head(rows)
        elif full_path.suffix == ".csv":
            df = pd.read_csv(full_path, nrows=rows)
        else:
            return {"error": f"Unsupported format: {full_path.suffix}"}
    except Exception as e:
        return {"error": str(e)}

    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "rows": df.fillna("").to_dict(orient="records"),
        "total_rows": len(df),
    }


@router.get("/data/sources/{source_name}/stats")
async def source_stats(request: Request, source_name: str, project: str | None = None):
    """Column-level statistics for a data source."""
    project_dir = resolve_project_dir_from_request(request, project)
    full_path = _resolve_source_file(project_dir, source_name)

    if not full_path:
        return {"error": f"Source '{source_name}' not found or file missing"}

    try:
        if full_path.suffix == ".parquet":
            df = pd.read_parquet(full_path)
        elif full_path.suffix == ".csv":
            df = pd.read_csv(full_path)
        else:
            return {"error": f"Unsupported format: {full_path.suffix}"}
    except Exception as e:
        return {"error": str(e)}

    stats = []
    for col in df.columns:
        col_stats = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(float(df[col].isna().mean()) * 100, 1),
            "unique_count": int(df[col].nunique()),
        }
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            col_stats["min"] = float(df[col].min()) if not df[col].isna().all() else None
            col_stats["max"] = float(df[col].max()) if not df[col].isna().all() else None
            col_stats["mean"] = round(float(df[col].mean()), 4) if not df[col].isna().all() else None
        stats.append(col_stats)

    return {"columns": stats, "total_rows": len(df)}
