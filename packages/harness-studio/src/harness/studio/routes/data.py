"""Data routes — read schema and profile from workspace data directory."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from harness.data.profiling.profiler import DataProfiler

router = APIRouter()


def _data_dir(request: Request) -> Path:
    return request.app.state.workspace_dir / "data"


@router.get("/schema")
def data_schema(request: Request):
    """Read data/clean/schema.json."""
    import json

    schema_path = _data_dir(request) / "clean" / "schema.json"
    if not schema_path.exists():
        raise HTTPException(status_code=404, detail="Schema not found")
    return json.loads(schema_path.read_text())


@router.get("/profile")
def data_profile(request: Request):
    """Run DataProfiler on data/clean/dataset.parquet."""
    dataset_path = _data_dir(request) / "clean" / "dataset.parquet"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_parquet(str(dataset_path))
    profiler = DataProfiler()
    profile = profiler.profile(df)
    return asdict(profile)
