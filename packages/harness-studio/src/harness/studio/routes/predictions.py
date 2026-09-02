"""Predictions routes — read predictions.parquet from version run dirs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter()


@router.post("/{version_id}/infer")
def infer(version_id: str, rows: list[dict], request: Request):
    """Run a fitted production bundle against raw JSON records."""
    from harness.ml.runners.production import ProductionBundle

    path = (
        request.app.state.workspace_dir
        / "versions"
        / version_id
        / "run"
        / "model.bundle"
    )
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Production bundle not found for version: {version_id}",
        )
    frame = pd.DataFrame(rows)
    try:
        bundle = ProductionBundle.load(path, trusted=True)
        predictions = bundle.predict(frame)
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if predictions.ndim == 1:
        result = [{"prediction": float(value)} for value in predictions]
    else:
        result = [
            {
                f"prediction_class_{index}": float(value)
                for index, value in enumerate(row)
            }
            for row in predictions
        ]
    if bundle.conformal_radius is not None:
        intervals = bundle.predict_interval(frame)
        result = intervals.to_dict(orient="records")
    return {"version_id": version_id, "predictions": result}


def _predictions_path(request: Request, version_id: str) -> Path:
    return (
        request.app.state.workspace_dir
        / "versions"
        / version_id
        / "run"
        / "predictions.parquet"
    )


@router.get("/{version_id}")
def version_predictions(
    version_id: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=10000),
):
    """Read predictions.parquet and return as JSON with pagination."""
    path = _predictions_path(request, version_id)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Predictions not found for version: {version_id}"
        )

    df = pd.read_parquet(str(path))
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    return {
        "version_id": version_id,
        "total": total,
        "page": page,
        "page_size": page_size,
        "columns": list(df.columns),
        "rows": page_df.to_dict(orient="records"),
    }


@router.get("/{version_id}/distribution")
def prediction_distribution(
    version_id: str,
    request: Request,
    bins: int = Query(20, ge=2, le=200),
):
    """Histogram of prediction values."""
    path = _predictions_path(request, version_id)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"Predictions not found for version: {version_id}"
        )

    df = pd.read_parquet(str(path))

    # Find prediction columns (anything with 'pred' in name or numeric)
    pred_cols = [c for c in df.columns if "pred" in c.lower()]
    if not pred_cols:
        # Fall back to all numeric columns
        pred_cols = list(df.select_dtypes(include="number").columns)

    distributions = {}
    for col in pred_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        counts, bin_edges = pd.cut(series, bins=bins, retbins=True)
        hist = counts.value_counts(sort=False)
        distributions[col] = {
            "bin_edges": [float(x) for x in bin_edges],
            "counts": [int(hist.get(interval, 0)) for interval in hist.index],
        }

    return {
        "version_id": version_id,
        "distributions": distributions,
    }
