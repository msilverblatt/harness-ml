"""Prediction browsing endpoints."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request
from harnessml.studio.errors import error_response
from harnessml.studio.routes.project import resolve_project_dir_from_request
from harnessml.studio.routes.runs import _detect_target_col, _load_predictions

router = APIRouter(tags=["predictions"])


def _find_prob_column(df: pd.DataFrame) -> str | None:
    """Find the best probability/prediction column in a predictions dataframe.

    Search order:
    1. prob_ensemble (standard ensemble column)
    2. Any prob_* column
    3. Columns containing 'probability', 'score', or 'pred' (case-insensitive)
    4. For binary classification: a single remaining numeric column
    """
    if "prob_ensemble" in df.columns:
        return "prob_ensemble"

    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if prob_cols:
        return prob_cols[0]

    # Broader search: columns containing common prediction-related names
    lower_map = {c.lower(): c for c in df.columns}
    for keyword in ("probability", "score", "prediction", "pred"):
        matches = [orig for low, orig in lower_map.items() if keyword in low and orig not in ("fold",)]
        if matches:
            return matches[0]

    # Last resort: if there's exactly one numeric column that isn't a known meta column,
    # treat it as the prediction column (common for binary classification)
    meta_cols = {"fold", "diff_prior"}
    numeric_cols = [
        c for c in df.select_dtypes(include=["float64", "float32", "int64", "int32"]).columns
        if c not in meta_cols
    ]
    # Exclude what looks like a target column (all 0/1 integers)
    candidate_cols = []
    for c in numeric_cols:
        vals = df[c].dropna()
        if len(vals) > 0 and not (vals.isin([0, 1]).all() and vals.dtype in ("int64", "int32")):
            candidate_cols.append(c)
    if len(candidate_cols) == 1:
        return candidate_cols[0]

    return None


def _latest_run_dir(project_dir: Path) -> Path | None:
    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return None
    run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
    return run_dirs[0] if run_dirs else None


def _resolve_run_dir(project_dir: Path, run_id: str | None) -> Path | None:
    if run_id:
        rd = project_dir / "outputs" / run_id
        return rd if rd.exists() else None
    return _latest_run_dir(project_dir)


@router.get("/predictions")
async def list_predictions(
    request: Request,
    run_id: str | None = None,
    page: int = 0,
    page_size: int = 50,
    project: str | None = None,
):
    try:
        project_dir = resolve_project_dir_from_request(request, project)
        run_dir = _resolve_run_dir(project_dir, run_id)
        if not run_dir:
            return {"error": "No runs found"}

        df = _load_predictions(run_dir)
        if df is None:
            return {"error": "No predictions found"}

        total = len(df)
        start = page * page_size
        end = min(start + page_size, total)
        page_df = df.iloc[start:end].copy()

        # Round floats for display
        for col in page_df.select_dtypes(include=["float64", "float32"]).columns:
            page_df[col] = page_df[col].round(4)

        return {
            "run_id": run_dir.name,
            "columns": list(df.columns),
            "rows": page_df.fillna("").to_dict(orient="records"),
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except Exception as e:
        return error_response(e)


@router.get("/predictions/distribution")
async def prediction_distribution(
    request: Request,
    run_id: str | None = None,
    bins: int = 20,
    project: str | None = None,
):
    import numpy as np

    project_dir = resolve_project_dir_from_request(request, project)
    run_dir = _resolve_run_dir(project_dir, run_id)
    if not run_dir:
        return {"error": "No runs found"}

    df = _load_predictions(run_dir)
    if df is None:
        return {"error": "No predictions found"}

    prob_col = _find_prob_column(df)
    if not prob_col:
        available = df.columns.tolist()
        return {"error": f"No probability column found. Available columns: {available}"}

    values = df[prob_col].dropna().values
    if len(values) == 0:
        return {"error": f"Column '{prob_col}' has no non-null values"}

    # Auto-detect range: use (0, 1) for probability-like values, otherwise data range
    v_min, v_max = float(values.min()), float(values.max())
    hist_range = (0, 1) if v_min >= 0 and v_max <= 1 else (v_min, v_max)
    counts, edges = np.histogram(values, bins=bins, range=hist_range)
    histogram = [
        {"bin_center": round(float((edges[i] + edges[i + 1]) / 2), 3), "count": int(counts[i])}
        for i in range(len(counts))
    ]

    return {
        "run_id": run_dir.name,
        "prob_column": prob_col,
        "histogram": histogram,
        "stats": {
            "mean": round(float(values.mean()), 4),
            "std": round(float(values.std()), 4),
            "median": round(float(np.median(values)), 4),
        },
    }


@router.get("/predictions/summary")
async def prediction_summary(
    request: Request,
    run_id: str | None = None,
    project: str | None = None,
):
    import numpy as np

    project_dir = resolve_project_dir_from_request(request, project)
    run_dir = _resolve_run_dir(project_dir, run_id)
    if not run_dir:
        return {"error": "No runs found"}

    df = _load_predictions(run_dir)
    if df is None:
        return {"error": "No predictions found"}

    target_col = _detect_target_col(df)
    prob_col = _find_prob_column(df)

    result: dict = {
        "run_id": run_dir.name,
        "total_predictions": len(df),
        "model_columns": [c.replace("prob_", "") for c in df.columns if c.startswith("prob_")],
    }

    if target_col and prob_col:
        y_true = df[target_col].values.astype(float)
        y_prob = df[prob_col].values.astype(float)

        unique = set(y_true)
        if unique.issubset({0.0, 1.0}):
            y_pred = (y_prob >= 0.5).astype(float)
            result["correct"] = int(np.sum(y_pred == y_true))
            result["accuracy"] = round(float(np.mean(y_pred == y_true)), 4)
            result["avg_confidence"] = round(float(np.mean(np.maximum(y_prob, 1 - y_prob))), 4)

    if "fold" in df.columns:
        fold_counts = df["fold"].value_counts().sort_index().to_dict()
        result["fold_counts"] = {str(k): int(v) for k, v in fold_counts.items()}

    return result
