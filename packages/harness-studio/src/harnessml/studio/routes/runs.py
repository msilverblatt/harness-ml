"""Run output endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(tags=["runs"])


@router.get("/runs")
async def list_runs(request: Request):
    project_dir = Path(request.app.state.project_dir)
    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return []
    runs = []
    for d in sorted(outputs_dir.iterdir(), reverse=True):
        if d.is_dir():
            metrics_path = d / "pooled_metrics.json"
            metrics = {}
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text())
                except json.JSONDecodeError:
                    pass
            runs.append({"id": d.name, "metrics": metrics})
    return runs


@router.get("/runs/{run_id}/metrics")
async def run_metrics(request: Request, run_id: str):
    project_dir = Path(request.app.state.project_dir)
    metrics_path = project_dir / "outputs" / run_id / "pooled_metrics.json"
    if not metrics_path.exists():
        return {"error": f"Metrics not found for run {run_id}"}
    return json.loads(metrics_path.read_text())


@router.get("/runs/{run_id}/correlations")
async def run_correlations(request: Request, run_id: str):
    """Compute model prediction correlation matrix from predictions parquet."""
    project_dir = Path(request.app.state.project_dir)
    run_dir = project_dir / "outputs" / run_id

    # Find predictions file
    pred_path = run_dir / "predictions" / "predictions.parquet"
    if not pred_path.exists():
        # Try per-fold files
        pred_dir = run_dir / "predictions"
        if pred_dir.exists():
            import pandas as pd
            frames = []
            for f in sorted(pred_dir.glob("*.parquet")):
                frames.append(pd.read_parquet(f))
            if frames:
                df = pd.concat(frames, ignore_index=True)
            else:
                return {"error": "No prediction files found"}
        else:
            return {"error": "No predictions directory found"}
    else:
        import pandas as pd
        df = pd.read_parquet(pred_path)

    # Find prob_ columns (model predictions)
    prob_cols = [c for c in df.columns if c.startswith("prob_") and c != "prob_ensemble"]
    if len(prob_cols) < 2:
        return {"models": [], "matrix": []}

    corr = df[prob_cols].corr()
    model_names = [c.replace("prob_", "") for c in prob_cols]
    return {
        "models": model_names,
        "matrix": corr.values.tolist(),
    }


@router.get("/runs/{run_id}/calibration")
async def run_calibration(request: Request, run_id: str, bins: int = 10):
    """Compute calibration curve data from predictions."""
    project_dir = Path(request.app.state.project_dir)
    run_dir = project_dir / "outputs" / run_id

    pred_path = run_dir / "predictions" / "predictions.parquet"
    if not pred_path.exists():
        return {"error": "No predictions file found"}

    import numpy as np
    import pandas as pd

    df = pd.read_parquet(pred_path)

    # Need target column and ensemble prob
    target_col = None
    for candidate in ["target", "y", "label", "Survived"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        return {"error": "Could not identify target column"}

    prob_col = "prob_ensemble"
    if prob_col not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if prob_cols:
            prob_col = prob_cols[0]
        else:
            return {"error": "No probability columns found"}

    y_true = df[target_col].values
    y_prob = df[prob_col].values

    # Bin predictions
    bin_edges = np.linspace(0, 1, bins + 1)
    curve = []
    for i in range(bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() > 0:
            curve.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "predicted": float(y_prob[mask].mean()),
                "actual": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            })

    return {"calibration": curve, "prob_column": prob_col}
