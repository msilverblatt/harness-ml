"""Model configuration endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Request
from harnessml.studio.routes.project import _load_yaml, resolve_project_dir_from_request

router = APIRouter(tags=["models"])


@router.get("/models")
async def list_models(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    models_cfg = _load_yaml(project_dir / "config" / "models.yaml")
    all_models = models_cfg.get("models", {})

    # Load experimental models if they exist
    experimental_cfg = _load_yaml(project_dir / "config" / "models" / "experimental.yaml")

    result = []
    for name, mdef in all_models.items():
        if not isinstance(mdef, dict):
            continue
        result.append({
            "name": name,
            "type": mdef.get("type", "unknown"),
            "mode": mdef.get("mode"),
            "params": mdef.get("params", {}),
            "features": mdef.get("features", []),
            "feature_count": len(mdef.get("features", [])),
            "active": mdef.get("active", True),
            "include_in_ensemble": mdef.get("include_in_ensemble", True),
            "source": "production",
        })

    for name, mdef in experimental_cfg.items():
        if not isinstance(mdef, dict) or name in {m["name"] for m in result}:
            continue
        result.append({
            "name": name,
            "type": mdef.get("type", "unknown"),
            "mode": mdef.get("mode"),
            "params": mdef.get("params", {}),
            "features": mdef.get("features", []),
            "feature_count": len(mdef.get("features", [])),
            "active": mdef.get("active", True),
            "include_in_ensemble": mdef.get("include_in_ensemble", False),
            "source": "experimental",
        })

    return result


@router.get("/models/{model_name}/metrics")
async def model_metrics(request: Request, model_name: str, project: str | None = None):
    """Get per-model metrics from latest run predictions."""
    import numpy as np
    import pandas as pd

    project_dir = resolve_project_dir_from_request(request, project)
    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return {"error": "No runs found"}

    # Find latest run with predictions
    run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
    for run_dir in run_dirs:
        pred_dir = run_dir / "predictions"
        if not pred_dir.exists():
            continue

        frames = []
        for f in sorted(pred_dir.glob("*.parquet")):
            frames.append(pd.read_parquet(f))
        if not frames:
            continue

        df = pd.concat(frames, ignore_index=True)
        prob_col = f"prob_{model_name}"
        if prob_col not in df.columns:
            return {"error": f"No predictions for model '{model_name}'"}

        # Find target column
        prob_cols = {c for c in df.columns if c.startswith("prob_")}
        meta_cols = {"fold", "diff_prior"}
        target_col = None
        for col in df.columns:
            if col not in prob_cols and col not in meta_cols:
                target_col = col
                break

        if target_col is None:
            return {"error": "Could not detect target column"}

        y_true = df[target_col].values.astype(float)
        y_prob = df[prob_col].values.astype(float)

        metrics: dict[str, float] = {}

        # Binary classification metrics
        unique = set(y_true)
        if unique.issubset({0.0, 1.0}):
            y_pred = (y_prob >= 0.5).astype(float)
            metrics["accuracy"] = round(float(np.mean(y_pred == y_true)), 4)
            metrics["brier"] = round(float(np.mean((y_prob - y_true) ** 2)), 4)
        else:
            # Regression
            metrics["rmse"] = round(float(np.sqrt(np.mean((y_prob - y_true) ** 2))), 4)
            metrics["mae"] = round(float(np.mean(np.abs(y_prob - y_true))), 4)

        return {
            "model": model_name,
            "run_id": run_dir.name,
            "metrics": metrics,
            "n_predictions": len(df),
        }

    return {"error": "No runs with predictions found"}
