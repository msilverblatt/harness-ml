"""Ensemble analysis endpoints."""
from __future__ import annotations

import json as _json
from pathlib import Path

from fastapi import APIRouter, Request
from harnessml.studio.routes.project import _load_yaml, resolve_project_dir_from_request

router = APIRouter(tags=["ensemble"])


def _latest_run_id(project_dir: Path) -> str | None:
    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return None
    run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
    return run_dirs[0].name if run_dirs else None


@router.get("/ensemble/config")
async def ensemble_config(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    models_cfg = _load_yaml(project_dir / "config" / "models.yaml")
    pipeline = _load_yaml(project_dir / "config" / "pipeline.yaml")
    ensemble_cfg = _load_yaml(project_dir / "config" / "ensemble.yaml")

    ensemble = ensemble_cfg.get("ensemble", models_cfg.get("ensemble", pipeline.get("ensemble", {})))
    if not isinstance(ensemble, dict):
        ensemble = {}

    backtest = pipeline.get("backtest", {})

    # Resolve task_type from pipeline config
    pipeline_inner = pipeline.get("pipeline", {})
    task_type = (
        pipeline_inner.get("task")
        or pipeline.get("task")
        or None
    )

    # Resolve target_column from pipeline config
    data_section = pipeline_inner.get("data", pipeline.get("data", {}))
    if isinstance(data_section, dict):
        target_column = data_section.get("target") or pipeline_inner.get("target") or pipeline.get("target")
    else:
        target_column = pipeline_inner.get("target") or pipeline.get("target")

    # Count active models
    models_list = models_cfg.get("models", [])
    if isinstance(models_list, list):
        model_count = sum(
            1 for m in models_list
            if isinstance(m, dict) and m.get("active", True)
        )
    else:
        model_count = 0

    return {
        "method": ensemble.get("method", "stacking"),
        "meta_learner": ensemble.get("meta_learner", {}),
        "temperature": ensemble.get("temperature"),
        "clip_floor": ensemble.get("clip_floor"),
        "calibration": ensemble.get("calibration", {}),
        "cv_strategy": backtest.get("cv_strategy"),
        "fold_column": backtest.get("fold_column"),
        "metrics": backtest.get("metrics", []),
        "exclude_models": ensemble.get("exclude_models", []),
        "pre_calibration": ensemble.get("pre_calibration", {}),
        "task_type": task_type,
        "target_column": target_column,
        "model_count": model_count,
    }


@router.get("/ensemble/weights")
async def ensemble_weights(request: Request, project: str | None = None):
    """Meta-learner coefficients from latest run."""
    project_dir = resolve_project_dir_from_request(request, project)
    run_id = _latest_run_id(project_dir)
    if not run_id:
        return {"error": "No runs found"}

    metrics_path = project_dir / "outputs" / run_id / "diagnostics" / "pooled_metrics.json"
    if not metrics_path.exists():
        return {"error": "No metrics found"}

    raw = _json.loads(metrics_path.read_text())
    coefficients = raw.get("meta_coefficients", {})

    # Extract all pooled metrics (exclude meta_coefficients — they're already in coefficients)
    pooled_metrics = {k: v for k, v in raw.items() if k != "meta_coefficients"}

    # Detect task_type from metrics keys if possible
    task_type = None
    if "brier" in raw or "accuracy" in raw or "ece" in raw:
        task_type = "binary"
    elif "rmse" in raw or "mae" in raw or "r2" in raw:
        task_type = "regression"

    return {
        "run_id": run_id,
        "coefficients": coefficients,
        "pooled_metrics": pooled_metrics,
        "task_type": task_type,
    }
