"""Project config and DAG endpoints."""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, Request

router = APIRouter(tags=["project"])


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


@router.get("/project/config")
async def project_config(request: Request):
    project_dir = Path(request.app.state.project_dir)
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    models = _load_yaml(config_dir / "models.yaml")
    return {"pipeline": pipeline, "models": models}


@router.get("/project/dag")
async def project_dag(request: Request):
    project_dir = Path(request.app.state.project_dir)
    return build_dag(project_dir)


@router.get("/project/status")
async def project_status(request: Request):
    project_dir = Path(request.app.state.project_dir)
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")

    try:
        from harnessml.core.runner.workflow_tracker import WorkflowTracker
        tracker = WorkflowTracker(project_dir, workflow_config=pipeline.get("workflow", {}))
        status = tracker.get_status()
        return {
            "phase": status.current_phase,
            "model_types_tried": status.model_types_tried,
            "experiments_run": status.experiments_run,
            "has_baseline": status.has_baseline,
            "feature_discovery_done": status.feature_discovery_done,
        }
    except Exception:
        return {"phase": "unknown", "model_types_tried": 0, "experiments_run": 0, "has_baseline": False, "feature_discovery_done": False}


def build_dag(project_dir: Path) -> dict:
    """Build pipeline DAG from project config files."""
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    models_cfg = _load_yaml(config_dir / "models.yaml")

    nodes = []
    edges = []

    # Data sources
    sources = pipeline.get("data", {}).get("sources", {})
    if isinstance(sources, dict):
        for name, src in sources.items():
            nodes.append({"id": f"source_{name}", "type": "source", "label": name, "data": src if isinstance(src, dict) else {"path": str(src)}})

    # Feature store
    features_cfg = pipeline.get("features", {})
    feature_count = len(features_cfg) if isinstance(features_cfg, dict) else 0
    nodes.append({"id": "feature_store", "type": "features", "label": "Feature Store", "data": {"count": feature_count}})
    for src_name in (sources or {}):
        edges.append({"source": f"source_{src_name}", "target": "feature_store"})

    # If no explicit sources, add a generic data node
    if not sources:
        data_file = pipeline.get("data", {}).get("features_file", "features.parquet")
        nodes.append({"id": "source_data", "type": "source", "label": data_file, "data": {}})
        edges.append({"source": "source_data", "target": "feature_store"})

    # Models
    all_models = models_cfg.get("models", {})
    for name, mdef in all_models.items():
        if not isinstance(mdef, dict):
            continue
        nodes.append({
            "id": f"model_{name}", "type": "model", "label": name,
            "data": {
                "model_type": mdef.get("type", "unknown"),
                "features": mdef.get("features", []),
                "active": mdef.get("active", True),
            },
        })
        edges.append({"source": "feature_store", "target": f"model_{name}"})

    # Ensemble
    ensemble = pipeline.get("ensemble", {})
    if not isinstance(ensemble, dict):
        ensemble = {}
    nodes.append({"id": "ensemble", "type": "ensemble", "label": "Ensemble", "data": {"method": ensemble.get("method", "stacking")}})
    for name, mdef in all_models.items():
        if isinstance(mdef, dict) and mdef.get("active", True):
            edges.append({"source": f"model_{name}", "target": "ensemble"})

    # Calibration + Output
    calibration = ensemble.get("calibration", {})
    if calibration:
        nodes.append({"id": "calibration", "type": "calibration", "label": "Calibration", "data": calibration})
        edges.append({"source": "ensemble", "target": "calibration"})
        nodes.append({"id": "output", "type": "output", "label": "Predictions"})
        edges.append({"source": "calibration", "target": "output"})
    else:
        nodes.append({"id": "output", "type": "output", "label": "Predictions"})
        edges.append({"source": "ensemble", "target": "output"})

    return {"nodes": nodes, "edges": edges}
