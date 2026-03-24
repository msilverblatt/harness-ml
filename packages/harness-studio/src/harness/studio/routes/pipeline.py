"""Pipeline config routes — read workspace config files directly."""
from __future__ import annotations

from fastapi import APIRouter, Request

from harness.app.workspace.config import ConfigManager
from harness.ml.runners.dag import ModelDAG

router = APIRouter()


def _config(request: Request) -> ConfigManager:
    return ConfigManager(request.app.state.workspace_dir)


@router.get("/config")
def pipeline_config(request: Request):
    """Current pipeline config: project + models + ensemble + features."""
    cm = _config(request)
    project = cm.read_project()
    models = cm.read_models()
    ensemble = cm.read_ensemble()
    features = cm.read_features()

    models_dict = {}
    for name, m in models.models.items():
        d = m.model_dump(exclude_defaults=True)
        d.pop("name", None)
        models_dict[name] = d

    features_dict = {}
    for name, f in features.features.items():
        d = f.model_dump(exclude_defaults=True, mode="json")
        d.pop("name", None)
        features_dict[name] = d

    return {
        "project": project.model_dump(),
        "models": models_dict,
        "ensemble": ensemble.model_dump(),
        "features": features_dict,
    }


@router.get("/dag")
def pipeline_dag(request: Request):
    """Model dependency DAG from models config."""
    cm = _config(request)
    models = cm.read_models()
    dag = ModelDAG(models.models)
    waves = dag.topological_waves()
    errors = dag.validate()

    nodes = []
    edges = []
    for name, m in models.models.items():
        nodes.append({"name": name, "model_type": m.model_type, "active": m.active})
        for dep in m.depends_on:
            edges.append({"from": dep, "to": name})

    return {
        "nodes": nodes,
        "edges": edges,
        "waves": waves,
        "errors": errors,
    }
