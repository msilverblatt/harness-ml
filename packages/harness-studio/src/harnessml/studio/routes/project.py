"""Project config and DAG endpoints."""
from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Request
from harnessml.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["project"])


def resolve_project_dir_from_request(request: Request, project: str | None) -> Path:
    """Resolve a project name to its on-disk directory.

    Falls back to app.state.project_dir for standalone (single-project) mode.
    """
    if project:
        store = request.app.state.event_store
        if store is not None:
            d = store.get_project_dir(project)
            if d:
                return Path(d)
        raise HTTPException(404, f"project '{project}' not found in registry")
    pd = getattr(request.app.state, "project_dir", None)
    if pd:
        return Path(pd)
    raise HTTPException(400, "project parameter required")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


@router.get("/projects")
async def list_projects(request: Request):
    """List all registered projects."""
    store = request.app.state.event_store
    if store is None:
        return []
    return store.list_projects_with_dirs()


@router.get("/project/config")
async def project_config(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    models = _load_yaml(config_dir / "models.yaml")
    return {"pipeline": pipeline, "models": models}


@router.get("/project/dag")
async def project_dag(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    return build_dag(project_dir)


@router.get("/project/status")
async def project_status(request: Request, project: str | None = None):
    import json as _json

    project_dir = resolve_project_dir_from_request(request, project)
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    models_cfg = _load_yaml(config_dir / "models.yaml")

    # Project name from pipeline or directory
    project_name = pipeline.get("name") or pipeline.get("data", {}).get("name") or project_dir.name

    # Task type and target
    task = pipeline.get("data", {}).get("task", "unknown")
    target_column = pipeline.get("data", {}).get("target_column")

    # Count models
    all_models = models_cfg.get("models", {})
    model_types = set()
    for mdef in all_models.values():
        if isinstance(mdef, dict):
            model_types.add(mdef.get("type", "unknown"))
    active_models = sum(1 for m in all_models.values() if isinstance(m, dict) and m.get("active", True))

    # Count experiments from journal
    experiments_run = 0
    journal_path = project_dir / "experiments" / "journal.jsonl"
    if journal_path.exists():
        for line in journal_path.read_text().strip().split("\n"):
            if line.strip():
                experiments_run += 1

    # Count runs
    outputs_dir = project_dir / "outputs"
    run_count = sum(1 for d in outputs_dir.iterdir() if d.is_dir()) if outputs_dir.exists() else 0

    # Feature count
    feature_defs = pipeline.get("data", {}).get("feature_defs", {})
    feature_count = len(feature_defs) if isinstance(feature_defs, dict) else 0
    if feature_count == 0:
        # Count from model feature lists
        all_feats: set[str] = set()
        for mdef in all_models.values():
            if isinstance(mdef, dict):
                all_feats.update(mdef.get("features", []))
        feature_count = len(all_feats)

    # Latest metrics — best from runs or experiments
    _LOWER_IS_BETTER = {"brier", "ece", "log_loss", "mae", "mse", "rmse"}
    candidates: list[dict] = []

    # Collect from run outputs
    if outputs_dir.exists():
        run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
        for rd in run_dirs:
            mp = rd / "diagnostics" / "pooled_metrics.json"
            if mp.exists():
                try:
                    raw = _json.loads(mp.read_text())
                    metrics = raw.get("ensemble", raw) if isinstance(raw.get("ensemble"), dict) else raw
                    if metrics:
                        candidates.append(metrics)
                except (_json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning("failed to parse pooled_metrics.json", path=str(mp), error=str(e))
                break

    # Collect from experiment result files
    exp_dir = project_dir / "experiments"
    if exp_dir.exists():
        for child in exp_dir.iterdir():
            if not child.is_dir():
                continue
            results_path = child / "results.json"
            if results_path.exists():
                try:
                    results = _json.loads(results_path.read_text())
                    exp_metrics = results.get("metrics", {})
                    if exp_metrics:
                        candidates.append(exp_metrics)
                except (_json.JSONDecodeError, KeyError, ValueError):
                    continue

    # Pick best candidate by primary metric
    latest_metrics: dict = {}
    if candidates:
        # Determine primary metric
        primary_metric: str | None = None
        for c in candidates:
            if "brier" in c:
                primary_metric = "brier"
                break
            if "accuracy" in c:
                primary_metric = "accuracy"
                break
        if primary_metric is None:
            primary_metric = next(iter(candidates[0]), None)

        if primary_metric:
            lower = primary_metric in _LOWER_IS_BETTER
            best_val: float | None = None
            for c in candidates:
                val = c.get(primary_metric)
                if val is None:
                    continue
                if best_val is None or (lower and val < best_val) or (not lower and val > best_val):
                    best_val = val
                    latest_metrics = c
        else:
            latest_metrics = candidates[0]

    return {
        "project_name": project_name,
        "task": task,
        "target_column": target_column,
        "model_types_tried": len(model_types),
        "active_models": active_models,
        "experiments_run": experiments_run,
        "run_count": run_count,
        "feature_count": feature_count,
        "latest_metrics": latest_metrics,
    }


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    import json
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.warning("failed to load JSON file", path=str(path), error=str(e))
        return {}


def _collect_sources(project_dir: Path, pipeline: dict) -> list[dict]:
    """Collect data sources from source_registry.json and pipeline config."""
    sources = []

    # Source registry (primary source of truth)
    registry = _load_json(project_dir / "data" / "source_registry.json")
    seen_names: set[str] = set()
    registry_sources = registry.get("sources", registry if isinstance(registry, list) else [])
    if isinstance(registry_sources, list):
        for src in registry_sources:
            name = src.get("name", "unknown")
            if name in seen_names:
                continue
            seen_names.add(name)
            sources.append({
                "name": name,
                "path": src.get("path", ""),
                "columns": src.get("columns_added", []),
                "rows": src.get("rows", 0),
                "adapter": src.get("adapter", "file"),
            })

    # Pipeline data.sources as fallback
    pipe_sources = pipeline.get("data", {}).get("sources", {})
    if isinstance(pipe_sources, dict):
        for name, src in pipe_sources.items():
            if name not in seen_names:
                sources.append({
                    "name": name,
                    "path": src.get("path", str(src)) if isinstance(src, dict) else str(src),
                    "columns": [],
                    "rows": 0,
                    "adapter": src.get("adapter", "file") if isinstance(src, dict) else "file",
                })

    return sources


def _collect_feature_defs(pipeline: dict) -> list[dict]:
    """Collect feature definitions from pipeline config."""
    feature_defs = pipeline.get("data", {}).get("feature_defs", {})
    if not isinstance(feature_defs, dict):
        return []
    features = []
    for name, fdef in feature_defs.items():
        if not isinstance(fdef, dict):
            continue
        features.append({
            "name": name,
            "type": fdef.get("type", "instance"),
            "formula": fdef.get("formula"),
            "category": fdef.get("category", "general"),
            "enabled": fdef.get("enabled", True),
            "nan_strategy": fdef.get("nan_strategy"),
        })
    return features


def build_dag(project_dir: Path) -> dict:
    """Build rich pipeline DAG from project config files."""
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    models_cfg = _load_yaml(config_dir / "models.yaml")
    data_cfg = pipeline.get("data", {})

    nodes = []
    edges = []

    # --- Data sources ---
    sources = _collect_sources(project_dir, pipeline)
    if sources:
        for src in sources:
            nodes.append({
                "id": f"source_{src['name']}",
                "type": "source",
                "label": src["name"],
                "data": {
                    "path": src["path"],
                    "columns": src["columns"],
                    "rows": src["rows"],
                    "adapter": src["adapter"],
                },
            })
    else:
        # Fallback: generic data node
        data_file = data_cfg.get("features_file", "features.parquet")
        nodes.append({
            "id": "source_data", "type": "source", "label": data_file,
            "data": {"path": data_file, "columns": [], "rows": 0, "adapter": "file"},
        })

    # --- Views / transforms ---
    views = data_cfg.get("views", {})
    if isinstance(views, dict) and views:
        for view_name, view_def in views.items():
            steps = []
            if isinstance(view_def, dict):
                steps = list(view_def.get("steps", {}).keys()) if isinstance(view_def.get("steps"), dict) else []
                if isinstance(view_def.get("steps"), list):
                    steps = [s.get("type", "?") if isinstance(s, dict) else str(s) for s in view_def["steps"]]
            nodes.append({
                "id": f"view_{view_name}", "type": "view", "label": view_name,
                "data": {"steps": steps, "step_count": len(steps)},
            })
            # Views connect sources to feature store
            for src in sources:
                edges.append({"source": f"source_{src['name']}", "target": f"view_{view_name}"})
            edges.append({"source": f"view_{view_name}", "target": "feature_store"})

    # --- Feature store ---
    feature_defs = _collect_feature_defs(pipeline)
    all_models = models_cfg.get("models", {})

    # Classify features as raw (passthrough) vs engineered (has formula)
    if feature_defs:
        enabled_defs = [f for f in feature_defs if f["enabled"]]
        feature_count = len(enabled_defs)
        raw_features = [f["name"] for f in enabled_defs if not f["formula"]]
        engineered_features = [
            {"name": f["name"], "formula": f["formula"]}
            for f in enabled_defs if f["formula"]
        ]
        feature_names = [f["name"] for f in enabled_defs]
    else:
        all_feat_set: set[str] = set()
        for mdef in all_models.values():
            if isinstance(mdef, dict):
                all_feat_set.update(mdef.get("features", []))
        feature_count = len(all_feat_set)
        feature_names = sorted(all_feat_set)
        raw_features = feature_names
        engineered_features = []

    nodes.append({
        "id": "feature_store", "type": "features", "label": "Feature Store",
        "data": {
            "count": feature_count,
            "features": feature_names,
            "raw_features": raw_features,
            "engineered_features": engineered_features,
            "target_column": data_cfg.get("target_column"),
            "task": data_cfg.get("task"),
        },
    })

    # Connect sources directly to feature store if no views
    if not (isinstance(views, dict) and views):
        for src in sources:
            edges.append({"source": f"source_{src['name']}", "target": "feature_store"})
        if not sources:
            edges.append({"source": "source_data", "target": "feature_store"})

    # --- Detect active experiment overlay models ---
    experiment_models: dict[str, dict] = {}  # model_name -> overlay def
    active_experiment_id: str | None = None
    experiments_dir = project_dir / "experiments"
    journal_path = experiments_dir / "journal.jsonl"
    if journal_path.exists():
        import json as _json
        # Find the most recent experiment (last line in journal)
        lines = [l for l in journal_path.read_text().strip().split("\n") if l.strip()]
        for line in reversed(lines):
            try:
                entry = _json.loads(line)
            except (_json.JSONDecodeError, ValueError) as e:
                logger.warning("failed to parse journal entry", error=str(e))
                continue
            exp_id = entry.get("experiment_id", "")
            if not exp_id:
                continue
            overlay_path = experiments_dir / exp_id / "overlay.yaml"
            if overlay_path.exists():
                overlay = yaml.safe_load(overlay_path.read_text()) or {}
                overlay_models = overlay.get("models", {})
                if isinstance(overlay_models, dict) and overlay_models:
                    active_experiment_id = exp_id
                    experiment_models = overlay_models
                    break

    # --- Models ---
    for name, mdef in all_models.items():
        if not isinstance(mdef, dict):
            continue
        model_features = mdef.get("features", [])
        params = mdef.get("params", {})
        is_experiment = name in experiment_models
        nodes.append({
            "id": f"model_{name}", "type": "model", "label": name,
            "data": {
                "model_type": mdef.get("type", "unknown"),
                "mode": mdef.get("mode"),
                "features": model_features,
                "feature_count": len(model_features),
                "active": mdef.get("active", True),
                "params": {k: v for k, v in params.items()},
                **({"_experiment": active_experiment_id} if is_experiment else {}),
            },
        })
        edges.append({"source": "feature_store", "target": f"model_{name}"})

    # Add experiment-only models (new models not in production config)
    for name, mdef in experiment_models.items():
        if name in all_models or not isinstance(mdef, dict):
            continue
        model_features = mdef.get("features", [])
        params = mdef.get("params", {})
        nodes.append({
            "id": f"model_{name}", "type": "model", "label": name,
            "data": {
                "model_type": mdef.get("type", "unknown"),
                "mode": mdef.get("mode"),
                "features": model_features,
                "feature_count": len(model_features),
                "active": mdef.get("active", True),
                "params": {k: v for k, v in params.items()},
                "_experiment": active_experiment_id,
                "_experiment_new": True,
            },
        })
        edges.append({"source": "feature_store", "target": f"model_{name}"})

    # --- Ensemble ---
    ensemble = models_cfg.get("ensemble", pipeline.get("ensemble", {}))
    if not isinstance(ensemble, dict):
        ensemble = {}
    backtest = pipeline.get("backtest", {})

    nodes.append({
        "id": "ensemble", "type": "ensemble", "label": "Ensemble",
        "data": {
            "method": ensemble.get("method", "stacking"),
            "model_count": sum(1 for m in all_models.values() if isinstance(m, dict) and m.get("active", True)),
            "cv_strategy": backtest.get("cv_strategy"),
            "metrics": backtest.get("metrics", []),
            "fold_column": backtest.get("fold_column"),
        },
    })
    for name, mdef in all_models.items():
        if isinstance(mdef, dict) and mdef.get("active", True):
            edges.append({"source": f"model_{name}", "target": "ensemble"})
    # Connect experiment-only models to ensemble
    for name, mdef in experiment_models.items():
        if name not in all_models and isinstance(mdef, dict) and mdef.get("active", True):
            edges.append({"source": f"model_{name}", "target": "ensemble"})

    # --- Calibration ---
    calibration = ensemble.get("calibration", {})
    if calibration:
        cal_data = calibration if isinstance(calibration, dict) else {"type": str(calibration)}
        nodes.append({"id": "calibration", "type": "calibration", "label": "Calibration", "data": cal_data})
        edges.append({"source": "ensemble", "target": "calibration"})
        nodes.append({"id": "output", "type": "output", "label": "Predictions", "data": {"target": data_cfg.get("target_column")}})
        edges.append({"source": "calibration", "target": "output"})
    else:
        nodes.append({"id": "output", "type": "output", "label": "Predictions", "data": {"target": data_cfg.get("target_column")}})
        edges.append({"source": "ensemble", "target": "output"})

    return {"nodes": nodes, "edges": edges}
