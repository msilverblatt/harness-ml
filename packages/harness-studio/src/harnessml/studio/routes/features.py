"""Feature store endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request
from harnessml.studio.routes.project import _load_yaml, resolve_project_dir_from_request

router = APIRouter(tags=["features"])


def _collect_features(project_dir: Path) -> list[dict]:
    """Collect features from cache manifest and pipeline config."""
    features = []
    seen = set()

    # Cache manifest (primary — has computed features)
    manifest_path = project_dir / "data" / "features" / "cache" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            for name, info in manifest.items():
                seen.add(name)
                features.append({
                    "name": name,
                    "type": info.get("feature_type", "instance"),
                    "source": info.get("source"),
                    "derived_from": info.get("derived_from", []),
                    "formula": None,
                    "category": "general",
                    "enabled": True,
                })
        except (json.JSONDecodeError, AttributeError):
            pass

    # Pipeline feature_defs (has formulas, categories)
    pipeline = _load_yaml(project_dir / "config" / "pipeline.yaml")
    feature_defs = pipeline.get("data", {}).get("feature_defs", {})
    if isinstance(feature_defs, dict):
        for name, fdef in feature_defs.items():
            if not isinstance(fdef, dict):
                continue
            if name in seen:
                # Enrich existing entry with formula/category
                for f in features:
                    if f["name"] == name:
                        f["formula"] = fdef.get("formula")
                        f["category"] = fdef.get("category", "general")
                        f["enabled"] = fdef.get("enabled", True)
                        break
            else:
                seen.add(name)
                features.append({
                    "name": name,
                    "type": fdef.get("type", "instance"),
                    "source": None,
                    "derived_from": [],
                    "formula": fdef.get("formula"),
                    "category": fdef.get("category", "general"),
                    "enabled": fdef.get("enabled", True),
                })

    return features


def _get_model_feature_usage(project_dir: Path) -> dict[str, list[str]]:
    """Map feature name -> list of model names that use it."""
    models_cfg = _load_yaml(project_dir / "config" / "models.yaml")
    all_models = models_cfg.get("models", {})
    usage: dict[str, list[str]] = {}
    for model_name, mdef in all_models.items():
        if not isinstance(mdef, dict):
            continue
        for feat in mdef.get("features", []):
            usage.setdefault(feat, []).append(model_name)
    return usage


@router.get("/features/importance")
async def feature_importance(request: Request, project: str | None = None):
    """Return feature importances from pre-computed diagnostics if available."""
    project_dir = resolve_project_dir_from_request(request, project)

    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return {"features": [], "source": "none"}

    run_dirs = sorted([d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True)
    for run_dir in run_dirs:
        diag_path = run_dir / "diagnostics" / "feature_importance.json"
        if not diag_path.exists():
            continue
        try:
            raw = json.loads(diag_path.read_text())
            features = []
            if isinstance(raw, dict):
                sorted_items = sorted(raw.items(), key=lambda x: x[1], reverse=True)
                max_val = sorted_items[0][1] if sorted_items else 1.0
                for rank, (name, score) in enumerate(sorted_items, 1):
                    features.append({
                        "name": name,
                        "importance": round(score / max_val, 4) if max_val else 0,
                        "rank": rank,
                    })
            elif isinstance(raw, list):
                features = raw
            if features:
                return {"features": features, "source": "model"}
        except (json.JSONDecodeError, AttributeError):
            continue

    return {"features": [], "source": "none"}


@router.get("/features")
async def list_features(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    features = _collect_features(project_dir)
    usage = _get_model_feature_usage(project_dir)

    for f in features:
        f["used_by"] = usage.get(f["name"], [])

    return features


@router.get("/features/summary")
async def feature_summary(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    features = _collect_features(project_dir)
    usage = _get_model_feature_usage(project_dir)

    by_type: dict[str, int] = {}
    by_category: dict[str, int] = {}
    used_count = 0
    for f in features:
        by_type[f["type"]] = by_type.get(f["type"], 0) + 1
        by_category[f["category"]] = by_category.get(f["category"], 0) + 1
        if f["name"] in usage:
            used_count += 1

    return {
        "total": len(features),
        "used_by_models": used_count,
        "unused": len(features) - used_count,
        "by_type": by_type,
        "by_category": by_category,
    }
