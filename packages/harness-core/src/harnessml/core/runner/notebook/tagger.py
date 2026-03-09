from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


def _load_models_and_features(project_dir: Path) -> tuple[list[str], list[str]]:
    """Load model names and feature names from project config.

    Model names come from:
    - Top-level keys in config/models.yaml under ``models:``
    - Filename stems of config/models/*.yaml

    Feature names come from ``features`` lists inside model dicts in
    config/models.yaml.
    """
    model_names: list[str] = []
    feature_names: list[str] = []

    # config/models.yaml
    models_yaml = project_dir / "config" / "models.yaml"
    try:
        data = yaml.safe_load(models_yaml.read_text())
        if isinstance(data, dict):
            models_dict = data.get("models") or {}
            if isinstance(models_dict, dict):
                for name, cfg in models_dict.items():
                    model_names.append(str(name))
                    if isinstance(cfg, dict):
                        feats = cfg.get("features")
                        if isinstance(feats, list):
                            for f in feats:
                                feature_names.append(str(f))
    except Exception:
        pass

    # config/models/*.yaml (filename stems)
    models_dir = project_dir / "config" / "models"
    try:
        if models_dir.is_dir():
            for p in models_dir.glob("*.yaml"):
                model_names.append(p.stem)
    except Exception:
        pass

    # Deduplicate preserving order
    seen: set[str] = set()
    deduped_models: list[str] = []
    for m in model_names:
        if m not in seen:
            seen.add(m)
            deduped_models.append(m)

    seen_f: set[str] = set()
    deduped_features: list[str] = []
    for f in feature_names:
        if f not in seen_f:
            seen_f.add(f)
            deduped_features.append(f)

    return deduped_models, deduped_features


def _load_source_names(project_dir: Path) -> list[str]:
    """Load source names from data/source_registry.json."""
    registry_path = project_dir / "data" / "source_registry.json"
    try:
        data = json.loads(registry_path.read_text())
        if isinstance(data, dict):
            return list(data.keys())
    except Exception:
        pass
    return []


def auto_tag(content: str, project_dir: Path) -> list[str]:
    """Scan notebook entry content for known entity names and return tags.

    Detection is case-insensitive. Results are deduplicated preserving order.
    Missing config/data directories are silently skipped.
    """
    tags: list[str] = []
    seen: set[str] = set()
    content_lower = content.lower()

    def _add(tag: str) -> None:
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    # Model names
    model_names, feature_names = _load_models_and_features(project_dir)
    for name in model_names:
        if name.lower() in content_lower:
            _add(f"model:{name}")

    # Feature names
    for name in feature_names:
        if name.lower() in content_lower:
            _add(f"feature:{name}")

    # Source names
    source_names = _load_source_names(project_dir)
    for name in source_names:
        if name.lower() in content_lower:
            _add(f"source:{name}")

    # Experiment IDs
    for match in re.finditer(r"\bexp-\d+\b", content, re.IGNORECASE):
        _add(f"experiment:{match.group()}")

    return tags
