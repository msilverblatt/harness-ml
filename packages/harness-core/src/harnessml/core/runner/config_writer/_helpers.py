"""Shared helpers for config_writer submodules."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _save_yaml(path: Path, data: dict) -> None:
    """Save data to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))


def _get_config_dir(project_dir: Path) -> Path:
    """Resolve the config directory from project dir."""
    config_dir = project_dir / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")
    return config_dir


def _invalidate_view_cache(project_dir: Path, name: str) -> None:
    """Remove cached parquet files for a view (and downstream dependents).

    The fingerprint-based cache is self-invalidating on next resolve, but this
    proactively removes stale files so they don't consume disk space.
    """
    cache_dir = Path(project_dir) / "data" / "views" / ".cache"
    if not cache_dir.exists():
        return
    for f in cache_dir.glob(f"{name}_*.parquet"):
        f.unlink()


def _persist_feature_defs(project_dir: Path, config) -> None:
    """Write feature_defs back to pipeline.yaml so the runner picks them up."""
    config_dir = _get_config_dir(project_dir)
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}

    if config.feature_defs:
        data["data"]["feature_defs"] = {
            name: feat.model_dump(mode="json")
            for name, feat in config.feature_defs.items()
        }
    elif "feature_defs" in data.get("data", {}):
        del data["data"]["feature_defs"]

    _save_yaml(pipeline_path, data)


def _expand_dot_keys(flat: dict) -> dict:
    """Expand a flat dict with dot-notation keys into nested dicts.

    Keys that don't contain dots are left as-is.  Keys that do contain dots
    are expanded into nested structures using ``set_nested_key``.

    Example::

        _expand_dot_keys({"models.xgb.features": ["a", "b"]})
        # -> {"models": {"xgb": {"features": ["a", "b"]}}}
    """
    from harnessml.core.runner.sweep import set_nested_key

    result: dict = {}
    for key, value in flat.items():
        if "." in key:
            set_nested_key(result, key, value)
        else:
            result[key] = value
    return result


def _get_source_registry(project_dir: Path):
    """Lazily import and return a SourceRegistry for the project."""
    from harnessml.core.runner.sources.registry import SourceRegistry

    config_dir = _get_config_dir(Path(project_dir))
    return SourceRegistry(config_dir)


def _get_freshness_tracker(project_dir: Path):
    """Lazily import and return a FreshnessTracker for the project."""
    from harnessml.core.runner.sources.freshness import FreshnessTracker

    config_dir = _get_config_dir(Path(project_dir))
    state_file = config_dir / "sources_state.json"
    return FreshnessTracker(state_file)


# Metrics where lower values are better -- used for delta direction indicators.
_LOWER_IS_BETTER = frozenset({
    "brier", "brier_score", "log_loss", "logloss", "mae", "mse", "rmse",
    "ece", "expected_calibration_error", "mape", "smape", "rae",
})
