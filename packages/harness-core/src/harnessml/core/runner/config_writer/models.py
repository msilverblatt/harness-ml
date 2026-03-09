"""Model and ensemble configuration operations."""
from __future__ import annotations

import logging
from pathlib import Path

import yaml
from harnessml.core.runner.config_writer._helpers import (
    _get_config_dir,
    _load_yaml,
    _save_yaml,
)

logger = logging.getLogger(__name__)


def _check_feature_existence(
    project_dir: Path,
    features: list[str],
) -> list[str]:
    """Check feature names against project config.

    Returns a list of warning strings for unknown features. Returns an
    empty list if all features are known or if the config cannot be loaded.
    """
    try:
        config_dir = _get_config_dir(Path(project_dir))
    except FileNotFoundError:
        return []

    pipeline_data = _load_yaml(config_dir / "pipeline.yaml")
    feature_defs = pipeline_data.get("data", {}).get("feature_defs", {})

    if not feature_defs:
        # No declarative features defined — can't validate
        return []

    known_features = set(feature_defs.keys())
    unknown = [f for f in features if f not in known_features]

    if unknown:
        return [
            f"**Warning**: Unknown feature(s) not found in feature_defs: "
            f"{', '.join(f'`{u}`' for u in unknown)}. "
            f"They may be raw columns or misspelled."
        ]
    return []


def add_model(
    project_dir: Path,
    name: str,
    *,
    model_type: str | None = None,
    preset: str | None = None,
    features: list[str] | None = None,
    params: dict | None = None,
    active: bool = True,
    include_in_ensemble: bool = True,
    mode: str | None = None,
    prediction_type: str | None = None,
    cdf_scale: float | None = None,
    zero_fill_features: list[str] | None = None,
    class_weight: str | dict | None = None,
) -> str:
    """Add a model to models.yaml.

    Either model_type or preset must be specified. If preset is given,
    applies the preset defaults and merges any overrides.

    Returns markdown confirmation.
    """
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    if "models" not in data:
        data["models"] = {}

    if name in data["models"]:
        return f"**Error**: Model `{name}` already exists. Use a different name or remove it first."

    model_def: dict = {}

    if preset:
        from harnessml.core.runner.scaffold.presets import apply_preset
        model_def = apply_preset(preset, overrides=params or {})
    elif model_type:
        model_def["type"] = model_type
        if mode is not None:
            model_def["mode"] = mode
        if prediction_type is not None:
            model_def["prediction_type"] = prediction_type
        if params:
            model_def["params"] = params
    else:
        return "**Error**: Either `model_type` or `preset` must be specified."

    if features:
        model_def["features"] = features
    model_def["active"] = active
    model_def["include_in_ensemble"] = include_in_ensemble
    if cdf_scale is not None:
        model_def["cdf_scale"] = cdf_scale
    if zero_fill_features is not None:
        model_def["zero_fill_features"] = zero_fill_features
    if class_weight is not None:
        model_def["class_weight"] = class_weight

    data["models"][name] = model_def
    _save_yaml(models_path, data)

    # Check for unknown features (warning only, not blocking)
    warnings: list[str] = []
    if features:
        warnings = _check_feature_existence(project_dir, features)

    n_features = len(model_def.get("features", []))
    type_str = model_def.get("type", "unknown")
    preset_str = f" (preset: {preset})" if preset else ""

    result = (
        f"**Added model**: `{name}`\n"
        f"- Type: {type_str}{preset_str}\n"
        f"- Features: {n_features}\n"
        f"- Active: {active}\n"
        f"- Include in ensemble: {include_in_ensemble}"
    )
    if warnings:
        result += "\n\n" + "\n".join(warnings)
    return result


def remove_model(project_dir: Path, name: str, *, purge: bool = False) -> str:
    """Disable or permanently delete a model from models.yaml.

    By default (purge=False) sets active=False and include_in_ensemble=False.
    Pass purge=True to delete the model entry entirely.
    """
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        return f"**Error**: Model `{name}` not found."

    if purge:
        del models[name]
        _save_yaml(models_path, data)
        return f"**Removed model**: `{name}`"

    models[name]["active"] = False
    models[name]["include_in_ensemble"] = False
    _save_yaml(models_path, data)
    return (
        f"**Disabled model**: `{name}`\n"
        f"- Set `active: false`, `include_in_ensemble: false`\n"
        f"- Use `purge=True` to delete permanently."
    )


def update_model(
    project_dir: Path,
    name: str,
    *,
    features: list[str] | None = None,
    append_features: list[str] | None = None,
    remove_features: list[str] | None = None,
    params: dict | None = None,
    active: bool | None = None,
    include_in_ensemble: bool | None = None,
    mode: str | None = None,
    prediction_type: str | None = None,
    cdf_scale: float | None = None,
    zero_fill_features: list[str] | None = None,
    class_weight: str | dict | None = None,
    replace_params: bool = False,
) -> str:
    """Update an existing model in models.yaml.

    Only provided fields are merged -- None values keep the existing value.
    For params, does a dict merge by default. Set replace_params=True to
    fully replace the params dict instead of merging.

    Feature list helpers:
      - append_features: add to the existing feature list (skips duplicates).
      - remove_features: remove from the existing feature list.
    These are applied after ``features`` (if given), so they can be combined.
    """
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        return f"**Error**: Model `{name}` not found. Available: {sorted(models.keys())}"

    model_def = models[name]

    if features is not None:
        model_def["features"] = features
    if append_features:
        existing = model_def.get("features", [])
        model_def["features"] = existing + [f for f in append_features if f not in existing]
    if remove_features:
        existing = model_def.get("features", [])
        model_def["features"] = [f for f in existing if f not in remove_features]
    if params is not None:
        if replace_params:
            model_def["params"] = params
        else:
            existing_params = model_def.get("params", {})
            existing_params.update(params)
            model_def["params"] = existing_params
    if active is not None:
        model_def["active"] = active
    if include_in_ensemble is not None:
        model_def["include_in_ensemble"] = include_in_ensemble
    if mode is not None:
        model_def["mode"] = mode
    if prediction_type is not None:
        model_def["prediction_type"] = prediction_type
    if cdf_scale is not None:
        model_def["cdf_scale"] = cdf_scale
    if zero_fill_features is not None:
        model_def["zero_fill_features"] = zero_fill_features
    if class_weight is not None:
        model_def["class_weight"] = class_weight

    _save_yaml(models_path, data)

    status = "active" if model_def.get("active", True) else "inactive"
    n_feat = len(model_def.get("features", []))
    in_ens = "yes" if model_def.get("include_in_ensemble", True) else "no"

    return (
        f"**Updated model**: `{name}`\n"
        f"- Type: {model_def.get('type', '?')}\n"
        f"- Status: {status}\n"
        f"- Features: {n_feat}\n"
        f"- Ensemble: {in_ens}\n"
        f"- Params: {model_def.get('params', {})}"
    )


def show_models(project_dir: Path) -> str:
    """List all models with type, status, and feature count."""
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if not models:
        return "No models configured."

    lines = ["## Models\n"]
    lines.append("| Name | Type | Status | Features | Ensemble |")
    lines.append("|------|------|--------|----------|----------|")

    for name, m in sorted(models.items()):
        status = "active" if m.get("active", True) else "inactive"
        n_feat = len(m.get("features", []))
        in_ens = "yes" if m.get("include_in_ensemble", True) else "no"
        lines.append(f"| {name} | {m.get('type', '?')} | {status} | {n_feat} | {in_ens} |")

    return "\n".join(lines)


def show_model(project_dir: Path, name: str) -> str:
    """Show full configuration for a single model."""
    config_dir = _get_config_dir(Path(project_dir))
    models_path = config_dir / "models.yaml"
    data = _load_yaml(models_path)

    models = data.get("models", {})
    if name not in models:
        available = sorted(models.keys())
        return f"**Error**: Model `{name}` not found. Available: {', '.join(available) or '(none)'}"

    model_config = models[name]

    lines = [f"## Model: `{name}`\n"]
    lines.append(f"- **Type**: {model_config.get('type', '?')}")
    lines.append(f"- **Active**: {model_config.get('active', True)}")
    lines.append(f"- **In Ensemble**: {model_config.get('include_in_ensemble', True)}")

    features = model_config.get("features", [])
    lines.append(f"\n### Features ({len(features)})\n")
    for f in features:
        lines.append(f"- {f}")

    params = model_config.get("params", {})
    if params:
        lines.append("\n### Parameters\n")
        lines.append(f"```yaml\n{yaml.dump(params, default_flow_style=False)}```")

    # Show any other config keys
    skip_keys = {"type", "active", "include_in_ensemble", "features", "params"}
    extras = {k: v for k, v in model_config.items() if k not in skip_keys}
    if extras:
        lines.append("\n### Other Settings\n")
        lines.append(f"```yaml\n{yaml.dump(extras, default_flow_style=False)}```")

    return "\n".join(lines)


def show_presets() -> str:
    """List available model presets."""
    from harnessml.core.runner.scaffold.presets import get_preset, list_presets

    preset_names = list_presets()
    if not preset_names:
        return "No presets available."

    lines = ["## Model Presets\n"]
    lines.append("| Name | Type | Mode |")
    lines.append("|------|------|------|")
    for name in preset_names:
        preset = get_preset(name)
        lines.append(
            f"| {name} | {preset.get('type', '?')} | {preset.get('mode', '?')} |"
        )

    return "\n".join(lines)


def configure_ensemble(
    project_dir: Path,
    *,
    method: str | None = None,
    temperature: float | None = None,
    exclude_models: list[str] | None = None,
    calibration: str | None = None,
    pre_calibration: dict | None = None,
    prior_feature: str | None = None,
    spline_prob_max: float | None = None,
    spline_n_bins: int | None = None,
    **kwargs,
) -> str:
    """Update ensemble.yaml configuration.

    Parameters
    ----------
    calibration : str | None
        Post-ensemble calibration method: 'spline', 'isotonic', 'platt', 'none'.
    pre_calibration : dict | None
        Per-model pre-calibration applied before the meta-learner.
        Maps model_name -> method, e.g. {"xgb_core": "platt"}.
        Valid methods: 'platt', 'isotonic', 'spline', 'none'.
    """
    config_dir = _get_config_dir(Path(project_dir))
    ensemble_path = config_dir / "ensemble.yaml"
    data = _load_yaml(ensemble_path)

    if "ensemble" not in data:
        data["ensemble"] = {}

    ens = data["ensemble"]

    if method is not None:
        ens["method"] = method
    if temperature is not None:
        ens["temperature"] = temperature
    if exclude_models is not None:
        ens["exclude_models"] = exclude_models
    if calibration is not None:
        ens["calibration"] = calibration
    if pre_calibration is not None:
        ens["pre_calibration"] = pre_calibration
    if prior_feature is not None:
        ens["prior_feature"] = prior_feature
    if spline_prob_max is not None:
        ens["spline_prob_max"] = spline_prob_max
    if spline_n_bins is not None:
        ens["spline_n_bins"] = spline_n_bins

    for key, val in kwargs.items():
        ens[key] = val

    _save_yaml(ensemble_path, data)

    pre_cal_models = list(ens.get("pre_calibration", {}).keys())
    return (
        f"**Updated ensemble config**\n"
        f"- Method: {ens.get('method', 'average')}\n"
        f"- Temperature: {ens.get('temperature', 1.0)}\n"
        f"- Calibration: {ens.get('calibration', 'spline')}\n"
        f"- Pre-calibration: {pre_cal_models or 'none'}"
    )
