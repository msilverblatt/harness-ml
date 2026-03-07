"""Handler for manage_models tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import (
    validate_enum, validate_required, collect_hints, format_response_with_hints,
)


def _handle_add(*, name, model_type, preset, features, params, active, include_in_ensemble, mode, prediction_type, cdf_scale, zero_fill_features, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    parsed_params = parse_json_param(params)
    kw = dict(
        model_type=model_type,
        preset=preset,
        features=features,
        params=parsed_params,
        active=True if active is None else active,
        include_in_ensemble=True if include_in_ensemble is None else include_in_ensemble,
        mode=mode,
        prediction_type=prediction_type,
    )
    if cdf_scale is not None:
        kw["cdf_scale"] = cdf_scale
    if zero_fill_features is not None:
        kw["zero_fill_features"] = zero_fill_features
    return cw.add_model(resolve_project_dir(project_dir), name, **kw)


def _handle_update(*, name, features, params, active, include_in_ensemble, mode, prediction_type, cdf_scale, zero_fill_features, replace_params=False, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    parsed_params = parse_json_param(params)
    kw = dict(
        features=features,
        params=parsed_params,
        active=active,
        include_in_ensemble=include_in_ensemble,
        mode=mode,
        prediction_type=prediction_type,
        replace_params=replace_params,
    )
    if cdf_scale is not None:
        kw["cdf_scale"] = cdf_scale
    if zero_fill_features is not None:
        kw["zero_fill_features"] = zero_fill_features
    return cw.update_model(resolve_project_dir(project_dir), name, **kw)


def _handle_remove(*, name, purge, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    return cw.remove_model(resolve_project_dir(project_dir), name, purge=purge)


def _handle_list(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.show_models(resolve_project_dir(project_dir))


def _handle_presets(**_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.show_presets()


def _handle_add_batch(*, items, project_dir, **_kwargs):
    """Add multiple models in one call."""
    if not items:
        return "**Error**: `items` (JSON array of model configs) is required for add_batch."
    parsed = parse_json_param(items) if isinstance(items, str) else items
    results, errors = [], []
    for i, item in enumerate(parsed):
        try:
            item_name = item.get("name", f"item_{i}")
            result = _handle_add(project_dir=project_dir, **_with_defaults(item))
            if isinstance(result, str) and result.startswith("**Error**"):
                errors.append(f"{item_name}: {result}")
            else:
                results.append(item_name)
        except Exception as e:
            errors.append(f"{item.get('name', f'item_{i}')}: {e}")
    summary = f"Added {len(results)} model(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


def _handle_update_batch(*, items, project_dir, **_kwargs):
    """Update multiple models in one call."""
    if not items:
        return "**Error**: `items` (JSON array of model configs) is required for update_batch."
    parsed = parse_json_param(items) if isinstance(items, str) else items
    results, errors = [], []
    for i, item in enumerate(parsed):
        try:
            item_name = item.get("name", f"item_{i}")
            result = _handle_update(project_dir=project_dir, **_with_defaults(item, for_update=True))
            if isinstance(result, str) and result.startswith("**Error**"):
                errors.append(f"{item_name}: {result}")
            else:
                results.append(item_name)
        except Exception as e:
            errors.append(f"{item.get('name', f'item_{i}')}: {e}")
    summary = f"Updated {len(results)} model(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


def _handle_remove_batch(*, items, project_dir, **_kwargs):
    """Remove multiple models in one call."""
    if not items:
        return "**Error**: `items` (JSON array of {name, purge?}) is required for remove_batch."
    parsed = parse_json_param(items) if isinstance(items, str) else items
    results, errors = [], []
    for i, item in enumerate(parsed):
        try:
            item_name = item.get("name", f"item_{i}")
            purge = item.get("purge", False)
            _handle_remove(name=item_name, purge=purge, project_dir=project_dir)
            results.append(item_name)
        except Exception as e:
            errors.append(f"{item.get('name', f'item_{i}')}: {e}")
    summary = f"Removed {len(results)} model(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


def _handle_clone(*, name, items, project_dir, **_kwargs):
    """Clone an existing model with a new name and optional overrides."""
    from easyml.core.runner import config_writer as cw
    import yaml

    err = validate_required(name, "name (source model)")
    if err:
        return err
    if not items:
        return "**Error**: `items` (JSON with {new_name, ...overrides}) is required for clone."
    parsed = parse_json_param(items) if isinstance(items, str) else items
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    new_name = parsed.get("new_name")
    if not new_name:
        return "**Error**: `new_name` is required in the items object for clone."

    proj = resolve_project_dir(project_dir)
    config_path = proj / "config" / "pipeline.yaml"
    if not config_path.exists():
        return "**Error**: No pipeline.yaml found. Run `configure(action='init')` first."

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", {})
    if name not in models:
        return f"**Error**: Source model `{name}` not found. Available: {', '.join(sorted(models.keys()))}"

    # Deep copy the source model config
    import copy
    new_config = copy.deepcopy(models[name])

    # Apply overrides from the items object
    overrides = {k: v for k, v in parsed.items() if k != "new_name"}
    if "params" in overrides:
        overrides["params"] = parse_json_param(overrides["params"])
    for k, v in overrides.items():
        if k == "params" and isinstance(v, dict) and isinstance(new_config.get("params"), dict):
            new_config["params"].update(v)
        else:
            new_config[k] = v

    # Write the cloned model
    models[new_name] = new_config
    cfg["models"] = models
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return f"Cloned model `{name}` as `{new_name}` with {len(overrides)} override(s)."


def _with_defaults(item: dict, for_update: bool = False) -> dict:
    """Fill in default None values for fields not in the item dict."""
    defaults = {
        "name": None,
        "model_type": None,
        "preset": None,
        "features": None,
        "params": None,
        "active": None,
        "include_in_ensemble": None,
        "mode": None,
        "prediction_type": None,
        "cdf_scale": None,
        "zero_fill_features": None,
        "replace_params": None,
    }
    result = {k: item.get(k, v) for k, v in defaults.items()}
    return result


def _handle_show(*, name, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw
    err = validate_required(name, "name")
    if err:
        return err
    return cw.show_model(resolve_project_dir(project_dir), name)


ACTIONS = {
    "add": _handle_add,
    "update": _handle_update,
    "remove": _handle_remove,
    "list": _handle_list,
    "show": _handle_show,
    "presets": _handle_presets,
    "add_batch": _handle_add_batch,
    "update_batch": _handle_update_batch,
    "remove_batch": _handle_remove_batch,
    "clone": _handle_clone,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_models action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="models", **kwargs)
    return format_response_with_hints(result, hints)
