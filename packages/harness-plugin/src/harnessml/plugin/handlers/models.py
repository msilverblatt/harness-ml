"""Handler for manage_models tool."""
from __future__ import annotations

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    validate_required,
)
from protomcp import action, tool_group


def _handle_add(*, name, model_type, preset, features, params, active, include_in_ensemble, mode, prediction_type, cdf_scale, zero_fill_features, class_weight=None, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

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
    if class_weight is not None:
        kw["class_weight"] = class_weight
    return cw.add_model(resolve_project_dir(project_dir), name, **kw)


def _handle_update(*, name, features, append_features=None, remove_features=None, params, active, include_in_ensemble, mode, prediction_type, cdf_scale, zero_fill_features, class_weight=None, replace_params=False, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    parsed_params = parse_json_param(params)
    kw = dict(
        features=features,
        append_features=append_features,
        remove_features=remove_features,
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
    if class_weight is not None:
        kw["class_weight"] = class_weight
    return cw.update_model(resolve_project_dir(project_dir), name, **kw)


def _handle_remove(*, name, purge, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    return cw.remove_model(resolve_project_dir(project_dir), name, purge=purge)


def _handle_list(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.show_models(resolve_project_dir(project_dir))


def _handle_presets(**_kwargs):
    from harnessml.core.runner import config_writer as cw

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


def _handle_clone(*, name, new_name=None, features=None, params=None, active=None,
                   include_in_ensemble=None, mode=None, prediction_type=None,
                   cdf_scale=None, zero_fill_features=None, class_weight=None,
                   project_dir, **_kwargs):
    """Clone an existing model with a new name and optional overrides."""
    import copy

    import yaml

    err = validate_required(name, "name (source model)")
    if err:
        return err
    if not new_name:
        return "**Error**: `new_name` is required for clone."

    proj = resolve_project_dir(project_dir)
    config_path = proj / "config" / "models.yaml"
    if not config_path.exists():
        return "**Error**: No models.yaml found. Run `configure(action='init')` first."

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    models = cfg.get("models", {})
    if name not in models:
        return f"**Error**: Source model `{name}` not found. Available: {', '.join(sorted(models.keys()))}"

    if new_name in models:
        return f"**Error**: Model `{new_name}` already exists. Use a different name or remove it first."

    # Deep copy the source model config
    new_config = copy.deepcopy(models[name])

    # Apply overrides from explicit parameters
    parsed_params = parse_json_param(params) if params is not None else None
    override_count = 0
    if features is not None:
        new_config["features"] = features
        override_count += 1
    if parsed_params is not None:
        if isinstance(parsed_params, dict) and isinstance(new_config.get("params"), dict):
            new_config["params"].update(parsed_params)
        else:
            new_config["params"] = parsed_params
        override_count += 1
    if active is not None:
        new_config["active"] = active
        override_count += 1
    if include_in_ensemble is not None:
        new_config["include_in_ensemble"] = include_in_ensemble
        override_count += 1
    if mode is not None:
        new_config["mode"] = mode
        override_count += 1
    if prediction_type is not None:
        new_config["prediction_type"] = prediction_type
        override_count += 1
    if cdf_scale is not None:
        new_config["cdf_scale"] = cdf_scale
        override_count += 1
    if zero_fill_features is not None:
        new_config["zero_fill_features"] = zero_fill_features
        override_count += 1
    if class_weight is not None:
        new_config["class_weight"] = class_weight
        override_count += 1

    # Write the cloned model
    models[new_name] = new_config
    cfg["models"] = models
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    return f"Cloned model `{name}` as `{new_name}` with {override_count} override(s)."


def _with_defaults(item: dict, for_update: bool = False) -> dict:
    """Fill in default None values for fields not in the item dict."""
    defaults = {
        "name": None,
        "model_type": None,
        "preset": None,
        "features": None,
        "append_features": None,
        "remove_features": None,
        "params": None,
        "active": None,
        "include_in_ensemble": None,
        "mode": None,
        "prediction_type": None,
        "cdf_scale": None,
        "zero_fill_features": None,
        "class_weight": None,
        "replace_params": None,
    }
    result = {k: item.get(k, v) for k, v in defaults.items()}
    return result


def _handle_show(*, name, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    err = validate_required(name, "name")
    if err:
        return err
    return cw.show_model(resolve_project_dir(project_dir), name)


@tool_group("models", description="Manage ML models in the pipeline.")
class ModelsGroup:

    @action("add", description="Add a new model.", requires=["name"])
    def add(self, *, name=None, model_type=None, preset=None, features=None, params=None,
            active=None, include_in_ensemble=None, mode=None, prediction_type=None,
            cdf_scale=None, zero_fill_features=None, class_weight=None, project_dir=None, **kw):
        return _handle_add(name=name, model_type=model_type, preset=preset, features=features,
                           params=params, active=active, include_in_ensemble=include_in_ensemble,
                           mode=mode, prediction_type=prediction_type, cdf_scale=cdf_scale,
                           zero_fill_features=zero_fill_features, class_weight=class_weight,
                           project_dir=project_dir, **kw)

    @action("update", description="Update an existing model.", requires=["name"])
    def update(self, *, name=None, features=None, append_features=None, remove_features=None,
               params=None, active=None, include_in_ensemble=None, mode=None,
               prediction_type=None, cdf_scale=None, zero_fill_features=None,
               class_weight=None, replace_params=False, project_dir=None, **kw):
        return _handle_update(name=name, features=features, append_features=append_features,
                              remove_features=remove_features, params=params, active=active,
                              include_in_ensemble=include_in_ensemble, mode=mode,
                              prediction_type=prediction_type, cdf_scale=cdf_scale,
                              zero_fill_features=zero_fill_features, class_weight=class_weight,
                              replace_params=replace_params, project_dir=project_dir, **kw)

    @action("remove", description="Remove a model.", requires=["name"])
    def remove(self, *, name=None, purge=None, project_dir=None, **kw):
        return _handle_remove(name=name, purge=purge, project_dir=project_dir, **kw)

    @action("list", description="List all models.")
    def list_models(self, *, project_dir=None, **kw):
        return _handle_list(project_dir=project_dir, **kw)

    @action("show", description="Show details of a model.", requires=["name"])
    def show(self, *, name=None, project_dir=None, **kw):
        return _handle_show(name=name, project_dir=project_dir, **kw)

    @action("presets", description="Show available model presets.")
    def presets(self, **kw):
        return _handle_presets(**kw)

    @action("add_batch", description="Add multiple models in one call.")
    def add_batch(self, *, items=None, project_dir=None, **kw):
        return _handle_add_batch(items=items, project_dir=project_dir, **kw)

    @action("update_batch", description="Update multiple models in one call.")
    def update_batch(self, *, items=None, project_dir=None, **kw):
        return _handle_update_batch(items=items, project_dir=project_dir, **kw)

    @action("remove_batch", description="Remove multiple models in one call.")
    def remove_batch(self, *, items=None, project_dir=None, **kw):
        return _handle_remove_batch(items=items, project_dir=project_dir, **kw)

    @action("clone", description="Clone an existing model.", requires=["name"])
    def clone(self, *, name=None, new_name=None, features=None, params=None, active=None,
              include_in_ensemble=None, mode=None, prediction_type=None, cdf_scale=None,
              zero_fill_features=None, class_weight=None, project_dir=None, **kw):
        return _handle_clone(name=name, new_name=new_name, features=features, params=params,
                             active=active, include_in_ensemble=include_in_ensemble, mode=mode,
                             prediction_type=prediction_type, cdf_scale=cdf_scale,
                             zero_fill_features=zero_fill_features, class_weight=class_weight,
                             project_dir=project_dir, **kw)
