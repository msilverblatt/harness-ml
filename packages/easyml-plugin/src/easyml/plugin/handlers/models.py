"""Handler for manage_models tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import validate_enum, validate_required


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


def _handle_update(*, name, features, params, active, include_in_ensemble, mode, prediction_type, cdf_scale, zero_fill_features, project_dir, **_kwargs):
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


ACTIONS = {
    "add": _handle_add,
    "update": _handle_update,
    "remove": _handle_remove,
    "list": _handle_list,
    "presets": _handle_presets,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_models action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    return ACTIONS[action](**kwargs)
