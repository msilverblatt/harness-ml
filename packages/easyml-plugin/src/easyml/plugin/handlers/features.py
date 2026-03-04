"""Handler for manage_features tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import (
    validate_enum, validate_required, collect_hints, format_response_with_hints,
)


def _handle_add(*, name, formula, type, source, column, condition, pairwise_mode, category, description, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(name, "name")
    if err:
        return err
    if not formula and not type and not source and not condition:
        return (
            "**Error**: Provide at least one of: type, formula, source, or condition."
        )
    return cw.add_feature(
        resolve_project_dir(project_dir),
        name,
        formula,
        type=type,
        source=source,
        column=column,
        condition=condition,
        pairwise_mode=pairwise_mode,
        category=category,
        description=description,
    )


def _handle_add_batch(*, features, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(features, "features")
    if err:
        return err
    parsed = parse_json_param(features)
    return cw.add_features_batch(resolve_project_dir(project_dir), parsed)


def _handle_test_transformations(*, features, test_interactions, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(features, "features")
    if err:
        return err
    parsed = parse_json_param(features)
    return cw.test_feature_transformations(
        resolve_project_dir(project_dir),
        parsed,
        test_interactions=test_interactions,
    )


def _handle_discover(*, top_n, method, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.discover_features(
        resolve_project_dir(project_dir),
        top_n=top_n,
        method=method,
    )


def _handle_diversity(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.feature_diversity_report(resolve_project_dir(project_dir))


def _handle_auto_search(*, features, search_types, top_n, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    parsed_features = parse_json_param(features) if features else None
    parsed_search_types = parse_json_param(search_types) if search_types else None
    return cw.auto_search_features(
        resolve_project_dir(project_dir),
        features=parsed_features,
        search_types=parsed_search_types,
        top_n=top_n,
    )


ACTIONS = {
    "add": _handle_add,
    "add_batch": _handle_add_batch,
    "test_transformations": _handle_test_transformations,
    "discover": _handle_discover,
    "diversity": _handle_diversity,
    "auto_search": _handle_auto_search,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_features action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="features", **kwargs)
    return format_response_with_hints(result, hints)
