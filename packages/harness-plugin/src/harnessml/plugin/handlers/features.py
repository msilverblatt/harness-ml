"""Handler for manage_features tool."""
from __future__ import annotations

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    collect_hints,
    format_response_with_hints,
    validate_enum,
    validate_required,
)


def _handle_add(*, name, formula, type, source, column, condition, pairwise_mode, category, description, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

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
    from harnessml.core.runner import config_writer as cw

    err = validate_required(features, "features")
    if err:
        return err
    parsed = parse_json_param(features)
    return cw.add_features_batch(resolve_project_dir(project_dir), parsed)


def _handle_test_transformations(*, features, test_interactions, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    err = validate_required(features, "features")
    if err:
        return err
    parsed = parse_json_param(features)
    return cw.test_feature_transformations(
        resolve_project_dir(project_dir),
        parsed,
        test_interactions=test_interactions,
    )


async def _handle_discover(*, top_n, method, ctx, project_dir, **_kwargs):
    import asyncio

    from harnessml.core.runner import config_writer as cw

    loop = asyncio.get_running_loop()

    def _progress_callback(current, total, message):
        import logging
        logging.getLogger(__name__).info("Feature discovery progress: %s", message)
        if ctx is not None:
            asyncio.run_coroutine_threadsafe(
                ctx.report_progress(progress=current, total=total, message=message),
                loop,
            )

    if ctx is not None:
        await ctx.report_progress(progress=0, total=1, message="Starting feature discovery...")

    result = await loop.run_in_executor(
        None,
        lambda: cw.discover_features(
            resolve_project_dir(project_dir),
            top_n=top_n,
            method=method,
            on_progress=_progress_callback,
        ),
    )

    if ctx is not None:
        await ctx.report_progress(progress=1, total=1, message="Feature discovery complete.")

    return result


def _handle_diversity(*, project_dir, **_kwargs):
    import yaml
    from harnessml.core.runner.feature_diversity import format_diversity_report

    proj = resolve_project_dir(project_dir)
    config_path = proj / "config" / "pipeline.yaml"
    if not config_path.exists():
        return "**Error**: No pipeline.yaml found. Run `configure(action='init')` first."

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    models = cfg.get("models", {})
    if not models:
        return "**Error**: No models configured. Add models first with `models(action='add', ...)`."

    return format_diversity_report(models)


def _handle_auto_search(*, features, search_types, top_n, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

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


async def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_features action."""
    import asyncio

    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    hints = collect_hints(action, tool="features", **kwargs)
    return format_response_with_hints(result, hints)
