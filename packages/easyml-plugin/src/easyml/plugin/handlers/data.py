"""Handler for manage_data tool."""
from __future__ import annotations

from easyml.plugin.handlers._common import resolve_project_dir, parse_json_param
from easyml.plugin.handlers._validation import (
    validate_enum, validate_required, collect_hints, format_response_with_hints,
)


def _handle_add(*, data_path, join_on, prefix, auto_clean, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    err = validate_required(data_path, "data_path")
    if err:
        return err
    return cw.add_dataset(
        resolve_project_dir(project_dir),
        data_path,
        join_on=join_on,
        prefix=prefix,
        auto_clean=auto_clean,
    )


def _handle_validate(*, data_path, project_dir, **_kwargs):
    err = validate_required(data_path, "data_path")
    if err:
        return err
    from easyml.core.runner.data_ingest import validate_dataset

    return validate_dataset(resolve_project_dir(project_dir), data_path)


def _handle_fill_nulls(*, column, strategy, value, project_dir, **_kwargs):
    err = validate_required(column, "column")
    if err:
        return err
    from easyml.core.runner.data_ingest import fill_nulls

    return fill_nulls(
        resolve_project_dir(project_dir),
        column,
        strategy=strategy,
        value=value,
    )


def _handle_drop_duplicates(*, columns, project_dir, **_kwargs):
    from easyml.core.runner.data_ingest import drop_duplicates

    return drop_duplicates(resolve_project_dir(project_dir), columns=columns)


def _handle_rename(*, mapping, project_dir, **_kwargs):
    err = validate_required(mapping, "mapping")
    if err:
        return err
    from easyml.core.runner.data_ingest import rename_columns

    parsed = parse_json_param(mapping)
    return rename_columns(resolve_project_dir(project_dir), parsed)


def _handle_profile(*, category, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.profile_data(resolve_project_dir(project_dir), category=category)


def _handle_list_features(*, prefix, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.available_features(resolve_project_dir(project_dir), prefix=prefix)


def _handle_status(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.feature_store_status(resolve_project_dir(project_dir))


def _handle_list_sources(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.list_sources(resolve_project_dir(project_dir))


def _handle_add_source(*, name, data_path, format, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    err = validate_required(data_path, "data_path")
    if err:
        return err
    from easyml.core.runner import config_writer as cw

    return cw.add_source(
        resolve_project_dir(project_dir),
        name,
        data_path,
        format=format,
    )


def _handle_add_view(*, name, source, steps, description, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    err = validate_required(source, "source")
    if err:
        return err
    from easyml.core.runner import config_writer as cw

    parsed_steps = parse_json_param(steps) or []
    return cw.add_view(
        resolve_project_dir(project_dir),
        name,
        source,
        parsed_steps,
        description=description,
    )


def _handle_update_view(*, name, source, steps, description, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from easyml.core.runner import config_writer as cw

    parsed_steps = None
    if steps is not None:
        parsed_steps = parse_json_param(steps)
    return cw.update_view(
        resolve_project_dir(project_dir),
        name,
        source=source,
        steps=parsed_steps,
        description=description if description else None,
    )


def _handle_remove_view(*, name, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from easyml.core.runner import config_writer as cw

    return cw.remove_view(resolve_project_dir(project_dir), name)


def _handle_list_views(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.list_views(resolve_project_dir(project_dir))


def _handle_preview_view(*, name, n_rows, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from easyml.core.runner import config_writer as cw

    return cw.preview_view(resolve_project_dir(project_dir), name, n_rows=n_rows)


def _handle_set_features_view(*, name, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from easyml.core.runner import config_writer as cw

    return cw.set_features_view(resolve_project_dir(project_dir), name)


def _handle_view_dag(*, project_dir, **_kwargs):
    from easyml.core.runner import config_writer as cw

    return cw.view_dag(resolve_project_dir(project_dir))


def _handle_add_sources_batch(*, sources, project_dir, **_kwargs):
    """Register multiple data sources in one call."""
    if not sources:
        return "**Error**: `sources` (JSON array of {name, data_path, format?}) is required for add_sources_batch."
    parsed = parse_json_param(sources) if isinstance(sources, str) else sources
    results, errors = [], []
    for i, src in enumerate(parsed):
        try:
            src_name = src.get("name", f"source_{i}")
            src_path = src.get("data_path")
            src_format = src.get("format", "auto")
            if not src_path:
                errors.append(f"{src_name}: missing `data_path`")
                continue
            _handle_add_source(
                name=src_name,
                data_path=src_path,
                format=src_format,
                project_dir=project_dir,
            )
            results.append(src_name)
        except Exception as e:
            errors.append(f"{src.get('name', f'source_{i}')}: {e}")
    summary = f"Added {len(results)} source(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


def _handle_fill_nulls_batch(*, columns, project_dir, **_kwargs):
    """Fill nulls in multiple columns in one call."""
    if not columns:
        return "**Error**: `columns` (JSON array of {column, strategy?, value?}) is required for fill_nulls_batch."
    parsed = parse_json_param(columns) if isinstance(columns, str) else columns
    results, errors = [], []
    for i, item in enumerate(parsed):
        try:
            col_name = item.get("column", f"column_{i}")
            strat = item.get("strategy", "median")
            val = item.get("value")
            _handle_fill_nulls(
                column=col_name,
                strategy=strat,
                value=val,
                project_dir=project_dir,
            )
            results.append(col_name)
        except Exception as e:
            errors.append(f"{item.get('column', f'column_{i}')}: {e}")
    summary = f"Filled nulls in {len(results)} column(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


def _handle_add_views_batch(*, views, project_dir, **_kwargs):
    """Declare multiple views in one call."""
    if not views:
        return "**Error**: `views` (JSON array of {name, source, steps?, description?}) is required for add_views_batch."
    parsed = parse_json_param(views) if isinstance(views, str) else views
    results, errors = [], []
    for i, view in enumerate(parsed):
        try:
            view_name = view.get("name", f"view_{i}")
            view_source = view.get("source")
            view_steps = view.get("steps")
            view_desc = view.get("description", "")
            if not view_source:
                errors.append(f"{view_name}: missing `source`")
                continue
            _handle_add_view(
                name=view_name,
                source=view_source,
                steps=view_steps,
                description=view_desc,
                project_dir=project_dir,
            )
            results.append(view_name)
        except Exception as e:
            errors.append(f"{view.get('name', f'view_{i}')}: {e}")
    summary = f"Added {len(results)} view(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


ACTIONS = {
    "add": _handle_add,
    "validate": _handle_validate,
    "fill_nulls": _handle_fill_nulls,
    "drop_duplicates": _handle_drop_duplicates,
    "rename": _handle_rename,
    "profile": _handle_profile,
    "list_features": _handle_list_features,
    "status": _handle_status,
    "list_sources": _handle_list_sources,
    "add_source": _handle_add_source,
    "add_view": _handle_add_view,
    "update_view": _handle_update_view,
    "remove_view": _handle_remove_view,
    "list_views": _handle_list_views,
    "preview_view": _handle_preview_view,
    "set_features_view": _handle_set_features_view,
    "view_dag": _handle_view_dag,
    "add_sources_batch": _handle_add_sources_batch,
    "fill_nulls_batch": _handle_fill_nulls_batch,
    "add_views_batch": _handle_add_views_batch,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_data action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="data", **kwargs)
    return format_response_with_hints(result, hints)
