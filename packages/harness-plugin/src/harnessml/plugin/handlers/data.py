"""Handler for manage_data tool."""
from __future__ import annotations

from harnessml.core.logging import get_logger
from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    collect_hints,
    format_response_with_hints,
    validate_enum,
    validate_required,
)

logger = get_logger(__name__)


def _handle_add(*, data_path, join_on, prefix, auto_clean, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

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
    from harnessml.core.runner.data.ingest import validate_dataset

    return validate_dataset(resolve_project_dir(project_dir), data_path)


def _handle_fill_nulls(*, column, strategy, value, project_dir, **_kwargs):
    err = validate_required(column, "column")
    if err:
        return err
    from harnessml.core.runner.data.ingest import fill_nulls

    return fill_nulls(
        resolve_project_dir(project_dir),
        column,
        strategy=strategy,
        value=value,
    )


def _handle_drop_duplicates(*, columns, project_dir, **_kwargs):
    from harnessml.core.runner.data.ingest import drop_duplicates

    return drop_duplicates(resolve_project_dir(project_dir), columns=columns)


def _handle_rename(*, mapping, project_dir, **_kwargs):
    err = validate_required(mapping, "mapping")
    if err:
        return err
    from harnessml.core.runner.data.ingest import rename_columns

    parsed = parse_json_param(mapping)
    return rename_columns(resolve_project_dir(project_dir), parsed)


def _handle_drop_rows(*, column, condition, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.drop_rows(
        resolve_project_dir(project_dir),
        column=column,
        condition=condition or "null",
    )


def _handle_derive_column(*, name, expression, group_by, dtype, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    err = validate_required(expression, "expression")
    if err:
        return err
    from harnessml.core.runner import config_writer as cw

    return cw.derive_column(
        resolve_project_dir(project_dir),
        name,
        expression,
        group_by=group_by,
        dtype=dtype,
    )


def _handle_profile(*, category, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.profile_data(resolve_project_dir(project_dir), category=category)


def _handle_list_features(*, prefix, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.available_features(resolve_project_dir(project_dir), prefix=prefix)


def _handle_status(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.feature_store_status(resolve_project_dir(project_dir))


def _handle_list_sources(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.list_sources(resolve_project_dir(project_dir))


def _handle_add_source(*, name, data_path, format, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    err = validate_required(data_path, "data_path")
    if err:
        return err
    from harnessml.core.runner import config_writer as cw

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
    from harnessml.core.runner import config_writer as cw

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
    from harnessml.core.runner import config_writer as cw

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
    from harnessml.core.runner import config_writer as cw

    return cw.remove_view(resolve_project_dir(project_dir), name)


def _handle_list_views(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.list_views(resolve_project_dir(project_dir))


def _handle_preview_view(*, name, n_rows, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from harnessml.core.runner import config_writer as cw

    return cw.preview_view(resolve_project_dir(project_dir), name, n_rows=n_rows)


def _handle_set_features_view(*, name, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from harnessml.core.runner import config_writer as cw

    return cw.set_features_view(resolve_project_dir(project_dir), name)


def _handle_view_dag(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

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
        except (ValueError, KeyError, FileNotFoundError, OSError) as e:
            logger.exception("add_sources_batch failed for source", action="add_sources_batch")
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
        except (ValueError, KeyError, FileNotFoundError, OSError) as e:
            logger.exception("fill_nulls_batch failed for column", action="fill_nulls_batch")
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
        except (ValueError, KeyError, FileNotFoundError, OSError) as e:
            logger.exception("add_views_batch failed for view", action="add_views_batch")
            errors.append(f"{view.get('name', f'view_{i}')}: {e}")
    summary = f"Added {len(results)} view(s)."
    if errors:
        summary += f" {len(errors)} error(s):\n" + "\n".join(f"- {e}" for e in errors)
    return summary


def _handle_inspect(*, column, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.inspect_data(resolve_project_dir(project_dir), column=column)


def _handle_check_freshness(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.check_freshness(resolve_project_dir(project_dir))


def _handle_refresh(*, name, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from harnessml.core.runner import config_writer as cw

    return cw.refresh_source(resolve_project_dir(project_dir), name)


def _handle_refresh_all(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.refresh_all_sources(resolve_project_dir(project_dir))


def _handle_validate_source(*, name, project_dir, **_kwargs):
    err = validate_required(name, "name")
    if err:
        return err
    from harnessml.core.runner import config_writer as cw

    return cw.validate_source_data(resolve_project_dir(project_dir), name)


def _handle_sample(*, fraction, stratify_column, seed, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    if fraction is None:
        return "**Error**: `fraction` (0.0-1.0) is required for sample."
    return cw.sample_data(
        resolve_project_dir(project_dir),
        fraction=fraction,
        stratify_column=stratify_column,
        seed=seed or 42,
    )


def _handle_restore(*, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw
    return cw.restore_full_data(resolve_project_dir(project_dir))


def _handle_fetch_url(*, data_path, name, project_dir, **_kwargs):
    """Download a file from a URL to the raw data directory."""
    from harnessml.core.runner import config_writer as cw
    err = validate_required(data_path, "data_path (the URL)")
    if err:
        return err
    return cw.fetch_url(
        resolve_project_dir(project_dir),
        data_path,
        filename=name,
    )


def _handle_upload_drive(*, files, folder_id, folder_name, name, project_dir, **_kwargs):
    if not files:
        return "**Error**: `files` (list of file paths) is required for upload_drive."
    parsed = parse_json_param(files) if isinstance(files, str) else files
    from pathlib import Path

    from harnessml.core.runner.drives.drive import create_folder, upload_file

    pdir = resolve_project_dir(project_dir)
    credentials_dir = pdir / ".harnessml"

    target_folder_id = folder_id
    if folder_name and not folder_id:
        folder_result = create_folder(folder_name, credentials_dir=credentials_dir)
        target_folder_id = folder_result["id"]

    results = []
    for f in parsed:
        fp = Path(f)
        if not fp.is_absolute():
            fp = pdir / fp
        r = upload_file(fp, folder_id=target_folder_id, credentials_dir=credentials_dir)
        results.append(r)

    lines = [f"Uploaded {len(results)} file(s) to Google Drive:"]
    for r in results:
        line = f"- **{r['name']}** (ID: `{r['id']}`)"
        if "colab_url" in r:
            line += f"\n  Colab: {r['colab_url']}"
        lines.append(line)
    return "\n".join(lines)


def _handle_snapshot(*, name, project_dir, **_kwargs):
    """Snapshot all config YAMLs and features parquet for later restore."""
    import shutil
    from datetime import datetime

    pdir = resolve_project_dir(project_dir)
    snapshot_name = name or datetime.now().strftime("snap_%Y%m%d_%H%M%S")
    snap_dir = pdir / "snapshots" / snapshot_name
    snap_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    config_dir = pdir / "config"
    if config_dir.is_dir():
        snap_config_dir = snap_dir / "config"
        snap_config_dir.mkdir(parents=True, exist_ok=True)
        for yaml_file in config_dir.glob("*.yaml"):
            shutil.copy2(yaml_file, snap_config_dir / yaml_file.name)
            copied.append(f"config/{yaml_file.name}")

    features_parquet = pdir / "features.parquet"
    if features_parquet.is_file():
        shutil.copy2(features_parquet, snap_dir / "features.parquet")
        copied.append("features.parquet")

    if not copied:
        return "**Warning**: No config YAMLs or features.parquet found to snapshot."

    lines = [f"Snapshot **{snapshot_name}** created with {len(copied)} file(s):"]
    for f in copied:
        lines.append(f"- {f}")
    return "\n".join(lines)


def _handle_restore_snapshot(*, name, project_dir, **_kwargs):
    """Restore config YAMLs and features parquet from a named snapshot."""
    import shutil

    err = validate_required(name, "name")
    if err:
        return err

    pdir = resolve_project_dir(project_dir)
    snap_dir = pdir / "snapshots" / name
    if not snap_dir.is_dir():
        available = []
        snapshots_root = pdir / "snapshots"
        if snapshots_root.is_dir():
            available = [d.name for d in snapshots_root.iterdir() if d.is_dir()]
        msg = f"**Error**: Snapshot `{name}` not found."
        if available:
            msg += f" Available snapshots: {', '.join(sorted(available))}"
        return msg

    restored = []

    snap_config_dir = snap_dir / "config"
    if snap_config_dir.is_dir():
        config_dir = pdir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        for yaml_file in snap_config_dir.glob("*.yaml"):
            shutil.copy2(yaml_file, config_dir / yaml_file.name)
            restored.append(f"config/{yaml_file.name}")

    snap_features = snap_dir / "features.parquet"
    if snap_features.is_file():
        shutil.copy2(snap_features, pdir / "features.parquet")
        restored.append("features.parquet")

    if not restored:
        return f"**Warning**: Snapshot `{name}` exists but contains no restorable files."

    lines = [f"Restored **{name}** — {len(restored)} file(s):"]
    for f in restored:
        lines.append(f"- {f}")
    return "\n".join(lines)


def _handle_upload_kaggle(*, files, dataset_slug, title, name, project_dir, **_kwargs):
    if not dataset_slug:
        return "**Error**: `dataset_slug` (e.g. 'username/dataset-name') is required."
    if not files:
        return "**Error**: `files` (list of file paths) is required for upload_kaggle."
    parsed = parse_json_param(files) if isinstance(files, str) else files
    from pathlib import Path

    from harnessml.core.runner.drives.kaggle import upload_dataset

    pdir = resolve_project_dir(project_dir)
    resolved_files = []
    for f in parsed:
        fp = Path(f)
        if not fp.is_absolute():
            fp = pdir / fp
        resolved_files.append(fp)

    result = upload_dataset(
        files=resolved_files,
        dataset_slug=dataset_slug,
        title=title or dataset_slug.split("/")[-1],
    )
    return f"Uploaded {result['files']} file(s) to Kaggle dataset `{result['slug']}`."


ACTIONS = {
    "add": _handle_add,
    "validate": _handle_validate,
    "fill_nulls": _handle_fill_nulls,
    "drop_duplicates": _handle_drop_duplicates,
    "drop_rows": _handle_drop_rows,
    "rename": _handle_rename,
    "derive_column": _handle_derive_column,
    "inspect": _handle_inspect,
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
    "sample": _handle_sample,
    "restore": _handle_restore,
    "check_freshness": _handle_check_freshness,
    "refresh": _handle_refresh,
    "refresh_all": _handle_refresh_all,
    "validate_source": _handle_validate_source,
    "fetch_url": _handle_fetch_url,
    "upload_drive": _handle_upload_drive,
    "upload_kaggle": _handle_upload_kaggle,
    "snapshot": _handle_snapshot,
    "restore_snapshot": _handle_restore_snapshot,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a manage_data action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="data", **kwargs)
    return format_response_with_hints(result, hints)
