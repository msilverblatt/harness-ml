"""Handler for manage_data tool."""
from __future__ import annotations

from harnessml.core.logging import get_logger
from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    validate_required,
)
from protomcp import action, tool_group

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


def _handle_detect_outliers(*, column=None, method=None, threshold=None, project_dir, **_kwargs):
    from harnessml.core.runner import config_writer as cw

    return cw.detect_outliers(
        resolve_project_dir(project_dir),
        column=column,
        method=method or "iqr",
        threshold=float(threshold) if threshold is not None else None,
    )


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


@tool_group("data", description="Manage data sources, views, and transformations.")
class DataGroup:

    @action("add", description="Add a dataset.", requires=["data_path"])
    def add(self, *, data_path=None, join_on=None, prefix=None, auto_clean=None, project_dir=None, **kw):
        return _handle_add(data_path=data_path, join_on=join_on, prefix=prefix, auto_clean=auto_clean, project_dir=project_dir, **kw)

    @action("validate", description="Validate a dataset.", requires=["data_path"])
    def validate(self, *, data_path=None, project_dir=None, **kw):
        return _handle_validate(data_path=data_path, project_dir=project_dir, **kw)

    @action("fill_nulls", description="Fill null values in a column.", requires=["column"])
    def fill_nulls(self, *, column=None, strategy=None, value=None, project_dir=None, **kw):
        return _handle_fill_nulls(column=column, strategy=strategy, value=value, project_dir=project_dir, **kw)

    @action("drop_duplicates", description="Drop duplicate rows.")
    def drop_duplicates(self, *, columns=None, project_dir=None, **kw):
        return _handle_drop_duplicates(columns=columns, project_dir=project_dir, **kw)

    @action("detect_outliers", description="Detect outliers in data.")
    def detect_outliers(self, *, column=None, method=None, threshold=None, project_dir=None, **kw):
        return _handle_detect_outliers(column=column, method=method, threshold=threshold, project_dir=project_dir, **kw)

    @action("drop_rows", description="Drop rows matching a condition.")
    def drop_rows(self, *, column=None, condition=None, project_dir=None, **kw):
        return _handle_drop_rows(column=column, condition=condition, project_dir=project_dir, **kw)

    @action("rename", description="Rename columns.", requires=["mapping"])
    def rename(self, *, mapping=None, project_dir=None, **kw):
        return _handle_rename(mapping=mapping, project_dir=project_dir, **kw)

    @action("derive_column", description="Create a derived column.", requires=["name", "expression"])
    def derive_column(self, *, name=None, expression=None, group_by=None, dtype=None, project_dir=None, **kw):
        return _handle_derive_column(name=name, expression=expression, group_by=group_by, dtype=dtype, project_dir=project_dir, **kw)

    @action("inspect", description="Inspect data or a specific column.")
    def inspect(self, *, column=None, project_dir=None, **kw):
        return _handle_inspect(column=column, project_dir=project_dir, **kw)

    @action("profile", description="Profile the dataset.")
    def profile(self, *, category=None, project_dir=None, **kw):
        return _handle_profile(category=category, project_dir=project_dir, **kw)

    @action("list_features", description="List available features.")
    def list_features(self, *, prefix=None, project_dir=None, **kw):
        return _handle_list_features(prefix=prefix, project_dir=project_dir, **kw)

    @action("status", description="Show feature store status.")
    def status(self, *, project_dir=None, **kw):
        return _handle_status(project_dir=project_dir, **kw)

    @action("list_sources", description="List registered data sources.")
    def list_sources(self, *, project_dir=None, **kw):
        return _handle_list_sources(project_dir=project_dir, **kw)

    @action("add_source", description="Register a data source.", requires=["name", "data_path"])
    def add_source(self, *, name=None, data_path=None, format=None, project_dir=None, **kw):
        return _handle_add_source(name=name, data_path=data_path, format=format, project_dir=project_dir, **kw)

    @action("add_view", description="Declare a data view.", requires=["name", "source"])
    def add_view(self, *, name=None, source=None, steps=None, description=None, project_dir=None, **kw):
        return _handle_add_view(name=name, source=source, steps=steps, description=description, project_dir=project_dir, **kw)

    @action("update_view", description="Update a data view.", requires=["name"])
    def update_view(self, *, name=None, source=None, steps=None, description=None, project_dir=None, **kw):
        return _handle_update_view(name=name, source=source, steps=steps, description=description, project_dir=project_dir, **kw)

    @action("remove_view", description="Remove a data view.", requires=["name"])
    def remove_view(self, *, name=None, project_dir=None, **kw):
        return _handle_remove_view(name=name, project_dir=project_dir, **kw)

    @action("list_views", description="List all data views.")
    def list_views(self, *, project_dir=None, **kw):
        return _handle_list_views(project_dir=project_dir, **kw)

    @action("preview_view", description="Preview a data view.", requires=["name"])
    def preview_view(self, *, name=None, n_rows=None, project_dir=None, **kw):
        return _handle_preview_view(name=name, n_rows=n_rows, project_dir=project_dir, **kw)

    @action("set_features_view", description="Set features view.", requires=["name"])
    def set_features_view(self, *, name=None, project_dir=None, **kw):
        return _handle_set_features_view(name=name, project_dir=project_dir, **kw)

    @action("view_dag", description="Show view dependency DAG.")
    def view_dag(self, *, project_dir=None, **kw):
        return _handle_view_dag(project_dir=project_dir, **kw)

    @action("add_sources_batch", description="Register multiple data sources.")
    def add_sources_batch(self, *, sources=None, project_dir=None, **kw):
        return _handle_add_sources_batch(sources=sources, project_dir=project_dir, **kw)

    @action("fill_nulls_batch", description="Fill nulls in multiple columns.")
    def fill_nulls_batch(self, *, columns=None, project_dir=None, **kw):
        return _handle_fill_nulls_batch(columns=columns, project_dir=project_dir, **kw)

    @action("add_views_batch", description="Declare multiple views.")
    def add_views_batch(self, *, views=None, project_dir=None, **kw):
        return _handle_add_views_batch(views=views, project_dir=project_dir, **kw)

    @action("sample", description="Sample the dataset.")
    def sample(self, *, fraction=None, stratify_column=None, seed=None, project_dir=None, **kw):
        return _handle_sample(fraction=fraction, stratify_column=stratify_column, seed=seed, project_dir=project_dir, **kw)

    @action("restore", description="Restore full dataset after sampling.")
    def restore(self, *, project_dir=None, **kw):
        return _handle_restore(project_dir=project_dir, **kw)

    @action("check_freshness", description="Check data freshness.")
    def check_freshness(self, *, project_dir=None, **kw):
        return _handle_check_freshness(project_dir=project_dir, **kw)

    @action("refresh", description="Refresh a data source.", requires=["name"])
    def refresh(self, *, name=None, project_dir=None, **kw):
        return _handle_refresh(name=name, project_dir=project_dir, **kw)

    @action("refresh_all", description="Refresh all data sources.")
    def refresh_all(self, *, project_dir=None, **kw):
        return _handle_refresh_all(project_dir=project_dir, **kw)

    @action("validate_source", description="Validate a data source.", requires=["name"])
    def validate_source(self, *, name=None, project_dir=None, **kw):
        return _handle_validate_source(name=name, project_dir=project_dir, **kw)

    @action("fetch_url", description="Download a file from a URL.", requires=["data_path"])
    def fetch_url(self, *, data_path=None, name=None, project_dir=None, **kw):
        return _handle_fetch_url(data_path=data_path, name=name, project_dir=project_dir, **kw)

    @action("upload_drive", description="Upload files to Google Drive.")
    def upload_drive(self, *, files=None, folder_id=None, folder_name=None, name=None, project_dir=None, **kw):
        return _handle_upload_drive(files=files, folder_id=folder_id, folder_name=folder_name, name=name, project_dir=project_dir, **kw)

    @action("upload_kaggle", description="Upload files to Kaggle.")
    def upload_kaggle(self, *, files=None, dataset_slug=None, title=None, name=None, project_dir=None, **kw):
        return _handle_upload_kaggle(files=files, dataset_slug=dataset_slug, title=title, name=name, project_dir=project_dir, **kw)

    @action("snapshot", description="Snapshot config and features.")
    def snapshot(self, *, name=None, project_dir=None, **kw):
        return _handle_snapshot(name=name, project_dir=project_dir, **kw)

    @action("restore_snapshot", description="Restore from a snapshot.", requires=["name"])
    def restore_snapshot(self, *, name=None, project_dir=None, **kw):
        return _handle_restore_snapshot(name=name, project_dir=project_dir, **kw)
