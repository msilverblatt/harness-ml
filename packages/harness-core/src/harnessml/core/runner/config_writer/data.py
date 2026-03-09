"""Data management, inspection, and configuration operations."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from harnessml.core.runner.config_writer._helpers import (
    _get_config_dir,
    _load_yaml,
    _save_yaml,
)

logger = logging.getLogger(__name__)


def update_data_config(
    project_dir: Path,
    *,
    target_column: str | None = None,
    key_columns: list[str] | None = None,
    time_column: str | None = None,
) -> str:
    """Update data section of pipeline.yaml (target_column, key_columns, time_column)."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}

    d = data["data"]
    updates = []

    if target_column is not None:
        d["target_column"] = target_column
        updates.append(f"- Target column: {target_column}")
    if key_columns is not None:
        d["key_columns"] = key_columns
        updates.append(f"- Key columns: {key_columns}")
    if time_column is not None:
        d["time_column"] = time_column
        updates.append(f"- Time column: {time_column}")

    if not updates:
        return "**No changes** — provide at least one of: target_column, key_columns, time_column."

    _save_yaml(pipeline_path, data)

    return "**Updated data config**\n" + "\n".join(updates)


def configure_exclude_columns(
    project_dir: Path,
    *,
    add_columns: list[str] | None = None,
    remove_columns: list[str] | None = None,
) -> str:
    """Add or remove columns from data.exclude_columns in pipeline.yaml.

    Excluded columns are never used as model features or in feature discovery.
    Use this to mark regression target columns (e.g. 'margin') or ID columns
    that should not be treated as predictive features.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    current = list(data.get("data", {}).get("exclude_columns", []))
    current_set = set(current)

    added, removed = [], []
    if add_columns:
        for col in add_columns:
            if col not in current_set:
                current.append(col)
                current_set.add(col)
                added.append(col)

    if remove_columns:
        for col in remove_columns:
            if col in current_set:
                current.remove(col)
                current_set.discard(col)
                removed.append(col)

    if "data" not in data:
        data["data"] = {}
    data["data"]["exclude_columns"] = current
    _save_yaml(pipeline_path, data)

    lines = ["**Updated `data.exclude_columns`**"]
    if added:
        lines.append(f"- Added: {added}")
    if removed:
        lines.append(f"- Removed: {removed}")
    lines.append(f"- Current list: {current}")
    return "\n".join(lines)


def configure_denylist(
    project_dir: Path,
    *,
    add_columns: list[str] | None = None,
    remove_columns: list[str] | None = None,
) -> str:
    """Add or remove columns from the guardrails feature leakage denylist.

    The denylist is checked by check_guardrails() -- any model whose feature
    list contains a denied column causes a FAIL.
    """
    project_dir = Path(project_dir)
    sources_path = _get_config_dir(project_dir) / "sources.yaml"
    data = _load_yaml(sources_path)

    if "guardrails" not in data:
        data["guardrails"] = {}
    current = list(data["guardrails"].get("feature_leakage_denylist", []))
    current_set = set(current)

    added, removed = [], []
    if add_columns:
        for col in add_columns:
            if col not in current_set:
                current.append(col)
                current_set.add(col)
                added.append(col)

    if remove_columns:
        for col in remove_columns:
            if col in current_set:
                current.remove(col)
                current_set.discard(col)
                removed.append(col)

    data["guardrails"]["feature_leakage_denylist"] = current
    _save_yaml(sources_path, data)

    lines = ["**Updated `guardrails.feature_leakage_denylist`**"]
    if added:
        lines.append(f"- Added: {added}")
    if removed:
        lines.append(f"- Removed: {removed}")
    lines.append(f"- Current denylist: {current}")
    return "\n".join(lines)


def add_target(
    project_dir: Path,
    name: str,
    *,
    column: str,
    task: str = "binary",
    metrics: list[str] | None = None,
) -> str:
    """Add a named target profile to data.targets in pipeline.yaml.

    Returns markdown confirmation.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}
    if "targets" not in data["data"]:
        data["data"]["targets"] = {}

    target_def: dict = {"column": column, "task": task}
    if metrics:
        target_def["metrics"] = metrics

    data["data"]["targets"][name] = target_def
    _save_yaml(pipeline_path, data)

    lines = [f"**Added target profile `{name}`**"]
    lines.append(f"- Column: `{column}`")
    lines.append(f"- Task: `{task}`")
    if metrics:
        lines.append(f"- Metrics: {metrics}")
    return "\n".join(lines)


def list_targets(project_dir: Path) -> str:
    """List all named target profiles from pipeline.yaml.

    Returns markdown-formatted list.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    targets = data.get("data", {}).get("targets", {})
    active_column = data.get("data", {}).get("target_column", "result")
    active_task = data.get("data", {}).get("task", "classification")

    if not targets:
        return (
            f"**No named target profiles defined.**\n"
            f"- Default target: `{active_column}` (task: `{active_task}`)\n"
            f"- Use `add_target` to define named profiles."
        )

    lines = ["**Target Profiles**\n"]
    for name, tgt in targets.items():
        col = tgt.get("column", "?")
        task = tgt.get("task", "binary")
        metrics = tgt.get("metrics", [])
        active_marker = " **(active)**" if col == active_column and task == active_task else ""
        line = f"- **{name}**{active_marker}: column=`{col}`, task=`{task}`"
        if metrics:
            line += f", metrics={metrics}"
        lines.append(line)

    return "\n".join(lines)


def set_active_target(project_dir: Path, name: str) -> str:
    """Set a named target profile as the active target.

    Updates data.target_column, data.task, and optionally backtest.metrics.
    Returns markdown confirmation.
    """
    project_dir = Path(project_dir)
    pipeline_path = _get_config_dir(project_dir) / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    targets = data.get("data", {}).get("targets", {})
    if name not in targets:
        available = ", ".join(sorted(targets.keys())) if targets else "(none defined)"
        return f"**Error**: Unknown target `{name}`. Available targets: {available}"

    tgt = targets[name]
    data["data"]["target_column"] = tgt["column"]
    data["data"]["task"] = tgt.get("task", "binary")

    metrics = tgt.get("metrics", [])
    if metrics:
        if "backtest" not in data:
            data["backtest"] = {}
        data["backtest"]["metrics"] = metrics

    _save_yaml(pipeline_path, data)

    lines = [f"**Activated target profile `{name}`**"]
    lines.append(f"- target_column: `{tgt['column']}`")
    lines.append(f"- task: `{tgt.get('task', 'binary')}`")
    if metrics:
        lines.append(f"- backtest.metrics updated to: {metrics}")
    return "\n".join(lines)


def add_dataset(
    project_dir: Path,
    data_path: str,
    *,
    join_on: list[str] | None = None,
    prefix: str | None = None,
    features_dir: str | None = None,
    auto_clean: bool = False,
) -> str:
    """Add a new dataset by merging into the features parquet.

    Uses DataPipeline when sources are configured in DataConfig,
    falls back to direct ingest otherwise.
    """
    from harnessml.core.runner.data_utils import load_data_config

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = None

    # Use DataPipeline if config has sources configured
    if config is not None and config.sources:
        from harnessml.core.runner.data_pipeline import DataPipeline
        from harnessml.core.runner.schema import SourceConfig

        pipeline = DataPipeline(project_dir, config)
        name = Path(data_path).stem
        source = SourceConfig(name=name, path=data_path, join_on=join_on)
        pipeline.config.sources[name] = source
        result = pipeline.refresh(sources=[name])
        cols = result.columns_added.get(name, [])
        lines = [f"## Ingested: {name}\n"]
        lines.append(f"- **Columns added**: {len(cols)}")
        if cols:
            cols_preview = ", ".join(cols[:10])
            if len(cols) > 10:
                cols_preview += f", ... (+{len(cols) - 10} more)"
            lines.append(f"- **Columns**: {cols_preview}")
        lines.append("- **Source registered** in pipeline config")
        if result.errors:
            lines.append("\n### Errors\n")
            for src, err in result.errors.items():
                lines.append(f"- {src}: {err}")
        return "\n".join(lines)

    # Fallback to direct ingest
    from harnessml.core.runner.data_ingest import ingest_dataset
    result = ingest_dataset(
        project_dir=project_dir,
        data_path=data_path,
        join_on=join_on,
        prefix=prefix,
        features_dir=features_dir,
        auto_clean=auto_clean,
    )
    return result.format_summary()


def derive_column(
    project_dir: Path,
    name: str,
    expression: str,
    *,
    group_by: str | None = None,
    dtype: str | None = None,
) -> str:
    """Derive a new column from a pandas expression and save to the feature store."""
    from harnessml.core.runner.data_ingest import derive_column as _derive

    return _derive(
        Path(project_dir),
        name,
        expression,
        group_by=group_by,
        dtype=dtype,
    )


def drop_rows(
    project_dir: Path,
    *,
    column: str | None = None,
    condition: str = "null",
) -> str:
    """Drop rows from the feature store by condition."""
    from harnessml.core.runner.data_ingest import drop_rows as _drop

    return _drop(
        Path(project_dir),
        column=column,
        condition=condition,
    )


def sample_data(project_dir: Path, *, fraction=0.1, stratify_column=None, seed=42) -> str:
    """Sample the feature store for fast iteration."""
    from harnessml.core.runner.data_ingest import sample_data as _sample
    return _sample(Path(project_dir), fraction=fraction, stratify_column=stratify_column, seed=seed)


def restore_full_data(project_dir: Path) -> str:
    """Restore the full feature store from backup."""
    from harnessml.core.runner.data_ingest import restore_full_data as _restore
    return _restore(Path(project_dir))


def fetch_url(project_dir: Path, url: str, *, filename: str | None = None) -> str:
    """Download a file from a URL to the project's raw data directory."""
    import urllib.request
    from urllib.parse import urlparse

    project_dir = Path(project_dir)
    raw_dir = project_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not filename:
        filename = Path(urlparse(url).path).name or "download.csv"

    dest = raw_dir / filename
    urllib.request.urlretrieve(url, dest)

    size_mb = dest.stat().st_size / (1024 * 1024)
    return f"Downloaded `{filename}` ({size_mb:.1f} MB) to `{dest.relative_to(project_dir)}`."


def inspect_data(project_dir: Path, *, column: str | None = None) -> str:
    """Inspect the features dataset.

    Without column: overview of all columns (shape, dtypes, null counts).
    With column: detailed statistics for that specific column.
    """
    from harnessml.core.runner.data_utils import get_features_df, load_data_config
    from harnessml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    if column is not None:
        return _inspect_column(df, column)
    return _inspect_overview(df)


def _inspect_overview(df) -> str:
    """Return a markdown overview of all columns in the DataFrame."""
    n_rows, n_cols = df.shape
    lines = [
        f"## Data Overview: {n_rows:,} rows x {n_cols} columns\n",
        "| Column | Dtype | Nulls | Null% |",
        "|--------|-------|-------|-------|",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / n_rows * 100) if n_rows > 0 else 0.0
        lines.append(f"| {col} | {dtype} | {null_count:,} | {null_pct:.1f}% |")
    return "\n".join(lines)


def _inspect_column(df, column: str) -> str:
    """Return detailed statistics for a single column."""
    if column not in df.columns:
        available = ", ".join(f"`{c}`" for c in sorted(df.columns))
        return f"**Error**: Column `{column}` not found. Available columns: {available}"

    series = df[column]
    dtype = str(series.dtype)
    non_null = int(series.notna().sum())
    n_unique = int(series.nunique())
    n_rows = len(series)

    lines = [
        f"## Column: `{column}`\n",
        f"- **Dtype**: {dtype}",
        f"- **Non-null**: {non_null:,} / {n_rows:,}",
        f"- **Unique values**: {n_unique:,}",
    ]

    import pandas as pd

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        lines.append("")
        lines.append("### Statistics")
        lines.append(f"- **Mean**: {desc['mean']:.4f}")
        lines.append(f"- **Std**: {desc['std']:.4f}")
        lines.append(f"- **Min**: {desc['min']:.4f}")
        lines.append(f"- **25%**: {desc['25%']:.4f}")
        lines.append(f"- **50%**: {desc['50%']:.4f}")
        lines.append(f"- **75%**: {desc['75%']:.4f}")
        lines.append(f"- **Max**: {desc['max']:.4f}")

    if n_unique <= 20:
        lines.append("")
        lines.append("### Value Counts")
        vc = series.value_counts(dropna=False)
        for val, count in vc.items():
            pct = count / n_rows * 100 if n_rows > 0 else 0.0
            label = "NaN" if pd.isna(val) else str(val)
            lines.append(f"- `{label}`: {count:,} ({pct:.1f}%)")
    elif n_unique > 20:
        lines.append("")
        lines.append("### Sample Values (first 10)")
        for val in series.dropna().unique()[:10]:
            lines.append(f"- `{val}`")

    return "\n".join(lines)


def profile_data(project_dir: Path, category: str | None = None) -> str:
    """Profile the features dataset."""
    from harnessml.core.runner.data_utils import get_features_df, load_data_config
    from harnessml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."

    from harnessml.core.runner.data_profiler import profile_dataset

    profile = profile_dataset(config=config, df=df)

    if category:
        return profile.format_columns(category=category)
    return profile.format_summary()


def available_features(
    project_dir: Path,
    prefix: str | None = None,
    type_filter: str | None = None,
) -> str:
    """List available feature columns from the dataset.

    If the project uses the declarative feature store, shows features
    grouped by type. Otherwise falls back to column listing.
    """
    from harnessml.core.runner.data_utils import get_features_df, load_data_config
    from harnessml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()

    # Check for declarative feature store
    if config.feature_defs:
        from harnessml.core.runner.feature_store import FeatureStore
        from harnessml.core.runner.schema import FeatureType

        store = FeatureStore(project_dir, config)
        ft = FeatureType(type_filter) if type_filter else None
        features = store.available(type_filter=ft)

        if not features:
            return "No declarative features registered."

        lines = [f"## Declarative Features ({len(features)})\n"]
        by_type: dict[str, list] = {}
        for f in features:
            by_type.setdefault(f.type.value, []).append(f)

        for ft_name, feats in by_type.items():
            lines.append(f"### {ft_name.title()} ({len(feats)})")
            for f in feats:
                lines.append(f"- `{f.name}` — {f.description or f.category}")
            lines.append("")

        return "\n".join(lines)

    # Fallback: flat column listing

    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature data found. Ingest data or set a features_view."
    cols = sorted(df.columns)
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]

    if not cols:
        return "No features found."

    lines = [f"## Available Features ({len(cols)} columns)\n"]
    for col in cols:
        lines.append(f"- `{col}`")
    return "\n".join(lines)


def feature_store_status(project_dir: Path) -> str:
    """Quick overview of the feature store state.

    Returns: row count, column count, target distribution,
    time column range, source count.
    """
    from harnessml.core.runner.data_utils import get_features_df, load_data_config
    from harnessml.core.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = DataConfig()


    try:
        df = get_features_df(project_dir, config)
    except FileNotFoundError:
        return "**Error**: No feature store found. Ingest data first with `manage_data(action='add')` or set a features_view."

    n_rows = len(df)
    n_cols = len(df.columns)

    lines = ["## Feature Store Status\n"]
    lines.append(f"- **Rows**: {n_rows}")
    lines.append(f"- **Columns**: {n_cols}")

    # Show source info
    if config.features_view:
        lines.append(f"- **Source**: view `{config.features_view}`")
    else:
        import os
        from datetime import datetime

        from harnessml.core.runner.data_utils import get_features_path
        parquet_path = get_features_path(project_dir, config)
        lines.append(f"- **File**: `{parquet_path.relative_to(project_dir)}`")
        if parquet_path.exists():
            mtime = os.path.getmtime(parquet_path)
            lines.append(f"- **Last modified**: {datetime.fromtimestamp(mtime).isoformat()}")

    # Target column distribution
    target_col = config.target_column
    if target_col and target_col in df.columns:
        dist = df[target_col].value_counts()
        lines.append(f"\n### Target Distribution (`{target_col}`)\n")
        for val, count in dist.items():
            pct = count / n_rows * 100
            lines.append(f"- {val}: {count} ({pct:.1f}%)")
    elif target_col:
        lines.append(f"\n*Target column `{target_col}` not found in data.*")

    # Time column range
    time_col = config.time_column
    if time_col and time_col in df.columns:
        lines.append(f"\n### Time Range (`{time_col}`)\n")
        lines.append(f"- Min: {df[time_col].min()}")
        lines.append(f"- Max: {df[time_col].max()}")
        lines.append(f"- Unique values: {df[time_col].nunique()}")

    # Source count
    registry_path = project_dir / "data" / "source_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        n_sources = len(registry.get("sources", []))
        lines.append(f"\n- **Ingested sources**: {n_sources}")

    return "\n".join(lines)


def list_sources(project_dir: Path) -> str:
    """List ingested data sources from the source registry.

    Reads data/source_registry.json and returns a summary of
    each source: name, path, columns added, row count.
    """
    project_dir = Path(project_dir)
    registry_path = project_dir / "data" / "source_registry.json"

    if not registry_path.exists():
        return "**No sources registered.** Ingest data with `manage_data(action='add')` first."

    registry = json.loads(registry_path.read_text())
    sources = registry.get("sources", [])

    if not sources:
        return "**No sources registered.** Ingest data with `manage_data(action='add')` first."

    lines = [f"## Data Sources ({len(sources)} registered)\n"]
    lines.append("| # | Name | Path | Columns Added | Rows | Bootstrap |")
    lines.append("|---|------|------|---------------|------|-----------|")

    for i, src in enumerate(sources, 1):
        name = src.get("name", "unknown")
        path = src.get("path", "\u2014")
        cols = src.get("columns_added", [])
        rows = src.get("rows", "\u2014")
        bootstrap = "Yes" if src.get("is_bootstrap") else "No"
        lines.append(f"| {i} | {name} | `{path}` | {len(cols)} | {rows} | {bootstrap} |")

    return "\n".join(lines)


def add_source(
    project_dir: Path,
    name: str,
    path: str,
    format: str = "auto",
) -> str:
    """Register a raw data source in pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}
    if "sources" not in data["data"]:
        data["data"]["sources"] = {}

    if name in data["data"]["sources"]:
        return f"**Error**: Source `{name}` already exists."

    # Verify file exists
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = Path(project_dir) / path
    if not source_path.exists():
        return f"**Error**: File not found: {source_path}"

    # Read to get row/column info
    import pandas as pd
    if path.endswith(".csv"):
        df = pd.read_csv(source_path, nrows=5)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(source_path).head(5)
    else:
        df = pd.read_csv(source_path, nrows=5)  # try csv

    data["data"]["sources"][name] = {
        "name": name,
        "path": path,
        "format": format,
    }
    _save_yaml(pipeline_path, data)

    return (
        f"**Added source**: `{name}`\n"
        f"- Path: {path}\n"
        f"- Columns ({len(df.columns)}): {', '.join(df.columns[:15])}"
        f"{'...' if len(df.columns) > 15 else ''}\n"
        f"- Format: {format}"
    )
