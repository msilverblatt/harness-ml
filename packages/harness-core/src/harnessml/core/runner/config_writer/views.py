"""View and source management operations."""
from __future__ import annotations

from pathlib import Path

from harnessml.core.runner.config_writer._helpers import (
    _get_config_dir,
    _invalidate_view_cache,
    _load_yaml,
    _save_yaml,
)


def add_view(
    project_dir: Path,
    name: str,
    source: str,
    steps: list[dict],
    description: str = "",
) -> str:
    """Declare a view in pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    if "data" not in data:
        data["data"] = {}
    if "views" not in data["data"]:
        data["data"]["views"] = {}

    if name in data["data"]["views"]:
        return f"**Error**: View `{name}` already exists. Remove it first or use a different name."

    # Validate source exists
    sources = data["data"].get("sources", {})
    views = data["data"].get("views", {})
    all_names = set(sources.keys()) | set(views.keys())
    if source not in all_names:
        return (
            f"**Error**: Source `{source}` not found. "
            f"Available: {sorted(all_names)}"
        )

    # Validate steps parse as TransformStep
    from harnessml.core.runner.schema import TransformStep
    from pydantic import TypeAdapter, ValidationError
    adapter = TypeAdapter(list[TransformStep])
    try:
        adapter.validate_python(steps)
    except ValidationError as e:
        return f"**Error**: Invalid steps:\n```\n{e}\n```"

    view_def = {"source": source, "steps": steps}
    if description:
        view_def["description"] = description

    data["data"]["views"][name] = view_def
    _save_yaml(pipeline_path, data)

    confirmation = (
        f"**Added view**: `{name}`\n"
        f"- Source: {source}\n"
        f"- Steps: {len(steps)}\n"
        f"- Description: {description or '(none)'}"
    )

    try:
        preview = preview_view(project_dir, name, n_rows=3)
        return confirmation + "\n\n" + preview
    except Exception:
        return confirmation


def remove_view(project_dir: Path, name: str) -> str:
    """Remove a view from pipeline.yaml."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    if name not in views:
        return f"**Error**: View `{name}` not found."

    del data["data"]["views"][name]

    # Clear features_view if it pointed to this view
    if data.get("data", {}).get("features_view") == name:
        data["data"]["features_view"] = None

    _save_yaml(pipeline_path, data)
    _invalidate_view_cache(project_dir, name)
    return f"**Removed view**: `{name}`"


def update_view(
    project_dir: Path,
    name: str,
    source: str | None = None,
    steps: list[dict] | None = None,
    description: str | None = None,
) -> str:
    """Update an existing view in pipeline.yaml.

    Only provided fields are merged -- None values keep the existing value.
    Returns updated confirmation with an inline preview.
    """
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    if name not in views:
        return f"**Error**: View `{name}` not found. Available: {sorted(views.keys())}"

    view_def = views[name]

    # Merge provided fields
    if source is not None:
        # Validate source exists
        sources = data["data"].get("sources", {})
        all_names = set(sources.keys()) | set(views.keys()) - {name}
        if source not in all_names:
            return (
                f"**Error**: Source `{source}` not found. "
                f"Available: {sorted(all_names)}"
            )
        view_def["source"] = source

    if steps is not None:
        from harnessml.core.runner.schema import TransformStep
        from pydantic import TypeAdapter, ValidationError
        adapter = TypeAdapter(list[TransformStep])
        try:
            adapter.validate_python(steps)
        except ValidationError as e:
            return f"**Error**: Invalid steps:\n```\n{e}\n```"
        view_def["steps"] = steps

    if description is not None:
        if description:
            view_def["description"] = description
        else:
            view_def.pop("description", None)

    data["data"]["views"][name] = view_def
    _save_yaml(pipeline_path, data)
    _invalidate_view_cache(project_dir, name)

    confirmation = (
        f"**Updated view**: `{name}`\n"
        f"- Source: {view_def.get('source', '?')}\n"
        f"- Steps: {len(view_def.get('steps', []))}\n"
        f"- Description: {view_def.get('description', '(none)')}"
    )

    try:
        preview = preview_view(project_dir, name, n_rows=3)
        return confirmation + "\n\n" + preview
    except Exception:
        return confirmation


def list_views(project_dir: Path) -> str:
    """List all views with descriptions and dependency info."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    features_view = data.get("data", {}).get("features_view")

    if not views:
        return "No views defined. Use `add_view` to create one."

    lines = [f"## Views ({len(views)})\n"]
    for name, view_def in views.items():
        marker = " **[prediction table]**" if name == features_view else ""
        source = view_def.get("source", "?")
        steps = view_def.get("steps", [])
        desc = view_def.get("description", "")
        lines.append(f"### `{name}`{marker}")
        lines.append(f"- Source: `{source}`")
        lines.append(f"- Steps: {len(steps)}")
        if desc:
            lines.append(f"- Description: {desc}")

        # Show step summary
        for i, step in enumerate(steps):
            op = step.get("op", "?")
            lines.append(f"  {i+1}. `{op}`")
        lines.append("")

    return "\n".join(lines)


def preview_view(project_dir: Path, name: str, n_rows: int = 5) -> str:
    """Materialize a view and show schema + first N rows."""
    from harnessml.core.runner.data.utils import load_data_config

    config = load_data_config(Path(project_dir))

    all_names = set(config.sources.keys()) | set(config.views.keys())
    if name not in all_names:
        return f"**Error**: `{name}` not found. Available: {sorted(all_names)}"

    try:
        from harnessml.core.runner.views.resolver import ViewResolver
        resolver = ViewResolver(project_dir, config)
        df = resolver.resolve(name)
    except Exception as e:
        return f"**Error resolving view `{name}`**: {e}"

    lines = [f"## Preview: `{name}`\n"]
    lines.append(f"- Rows: {len(df):,}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append("")

    # Schema
    lines.append("### Schema\n")
    lines.append("| Column | Type | Non-null | Sample |")
    lines.append("|--------|------|----------|--------|")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        sample = str(df[col].iloc[0]) if len(df) > 0 else ""
        if len(sample) > 30:
            sample = sample[:27] + "..."
        lines.append(f"| {col} | {dtype} | {non_null:,} | {sample} |")
    lines.append("")

    # Sample rows
    lines.append(f"### First {min(n_rows, len(df))} rows\n")
    lines.append(df.head(n_rows).to_markdown(index=False))

    return "\n".join(lines)


def set_features_view(project_dir: Path, name: str) -> str:
    """Set which view becomes the prediction table."""
    config_dir = _get_config_dir(Path(project_dir))
    pipeline_path = config_dir / "pipeline.yaml"
    data = _load_yaml(pipeline_path)

    views = data.get("data", {}).get("views", {})
    if name not in views:
        return f"**Error**: View `{name}` not found. Available: {sorted(views.keys())}"

    if "data" not in data:
        data["data"] = {}
    data["data"]["features_view"] = name
    _save_yaml(pipeline_path, data)

    return f"**Set features_view**: `{name}` is now the prediction table."


def view_dag(project_dir: Path) -> str:
    """Show the full view dependency graph."""
    from harnessml.core.runner.data.utils import load_data_config

    config = load_data_config(Path(project_dir))

    if not config.views:
        return "No views defined."

    lines = ["## View Dependency Graph\n"]
    lines.append("```")

    for name, view_def in config.views.items():
        deps = [view_def.source]
        for step in view_def.steps:
            step_dict = step if isinstance(step, dict) else step.model_dump()
            if "other" in step_dict:
                deps.append(step_dict["other"])

        marker = " [prediction table]" if name == config.features_view else ""
        lines.append(f"{name}{marker} <- {', '.join(deps)}")

    lines.append("```")

    # Show sources (leaf nodes)
    lines.append("\n### Sources (raw data)\n")
    for name, source in config.sources.items():
        path = source.path or "(no path)"
        lines.append(f"- `{name}`: {path}")

    return "\n".join(lines)
