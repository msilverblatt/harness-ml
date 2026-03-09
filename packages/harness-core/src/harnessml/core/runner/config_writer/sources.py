"""Source management: freshness, refresh, validation."""
from __future__ import annotations

from pathlib import Path

from harnessml.core.runner.config_writer._helpers import (
    _get_freshness_tracker,
    _get_source_registry,
)


def check_freshness(project_dir: Path) -> str:
    """Check freshness of all registered sources.

    Returns a markdown table of stale sources, or a confirmation that
    everything is fresh.
    """
    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)
    tracker = _get_freshness_tracker(project_dir)

    sources = registry.list_all()
    if not sources:
        return "**No sources registered.** Use `manage_data(action='add_source')` to register sources first."

    stale = tracker.check_all(sources)
    if not stale:
        return f"All **{len(sources)}** source(s) are fresh."

    lines = [f"## Stale Sources ({len(stale)} of {len(sources)})\n"]
    lines.append("| Source | Frequency | Last Fetched |")
    lines.append("|--------|-----------|--------------|")
    for s in stale:
        lines.append(f"| {s['name']} | {s['refresh_frequency']} | {s['last_fetched']} |")
    lines.append(
        "\nUse `manage_data(action='refresh', name='<source>')` to update a stale source."
    )
    return "\n".join(lines)


def refresh_source(project_dir: Path, name: str) -> str:
    """Fetch a single source using its adapter, validate, and update freshness.

    Returns a markdown summary of what was loaded and any validation issues.
    """
    from harnessml.core.runner.sources.adapters import ADAPTERS
    from harnessml.core.runner.sources.validation import validate_source

    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)
    tracker = _get_freshness_tracker(project_dir)

    src = registry.get(name)
    if src is None:
        return f"**Error**: Source '{name}' not found in registry."

    adapter_cls = ADAPTERS.get(src.source_type)
    if adapter_cls is None:
        return f"**Error**: No adapter for source type '{src.source_type}'."

    try:
        if src.source_type == "file":
            df = adapter_cls.load(src.path_pattern)
        elif src.source_type == "url":
            fmt = src.schema.get("format", "csv")
            auth_headers = src.auth.get("headers") if src.auth else None
            df = adapter_cls.load(src.path_pattern, format=fmt, auth_headers=auth_headers)
        elif src.source_type == "api":
            auth_headers = src.auth.get("headers") if src.auth else None
            pagination = src.schema.get("pagination")
            df = adapter_cls.load(
                src.path_pattern,
                rate_limit=src.rate_limit,
                auth_headers=auth_headers,
                pagination=pagination,
            )
        elif src.source_type == "computed":
            return "**Error**: Computed sources must be refreshed programmatically."
        else:
            return f"**Error**: Unknown source type '{src.source_type}'."
    except Exception as e:
        return f"**Error** refreshing '{name}': {e}"

    # Validate
    violations = validate_source(src, df)
    tracker.record_fetch(name, row_count=len(df))

    lines = [
        f"Refreshed **{name}**",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
    ]
    if violations:
        lines.append(f"\n### Validation Issues ({len(violations)})")
        for v in violations:
            lines.append(f"- [{v.severity}] {v.column}: {v.message}")
    else:
        lines.append("- Validation: passed")

    return "\n".join(lines)


def refresh_all_sources(project_dir: Path) -> str:
    """Fetch all stale sources in topological order.

    Returns a markdown summary with per-source results.
    """
    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)
    tracker = _get_freshness_tracker(project_dir)

    sources = registry.list_all()
    if not sources:
        return "**No sources registered.**"

    stale_set = {s["name"] for s in tracker.check_all(sources)}
    if not stale_set:
        return f"All **{len(sources)}** source(s) are fresh. Nothing to refresh."

    # Refresh in dependency order, but only stale ones
    order = registry.topological_order()
    results = []
    for name in order:
        if name not in stale_set:
            continue
        result = refresh_source(project_dir, name)
        results.append(f"### {name}\n{result}")

    return f"## Refresh Summary ({len(results)} source(s))\n\n" + "\n\n".join(results)


def validate_source_data(project_dir: Path, name: str) -> str:
    """Load a source and validate against its schema definition.

    Returns a markdown report of validation results.
    """
    from harnessml.core.runner.sources.adapters import ADAPTERS
    from harnessml.core.runner.sources.validation import validate_source

    project_dir = Path(project_dir)
    registry = _get_source_registry(project_dir)

    src = registry.get(name)
    if src is None:
        return f"**Error**: Source '{name}' not found in registry."

    if src.source_type != "file":
        return f"**Error**: Validation preview only supported for file sources (got '{src.source_type}')."

    adapter_cls = ADAPTERS.get(src.source_type)
    if adapter_cls is None:
        return f"**Error**: No adapter for source type '{src.source_type}'."

    try:
        df = adapter_cls.load(src.path_pattern)
    except Exception as e:
        return f"**Error** loading '{name}': {e}"

    violations = validate_source(src, df)

    lines = [
        f"## Validation: {name}",
        f"- Rows: {len(df):,}",
        f"- Columns: {len(df.columns)}",
    ]
    if not violations:
        lines.append("- Result: **all checks passed**")
    else:
        lines.append(f"\n### Issues ({len(violations)})")
        for v in violations:
            lines.append(f"- [{v.severity}] {v.column}: {v.message}")

    return "\n".join(lines)
