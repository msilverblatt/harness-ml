"""Handler for notebook tool."""
from __future__ import annotations

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    collect_hints,
    format_response_with_hints,
    validate_enum,
    validate_required,
)

_VALID_TYPES = {"theory", "finding", "research", "decision", "plan", "note"}


def _handle_write(*, type, content, tags=None, experiment_id=None, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    err = validate_required(content, "content")
    if err:
        return err
    err = validate_enum(type, _VALID_TYPES, "type")
    if err:
        return err

    parsed_tags = parse_json_param(tags) if tags else None

    try:
        store = NotebookStore(resolve_project_dir(project_dir))
        entry = store.write(
            type=type,
            content=content,
            tags=parsed_tags,
            experiment_id=experiment_id,
        )
    except ValueError as exc:
        return f"**Error**: {exc}"

    all_tags = sorted(set(entry.tags) | set(entry.auto_tags))
    preview = entry.content[:200]
    lines = [
        "**Notebook entry created**",
        "",
        f"- **ID**: `{entry.id}`",
        f"- **Type**: {entry.type.value}",
        f"- **Content**: {preview}",
    ]
    if all_tags:
        lines.append(f"- **Tags**: {', '.join(all_tags)}")
    return "\n".join(lines)


def _handle_read(*, type=None, tags=None, page=None, per_page=None, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    if type is not None:
        err = validate_enum(type, _VALID_TYPES, "type")
        if err:
            return err

    parsed_tags = parse_json_param(tags) if tags else None

    try:
        store = NotebookStore(resolve_project_dir(project_dir))
        entries = store.read(
            type=type,
            tags=parsed_tags,
            page=int(page) if page else 1,
            per_page=int(per_page) if per_page else 10,
        )
    except ValueError as exc:
        return f"**Error**: {exc}"

    if not entries:
        return "No notebook entries found."

    lines = [f"**Notebook entries** ({len(entries)} shown)\n"]
    for e in entries:
        all_tags = sorted(set(e.tags) | set(e.auto_tags))
        tag_str = f" | Tags: {', '.join(all_tags)}" if all_tags else ""
        lines.append(f"### `{e.id}` — {e.type.value} ({e.timestamp:%Y-%m-%d %H:%M}){tag_str}")
        lines.append(f"{e.content}\n")

    return "\n".join(lines)


def _handle_search(*, query=None, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    err = validate_required(query, "query")
    if err:
        return err

    try:
        store = NotebookStore(resolve_project_dir(project_dir))
        entries = store.search(query)
    except ValueError as exc:
        return f"**Error**: {exc}"

    if not entries:
        return f"No notebook entries matching '{query}'."

    lines = [f"**Search results for** '{query}' ({len(entries)} found)\n"]
    for e in entries:
        all_tags = sorted(set(e.tags) | set(e.auto_tags))
        tag_str = f" | Tags: {', '.join(all_tags)}" if all_tags else ""
        lines.append(f"### `{e.id}` — {e.type.value} ({e.timestamp:%Y-%m-%d %H:%M}){tag_str}")
        lines.append(f"{e.content}\n")

    return "\n".join(lines)


def _handle_strike(*, entry_id=None, reason=None, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    err = validate_required(entry_id, "entry_id")
    if err:
        return err
    err = validate_required(reason, "reason")
    if err:
        return err

    try:
        store = NotebookStore(resolve_project_dir(project_dir))
        entry = store.strike(entry_id, reason=reason)
    except ValueError as exc:
        return f"**Error**: {exc}"

    return (
        f"**Entry struck**\n\n"
        f"- **ID**: `{entry.id}`\n"
        f"- **Reason**: {entry.struck_reason}\n"
        f"- ~~{entry.content}~~"
    )


def _handle_summary(*, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    try:
        store = NotebookStore(resolve_project_dir(project_dir))
        data = store.summary()
    except ValueError as exc:
        return f"**Error**: {exc}"

    lines = ["**Notebook Summary**\n"]

    lines.append(f"- **Total entries**: {data['total_entries']}")
    lines.append(f"- **Struck entries**: {data['struck_entries']}\n")

    if data.get("latest_theory"):
        lines.append("### Current Theory")
        lines.append(f"{data['latest_theory']}\n")

    if data.get("latest_plan"):
        lines.append("### Current Plan")
        lines.append(f"{data['latest_plan']}\n")

    if data.get("recent_findings"):
        lines.append("### Recent Findings")
        for f in data["recent_findings"]:
            lines.append(f"- {f}")
        lines.append("")

    if data.get("entity_index"):
        lines.append("### Entity Index")
        for tag, count in sorted(data["entity_index"].items()):
            lines.append(f"- `{tag}`: {count}")

    return "\n".join(lines)


ACTIONS = {
    "write": _handle_write,
    "read": _handle_read,
    "search": _handle_search,
    "strike": _handle_strike,
    "summary": _handle_summary,
}


def dispatch(action: str, **kwargs) -> str:
    """Dispatch a notebook action."""
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err
    result = ACTIONS[action](**kwargs)
    hints = collect_hints(action, tool="notebook", **kwargs)
    return format_response_with_hints(result, hints)
