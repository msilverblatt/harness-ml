"""Project initialization/scaffold operations."""
from __future__ import annotations

import difflib
from pathlib import Path

_VALID_TASKS = {"binary", "classification", "multiclass", "regression", "ranking", "survival", "probabilistic"}


def scaffold_init(
    project_dir: Path,
    project_name: str | None = None,
    *,
    task: str = "classification",
    target_column: str = "result",
    key_columns: list[str] | None = None,
    time_column: str | None = None,
) -> str:
    """Initialize a new harnessml project via scaffold.

    Returns markdown confirmation or error.
    """
    project_dir = Path(project_dir)

    if task and task.lower() not in _VALID_TASKS:
        suggestion = difflib.get_close_matches(task.lower(), _VALID_TASKS, n=1, cutoff=0.4)
        hint = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
        return f"**Error**: Unknown task type '{task}'.{hint} Valid: {', '.join(sorted(_VALID_TASKS))}"

    # Warn if config directory already exists with files
    config_dir = project_dir / "config"
    if config_dir.exists() and any(config_dir.iterdir()):
        existing = [f.name for f in config_dir.iterdir() if f.is_file()]
        return (
            f"**Warning**: Project at `{project_dir}` already has config files: "
            f"{', '.join(sorted(existing)[:10])}.\n"
            f"Re-initializing may overwrite existing configuration. "
            f"Remove the `config/` directory first if you want a fresh start."
        )

    try:
        from harnessml.core.runner.scaffold.scaffold import scaffold_project

        scaffold_project(
            project_dir,
            project_name,
            task=task,
            target_column=target_column,
            key_columns=key_columns,
            time_column=time_column,
        )
    except FileExistsError:
        return f"**Error**: Directory `{project_dir}` already exists and is not empty. Provide a project_name to create a subdirectory, or point to an empty directory."
    except Exception as exc:
        return f"**Error**: Failed to initialize project: {exc}"

    actual_dir = project_dir
    if project_name and project_dir.exists() and (project_dir / project_name).exists():
        actual_dir = project_dir / project_name
    name = project_name or project_dir.name
    lines = [
        f"**Initialized project**: `{name}`",
        f"- Directory: `{actual_dir}`",
        f"- Task: {task}",
        f"- Target column: {target_column}",
    ]
    if key_columns:
        lines.append(f"- Key columns: {key_columns}")
    if time_column:
        lines.append(f"- Time column: {time_column}")
    lines.append(f"\nConfig files created in `{actual_dir}/config/`")

    return "\n".join(lines)
