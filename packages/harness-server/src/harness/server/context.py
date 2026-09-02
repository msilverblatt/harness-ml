from __future__ import annotations

from pathlib import Path
from threading import RLock

from harness.app.workspace.discovery import find_workspace
from harness.app.workspace.manager import WorkspaceManager

_lock = RLock()
_workspace_manager: WorkspaceManager | None = None


def get_workspace_manager() -> WorkspaceManager | None:
    global _workspace_manager
    with _lock:
        if _workspace_manager is None:
            workspace_dir = find_workspace()
            if workspace_dir:
                _workspace_manager = WorkspaceManager(workspace_dir)
        return _workspace_manager


def set_workspace(path: str | Path) -> WorkspaceManager:
    global _workspace_manager
    resolved = Path(path).expanduser().resolve()
    if not (resolved / "harness.yaml").exists():
        raise ValueError(f"Not a Harness workspace: {resolved}")
    with _lock:
        _workspace_manager = WorkspaceManager(resolved)
        return _workspace_manager


def initialize_workspace(
    path: str | Path, task_type: str, target_column: str
) -> WorkspaceManager:
    global _workspace_manager
    resolved = Path(path).expanduser().resolve()
    with _lock:
        _workspace_manager = WorkspaceManager.init(
            resolved, task_type=task_type, target_column=target_column
        )
        return _workspace_manager


def require_workspace() -> WorkspaceManager:
    workspace = get_workspace_manager()
    if workspace is None:
        raise RuntimeError(
            "No workspace found. Run project.init or workspace.open first."
        )
    return workspace


def clear_workspace() -> None:
    global _workspace_manager
    with _lock:
        _workspace_manager = None
