from __future__ import annotations

import json
import os
import socket
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout


class WorkspaceBusyError(RuntimeError):
    """Raised when another process owns the workspace mutation lock."""


class WorkspaceLock:
    def __init__(self, workspace_dir: Path, operation: str, timeout: float = 0):
        self._root = Path(workspace_dir)
        self._operation = operation
        self._timeout = timeout
        self._state_dir = self._root / ".harness"
        self._lock = FileLock(self._state_dir / "workspace.lock")
        self._metadata_path = self._state_dir / "workspace-lock.json"

    def __enter__(self) -> WorkspaceLock:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._lock.acquire(timeout=self._timeout)
        except Timeout as error:
            owner = read_lock_owner(self._root)
            detail = f": {owner}" if owner else ""
            raise WorkspaceBusyError(
                f"Workspace is busy with another mutation{detail}"
            ) from error
        atomic_write_json(
            self._metadata_path,
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "operation": self._operation,
                "acquired_at": datetime.now(UTC).isoformat(),
            },
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._metadata_path.unlink(missing_ok=True)
        self._lock.release()


def read_lock_owner(workspace_dir: Path) -> dict[str, Any] | None:
    path = Path(workspace_dir) / ".harness" / "workspace-lock.json"
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def atomic_write_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True))
