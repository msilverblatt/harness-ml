from pathlib import Path


def find_workspace(start: Path | None = None) -> Path | None:
    current = Path(start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "harness.yaml").exists():
            return current
        current = current.parent
    if (current / "harness.yaml").exists():
        return current
    return None
