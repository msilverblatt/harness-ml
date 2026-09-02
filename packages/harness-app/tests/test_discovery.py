from pathlib import Path
from harness.app.workspace.discovery import find_workspace


def test_find_workspace_from_subdir(tmp_path):
    """find_workspace returns root when called from a subdirectory."""
    root = tmp_path / "my_project"
    root.mkdir()
    (root / "harness.yaml").write_text("name: test\n")
    subdir = root / "a" / "b" / "c"
    subdir.mkdir(parents=True)

    result = find_workspace(start=subdir)
    assert result == root


def test_find_workspace_at_root(tmp_path):
    """find_workspace returns root when called directly at root."""
    root = tmp_path / "my_project"
    root.mkdir()
    (root / "harness.yaml").write_text("name: test\n")

    result = find_workspace(start=root)
    assert result == root


def test_find_workspace_returns_none(tmp_path):
    """find_workspace returns None when no harness.yaml exists."""
    empty = tmp_path / "empty"
    empty.mkdir()

    result = find_workspace(start=empty)
    assert result is None
