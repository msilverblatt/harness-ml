"""Tests for notebook Studio route."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from harnessml.studio.routes.notebook import _list_entries


@pytest.fixture
def project_with_notebook(tmp_path: Path) -> Path:
    """Create a project dir with config/ and notebook/entries.jsonl containing 3 entries."""
    (tmp_path / "config").mkdir()
    nb_dir = tmp_path / "notebook"
    nb_dir.mkdir()

    now = datetime.now(timezone.utc)

    entries = [
        {
            "id": "nb-001",
            "type": "finding",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "content": "Feature X is highly correlated with target.",
            "tags": ["feature-x"],
            "auto_tags": [],
            "struck": False,
            "struck_reason": None,
            "struck_at": None,
            "experiment_id": None,
        },
        {
            "id": "nb-002",
            "type": "theory",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "content": "Adding lag features should improve accuracy.",
            "tags": ["lag"],
            "auto_tags": ["model"],
            "struck": False,
            "struck_reason": None,
            "struck_at": None,
            "experiment_id": "exp-003",
        },
        {
            "id": "nb-003",
            "type": "decision",
            "timestamp": now.isoformat(),
            "content": "Dropping feature Y due to leakage.",
            "tags": ["feature-y"],
            "auto_tags": [],
            "struck": True,
            "struck_reason": "Revisited — feature Y is safe after all.",
            "struck_at": (now + timedelta(minutes=5)).isoformat(),
            "experiment_id": None,
        },
    ]

    with open(nb_dir / "entries.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return tmp_path


def test_list_entries_returns_all_including_struck(project_with_notebook: Path):
    """All entries including struck ones should be returned."""
    result = _list_entries(project_with_notebook)
    assert len(result) == 3
    ids = {e["id"] for e in result}
    assert ids == {"nb-001", "nb-002", "nb-003"}


def test_list_entries_newest_first(project_with_notebook: Path):
    """Entries should be sorted newest first by timestamp."""
    result = _list_entries(project_with_notebook)
    assert result[0]["id"] == "nb-003"
    assert result[1]["id"] == "nb-002"
    assert result[2]["id"] == "nb-001"


def test_list_entries_empty_project(tmp_path: Path):
    """An empty project (no notebook dir) should return an empty list."""
    (tmp_path / "config").mkdir()
    result = _list_entries(tmp_path)
    assert result == []


def test_struck_entries_have_metadata(project_with_notebook: Path):
    """Struck entries should have struck, struck_reason, and struck_at fields populated."""
    result = _list_entries(project_with_notebook)
    struck = [e for e in result if e.get("struck")]
    assert len(struck) == 1
    entry = struck[0]
    assert entry["id"] == "nb-003"
    assert entry["struck"] is True
    assert entry["struck_reason"] == "Revisited — feature Y is safe after all."
    assert entry["struck_at"] is not None
