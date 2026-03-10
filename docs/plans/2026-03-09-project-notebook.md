# Project Notebook Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a persistent project notebook that stores typed, tagged, searchable entries — giving the agent session-to-session memory about what it has learned.

**Architecture:** JSONL append-only storage (same pattern as experiment journal). Three-layer: core notebook module (schema + read/write/search/auto-tag) → MCP handler → MCP tool. Studio route for UI. Entries are immutable; strikethrough hides from agent reads but preserves history.

**Tech Stack:** Pydantic v2 schemas, JSONL file I/O, simple substring auto-tagging, FastAPI route for Studio.

---

### Task 1: Notebook Schema (Pydantic models)

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/notebook/__init__.py` (empty, namespace)
- Create: `packages/harness-core/src/harnessml/core/runner/notebook/schema.py`
- Test: `packages/harness-core/tests/runner/notebook/__init__.py` (empty)
- Test: `packages/harness-core/tests/runner/notebook/test_schema.py`

**Step 1: Write the failing tests**

```python
# tests/runner/notebook/test_schema.py
import pytest
from datetime import datetime, timezone


class TestNotebookEntry:
    def test_create_entry_with_required_fields(self):
        from harnessml.core.runner.notebook.schema import NotebookEntry, EntryType

        entry = NotebookEntry(
            id="nb-001",
            type=EntryType.FINDING,
            content="xgb_main struggles on fold 3",
        )
        assert entry.id == "nb-001"
        assert entry.type == EntryType.FINDING
        assert entry.content == "xgb_main struggles on fold 3"
        assert entry.tags == []
        assert entry.auto_tags == []
        assert entry.struck is False
        assert entry.struck_reason is None
        assert entry.struck_at is None
        assert entry.experiment_id is None
        assert isinstance(entry.timestamp, datetime)

    def test_create_entry_with_all_fields(self):
        from harnessml.core.runner.notebook.schema import NotebookEntry, EntryType

        entry = NotebookEntry(
            id="nb-002",
            type=EntryType.THEORY,
            content="The target is driven by temporal patterns",
            tags=["model:xgb_main"],
            auto_tags=["feature:age_group"],
            experiment_id="exp-005",
        )
        assert entry.tags == ["model:xgb_main"]
        assert entry.auto_tags == ["feature:age_group"]
        assert entry.experiment_id == "exp-005"

    def test_entry_type_enum(self):
        from harnessml.core.runner.notebook.schema import EntryType

        assert EntryType.THEORY == "theory"
        assert EntryType.FINDING == "finding"
        assert EntryType.RESEARCH == "research"
        assert EntryType.DECISION == "decision"
        assert EntryType.PLAN == "plan"
        assert EntryType.NOTE == "note"

    def test_empty_content_rejected(self):
        from harnessml.core.runner.notebook.schema import NotebookEntry, EntryType

        with pytest.raises(ValueError, match="content"):
            NotebookEntry(id="nb-001", type=EntryType.FINDING, content="")

    def test_whitespace_only_content_rejected(self):
        from harnessml.core.runner.notebook.schema import NotebookEntry, EntryType

        with pytest.raises(ValueError, match="content"):
            NotebookEntry(id="nb-001", type=EntryType.FINDING, content="   ")

    def test_serialization_roundtrip(self):
        from harnessml.core.runner.notebook.schema import NotebookEntry, EntryType

        entry = NotebookEntry(
            id="nb-001",
            type=EntryType.RESEARCH,
            content="Domain research finding",
            tags=["feature:age"],
        )
        json_str = entry.model_dump_json()
        restored = NotebookEntry.model_validate_json(json_str)
        assert restored.id == entry.id
        assert restored.type == entry.type
        assert restored.content == entry.content
        assert restored.tags == entry.tags

    def test_strike_fields(self):
        from harnessml.core.runner.notebook.schema import NotebookEntry, EntryType

        entry = NotebookEntry(
            id="nb-001",
            type=EntryType.FINDING,
            content="Outdated finding",
            struck=True,
            struck_reason="Superseded by nb-005",
            struck_at=datetime.now(timezone.utc),
        )
        assert entry.struck is True
        assert entry.struck_reason == "Superseded by nb-005"
        assert entry.struck_at is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/test_schema.py -v`
Expected: FAIL — module not found

**Step 3: Write the schema**

```python
# packages/harness-core/src/harnessml/core/runner/notebook/schema.py
"""Pydantic schemas for the project notebook."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, field_validator, Field


class EntryType(str, Enum):
    """Notebook entry types."""

    THEORY = "theory"
    FINDING = "finding"
    RESEARCH = "research"
    DECISION = "decision"
    PLAN = "plan"
    NOTE = "note"


class NotebookEntry(BaseModel):
    """A single notebook entry."""

    id: str
    type: EntryType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content: str
    tags: list[str] = Field(default_factory=list)
    auto_tags: list[str] = Field(default_factory=list)
    struck: bool = False
    struck_reason: str | None = None
    struck_at: datetime | None = None
    experiment_id: str | None = None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/test_schema.py -v`
Expected: PASS (all 8 tests)

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/runner/notebook/ packages/harness-core/tests/runner/notebook/
git commit -m "feat(notebook): add Pydantic schema for notebook entries"
```

---

### Task 2: Auto-Tagger

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/notebook/tagger.py`
- Test: `packages/harness-core/tests/runner/notebook/test_tagger.py`

**Step 1: Write the failing tests**

```python
# tests/runner/notebook/test_tagger.py
from pathlib import Path


class TestAutoTagger:
    def test_detects_model_names(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        # Create minimal config
        models_dir = tmp_path / "config" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "xgb_main.yaml").write_text("type: xgboost\n")
        (models_dir / "lr_base.yaml").write_text("type: logistic_regression\n")

        tags = auto_tag("xgb_main struggles on fold 3", tmp_path)
        assert "model:xgb_main" in tags
        assert "model:lr_base" not in tags

    def test_detects_model_names_from_single_models_yaml(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "models.yaml").write_text(
            "models:\n  xgb_main:\n    type: xgboost\n  rf_core:\n    type: random_forest\n"
        )

        tags = auto_tag("rf_core has high variance", tmp_path)
        assert "model:rf_core" in tags
        assert "model:xgb_main" not in tags

    def test_detects_feature_names(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "models.yaml").write_text(
            "models:\n  xgb:\n    type: xgboost\n    features: [age_group, income, zip_code]\n"
        )

        tags = auto_tag("age_group has bimodal distribution", tmp_path)
        assert "feature:age_group" in tags
        assert "feature:income" not in tags

    def test_detects_experiment_ids(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        tags = auto_tag("Based on exp-005 results, we should try L2", tmp_path)
        assert "experiment:exp-005" in tags

    def test_detects_source_names(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        import json

        (data_dir / "source_registry.json").write_text(
            json.dumps({"housing_csv": {"path": "data/raw/housing.csv"}})
        )

        tags = auto_tag("housing_csv has missing values in 3 columns", tmp_path)
        assert "source:housing_csv" in tags

    def test_case_insensitive(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "models.yaml").write_text(
            "models:\n  XGB_Main:\n    type: xgboost\n"
        )

        tags = auto_tag("xgb_main is overfitting", tmp_path)
        assert "model:XGB_Main" in tags

    def test_no_config_dir_returns_only_experiment_ids(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        tags = auto_tag("exp-001 showed improvement", tmp_path)
        assert "experiment:exp-001" in tags

    def test_no_duplicates(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        tags = auto_tag("exp-001 and exp-001 again", tmp_path)
        assert tags.count("experiment:exp-001") == 1

    def test_multiple_entity_types(self, tmp_path):
        from harnessml.core.runner.notebook.tagger import auto_tag

        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "models.yaml").write_text(
            "models:\n  xgb_main:\n    type: xgboost\n    features: [age]\n"
        )

        tags = auto_tag("xgb_main uses age, per exp-003", tmp_path)
        assert "model:xgb_main" in tags
        assert "feature:age" in tags
        assert "experiment:exp-003" in tags
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/test_tagger.py -v`
Expected: FAIL — module not found

**Step 3: Write the auto-tagger**

```python
# packages/harness-core/src/harnessml/core/runner/notebook/tagger.py
"""Auto-tag notebook entries by detecting entity references in content."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


def auto_tag(content: str, project_dir: Path) -> list[str]:
    """Scan content for known entity names and return tags.

    Detects: model names, feature names, source names, experiment IDs.
    """
    tags: list[str] = []
    content_lower = content.lower()

    # Models and features from config
    model_names, feature_names = _load_models_and_features(project_dir)
    for name in model_names:
        if name.lower() in content_lower:
            tags.append(f"model:{name}")
    for name in feature_names:
        if name.lower() in content_lower:
            tags.append(f"feature:{name}")

    # Sources from registry
    for name in _load_source_names(project_dir):
        if name.lower() in content_lower:
            tags.append(f"source:{name}")

    # Experiment IDs (exp-NNN pattern)
    for match in re.finditer(r"\bexp-\d+\b", content, re.IGNORECASE):
        tags.append(f"experiment:{match.group()}")

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)
    return unique


def _load_models_and_features(project_dir: Path) -> tuple[list[str], list[str]]:
    """Load model and feature names from config."""
    models: list[str] = []
    features: set[str] = set()

    config_dir = project_dir / "config"
    if not config_dir.exists():
        return models, list(features)

    # Single models.yaml
    models_yaml = config_dir / "models.yaml"
    if models_yaml.exists():
        try:
            data = yaml.safe_load(models_yaml.read_text()) or {}
            models_block = data.get("models", data)
            if isinstance(models_block, dict):
                for name, cfg in models_block.items():
                    models.append(name)
                    if isinstance(cfg, dict):
                        for f in cfg.get("features", []):
                            features.add(f)
        except Exception:
            pass

    # Per-model YAML files in config/models/
    models_dir = config_dir / "models"
    if models_dir.is_dir():
        for p in models_dir.glob("*.yaml"):
            models.append(p.stem)
            try:
                data = yaml.safe_load(p.read_text()) or {}
                for f in data.get("features", []):
                    features.add(f)
            except Exception:
                pass

    return models, list(features)


def _load_source_names(project_dir: Path) -> list[str]:
    """Load source names from source_registry.json."""
    registry_path = project_dir / "data" / "source_registry.json"
    if not registry_path.exists():
        return []
    try:
        data = json.loads(registry_path.read_text())
        if isinstance(data, dict):
            return list(data.keys())
    except Exception:
        pass
    return []
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/test_tagger.py -v`
Expected: PASS (all 9 tests)

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/runner/notebook/tagger.py packages/harness-core/tests/runner/notebook/test_tagger.py
git commit -m "feat(notebook): add auto-tagger for entity detection"
```

---

### Task 3: Notebook Store (JSONL read/write/search/strike)

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/notebook/store.py`
- Test: `packages/harness-core/tests/runner/notebook/test_store.py`

**Step 1: Write the failing tests**

```python
# tests/runner/notebook/test_store.py
import pytest
from pathlib import Path


class TestNotebookStore:
    @pytest.fixture
    def store(self, tmp_path):
        from harnessml.core.runner.notebook.store import NotebookStore

        return NotebookStore(tmp_path)

    def test_write_creates_file_and_returns_entry(self, store):
        entry = store.write(type="finding", content="Test finding", tags=["model:xgb"])
        assert entry.id.startswith("nb-")
        assert entry.type.value == "finding"
        assert entry.content == "Test finding"
        assert "model:xgb" in entry.tags
        assert (store.project_dir / "notebook" / "entries.jsonl").exists()

    def test_write_auto_increments_id(self, store):
        e1 = store.write(type="finding", content="First")
        e2 = store.write(type="finding", content="Second")
        # Extract numeric part
        n1 = int(e1.id.split("-")[1])
        n2 = int(e2.id.split("-")[1])
        assert n2 == n1 + 1

    def test_write_auto_tags(self, store):
        # Create config with a model
        config_dir = store.project_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "models.yaml").write_text(
            "models:\n  xgb_main:\n    type: xgboost\n"
        )

        entry = store.write(type="finding", content="xgb_main overfits on fold 3")
        assert "model:xgb_main" in entry.auto_tags

    def test_write_with_experiment_id(self, store):
        entry = store.write(
            type="finding", content="Result", experiment_id="exp-003"
        )
        assert entry.experiment_id == "exp-003"

    def test_read_all_excludes_struck(self, store):
        store.write(type="finding", content="Visible")
        e2 = store.write(type="finding", content="Will be struck")
        store.strike(e2.id, reason="Outdated")

        entries = store.read()
        assert len(entries) == 1
        assert entries[0].content == "Visible"

    def test_read_by_type(self, store):
        store.write(type="finding", content="A finding")
        store.write(type="theory", content="A theory")
        store.write(type="finding", content="Another finding")

        entries = store.read(type="finding")
        assert len(entries) == 2
        assert all(e.type.value == "finding" for e in entries)

    def test_read_by_tags(self, store):
        store.write(type="finding", content="About xgb", tags=["model:xgb"])
        store.write(type="finding", content="About rf", tags=["model:rf"])
        store.write(type="finding", content="About xgb again", tags=["model:xgb"])

        entries = store.read(tags=["model:xgb"])
        assert len(entries) == 2

    def test_read_by_tags_matches_auto_tags(self, store):
        config_dir = store.project_dir / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "models.yaml").write_text(
            "models:\n  xgb_main:\n    type: xgboost\n"
        )

        store.write(type="finding", content="xgb_main is great")
        entries = store.read(tags=["model:xgb_main"])
        assert len(entries) == 1

    def test_read_pagination(self, store):
        for i in range(15):
            store.write(type="finding", content=f"Entry {i}")

        page1 = store.read(page=1, per_page=10)
        page2 = store.read(page=2, per_page=10)
        assert len(page1) == 10
        assert len(page2) == 5

    def test_read_newest_first(self, store):
        store.write(type="finding", content="First")
        store.write(type="finding", content="Second")
        store.write(type="finding", content="Third")

        entries = store.read()
        assert entries[0].content == "Third"
        assert entries[2].content == "First"

    def test_search_full_text(self, store):
        store.write(type="finding", content="xgb overfits on fold 3")
        store.write(type="finding", content="rf is stable across folds")
        store.write(type="finding", content="fold 3 has temporal drift")

        results = store.search("fold 3")
        assert len(results) == 2

    def test_search_excludes_struck(self, store):
        e1 = store.write(type="finding", content="fold 3 issue")
        store.strike(e1.id, reason="Wrong")
        store.write(type="finding", content="fold 3 real issue")

        results = store.search("fold 3")
        assert len(results) == 1

    def test_search_case_insensitive(self, store):
        store.write(type="finding", content="XGB_MAIN overfits")

        results = store.search("xgb_main")
        assert len(results) == 1

    def test_strike_marks_entry(self, store):
        entry = store.write(type="finding", content="Old finding")
        store.strike(entry.id, reason="Superseded by nb-005")

        # Should not appear in normal read
        entries = store.read()
        assert len(entries) == 0

    def test_strike_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.strike("nb-999", reason="Test")

    def test_strike_already_struck_raises(self, store):
        entry = store.write(type="finding", content="To strike")
        store.strike(entry.id, reason="First strike")
        with pytest.raises(ValueError, match="already struck"):
            store.strike(entry.id, reason="Second strike")

    def test_summary(self, store):
        store.write(type="theory", content="Theory v1")
        store.write(type="theory", content="Theory v2 - updated")
        store.write(type="plan", content="Plan v1")
        for i in range(8):
            store.write(type="finding", content=f"Finding {i}")
        store.write(type="research", content="Research note")

        summary = store.summary()
        assert summary["latest_theory"] == "Theory v2 - updated"
        assert summary["latest_plan"] == "Plan v1"
        assert len(summary["recent_findings"]) == 5  # last 5 only
        assert summary["total_entries"] > 0
        assert summary["struck_entries"] == 0
        assert isinstance(summary["entity_index"], dict)

    def test_summary_empty_notebook(self, store):
        summary = store.summary()
        assert summary["latest_theory"] is None
        assert summary["latest_plan"] is None
        assert summary["recent_findings"] == []
        assert summary["total_entries"] == 0
        assert summary["struck_entries"] == 0

    def test_summary_entity_index(self, store):
        store.write(type="finding", content="About xgb", tags=["model:xgb"])
        store.write(type="finding", content="More xgb", tags=["model:xgb"])
        store.write(type="finding", content="About rf", tags=["model:rf"])

        summary = store.summary()
        index = summary["entity_index"]
        assert index["model:xgb"] == 2
        assert index["model:rf"] == 1

    def test_invalid_type_rejected(self, store):
        with pytest.raises(ValueError):
            store.write(type="invalid_type", content="Bad entry")

    def test_read_all_including_struck(self, store):
        store.write(type="finding", content="Visible")
        e2 = store.write(type="finding", content="Struck")
        store.strike(e2.id, reason="Test")

        entries = store.read_all(include_struck=True)
        assert len(entries) == 2

        struck = [e for e in entries if e.struck]
        assert len(struck) == 1
        assert struck[0].struck_reason == "Test"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/test_store.py -v`
Expected: FAIL — module not found

**Step 3: Write the store**

```python
# packages/harness-core/src/harnessml/core/runner/notebook/store.py
"""JSONL-backed notebook store with read, write, search, strike, and summary."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from harnessml.core.runner.notebook.schema import EntryType, NotebookEntry
from harnessml.core.runner.notebook.tagger import auto_tag


class NotebookStore:
    """Manages the project notebook stored as JSONL."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)
        self._dir = self.project_dir / "notebook"
        self._path = self._dir / "entries.jsonl"

    # ── write ──────────────────────────────────────────────────────────

    def write(
        self,
        *,
        type: str,
        content: str,
        tags: list[str] | None = None,
        experiment_id: str | None = None,
    ) -> NotebookEntry:
        """Append a new entry. Returns the created entry."""
        entry_type = EntryType(type)  # raises ValueError for invalid type
        entry_id = self._next_id()
        detected = auto_tag(content, self.project_dir)

        entry = NotebookEntry(
            id=entry_id,
            type=entry_type,
            content=content,
            tags=tags or [],
            auto_tags=detected,
            experiment_id=experiment_id,
        )

        self._dir.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    # ── read ───────────────────────────────────────────────────────────

    def read(
        self,
        *,
        type: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        per_page: int = 10,
    ) -> list[NotebookEntry]:
        """Read entries, excluding struck. Newest first. Filterable."""
        entries = [e for e in self._load_latest() if not e.struck]

        if type is not None:
            entry_type = EntryType(type)
            entries = [e for e in entries if e.type == entry_type]

        if tags:
            entries = [
                e
                for e in entries
                if any(t in e.tags or t in e.auto_tags for t in tags)
            ]

        # Newest first
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        # Paginate
        start = (page - 1) * per_page
        return entries[start : start + per_page]

    def read_all(self, *, include_struck: bool = False) -> list[NotebookEntry]:
        """Read all entries, optionally including struck ones."""
        entries = self._load_latest()
        if not include_struck:
            entries = [e for e in entries if not e.struck]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    # ── search ─────────────────────────────────────────────────────────

    def search(self, query: str) -> list[NotebookEntry]:
        """Full-text search across non-struck entries. Case-insensitive."""
        query_lower = query.lower()
        results = [
            e
            for e in self._load_latest()
            if not e.struck and query_lower in e.content.lower()
        ]
        results.sort(key=lambda e: e.timestamp, reverse=True)
        return results

    # ── strike ─────────────────────────────────────────────────────────

    def strike(self, entry_id: str, *, reason: str) -> NotebookEntry:
        """Strike through an entry. Appends a new record with struck=True."""
        latest = self._load_latest()
        entry = None
        for e in latest:
            if e.id == entry_id:
                entry = e
                break

        if entry is None:
            raise ValueError(f"Entry '{entry_id}' not found")
        if entry.struck:
            raise ValueError(f"Entry '{entry_id}' is already struck")

        struck_entry = entry.model_copy(
            update={
                "struck": True,
                "struck_reason": reason,
                "struck_at": datetime.now(timezone.utc),
            }
        )

        with open(self._path, "a") as f:
            f.write(struck_entry.model_dump_json() + "\n")

        return struck_entry

    # ── summary ────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return notebook summary for session start."""
        entries = self._load_latest()
        active = [e for e in entries if not e.struck]
        struck = [e for e in entries if e.struck]

        # Latest theory and plan (superseding types)
        latest_theory = None
        latest_plan = None
        for e in sorted(active, key=lambda e: e.timestamp, reverse=True):
            if e.type == EntryType.THEORY and latest_theory is None:
                latest_theory = e.content
            if e.type == EntryType.PLAN and latest_plan is None:
                latest_plan = e.content
            if latest_theory is not None and latest_plan is not None:
                break

        # Recent findings (last 5)
        findings = [e for e in active if e.type == EntryType.FINDING]
        findings.sort(key=lambda e: e.timestamp, reverse=True)
        recent_findings = [e.content for e in findings[:5]]

        # Entity index: tag -> count across active entries
        entity_index: dict[str, int] = {}
        for e in active:
            for tag in set(e.tags + e.auto_tags):
                entity_index[tag] = entity_index.get(tag, 0) + 1

        return {
            "latest_theory": latest_theory,
            "latest_plan": latest_plan,
            "recent_findings": recent_findings,
            "total_entries": len(active),
            "struck_entries": len(struck),
            "entity_index": entity_index,
        }

    # ── internals ──────────────────────────────────────────────────────

    def _load_latest(self) -> list[NotebookEntry]:
        """Load all entries, keeping latest version per ID."""
        if not self._path.exists():
            return []

        by_id: dict[str, NotebookEntry] = {}
        for line in self._path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = NotebookEntry.model_validate_json(line)
                by_id[entry.id] = entry
            except Exception:
                continue
        return list(by_id.values())

    def _next_id(self) -> str:
        """Generate the next entry ID."""
        entries = self._load_latest()
        if not entries:
            return "nb-001"
        max_num = 0
        for e in entries:
            try:
                num = int(e.id.split("-")[1])
                max_num = max(max_num, num)
            except (IndexError, ValueError):
                continue
        return f"nb-{max_num + 1:03d}"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/test_store.py -v`
Expected: PASS (all 21 tests)

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/runner/notebook/store.py packages/harness-core/tests/runner/notebook/test_store.py
git commit -m "feat(notebook): add JSONL-backed notebook store"
```

---

### Task 4: MCP Handler

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py`
- Test: `packages/harness-plugin/tests/handlers/test_notebook.py`

**Step 1: Write the failing tests**

```python
# packages/harness-plugin/tests/handlers/test_notebook.py
import pytest
from pathlib import Path


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory."""
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(
        "models:\n  xgb_main:\n    type: xgboost\n    features: [age, income]\n"
    )
    return str(tmp_path)


class TestNotebookHandler:
    def test_write_action(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        result = dispatch(
            "write",
            type="finding",
            content="xgb_main overfits on fold 3",
            project_dir=project_dir,
        )
        assert "nb-001" in result
        assert "finding" in result.lower()
        assert "model:xgb_main" in result  # auto-tag shown

    def test_write_requires_content(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        result = dispatch(
            "write", type="finding", content="", project_dir=project_dir
        )
        assert "error" in result.lower()

    def test_write_requires_valid_type(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        result = dispatch(
            "write", type="garbage", content="Test", project_dir=project_dir
        )
        assert "error" in result.lower()

    def test_read_action(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch("write", type="finding", content="Entry 1", project_dir=project_dir)
        dispatch("write", type="theory", content="Theory 1", project_dir=project_dir)

        result = dispatch("read", project_dir=project_dir)
        assert "Theory 1" in result
        assert "Entry 1" in result

    def test_read_filtered_by_type(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch("write", type="finding", content="A finding", project_dir=project_dir)
        dispatch("write", type="theory", content="A theory", project_dir=project_dir)

        result = dispatch("read", type="finding", project_dir=project_dir)
        assert "A finding" in result
        assert "A theory" not in result

    def test_read_filtered_by_tags(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch(
            "write",
            type="finding",
            content="About xgb",
            tags='["model:xgb"]',
            project_dir=project_dir,
        )
        dispatch(
            "write",
            type="finding",
            content="About rf",
            tags='["model:rf"]',
            project_dir=project_dir,
        )

        result = dispatch("read", tags='["model:xgb"]', project_dir=project_dir)
        assert "About xgb" in result
        assert "About rf" not in result

    def test_search_action(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch("write", type="finding", content="Fold 3 is weak", project_dir=project_dir)
        dispatch("write", type="finding", content="Fold 5 is strong", project_dir=project_dir)

        result = dispatch("search", query="fold 3", project_dir=project_dir)
        assert "Fold 3 is weak" in result
        assert "Fold 5 is strong" not in result

    def test_search_requires_query(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        result = dispatch("search", project_dir=project_dir)
        assert "error" in result.lower()

    def test_strike_action(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch("write", type="finding", content="Old finding", project_dir=project_dir)
        result = dispatch(
            "strike",
            entry_id="nb-001",
            reason="Superseded",
            project_dir=project_dir,
        )
        assert "struck" in result.lower() or "nb-001" in result

        # Verify it's hidden from reads
        read_result = dispatch("read", project_dir=project_dir)
        assert "Old finding" not in read_result

    def test_strike_requires_entry_id(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        result = dispatch("strike", reason="Test", project_dir=project_dir)
        assert "error" in result.lower()

    def test_strike_requires_reason(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch("write", type="finding", content="Entry", project_dir=project_dir)
        result = dispatch("strike", entry_id="nb-001", project_dir=project_dir)
        assert "error" in result.lower()

    def test_summary_action(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        dispatch("write", type="theory", content="Target driven by X", project_dir=project_dir)
        dispatch("write", type="plan", content="Try feature Y next", project_dir=project_dir)
        dispatch("write", type="finding", content="Finding 1", project_dir=project_dir)

        result = dispatch("summary", project_dir=project_dir)
        assert "Target driven by X" in result
        assert "Try feature Y next" in result
        assert "Finding 1" in result

    def test_summary_empty_notebook(self, project_dir):
        from harnessml.plugin.handlers.notebook import dispatch

        result = dispatch("summary", project_dir=project_dir)
        assert "no entries" in result.lower() or "empty" in result.lower()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harness-plugin/tests/handlers/test_notebook.py -v`
Expected: FAIL — module not found

**Step 3: Write the handler**

```python
# packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py
"""MCP handler for the project notebook."""

from __future__ import annotations

from harnessml.plugin.handlers._common import parse_json_param, resolve_project_dir
from harnessml.plugin.handlers._validation import (
    collect_hints,
    format_response_with_hints,
    validate_enum,
    validate_required,
)

_ENTRY_TYPES = {"theory", "finding", "research", "decision", "plan", "note"}


def _handle_write(*, type, content, tags=None, experiment_id=None, project_dir, **_kw):
    err = validate_required(content, "content")
    if err:
        return err
    err = validate_enum(type, _ENTRY_TYPES, "type")
    if err:
        return err

    from harnessml.core.runner.notebook.store import NotebookStore

    tags_list = parse_json_param(tags, "tags") if tags else []
    if isinstance(tags_list, str):
        return tags_list  # parse error

    store = NotebookStore(resolve_project_dir(project_dir))
    try:
        entry = store.write(
            type=type,
            content=content,
            tags=tags_list,
            experiment_id=experiment_id,
        )
    except ValueError as e:
        return f"**Error**: {e}"

    all_tags = sorted(set(entry.tags + entry.auto_tags))
    lines = [
        f"**Notebook entry created**: `{entry.id}` ({entry.type.value})",
        "",
        f"> {entry.content[:200]}{'...' if len(entry.content) > 200 else ''}",
    ]
    if all_tags:
        lines.append(f"\n**Tags**: {', '.join(f'`{t}`' for t in all_tags)}")
    return "\n".join(lines)


def _handle_read(*, type=None, tags=None, page=None, per_page=None, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    if type is not None:
        err = validate_enum(type, _ENTRY_TYPES, "type")
        if err:
            return err

    tags_list = parse_json_param(tags, "tags") if tags else None
    if isinstance(tags_list, str) and tags is not None:
        return tags_list

    store = NotebookStore(resolve_project_dir(project_dir))
    entries = store.read(
        type=type,
        tags=tags_list,
        page=int(page) if page else 1,
        per_page=int(per_page) if per_page else 10,
    )

    if not entries:
        return "No notebook entries found matching filters."

    lines = [f"**Notebook** — {len(entries)} entries\n"]
    for e in entries:
        all_tags = sorted(set(e.tags + e.auto_tags))
        tag_str = f"  tags: {', '.join(f'`{t}`' for t in all_tags)}" if all_tags else ""
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M")
        exp_str = f"  exp: `{e.experiment_id}`" if e.experiment_id else ""
        lines.append(f"### `{e.id}` — {e.type.value} ({ts})")
        lines.append(f"{e.content}")
        if tag_str or exp_str:
            lines.append(f"\n{tag_str}{exp_str}")
        lines.append("")
    return "\n".join(lines)


def _handle_search(*, query=None, project_dir, **_kw):
    err = validate_required(query, "query")
    if err:
        return err

    from harnessml.core.runner.notebook.store import NotebookStore

    store = NotebookStore(resolve_project_dir(project_dir))
    results = store.search(query)

    if not results:
        return f"No notebook entries matching '{query}'."

    lines = [f"**Search**: '{query}' — {len(results)} results\n"]
    for e in results:
        ts = e.timestamp.strftime("%Y-%m-%d %H:%M")
        lines.append(f"### `{e.id}` — {e.type.value} ({ts})")
        lines.append(f"{e.content}")
        lines.append("")
    return "\n".join(lines)


def _handle_strike(*, entry_id=None, reason=None, project_dir, **_kw):
    err = validate_required(entry_id, "entry_id")
    if err:
        return err
    err = validate_required(reason, "reason")
    if err:
        return err

    from harnessml.core.runner.notebook.store import NotebookStore

    store = NotebookStore(resolve_project_dir(project_dir))
    try:
        entry = store.strike(entry_id, reason=reason)
    except ValueError as e:
        return f"**Error**: {e}"

    return f"**Struck** `{entry.id}`: ~~{entry.content[:100]}~~\n\nReason: {reason}"


def _handle_summary(*, project_dir, **_kw):
    from harnessml.core.runner.notebook.store import NotebookStore

    store = NotebookStore(resolve_project_dir(project_dir))
    s = store.summary()

    if s["total_entries"] == 0:
        return "**Notebook**: Empty — no entries yet."

    lines = ["**Notebook Summary**\n"]

    lines.append("### Current Theory")
    lines.append(s["latest_theory"] or "_No theory entry yet._")
    lines.append("")

    lines.append("### Current Plan")
    lines.append(s["latest_plan"] or "_No plan entry yet._")
    lines.append("")

    if s["recent_findings"]:
        lines.append("### Recent Findings")
        for f in s["recent_findings"]:
            lines.append(f"- {f}")
        lines.append("")

    lines.append(f"**Total entries**: {s['total_entries']} active, {s['struck_entries']} struck")

    if s["entity_index"]:
        lines.append("\n### Entity Index")
        sorted_entities = sorted(s["entity_index"].items(), key=lambda x: -x[1])
        for tag, count in sorted_entities[:20]:
            lines.append(f"- `{tag}`: {count} entries")

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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harness-plugin/tests/handlers/test_notebook.py -v`
Expected: PASS (all 14 tests)

**Step 5: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py packages/harness-plugin/tests/handlers/test_notebook.py
git commit -m "feat(notebook): add MCP handler with write/read/search/strike/summary"
```

---

### Task 5: MCP Tool Registration

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/mcp_server.py` (add new tool function)

**Step 1: Add the `notebook` tool**

Add after the existing tool registrations (after the `competitions` tool). Follow the exact pattern of other tools:

```python
@mcp.tool()
@_safe_tool
async def notebook(
    action: str,
    ctx: Context,
    type: str | None = None,
    content: str | None = None,
    tags: str | None = None,
    query: str | None = None,
    entry_id: str | None = None,
    reason: str | None = None,
    experiment_id: str | None = None,
    page: int | None = None,
    per_page: int | None = None,
    project_dir: str | None = None,
) -> str:
    """Project notebook for persistent learnings across sessions.

    Actions:
      - "write": Add an entry. Requires type + content.
        type: theory | finding | research | decision | plan | note
        content: the entry text
        tags: optional JSON list of tags, e.g. '["model:xgb"]'
        experiment_id: optional link to an experiment
      - "read": Read entries (newest first, excludes struck).
        type: filter by entry type
        tags: filter by tags (JSON list)
        page: page number (default 1)
        per_page: entries per page (default 10)
      - "search": Full-text search. Requires query.
      - "strike": Hide an entry with a reason. Requires entry_id + reason.
      - "summary": Get current theory, plan, recent findings, and entity index.
        Call this at session start.
    """
    return _load_handler("notebook").dispatch(
        action,
        ctx=ctx,
        type=type,
        content=content,
        tags=tags,
        query=query,
        entry_id=entry_id,
        reason=reason,
        experiment_id=experiment_id,
        page=page,
        per_page=per_page,
        project_dir=project_dir,
    )
```

**Step 2: Verify MCP server still loads**

Run: `uv run python -c "from harnessml.plugin.mcp_server import mcp; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/mcp_server.py
git commit -m "feat(notebook): register notebook MCP tool"
```

---

### Task 6: Studio Route

**Files:**
- Create: `packages/harness-studio/src/harnessml/studio/routes/notebook.py`
- Modify: `packages/harness-studio/src/harnessml/studio/server.py` (register router)
- Test: `packages/harness-studio/tests/test_notebook_route.py`

**Step 1: Write the failing tests**

```python
# packages/harness-studio/tests/test_notebook_route.py
import json
import pytest
from pathlib import Path


@pytest.fixture
def project_with_notebook(tmp_path):
    """Create a project dir with notebook entries."""
    (tmp_path / "config").mkdir()
    notebook_dir = tmp_path / "notebook"
    notebook_dir.mkdir()

    entries = [
        {
            "id": "nb-001",
            "type": "theory",
            "timestamp": "2026-03-09T10:00:00Z",
            "content": "Target driven by temporal patterns",
            "tags": [],
            "auto_tags": ["feature:age"],
            "struck": False,
            "struck_reason": None,
            "struck_at": None,
            "experiment_id": None,
        },
        {
            "id": "nb-002",
            "type": "finding",
            "timestamp": "2026-03-09T11:00:00Z",
            "content": "xgb_main overfits fold 3",
            "tags": ["model:xgb_main"],
            "auto_tags": ["model:xgb_main"],
            "struck": False,
            "struck_reason": None,
            "struck_at": None,
            "experiment_id": "exp-005",
        },
        {
            "id": "nb-003",
            "type": "finding",
            "timestamp": "2026-03-09T12:00:00Z",
            "content": "Old finding",
            "tags": [],
            "auto_tags": [],
            "struck": True,
            "struck_reason": "Superseded",
            "struck_at": "2026-03-09T13:00:00Z",
            "experiment_id": None,
        },
    ]

    with open(notebook_dir / "entries.jsonl", "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return tmp_path


class TestNotebookRoute:
    def test_list_entries_returns_all_including_struck(self, project_with_notebook):
        from harnessml.studio.routes.notebook import _list_entries

        entries = _list_entries(project_with_notebook)
        assert len(entries) == 3

    def test_list_entries_newest_first(self, project_with_notebook):
        from harnessml.studio.routes.notebook import _list_entries

        entries = _list_entries(project_with_notebook)
        assert entries[0]["id"] == "nb-003"

    def test_list_entries_empty_project(self, tmp_path):
        from harnessml.studio.routes.notebook import _list_entries

        (tmp_path / "config").mkdir()
        entries = _list_entries(tmp_path)
        assert entries == []

    def test_struck_entries_have_metadata(self, project_with_notebook):
        from harnessml.studio.routes.notebook import _list_entries

        entries = _list_entries(project_with_notebook)
        struck = [e for e in entries if e.get("struck")]
        assert len(struck) == 1
        assert struck[0]["struck_reason"] == "Superseded"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harness-studio/tests/test_notebook_route.py -v`
Expected: FAIL — module not found

**Step 3: Write the route**

```python
# packages/harness-studio/src/harnessml/studio/routes/notebook.py
"""Studio route for project notebook entries."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

from harnessml.studio.routes.project import resolve_project_dir_from_request

router = APIRouter(tags=["notebook"])


def _list_entries(project_dir: Path) -> list[dict]:
    """Load all notebook entries (including struck) for Studio UI."""
    entries_path = project_dir / "notebook" / "entries.jsonl"
    if not entries_path.exists():
        return []

    by_id: dict[str, dict] = {}
    for line in entries_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            by_id[entry["id"]] = entry
        except (json.JSONDecodeError, KeyError):
            continue

    entries = list(by_id.values())
    entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return entries


@router.get("/notebook")
async def list_notebook_entries(request: Request, project: str | None = None):
    """Return all notebook entries including struck ones (for history UI)."""
    project_dir = resolve_project_dir_from_request(request, project)
    return _list_entries(project_dir)


@router.get("/notebook/{entry_id}")
async def get_notebook_entry(
    request: Request, entry_id: str, project: str | None = None
):
    """Return a single notebook entry by ID."""
    project_dir = resolve_project_dir_from_request(request, project)
    entries = _list_entries(project_dir)
    for entry in entries:
        if entry.get("id") == entry_id:
            return entry
    return {"error": f"Entry '{entry_id}' not found"}
```

**Step 4: Register the router in server.py**

Find where other routers are included (look for `app.include_router`) and add:

```python
from harnessml.studio.routes.notebook import router as notebook_router
app.include_router(notebook_router, prefix="/api")
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/harness-studio/tests/test_notebook_route.py -v`
Expected: PASS (all 4 tests)

**Step 6: Commit**

```bash
git add packages/harness-studio/src/harnessml/studio/routes/notebook.py packages/harness-studio/src/harnessml/studio/server.py packages/harness-studio/tests/test_notebook_route.py
git commit -m "feat(notebook): add Studio route for notebook entries"
```

---

### Task 7: Update Skills to Reference Notebook

**Files:**
- Modify: `skills/mindset/SKILL.md`
- Modify: `skills/run-experiment/SKILL.md`
- Modify: `skills/synthesis/SKILL.md`
- Modify: `skills/diagnosis/SKILL.md`

**Step 1: Update mindset skill**

Add to the "The Rules" section, after the "Always" list:

```markdown
## Session Start

At the start of every session, call:
```
notebook(action="summary")
```

This gives you the current theory, current plan, recent findings, and an index of all entities mentioned in the notebook. Read it before doing anything else — this is what you know so far.

## Session End

Before a session ends, update the notebook:
- Write a `theory` entry if your understanding of the target has changed
- Write a `plan` entry with ranked next steps
- Write `finding` entries for anything you learned that should persist
```

**Step 2: Update run-experiment skill**

In the "Log What You Learned" section (Step 5), add after the `experiments(action="log_result")` call:

```markdown
Also record the learning in the project notebook for cross-experiment synthesis:

```
notebook(action="write", type="finding", content="...", experiment_id="exp-001")
```

The experiment journal captures the structured result. The notebook captures the insight — what this means for the bigger picture.
```

**Step 3: Update diagnosis skill**

At the end of the "Forming the Next Hypothesis" section, add:

```markdown
## Record the Diagnosis

Write key diagnostic findings to the notebook so they persist across sessions:

```
notebook(action="write", type="finding", content="[diagnosis finding]", experiment_id="...")
```

Not every diagnostic detail — just the insights that should inform future work.
```

**Step 4: Update synthesis skill**

In "The Synthesis Questions" section, after the synthesis template, add:

```markdown
## Persist the Synthesis

Write the synthesis to the notebook:

```
notebook(action="write", type="theory", content="[current understanding of what drives the target]")
notebook(action="write", type="plan", content="[ranked next steps with reasoning]")
```

These entries become the starting point for the next session. The theory entry supersedes the previous one — only the latest is shown in the summary.
```

**Step 5: Commit**

```bash
git add skills/
git commit -m "feat(notebook): reference notebook tool in skills"
```

---

### Task 8: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest packages/harness-core/tests/runner/notebook/ packages/harness-plugin/tests/handlers/test_notebook.py packages/harness-studio/tests/test_notebook_route.py -v`
Expected: PASS (all ~45 tests)

**Step 2: Run full project test suite to check for regressions**

Run: `uv run pytest -x -q`
Expected: PASS (all existing tests + new notebook tests)

**Step 3: Commit if any fixes were needed**

```bash
git commit -m "fix(notebook): address test issues"
```
