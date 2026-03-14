"""Tests for notebook entry Pydantic schemas."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from harnessml.core.runner.notebook.schema import EntryType, NotebookEntry


class TestEntryTypeEnum:
    def test_entry_type_enum(self):
        """All 6 entry types exist with correct values."""
        assert EntryType.THEORY == "theory"
        assert EntryType.FINDING == "finding"
        assert EntryType.RESEARCH == "research"
        assert EntryType.DECISION == "decision"
        assert EntryType.PLAN == "plan"
        assert EntryType.NOTE == "note"
        assert EntryType.PHASE_TRANSITION == "phase_transition"
        assert len(EntryType) == 7


class TestNotebookEntry:
    def test_create_entry_with_required_fields(self):
        """Entry can be created with only required fields; defaults are correct."""
        entry = NotebookEntry(
            id="nb-001",
            type=EntryType.NOTE,
            content="Some observation",
        )
        assert entry.id == "nb-001"
        assert entry.type == EntryType.NOTE
        assert entry.content == "Some observation"
        assert isinstance(entry.timestamp, datetime)
        assert entry.tags == []
        assert entry.auto_tags == []
        assert entry.struck is False
        assert entry.struck_reason is None
        assert entry.struck_at is None
        assert entry.experiment_id is None

    def test_create_entry_with_all_fields(self):
        """Entry can be created with every field explicitly set."""
        ts = datetime(2026, 3, 9, 12, 0, 0, tzinfo=timezone.utc)
        struck_ts = datetime(2026, 3, 9, 14, 0, 0, tzinfo=timezone.utc)
        entry = NotebookEntry(
            id="nb-002",
            type=EntryType.DECISION,
            timestamp=ts,
            content="Switching to XGBoost",
            tags=["model-selection"],
            auto_tags=["decision"],
            struck=True,
            struck_reason="Superseded by LightGBM decision",
            struck_at=struck_ts,
            experiment_id="exp-042",
        )
        assert entry.id == "nb-002"
        assert entry.type == EntryType.DECISION
        assert entry.timestamp == ts
        assert entry.content == "Switching to XGBoost"
        assert entry.tags == ["model-selection"]
        assert entry.auto_tags == ["decision"]
        assert entry.struck is True
        assert entry.struck_reason == "Superseded by LightGBM decision"
        assert entry.struck_at == struck_ts
        assert entry.experiment_id == "exp-042"

    def test_empty_content_rejected(self):
        """Empty string content is rejected."""
        with pytest.raises(ValueError, match="[Cc]ontent"):
            NotebookEntry(id="nb-003", type=EntryType.NOTE, content="")

    def test_whitespace_only_content_rejected(self):
        """Whitespace-only content is rejected."""
        with pytest.raises(ValueError, match="[Cc]ontent"):
            NotebookEntry(id="nb-004", type=EntryType.NOTE, content="   \n\t  ")

    def test_serialization_roundtrip(self):
        """Entry survives JSON serialization and deserialization."""
        entry = NotebookEntry(
            id="nb-005",
            type=EntryType.RESEARCH,
            content="Investigating momentum features",
            tags=["features"],
            experiment_id="exp-100",
        )
        data = entry.model_dump(mode="json")
        restored = NotebookEntry.model_validate(data)
        assert restored.id == entry.id
        assert restored.type == entry.type
        assert restored.content == entry.content
        assert restored.tags == entry.tags
        assert restored.experiment_id == entry.experiment_id

    def test_strike_fields(self):
        """Struck entry has reason and timestamp populated."""
        ts = datetime(2026, 3, 9, 15, 0, 0, tzinfo=timezone.utc)
        entry = NotebookEntry(
            id="nb-006",
            type=EntryType.THEORY,
            content="Home-court advantage is constant",
            struck=True,
            struck_reason="Disproved by data",
            struck_at=ts,
        )
        assert entry.struck is True
        assert entry.struck_reason == "Disproved by data"
        assert entry.struck_at == ts

    def test_content_is_stripped(self):
        """Leading/trailing whitespace is stripped from content."""
        entry = NotebookEntry(
            id="nb-007",
            type=EntryType.FINDING,
            content="  trimmed content  ",
        )
        assert entry.content == "trimmed content"

    def test_type_from_string_value(self):
        """Entry type can be provided as a raw string."""
        entry = NotebookEntry(
            id="nb-008",
            type="plan",
            content="Next steps",
        )
        assert entry.type == EntryType.PLAN
