"""Pydantic schemas for the project notebook.

The notebook stores theories, findings, research notes, decisions, and plans.
Each entry is a self-contained NotebookEntry with metadata for filtering
and strikethrough support for invalidated entries.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class EntryType(str, Enum):
    """Type of notebook entry."""
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
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )
    content: str
    tags: list[str] = []
    auto_tags: list[str] = []
    struck: bool = False
    struck_reason: str | None = None
    struck_at: datetime | None = None
    experiment_id: str | None = None

    @field_validator("content")
    @classmethod
    def _validate_content(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Content must not be empty. Provide a meaningful notebook entry."
            )
        return v.strip()
