"""JSONL-backed notebook store.

Append-only log where each line is a full NotebookEntry snapshot.
When an entry is struck, a new line is appended with struck fields set.
Reading always returns the latest snapshot per entry ID.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from harnessml.core.runner.notebook.schema import EntryType, NotebookEntry
from harnessml.core.runner.notebook.tagger import auto_tag


class NotebookStore:
    """Manages the project notebook as a JSONL file."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self._jsonl_path = project_dir / "notebook" / "entries.jsonl"

    # ── public API ────────────────────────────────────────────────────

    def write(
        self,
        *,
        type: str,
        content: str,
        tags: list[str] | None = None,
        experiment_id: str | None = None,
    ) -> NotebookEntry:
        """Create a new notebook entry and append it to the JSONL file."""
        # Validate type via enum (raises ValueError for invalid)
        entry_type = EntryType(type)

        entry_id = self._next_id()
        auto_tags = auto_tag(content, self.project_dir)

        entry = NotebookEntry(
            id=entry_id,
            type=entry_type,
            content=content,
            tags=tags or [],
            auto_tags=auto_tags,
            experiment_id=experiment_id,
        )

        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._jsonl_path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

        return entry

    def read(
        self,
        *,
        type: str | None = None,
        tags: list[str] | None = None,
        page: int = 1,
        per_page: int = 10,
    ) -> list[NotebookEntry]:
        """Read entries with optional type/tag filters, excluding struck entries.

        Returns newest first, paginated.
        """
        entries = [e for e in self._load_latest() if not e.struck]

        if type is not None:
            entry_type = EntryType(type)
            entries = [e for e in entries if e.type == entry_type]

        if tags is not None:
            tag_set = set(tags)
            entries = [
                e
                for e in entries
                if tag_set & (set(e.tags) | set(e.auto_tags))
            ]

        # Newest first
        entries.sort(key=lambda e: e.timestamp, reverse=True)

        # Paginate
        start = (page - 1) * per_page
        return entries[start : start + per_page]

    def read_all(self, *, include_struck: bool = False) -> list[NotebookEntry]:
        """Return all entries, optionally including struck ones. Newest first."""
        entries = self._load_latest()
        if not include_struck:
            entries = [e for e in entries if not e.struck]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def search(self, query: str) -> list[NotebookEntry]:
        """Case-insensitive full-text search on content, excluding struck entries."""
        query_lower = query.lower()
        entries = [
            e
            for e in self._load_latest()
            if not e.struck and query_lower in e.content.lower()
        ]
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def strike(self, entry_id: str, *, reason: str) -> NotebookEntry:
        """Strike an entry by appending a new JSONL line with struck fields."""
        latest = {e.id: e for e in self._load_latest()}
        entry = latest.get(entry_id)

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

        with open(self._jsonl_path, "a") as f:
            f.write(struck_entry.model_dump_json() + "\n")

        return struck_entry

    def summary(self) -> dict:
        """Return a summary of the notebook state."""
        all_entries = self._load_latest()
        active = [e for e in all_entries if not e.struck]
        struck = [e for e in all_entries if e.struck]

        # Latest theory and plan (by timestamp)
        theories = sorted(
            [e for e in active if e.type == EntryType.THEORY],
            key=lambda e: e.timestamp,
            reverse=True,
        )
        plans = sorted(
            [e for e in active if e.type == EntryType.PLAN],
            key=lambda e: e.timestamp,
            reverse=True,
        )
        findings = sorted(
            [e for e in active if e.type == EntryType.FINDING],
            key=lambda e: e.timestamp,
            reverse=True,
        )

        # Entity index: tag -> count across active entries (both tags and auto_tags)
        entity_index: dict[str, int] = {}
        for e in active:
            for tag in set(e.tags) | set(e.auto_tags):
                entity_index[tag] = entity_index.get(tag, 0) + 1

        return {
            "latest_theory": theories[0].content if theories else None,
            "latest_plan": plans[0].content if plans else None,
            "recent_findings": [e.content for e in findings[:5]],
            "total_entries": len(active),
            "struck_entries": len(struck),
            "entity_index": entity_index,
        }

    # ── internal helpers ──────────────────────────────────────────────

    def _load_latest(self) -> list[NotebookEntry]:
        """Read JSONL and keep the latest version per ID."""
        if not self._jsonl_path.exists():
            return []

        snapshots: dict[str, NotebookEntry] = {}
        for line in self._jsonl_path.read_text().splitlines():
            line = line.strip()
            if line:
                entry = NotebookEntry.model_validate_json(line)
                snapshots[entry.id] = entry

        return list(snapshots.values())

    def _next_id(self) -> str:
        """Find max numeric ID and increment, format as nb-NNN."""
        entries = self._load_latest()
        if not entries:
            return "nb-001"

        max_num = 0
        for e in entries:
            # Parse "nb-NNN" -> NNN
            parts = e.id.split("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                max_num = max(max_num, int(parts[1]))

        return f"nb-{max_num + 1:03d}"
