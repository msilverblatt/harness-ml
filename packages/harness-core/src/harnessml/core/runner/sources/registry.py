"""Source registry -- track and manage data source definitions."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SourceDef:
    """Definition of a data source."""

    name: str
    source_type: str  # "file", "api", "url", "computed"
    path_pattern: str = ""
    refresh_frequency: str = "manual"  # hourly, daily, weekly, monthly, yearly, manual
    auth: dict[str, Any] = field(default_factory=dict)
    rate_limit: float = 0.0  # requests per second
    incremental: bool = False
    depends_on: list[str] = field(default_factory=list)
    schema: dict[str, Any] = field(default_factory=dict)  # required_columns, types
    leakage_notes: str = ""
    description: str = ""


# Fields that are omitted from YAML when they match their default (empty) value.
_OMIT_WHEN_EMPTY = ("auth", "depends_on", "schema", "leakage_notes", "description")


class SourceRegistry:
    """Registry of data sources with persistence to YAML."""

    def __init__(self, config_dir: Path):
        self._config_dir = Path(config_dir)
        self._sources_file = self._config_dir / "sources.yaml"
        self._sources: dict[str, SourceDef] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._sources_file.exists():
            data = yaml.safe_load(self._sources_file.read_text()) or {}
            for name, cfg in data.get("sources", {}).items():
                self._sources[name] = SourceDef(name=name, **cfg)

    def _save(self) -> None:
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {"sources": {}}
        for name, src in self._sources.items():
            d = asdict(src)
            d.pop("name")
            # Remove fields whose value matches the empty default
            for key in _OMIT_WHEN_EMPTY:
                if not d.get(key):
                    d.pop(key, None)
            data["sources"][name] = d
        self._sources_file.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False)
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, source: SourceDef) -> str:
        """Register a new source. Returns markdown confirmation or error."""
        if source.name in self._sources:
            return f"**Error**: Source '{source.name}' already exists. Use update instead."
        self._sources[source.name] = source
        self._save()
        return f"Added source **{source.name}** ({source.source_type})"

    def update(self, source: SourceDef) -> str:
        """Update an existing source definition."""
        if source.name not in self._sources:
            return f"**Error**: Source '{source.name}' not found."
        self._sources[source.name] = source
        self._save()
        return f"Updated source **{source.name}**"

    def get(self, name: str) -> SourceDef | None:
        return self._sources.get(name)

    def remove(self, name: str) -> str:
        """Remove a source by name. Returns markdown confirmation or error."""
        if name not in self._sources:
            return f"**Error**: Source '{name}' not found."
        del self._sources[name]
        self._save()
        return f"Removed source **{name}**"

    def list_all(self) -> list[SourceDef]:
        return list(self._sources.values())

    # ------------------------------------------------------------------
    # Dependency ordering
    # ------------------------------------------------------------------

    def topological_order(self) -> list[str]:
        """Return source names in dependency order (dependencies first)."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            src = self._sources.get(name)
            if src:
                for dep in src.depends_on:
                    visit(dep)
            order.append(name)

        for name in self._sources:
            visit(name)
        return order
