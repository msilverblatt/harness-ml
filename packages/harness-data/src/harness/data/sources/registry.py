"""Source registry — persists source declarations to sources.yaml."""

from __future__ import annotations
import yaml
from pathlib import Path
from harness.data.io import atomic_write_text
from harness.data.sources.protocol import SourceConfig


class SourceRegistry:
    """Manages source declarations. Persists to sources.yaml in the given directory."""

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._path = self._dir / "sources.yaml"
        self._sources: dict[str, SourceConfig] = {}
        self._load()

    def add(self, config: SourceConfig, *, overwrite: bool = False) -> None:
        if config.name in self._sources and not overwrite:
            raise ValueError(
                f"Source '{config.name}' already exists. Use overwrite=True to replace."
            )
        self._sources[config.name] = config
        self._save()

    def get(self, name: str) -> SourceConfig | None:
        return self._sources.get(name)

    def list_all(self) -> list[SourceConfig]:
        return list(self._sources.values())

    def remove(self, name: str) -> None:
        self._sources.pop(name, None)
        self._save()

    def _load(self) -> None:
        if not self._path.exists():
            return
        content = yaml.safe_load(self._path.read_text()) or {}
        sources = content.get("sources", {})
        for name, data in sources.items():
            data["name"] = name
            self._sources[name] = SourceConfig(**data)

    def _save(self) -> None:
        sources = {}
        for name, config in self._sources.items():
            d = config.model_dump(exclude_defaults=True)
            d.pop("name", None)
            sources[name] = d
        atomic_write_text(
            self._path,
            yaml.dump({"sources": sources}, default_flow_style=False, sort_keys=False),
        )
