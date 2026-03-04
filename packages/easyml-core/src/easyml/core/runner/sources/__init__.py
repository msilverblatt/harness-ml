"""Data source registry with leakage metadata and freshness checks."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

from easyml.core.schemas.contracts import SourceMeta


class SourceRegistry:
    """Registry for data sources with leakage metadata and freshness tracking.

    Similar pattern to FeatureRegistry -- decorators register source functions
    with metadata, then run/inspect them later.
    """

    def __init__(self) -> None:
        self._sources: dict[str, SourceMeta] = {}
        self._functions: dict[str, Callable] = {}

    def register(
        self,
        name: str,
        category: str,
        outputs: list[str],
        temporal_safety: str,
        leakage_notes: str = "",
    ) -> Callable:
        """Decorator to register a data source function.

        The decorated function should accept (output_dir: Path, config: dict).
        """

        def decorator(fn: Callable) -> Callable:
            meta = SourceMeta(
                name=name,
                category=category,
                outputs=outputs,
                temporal_safety=temporal_safety,
                leakage_notes=leakage_notes,
            )
            self._sources[name] = meta
            self._functions[name] = fn
            return fn

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._sources

    def get_metadata(self, name: str) -> SourceMeta:
        """Return metadata for a registered source."""
        if name not in self._sources:
            raise KeyError(f"Source '{name}' not registered")
        return self._sources[name]

    def list_sources(self, category: str | None = None) -> list[SourceMeta]:
        """List all registered sources, optionally filtered by category."""
        sources = list(self._sources.values())
        if category is not None:
            sources = [s for s in sources if s.category == category]
        return sources

    def check_freshness(self, name: str) -> dict[str, Any]:
        """Check freshness of a source's output paths.

        Returns dict with:
        - exists: bool -- whether any output path exists
        - age_hours: float | None -- age of oldest output in hours
        - paths: dict mapping each output path to its status
        """
        if name not in self._sources:
            raise KeyError(f"Source '{name}' not registered")

        meta = self._sources[name]
        now = time.time()
        path_statuses: dict[str, dict[str, Any]] = {}
        any_exists = False
        oldest_mtime: float | None = None

        for output_path in meta.outputs:
            p = Path(output_path)
            if p.exists():
                any_exists = True
                # For directories, find the newest file inside
                if p.is_dir():
                    mtimes = []
                    for root, _dirs, files in os.walk(p):
                        for f in files:
                            fp = Path(root) / f
                            mtimes.append(fp.stat().st_mtime)
                    mtime = max(mtimes) if mtimes else p.stat().st_mtime
                else:
                    mtime = p.stat().st_mtime

                age_hours = (now - mtime) / 3600
                path_statuses[str(p)] = {"exists": True, "age_hours": age_hours}

                if oldest_mtime is None or mtime < oldest_mtime:
                    oldest_mtime = mtime
            else:
                path_statuses[str(p)] = {"exists": False, "age_hours": None}

        overall_age = (now - oldest_mtime) / 3600 if oldest_mtime is not None else None

        return {
            "exists": any_exists,
            "age_hours": overall_age,
            "paths": path_statuses,
        }

    def run(self, name: str, output_dir: Path | str, config: dict[str, Any]) -> None:
        """Execute a registered source function."""
        if name not in self._functions:
            raise KeyError(f"Source '{name}' not registered")

        output_dir = Path(output_dir)
        self._functions[name](output_dir, config)
