"""Feature manifest — tracks source hashes and cache paths for incremental builds."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FeatureManifest:
    """Persisted JSON mapping of feature_name -> {source_hash, cache_path}.

    Used by :class:`FeatureBuilder` to decide whether a cached parquet
    file is still valid or needs recomputation.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._entries: dict[str, dict[str, str]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                self._entries = json.load(f)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._entries, f, indent=2)

    # ------------------------------------------------------------------
    # Lookup / update
    # ------------------------------------------------------------------

    def get(self, feature_name: str) -> dict[str, str] | None:
        """Return entry for *feature_name*, or None if not tracked."""
        return self._entries.get(feature_name)

    def set(self, feature_name: str, source_hash: str, cache_path: str) -> None:
        """Create or update the manifest entry for *feature_name*."""
        self._entries[feature_name] = {
            "source_hash": source_hash,
            "cache_path": cache_path,
        }
