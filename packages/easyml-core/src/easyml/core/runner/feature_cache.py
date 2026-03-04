"""Content-addressed feature cache with manifest and cascade invalidation.

Stores computed feature Series as individual parquet files, organized by
feature type (team/, pairwise/, matchup/, regime/). A JSON manifest tracks
cache keys and dependency relationships for cascade invalidation.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_FEATURE_TYPES = ("team", "pairwise", "matchup", "regime")


@dataclass
class CacheEntry:
    """A single entry in the feature cache manifest."""
    cache_key: str
    path: str
    feature_type: str
    source: str | None = None
    derived_from: list[str] = field(default_factory=list)
    derivatives: list[str] = field(default_factory=list)


class FeatureCache:
    """Content-addressed cache for computed features.

    Features are stored as individual parquet files organized by type.
    A manifest.json tracks cache keys and dependency relationships.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._manifest_path = self._cache_dir / "manifest.json"
        self._entries: dict[str, CacheEntry] = {}

        # Create directory structure
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        for ft in _FEATURE_TYPES:
            (self._cache_dir / ft).mkdir(exist_ok=True)

        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load manifest from disk."""
        if self._manifest_path.exists():
            raw = json.loads(self._manifest_path.read_text())
            self._entries = {
                name: CacheEntry(**entry) for name, entry in raw.items()
            }

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        raw = {}
        for name, entry in self._entries.items():
            raw[name] = {
                "cache_key": entry.cache_key,
                "path": entry.path,
                "feature_type": entry.feature_type,
                "source": entry.source,
                "derived_from": entry.derived_from,
                "derivatives": entry.derivatives,
            }
        self._manifest_path.write_text(json.dumps(raw, indent=2))

    def compute_key(
        self,
        *,
        name: str,
        feature_type: str,
        source: str | None = None,
        column: str | None = None,
        formula: str | None = None,
        condition: str | None = None,
        pairwise_mode: str | None = None,
        extra: str = "",
    ) -> str:
        """Compute a deterministic cache key from feature definition."""
        components = [
            name,
            feature_type,
            source or "",
            column or "",
            formula or "",
            condition or "",
            pairwise_mode or "",
            extra,
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, name: str, expected_key: str) -> pd.Series | None:
        """Return cached feature Series if key matches, else None."""
        entry = self._entries.get(name)
        if entry is None or entry.cache_key != expected_key:
            return None

        parquet_path = self._cache_dir / entry.path
        if not parquet_path.exists():
            return None

        try:
            df = pd.read_parquet(parquet_path)
            return df.iloc[:, 0]
        except Exception:
            logger.warning("Failed to read cache for %s", name)
            return None

    def put(
        self,
        name: str,
        cache_key: str,
        data: pd.Series,
        *,
        feature_type: str,
        source: str | None = None,
        derived_from: list[str] | None = None,
        derivatives: list[str] | None = None,
    ) -> None:
        """Store a computed feature in the cache."""
        rel_path = f"{feature_type}/{name}.parquet"
        abs_path = self._cache_dir / rel_path

        # Save data as single-column parquet
        df = pd.DataFrame({name: data})
        df.to_parquet(abs_path, index=False)

        self._entries[name] = CacheEntry(
            cache_key=cache_key,
            path=rel_path,
            feature_type=feature_type,
            source=source,
            derived_from=derived_from or [],
            derivatives=derivatives or [],
        )
        self._save_manifest()

    def invalidate(self, name: str) -> None:
        """Invalidate a feature and cascade to its derivatives."""
        entry = self._entries.get(name)
        if entry is None:
            return

        # Cascade to derivatives
        for derivative in list(entry.derivatives):
            self.invalidate(derivative)

        # Remove cached file
        parquet_path = self._cache_dir / entry.path
        if parquet_path.exists():
            parquet_path.unlink()

        # Remove from manifest
        del self._entries[name]

        # Clean up parent's derivatives list
        for parent_name in entry.derived_from:
            parent = self._entries.get(parent_name)
            if parent and name in parent.derivatives:
                parent.derivatives.remove(name)

        self._save_manifest()

    def list_cached(self, *, feature_type: str | None = None) -> list[str]:
        """List cached feature names, optionally filtered by type."""
        names = []
        for name, entry in self._entries.items():
            if feature_type is None or entry.feature_type == feature_type:
                names.append(name)
        return sorted(names)
