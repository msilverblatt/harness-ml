"""Content-addressable prediction cache.

Stores per-model, per-fold predictions keyed by fingerprint.

Layout::

    cache_dir/
        {model_name}/
            {fold_value}/
                {fingerprint}.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PredictionCache:
    """Cache for model predictions keyed by content fingerprint."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def lookup(
        self,
        model_name: str,
        fold_value: int,
        fingerprint: str,
    ) -> pd.DataFrame | None:
        """Return cached predictions or ``None`` on miss / corruption."""
        path = self._path(model_name, fold_value, fingerprint)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Corrupt cache entry, treating as miss: %s (%s)", path, exc
            )
            return None

    def store(
        self,
        model_name: str,
        fold_value: int,
        fingerprint: str,
        predictions: pd.DataFrame,
    ) -> Path:
        """Store predictions in the cache.  Overwrites if fingerprint exists."""
        path = self._path(model_name, fold_value, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(path, index=False)
        return path

    def prune(self, keep_last_n: int = 5) -> int:
        """Remove old cache entries per (model, fold_value), keeping newest *N*.

        Returns the number of entries removed.
        """
        removed = 0
        if not self.cache_dir.exists():
            return removed

        for model_dir in sorted(self.cache_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for fold_dir in sorted(model_dir.iterdir()):
                if not fold_dir.is_dir():
                    continue
                entries = sorted(
                    fold_dir.glob("*.parquet"),
                    key=lambda p: p.stat().st_mtime,
                )
                to_remove = entries[: max(0, len(entries) - keep_last_n)]
                for entry in to_remove:
                    entry.unlink()
                    removed += 1
        return removed

    def clear(self) -> int:
        """Remove all cached entries.  Returns count removed."""
        return self.prune(keep_last_n=0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path(self, model_name: str, fold_value: int, fingerprint: str) -> Path:
        return (
            self.cache_dir / model_name / str(fold_value) / f"{fingerprint}.parquet"
        )
