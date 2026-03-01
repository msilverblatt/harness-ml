"""Feature builder — computes features with manifest-based incremental caching."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from easyml.features.manifest import FeatureManifest
from easyml.features.registry import FeatureRegistry


class FeatureBuilder:
    """Builds all registered features, caching results per source-hash.

    On each call to :meth:`build_all`, the builder checks a JSON manifest
    to see whether each feature's source code has changed since the last
    build.  Unchanged features are loaded from parquet cache; changed
    features are recomputed, cached, and the manifest is updated.
    """

    def __init__(
        self,
        *,
        registry: FeatureRegistry,
        cache_dir: Path,
        manifest_path: Path,
    ) -> None:
        self._registry = registry
        self._cache_dir = Path(cache_dir)
        self._manifest = FeatureManifest(manifest_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_all(
        self,
        raw_data: pd.DataFrame,
        config: dict[str, Any],
    ) -> pd.DataFrame:
        """Compute all registered features, returning a merged DataFrame.

        The result is joined on ``entity_id`` + ``period_id``.
        """
        feature_dfs: list[pd.DataFrame] = []
        join_cols = ["entity_id", "period_id"]

        for meta in self._registry.list_features():
            name = meta.name
            current_hash = self._registry.source_hash(name)

            cached_df = self._try_load_cache(name, current_hash)
            if cached_df is not None:
                feature_dfs.append(cached_df)
                continue

            # Cache miss — recompute
            fn = self._registry.get_fn(name)
            result_df = fn(raw_data, config)
            self._save_cache(name, current_hash, result_df)
            feature_dfs.append(result_df)

        self._manifest.save()

        # Merge all feature outputs
        if not feature_dfs:
            return raw_data[join_cols].copy()

        merged = feature_dfs[0]
        for df in feature_dfs[1:]:
            merged = merged.merge(df, on=join_cols, how="outer")

        return merged

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _try_load_cache(
        self, feature_name: str, current_hash: str
    ) -> pd.DataFrame | None:
        entry = self._manifest.get(feature_name)
        if entry is None:
            return None
        if entry["source_hash"] != current_hash:
            return None
        cache_path = Path(entry["cache_path"])
        if not cache_path.exists():
            return None
        return pd.read_parquet(cache_path)

    def _save_cache(
        self, feature_name: str, source_hash: str, df: pd.DataFrame
    ) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"{feature_name}.parquet"
        df.to_parquet(cache_path, index=False)
        self._manifest.set(
            feature_name,
            source_hash=source_hash,
            cache_path=str(cache_path),
        )
