"""View resolver — lazy DAG resolution with fingerprint-based caching.

Resolves views from their declarative definitions by walking the dependency
graph, loading sources, executing transform steps, and caching results.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from harnessml.core.runner.schema import DataConfig, SourceConfig, ViewDef
from harnessml.core.runner.view_executor import execute_step

logger = logging.getLogger(__name__)


class ViewResolver:
    """Lazy view resolver with fingerprint-based disk caching.

    Parameters
    ----------
    project_dir : Path
        Root project directory. Source file paths are resolved relative to this.
    config : DataConfig
        Data configuration containing sources, views, features_view.
    cache_dir : Path | None
        Where to cache materialized views. Defaults to project_dir/data/views/.cache
    """

    def __init__(
        self,
        project_dir: str | Path,
        config: DataConfig,
        cache_dir: Path | None = None,
    ) -> None:
        self.project_dir = Path(project_dir)
        self.config = config
        self._sources = config.sources
        self._views = config.views
        self._cache_dir = cache_dir or (self.project_dir / "data" / "views" / ".cache")

        # In-memory cache for current session
        self._memory_cache: dict[str, pd.DataFrame] = {}

    def resolve(self, name: str) -> pd.DataFrame:
        """Resolve a source or view name to a DataFrame.

        Resolution is lazy: results are cached in memory and on disk.
        A name can be either a source (raw file) or a view (transform chain).

        Raises ValueError if name is not found in sources or views.
        """
        # Check memory cache first
        if name in self._memory_cache:
            return self._memory_cache[name].copy()

        if name in self._sources:
            df = self._load_source(name)
        elif name in self._views:
            df = self._resolve_view(name)
        else:
            raise ValueError(
                f"Unknown name '{name}'. Not a source or view. "
                f"Sources: {sorted(self._sources)}, Views: {sorted(self._views)}"
            )

        self._memory_cache[name] = df
        return df.copy()

    def _load_source(self, name: str) -> pd.DataFrame:
        """Load a raw data source from disk."""
        source = self._sources[name]
        path = self._resolve_path(source.path)

        if not path.exists():
            raise FileNotFoundError(f"Source '{name}' file not found: {path}")

        fmt = source.format
        if fmt == "auto":
            fmt = path.suffix.lstrip(".").lower()
            if fmt in ("xlsx", "xls"):
                fmt = "excel"

        if fmt == "csv":
            return pd.read_csv(path)
        elif fmt == "parquet":
            return pd.read_parquet(path)
        elif fmt == "excel":
            return pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format '{fmt}' for source '{name}'")

    def _resolve_view(self, name: str) -> pd.DataFrame:
        """Resolve a view by executing its steps on its source."""
        view_def = self._views[name]

        # Check disk cache
        fingerprint = self._compute_fingerprint(name)
        cached_df = self._load_cache(name, fingerprint)
        if cached_df is not None:
            logger.debug("Cache hit for view '%s'", name)
            return cached_df

        # Resolve the source (recursion handles view chains)
        df = self.resolve(view_def.source)

        # Apply steps
        for step in view_def.steps:
            df = execute_step(df, step, resolver=self.resolve)

        # Cache to disk if enabled
        if view_def.cache:
            self._save_cache(name, fingerprint, df)

        return df

    def _compute_fingerprint(self, name: str) -> str:
        """Compute a fingerprint for a view based on its definition + upstream state."""
        view_def = self._views[name]
        parts: list[str] = []

        # View definition itself
        parts.append(json.dumps(view_def.model_dump(mode="json"), sort_keys=True))

        # Upstream fingerprints
        deps = self._get_direct_deps(name)
        for dep_name in sorted(deps):
            if dep_name in self._sources:
                # Source fingerprint = file mtime + size
                source = self._sources[dep_name]
                path = self._resolve_path(source.path)
                if path.exists():
                    stat = path.stat()
                    parts.append(f"source:{dep_name}:{stat.st_mtime}:{stat.st_size}")
                else:
                    parts.append(f"source:{dep_name}:missing")
            elif dep_name in self._views:
                # Recursively get upstream fingerprint
                parts.append(f"view:{dep_name}:{self._compute_fingerprint(dep_name)}")

        h = hashlib.sha256("|".join(parts).encode())
        return h.hexdigest()[:16]

    def _get_direct_deps(self, name: str) -> set[str]:
        """Get the direct dependencies of a view."""
        if name not in self._views:
            return set()
        view_def = self._views[name]
        deps = {view_def.source}
        for step in view_def.steps:
            if hasattr(step, "other"):
                deps.add(step.other)
        return deps

    def dependency_graph(self) -> dict[str, set[str]]:
        """Return the full dependency graph for all views."""
        graph: dict[str, set[str]] = {}
        for name in self._views:
            graph[name] = self._get_direct_deps(name)
        return graph

    def invalidate(self, name: str) -> None:
        """Remove a view from memory and disk cache, plus all downstream views."""
        # Remove from memory
        self._memory_cache.pop(name, None)

        # Remove from disk
        if self._cache_dir.exists():
            for f in self._cache_dir.glob(f"{name}_*.parquet"):
                f.unlink()

        # Invalidate downstream views
        for view_name, view_deps in self.dependency_graph().items():
            if name in view_deps:
                self.invalidate(view_name)

    def _resolve_path(self, path: str | None) -> Path:
        """Resolve a path relative to project_dir."""
        if path is None:
            raise ValueError("Source path is None")
        p = Path(path)
        if not p.is_absolute():
            p = self.project_dir / p
        return p

    def _load_cache(self, name: str, fingerprint: str) -> pd.DataFrame | None:
        """Load from disk cache if fingerprint matches."""
        cache_path = self._cache_dir / f"{name}_{fingerprint}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def _save_cache(self, name: str, fingerprint: str, df: pd.DataFrame) -> None:
        """Save to disk cache."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Remove old cache files for this view
        for f in self._cache_dir.glob(f"{name}_*.parquet"):
            f.unlink()
        cache_path = self._cache_dir / f"{name}_{fingerprint}.parquet"
        df.to_parquet(cache_path, index=False)
