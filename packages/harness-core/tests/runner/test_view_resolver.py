"""Tests for the view resolver — DAG resolution with fingerprint caching."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from harnessml.core.runner.schema import (
    DataConfig,
    FilterStep,
    JoinStep,
    SourceConfig,
    ViewDef,
)
from harnessml.core.runner.views.resolver import ViewResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_project(tmp_path: Path) -> DataConfig:
    """Create raw CSV files and return a DataConfig with sources and views."""
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    df1 = pd.DataFrame({"id": [1, 2, 3], "score": [10, 20, 30]})
    df1.to_csv(raw_dir / "scores.csv", index=False)

    df2 = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    df2.to_csv(raw_dir / "names.csv", index=False)

    config = DataConfig(
        sources={
            "scores": SourceConfig(name="scores", path="data/raw/scores.csv"),
            "names": SourceConfig(name="names", path="data/raw/names.csv"),
        },
        views={
            "high_scores": ViewDef(
                source="scores",
                steps=[FilterStep(expr="score > 15")],
            ),
            "enriched": ViewDef(
                source="high_scores",
                steps=[JoinStep(other="names", on=["id"])],
            ),
        },
    )
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResolveSource:
    """test_resolve_source — Resolve a raw CSV source by name."""

    def test_resolve_source(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        resolver = ViewResolver(tmp_path, config)

        df = resolver.resolve("scores")
        assert list(df.columns) == ["id", "score"]
        assert len(df) == 3
        assert list(df["id"]) == [1, 2, 3]
        assert list(df["score"]) == [10, 20, 30]


class TestResolveSimpleView:
    """test_resolve_simple_view — View with filter step on a source."""

    def test_resolve_simple_view(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        resolver = ViewResolver(tmp_path, config)

        df = resolver.resolve("high_scores")
        assert len(df) == 2
        assert list(df["score"]) == [20, 30]
        assert list(df["id"]) == [2, 3]


class TestResolveChainedViews:
    """test_resolve_chained_views — A -> B -> C chain (B reads A, C reads B)."""

    def test_resolve_chained_views(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        resolver = ViewResolver(tmp_path, config)

        # "enriched" reads "high_scores" which reads "scores"
        df = resolver.resolve("enriched")
        assert len(df) == 2
        assert "name" in df.columns
        assert list(df["id"]) == [2, 3]
        assert list(df["name"]) == ["b", "c"]
        assert list(df["score"]) == [20, 30]


class TestResolveViewWithJoin:
    """test_resolve_view_with_join — View that joins two sources."""

    def test_resolve_view_with_join(self, tmp_path: Path):
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)

        pd.DataFrame({"id": [1, 2], "val": [100, 200]}).to_csv(
            raw_dir / "left.csv", index=False
        )
        pd.DataFrame({"id": [1, 2], "label": ["x", "y"]}).to_csv(
            raw_dir / "right.csv", index=False
        )

        config = DataConfig(
            sources={
                "left": SourceConfig(name="left", path="data/raw/left.csv"),
                "right": SourceConfig(name="right", path="data/raw/right.csv"),
            },
            views={
                "joined": ViewDef(
                    source="left",
                    steps=[JoinStep(other="right", on=["id"])],
                ),
            },
        )
        resolver = ViewResolver(tmp_path, config)
        df = resolver.resolve("joined")
        assert len(df) == 2
        assert list(df.columns) == ["id", "val", "label"]
        assert list(df["label"]) == ["x", "y"]


class TestMemoryCacheHit:
    """test_memory_cache_hit — Resolving same name twice returns cached copy."""

    def test_memory_cache_hit(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        resolver = ViewResolver(tmp_path, config)

        df1 = resolver.resolve("scores")
        df2 = resolver.resolve("scores")

        # Both return equal data
        pd.testing.assert_frame_equal(df1, df2)

        # But they are copies (mutating one does not affect the other)
        df1["score"] = [999, 999, 999]
        df3 = resolver.resolve("scores")
        assert list(df3["score"]) == [10, 20, 30]


class TestDiskCacheHit:
    """test_disk_cache_hit — Create new resolver instance, cached parquet is found."""

    def test_disk_cache_hit(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        cache_dir = tmp_path / "cache"

        # First resolver materializes and caches
        resolver1 = ViewResolver(tmp_path, config, cache_dir=cache_dir)
        df1 = resolver1.resolve("high_scores")

        # Verify cache file was written
        cache_files = list(cache_dir.glob("high_scores_*.parquet"))
        assert len(cache_files) == 1

        # Second resolver (fresh instance) should find disk cache
        resolver2 = ViewResolver(tmp_path, config, cache_dir=cache_dir)
        df2 = resolver2.resolve("high_scores")

        pd.testing.assert_frame_equal(df1, df2)


class TestFingerprintInvalidation:
    """test_fingerprint_invalidation — Modify source file, fingerprint changes, recomputes."""

    def test_fingerprint_invalidation(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        cache_dir = tmp_path / "cache"

        # First resolve to populate cache
        resolver1 = ViewResolver(tmp_path, config, cache_dir=cache_dir)
        df1 = resolver1.resolve("high_scores")
        assert len(df1) == 2

        # Get cached file name
        cache_files_before = list(cache_dir.glob("high_scores_*.parquet"))
        assert len(cache_files_before) == 1
        old_name = cache_files_before[0].name

        # Small delay so mtime actually changes
        time.sleep(0.05)

        # Modify the source file (add a row with score > 15)
        raw_path = tmp_path / "data" / "raw" / "scores.csv"
        df_modified = pd.DataFrame({"id": [1, 2, 3, 4], "score": [10, 20, 30, 40]})
        df_modified.to_csv(raw_path, index=False)

        # New resolver — fingerprint should differ, forcing recompute
        resolver2 = ViewResolver(tmp_path, config, cache_dir=cache_dir)
        df2 = resolver2.resolve("high_scores")
        assert len(df2) == 3  # now 20, 30, 40 pass the filter

        # Cache file should have a different fingerprint name
        cache_files_after = list(cache_dir.glob("high_scores_*.parquet"))
        assert len(cache_files_after) == 1
        new_name = cache_files_after[0].name
        assert old_name != new_name


class TestUnknownNameRaises:
    """test_unknown_name_raises — Resolving unknown name raises ValueError."""

    def test_unknown_name_raises(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        resolver = ViewResolver(tmp_path, config)

        with pytest.raises(ValueError, match="Unknown name 'nonexistent'"):
            resolver.resolve("nonexistent")


class TestDependencyGraph:
    """test_dependency_graph — Returns correct graph structure."""

    def test_dependency_graph(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        resolver = ViewResolver(tmp_path, config)

        graph = resolver.dependency_graph()

        assert "high_scores" in graph
        assert "enriched" in graph
        assert graph["high_scores"] == {"scores"}
        assert graph["enriched"] == {"high_scores", "names"}


class TestInvalidateCascades:
    """test_invalidate_cascades — Invalidating a source clears downstream views."""

    def test_invalidate_cascades(self, tmp_path: Path):
        config = _setup_project(tmp_path)
        cache_dir = tmp_path / "cache"
        resolver = ViewResolver(tmp_path, config, cache_dir=cache_dir)

        # Resolve both views to populate memory and disk caches
        resolver.resolve("high_scores")
        resolver.resolve("enriched")

        # Verify both are in memory cache
        assert "high_scores" in resolver._memory_cache
        assert "enriched" in resolver._memory_cache

        # Verify disk cache files exist
        assert len(list(cache_dir.glob("high_scores_*.parquet"))) == 1
        assert len(list(cache_dir.glob("enriched_*.parquet"))) == 1

        # Invalidate "high_scores" — should cascade to "enriched"
        resolver.invalidate("high_scores")

        # Both should be cleared from memory
        assert "high_scores" not in resolver._memory_cache
        assert "enriched" not in resolver._memory_cache

        # Both should be cleared from disk
        assert len(list(cache_dir.glob("high_scores_*.parquet"))) == 0
        assert len(list(cache_dir.glob("enriched_*.parquet"))) == 0

        # Sources should still be resolvable
        df = resolver.resolve("scores")
        assert len(df) == 3
