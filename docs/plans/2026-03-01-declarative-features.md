# Declarative Feature System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the flat formula-based feature system with a type-driven declarative feature store supporting team, pairwise, matchup, and regime features with automatic pairwise derivation and content-addressed caching.

**Architecture:** New `FeatureStore` class orchestrates feature lifecycle (declare → compute → cache → resolve). Features declare their semantic type; types drive computation rules. Team features auto-generate pairwise variants via the existing `PairwiseFeatureBuilder`. The store wraps the current `features.parquet` for backward compatibility while adding type-aware management on top.

**Tech Stack:** Python 3.11+, Pydantic v2, pandas, pyarrow, hashlib (SHA-256), existing `PairwiseFeatureBuilder` from `harnessml-features`

**Design doc:** `docs/plans/2026-03-01-declarative-features-design.md`

---

### Task 1: Add Declarative Feature Schemas

**Files:**
- Modify: `packages/harnessml-runner/src/harnessml/runner/schema.py`
- Test: `packages/harnessml-runner/tests/test_schema.py`

**Context:** The schema file currently has `FeatureDecl` (module+function based), `DataConfig`, `ModelDef` with `feature_sets: list[str]`, and various other configs. We're adding the new declarative types alongside the existing ones (not replacing).

**Step 1: Write failing tests**

Add to `test_schema.py`:

```python
class TestDeclarativeFeatureSchemas:
    """Test new declarative feature type schemas."""

    def test_feature_type_enum(self):
        from harnessml.runner.schema import FeatureType
        assert FeatureType.TEAM == "team"
        assert FeatureType.PAIRWISE == "pairwise"
        assert FeatureType.MATCHUP == "matchup"
        assert FeatureType.REGIME == "regime"

    def test_pairwise_mode_enum(self):
        from harnessml.runner.schema import PairwiseMode
        assert PairwiseMode.DIFF == "diff"
        assert PairwiseMode.RATIO == "ratio"
        assert PairwiseMode.BOTH == "both"
        assert PairwiseMode.NONE == "none"

    def test_feature_def_minimal(self):
        from harnessml.runner.schema import FeatureDef, FeatureType
        fd = FeatureDef(name="adj_em", type=FeatureType.TEAM)
        assert fd.name == "adj_em"
        assert fd.type == FeatureType.TEAM
        assert fd.pairwise_mode.value == "diff"  # default
        assert fd.enabled is True

    def test_feature_def_full(self):
        from harnessml.runner.schema import FeatureDef, FeatureType, PairwiseMode
        fd = FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="AdjEM",
            pairwise_mode=PairwiseMode.BOTH,
            category="efficiency",
            description="Adjusted efficiency margin",
            nan_strategy="zero",
        )
        assert fd.source == "kenpom"
        assert fd.column == "AdjEM"
        assert fd.pairwise_mode == PairwiseMode.BOTH
        assert fd.category == "efficiency"

    def test_feature_def_regime(self):
        from harnessml.runner.schema import FeatureDef, FeatureType
        fd = FeatureDef(
            name="late_season",
            type=FeatureType.REGIME,
            condition="day_num > 100",
        )
        assert fd.condition == "day_num > 100"
        assert fd.type == FeatureType.REGIME

    def test_feature_def_formula(self):
        from harnessml.runner.schema import FeatureDef, FeatureType
        fd = FeatureDef(
            name="em_tempo",
            type=FeatureType.PAIRWISE,
            formula="diff_adj_em * diff_adj_tempo",
        )
        assert fd.formula == "diff_adj_em * diff_adj_tempo"

    def test_feature_store_config_defaults(self):
        from harnessml.runner.schema import FeatureStoreConfig
        fsc = FeatureStoreConfig()
        assert fsc.cache_dir == "data/features/cache"
        assert fsc.auto_pairwise is True
        assert fsc.default_pairwise_mode.value == "diff"
        assert fsc.entity_a_column == "entity_a_id"
        assert fsc.entity_b_column == "entity_b_id"
        assert fsc.entity_column == "entity_id"
        assert fsc.period_column == "period_id"

    def test_data_config_feature_store(self):
        from harnessml.runner.schema import DataConfig, FeatureDef, FeatureType
        dc = DataConfig(
            feature_defs={
                "adj_em": FeatureDef(name="adj_em", type=FeatureType.TEAM, source="kenpom", column="AdjEM"),
            }
        )
        assert "adj_em" in dc.feature_defs
        assert dc.feature_store.auto_pairwise is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harnessml-runner/tests/test_schema.py::TestDeclarativeFeatureSchemas -v`
Expected: FAIL — `FeatureType`, `PairwiseMode`, `FeatureDef`, `FeatureStoreConfig` not defined

**Step 3: Implement schemas**

In `schema.py`, after the existing `FeatureDecl` class (line ~28), add:

```python
from enum import Enum

class FeatureType(str, Enum):
    """Semantic type of a declarative feature."""
    TEAM = "team"
    PAIRWISE = "pairwise"
    MATCHUP = "matchup"
    REGIME = "regime"


class PairwiseMode(str, Enum):
    """How to derive pairwise features from team features."""
    DIFF = "diff"
    RATIO = "ratio"
    BOTH = "both"
    NONE = "none"


class FeatureDef(BaseModel):
    """Declarative feature definition.

    Supports four semantic types:
    - team: Per-entity per-period metric. Auto-generates pairwise.
    - pairwise: Per-matchup (A vs B). Derived from team or custom formula.
    - matchup: Per-game context property.
    - regime: Temporal/contextual boolean flag.
    """
    name: str
    type: FeatureType
    source: str | None = None
    column: str | None = None
    formula: str | None = None
    condition: str | None = None
    pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    description: str = ""
    nan_strategy: str = "median"
    category: str = "general"
    enabled: bool = True


class FeatureStoreConfig(BaseModel):
    """Configuration for the declarative feature store."""
    cache_dir: str = "data/features/cache"
    auto_pairwise: bool = True
    default_pairwise_mode: PairwiseMode = PairwiseMode.DIFF
    entity_a_column: str = "entity_a_id"
    entity_b_column: str = "entity_b_id"
    entity_column: str = "entity_id"
    period_column: str = "period_id"
```

Add to `DataConfig` (after `default_cleaning` field):

```python
    # Declarative feature store
    feature_store: FeatureStoreConfig = FeatureStoreConfig()
    feature_defs: dict[str, FeatureDef] = {}
```

Update the `Enum` import at the top of the file — add `from enum import Enum` to imports.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harnessml-runner/tests/test_schema.py::TestDeclarativeFeatureSchemas -v`
Expected: PASS

**Step 5: Verify no existing tests break**

Run: `uv run pytest packages/harnessml-runner/tests/test_schema.py -v`
Expected: All tests PASS (new fields have defaults, no existing behavior changes)

**Step 6: Commit**

```bash
git add packages/harnessml-runner/src/harnessml/runner/schema.py packages/harnessml-runner/tests/test_schema.py
git commit -m "feat(schema): add FeatureType, PairwiseMode, FeatureDef, FeatureStoreConfig for declarative features"
```

---

### Task 2: Create Feature Cache Module

**Files:**
- Create: `packages/harnessml-runner/src/harnessml/runner/feature_cache.py`
- Create: `packages/harnessml-runner/tests/test_feature_cache.py`

**Context:** Content-addressed cache for computed features. Each feature gets a cache key derived from its definition and source data state. The manifest tracks `{feature_name: {cache_key, path, type, derived_from, derivatives}}` for cascade invalidation. Uses separate subdirectories per feature type.

**Step 1: Write failing tests**

Create `test_feature_cache.py`:

```python
"""Tests for feature cache with manifest and cascade invalidation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harnessml.runner.feature_cache import FeatureCache, CacheEntry


class TestCacheEntry:
    """Test CacheEntry data model."""

    def test_create_entry(self):
        entry = CacheEntry(
            cache_key="abc123",
            path="team/adj_em.parquet",
            feature_type="team",
            source="kenpom",
        )
        assert entry.cache_key == "abc123"
        assert entry.derived_from == []
        assert entry.derivatives == []

    def test_entry_with_derivatives(self):
        entry = CacheEntry(
            cache_key="abc123",
            path="team/adj_em.parquet",
            feature_type="team",
            derivatives=["diff_adj_em", "ratio_adj_em"],
        )
        assert entry.derivatives == ["diff_adj_em", "ratio_adj_em"]


class TestFeatureCache:
    """Test cache operations."""

    def test_init_creates_dirs(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        assert (tmp_path / "cache").exists()
        assert (tmp_path / "cache" / "team").exists()
        assert (tmp_path / "cache" / "pairwise").exists()
        assert (tmp_path / "cache" / "matchup").exists()
        assert (tmp_path / "cache" / "regime").exists()

    def test_put_and_get(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        series = pd.Series([1.0, 2.0, 3.0], name="adj_em")

        cache.put("adj_em", "key123", series, feature_type="team")

        result = cache.get("adj_em", "key123")
        assert result is not None
        pd.testing.assert_series_equal(result, series, check_names=False)

    def test_get_miss_wrong_key(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        series = pd.Series([1.0, 2.0], name="adj_em")
        cache.put("adj_em", "key123", series, feature_type="team")

        result = cache.get("adj_em", "different_key")
        assert result is None

    def test_get_miss_not_cached(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        assert cache.get("nonexistent", "key") is None

    def test_invalidate_feature(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        series = pd.Series([1.0, 2.0], name="adj_em")
        cache.put("adj_em", "key123", series, feature_type="team")

        cache.invalidate("adj_em")
        assert cache.get("adj_em", "key123") is None

    def test_invalidate_cascades(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        team_s = pd.Series([1.0, 2.0], name="adj_em")
        pair_s = pd.Series([0.5, -0.3], name="diff_adj_em")

        cache.put("adj_em", "k1", team_s, feature_type="team",
                  derivatives=["diff_adj_em"])
        cache.put("diff_adj_em", "k2", pair_s, feature_type="pairwise",
                  derived_from=["adj_em"])

        # Invalidating parent cascades to child
        cache.invalidate("adj_em")
        assert cache.get("adj_em", "k1") is None
        assert cache.get("diff_adj_em", "k2") is None

    def test_manifest_persists(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache1 = FeatureCache(cache_dir)
        cache1.put("adj_em", "k1", pd.Series([1.0]), feature_type="team")

        # Re-open cache from same dir
        cache2 = FeatureCache(cache_dir)
        assert cache2.get("adj_em", "k1") is not None

    def test_compute_cache_key(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        key1 = cache.compute_key(name="adj_em", feature_type="team",
                                 source="kenpom", column="AdjEM")
        key2 = cache.compute_key(name="adj_em", feature_type="team",
                                 source="kenpom", column="AdjEM")
        key3 = cache.compute_key(name="adj_em", feature_type="team",
                                 source="different", column="AdjEM")
        assert key1 == key2  # deterministic
        assert key1 != key3  # different source

    def test_list_cached(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        cache.put("adj_em", "k1", pd.Series([1.0]), feature_type="team")
        cache.put("diff_adj_em", "k2", pd.Series([0.5]), feature_type="pairwise")

        cached = cache.list_cached()
        assert "adj_em" in cached
        assert "diff_adj_em" in cached

    def test_list_cached_by_type(self, tmp_path):
        cache = FeatureCache(tmp_path / "cache")
        cache.put("adj_em", "k1", pd.Series([1.0]), feature_type="team")
        cache.put("diff_adj_em", "k2", pd.Series([0.5]), feature_type="pairwise")

        team_only = cache.list_cached(feature_type="team")
        assert "adj_em" in team_only
        assert "diff_adj_em" not in team_only
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harnessml-runner/tests/test_feature_cache.py -v`
Expected: FAIL — module `harnessml.runner.feature_cache` does not exist

**Step 3: Implement FeatureCache**

Create `feature_cache.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harnessml-runner/tests/test_feature_cache.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/harnessml-runner/src/harnessml/runner/feature_cache.py packages/harnessml-runner/tests/test_feature_cache.py
git commit -m "feat(runner): add FeatureCache with content-addressed storage and cascade invalidation"
```

---

### Task 3: Create FeatureStore Core — Team Features with Auto-Pairwise

**Files:**
- Create: `packages/harnessml-runner/src/harnessml/runner/feature_store.py`
- Create: `packages/harnessml-runner/tests/test_feature_store.py`

**Context:** The FeatureStore is the central orchestrator. This task implements the core: initialization, adding team features, computing pairwise derivatives, and basic resolution. It delegates to `FeatureCache` for caching and uses `PairwiseFeatureBuilder` from `harnessml-features` for diff/ratio computation.

The store works with two levels of data:
- **Entity-level**: Per-team per-period data loaded from sources (team features live here)
- **Matchup-level**: Per-game data in `features.parquet` (pairwise, matchup, regime features live here)

When a team feature is added, the store:
1. Loads entity-level data from the declared source
2. Caches the entity-level column
3. Uses PairwiseFeatureBuilder to derive matchup-level pairwise features
4. Registers the derived pairwise features in the registry

**Step 1: Write failing tests**

Create `test_feature_store.py`:

```python
"""Tests for FeatureStore — declarative feature management."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from harnessml.runner.feature_store import FeatureStore
from harnessml.runner.schema import (
    DataConfig,
    FeatureDef,
    FeatureStoreConfig,
    FeatureType,
    PairwiseMode,
    SourceConfig,
)


def _make_project(tmp_path: Path) -> Path:
    """Create a minimal project with entity-level and matchup-level data."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    # Create config
    config_dir = project_dir / "config"
    config_dir.mkdir()
    (config_dir / "pipeline.yaml").write_text(
        "data:\n  features_file: features.parquet\n  target_column: result\n"
    )

    # Entity-level data (team stats)
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    n_teams = 20
    n_seasons = 3
    entities = []
    for season in [2022, 2023, 2024]:
        for team_id in range(1, n_teams + 1):
            entities.append({
                "entity_id": team_id,
                "period_id": season,
                "adj_em": rng.standard_normal() * 10,
                "adj_tempo": rng.standard_normal() * 5 + 65,
                "win_rate": rng.uniform(0.2, 0.9),
            })
    entity_df = pd.DataFrame(entities)
    entity_df.to_parquet(raw_dir / "kenpom.parquet", index=False)

    # Matchup-level data (main features parquet)
    n_matchups = 200
    matchups = []
    for i in range(n_matchups):
        season = rng.choice([2022, 2023, 2024])
        team_a = rng.integers(1, n_teams + 1)
        team_b = rng.integers(1, n_teams + 1)
        while team_b == team_a:
            team_b = rng.integers(1, n_teams + 1)
        matchups.append({
            "entity_a_id": int(team_a),
            "entity_b_id": int(team_b),
            "period_id": int(season),
            "result": int(rng.integers(0, 2)),
            "day_num": int(rng.integers(1, 155)),
            "is_neutral": int(rng.integers(0, 2)),
        })
    matchup_df = pd.DataFrame(matchups)
    matchup_df.to_parquet(features_dir / "features.parquet", index=False)

    return project_dir


class TestFeatureStoreInit:
    """Test FeatureStore initialization."""

    def test_init_with_defaults(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)
        assert store is not None
        assert len(store.available()) == 0

    def test_init_with_feature_defs(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig(
            feature_defs={
                "adj_em": FeatureDef(
                    name="adj_em",
                    type=FeatureType.TEAM,
                    source="kenpom",
                    column="adj_em",
                ),
            },
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)
        assert len(store.available()) == 1


class TestTeamFeatures:
    """Test adding and computing team features."""

    def _make_config(self) -> DataConfig:
        return DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.parquet",
                ),
            },
        )

    def test_add_team_feature(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = self._make_config()
        store = FeatureStore(project_dir, config)

        result = store.add(FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="adj_em",
        ))

        assert result.name == "adj_em"
        assert isinstance(result.correlation, float)
        # Should have auto-generated pairwise
        available = store.available()
        names = [f.name for f in available]
        assert "adj_em" in names
        assert "diff_adj_em" in names

    def test_add_team_feature_with_both_pairwise(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = self._make_config()
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="adj_em",
            pairwise_mode=PairwiseMode.BOTH,
        ))

        names = [f.name for f in store.available()]
        assert "diff_adj_em" in names
        assert "ratio_adj_em" in names

    def test_add_team_feature_no_pairwise(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = self._make_config()
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="adj_em",
            pairwise_mode=PairwiseMode.NONE,
        ))

        names = [f.name for f in store.available()]
        assert "adj_em" in names
        assert "diff_adj_em" not in names

    def test_compute_team_pairwise(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = self._make_config()
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="adj_em",
        ))

        series = store.compute("diff_adj_em")
        assert len(series) > 0
        assert series.dtype in [np.float64, np.float32, float]

    def test_caching_works(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = self._make_config()
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="adj_em",
        ))

        # Compute once
        series1 = store.compute("diff_adj_em")
        # Compute again — should hit cache
        series2 = store.compute("diff_adj_em")
        pd.testing.assert_series_equal(series1, series2)

    def test_missing_source_raises(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig()  # no sources
        store = FeatureStore(project_dir, config)

        with pytest.raises(ValueError, match="source"):
            store.add(FeatureDef(
                name="adj_em",
                type=FeatureType.TEAM,
                source="nonexistent",
                column="adj_em",
            ))


class TestMatchupFeatures:
    """Test matchup feature support."""

    def test_add_matchup_feature_column(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)

        result = store.add(FeatureDef(
            name="neutral_site",
            type=FeatureType.MATCHUP,
            column="is_neutral",
        ))

        assert result.name == "neutral_site"
        series = store.compute("neutral_site")
        assert len(series) > 0


class TestRegimeFeatures:
    """Test regime feature support."""

    def test_add_regime_feature(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)

        result = store.add(FeatureDef(
            name="late_season",
            type=FeatureType.REGIME,
            condition="day_num > 100",
        ))

        assert result.name == "late_season"
        series = store.compute("late_season")
        assert set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0})


class TestFormulaFeatures:
    """Test formula-based features via the store."""

    def test_add_formula_pairwise(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)

        # First add a column to work with — use matchup column
        result = store.add(FeatureDef(
            name="day_squared",
            type=FeatureType.PAIRWISE,
            formula="day_num ** 2",
        ))

        assert result.name == "day_squared"
        series = store.compute("day_squared")
        assert len(series) > 0


class TestResolve:
    """Test feature resolution for model training."""

    def test_resolve_features(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em",
            type=FeatureType.TEAM,
            source="kenpom",
            column="adj_em",
        ))
        store.add(FeatureDef(
            name="late_season",
            type=FeatureType.REGIME,
            condition="day_num > 100",
        ))

        df = store.resolve(["diff_adj_em", "late_season"])
        assert "diff_adj_em" in df.columns
        assert "late_season" in df.columns

    def test_resolve_sets(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em", category="efficiency",
        ))
        store.add(FeatureDef(
            name="adj_tempo", type=FeatureType.TEAM,
            source="kenpom", column="adj_tempo", category="pace",
        ))

        cols = store.resolve_sets(["efficiency"])
        # Should include diff_adj_em (auto-pairwise from adj_em which is in "efficiency")
        assert "diff_adj_em" in cols
        assert "diff_adj_tempo" not in cols

    def test_available_by_type(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        ))
        store.add(FeatureDef(
            name="late_season", type=FeatureType.REGIME,
            condition="day_num > 100",
        ))

        team_feats = store.available(type_filter=FeatureType.TEAM)
        regime_feats = store.available(type_filter=FeatureType.REGIME)
        assert len(team_feats) == 1
        assert len(regime_feats) == 1


class TestRemoveFeature:
    """Test feature removal."""

    def test_remove_team_removes_derivatives(self, tmp_path):
        project_dir = _make_project(tmp_path)
        config = DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="adj_em",
        ))

        assert len(store.available()) >= 2  # team + pairwise

        store.remove("adj_em")

        assert len(store.available()) == 0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harnessml-runner/tests/test_feature_store.py -v`
Expected: FAIL — module `harnessml.runner.feature_store` does not exist

**Step 3: Implement FeatureStore**

Create `feature_store.py`:

```python
"""Declarative feature store — type-driven feature management.

The FeatureStore orchestrates the lifecycle of declarative features:
declare → compute → cache → resolve. Features declare their semantic
type (team, pairwise, matchup, regime), which drives computation rules.

Team features auto-generate pairwise variants. All computed features
are cached with content-addressed keys for incremental rebuilds.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from harnessml.runner.feature_cache import FeatureCache
from harnessml.runner.feature_engine import (
    FeatureResult,
    _check_redundancy,
    _compute_feature_stats,
    _resolve_formula,
)
from harnessml.runner.schema import (
    DataConfig,
    FeatureDef,
    FeatureStoreConfig,
    FeatureType,
    PairwiseMode,
    SourceConfig,
)

logger = logging.getLogger(__name__)


class FeatureStore:
    """Central feature management: declaration, computation, caching, resolution.

    Works with two levels of data:
    - Entity-level: Per-team per-period data loaded from sources
    - Matchup-level: Per-game data in features.parquet

    Team features are computed at entity level, then automatically derived
    into pairwise features at matchup level.
    """

    def __init__(self, project_dir: Path, config: DataConfig) -> None:
        self.project_dir = Path(project_dir)
        self.config = config
        self.store_config = config.feature_store

        # Registry of all declared features
        self._registry: dict[str, FeatureDef] = dict(config.feature_defs)

        # Cache
        cache_dir = Path(self.store_config.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = self.project_dir / cache_dir
        self._cache = FeatureCache(cache_dir)

        # Lazy-loaded data
        self._matchup_df: pd.DataFrame | None = None
        self._source_dfs: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_matchup_data(self) -> pd.DataFrame:
        """Load matchup-level data from features.parquet."""
        if self._matchup_df is None:
            from harnessml.runner.data_utils import get_features_path
            path = get_features_path(self.project_dir, self.config)
            if path.exists():
                self._matchup_df = pd.read_parquet(path)
            else:
                self._matchup_df = pd.DataFrame()
        return self._matchup_df

    def _load_source(self, source_name: str) -> pd.DataFrame:
        """Load entity-level data from a registered source."""
        if source_name in self._source_dfs:
            return self._source_dfs[source_name]

        source_config = self.config.sources.get(source_name)
        if source_config is None:
            raise ValueError(
                f"Unknown source '{source_name}'. "
                f"Available sources: {list(self.config.sources.keys())}"
            )

        if source_config.path is None:
            raise ValueError(f"Source '{source_name}' has no path configured.")

        source_path = Path(source_config.path)
        if not source_path.is_absolute():
            source_path = self.project_dir / source_path

        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        fmt = source_config.format
        if fmt == "auto":
            suffix = source_path.suffix.lower()
            if suffix == ".parquet":
                fmt = "parquet"
            elif suffix == ".csv":
                fmt = "csv"
            elif suffix in (".xlsx", ".xls"):
                fmt = "excel"
            else:
                fmt = "csv"

        if fmt == "parquet":
            df = pd.read_parquet(source_path)
        elif fmt == "csv":
            df = pd.read_csv(source_path)
        elif fmt == "excel":
            df = pd.read_excel(source_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        self._source_dfs[source_name] = df
        return df

    # ------------------------------------------------------------------
    # Declaration
    # ------------------------------------------------------------------

    def add(self, feature: FeatureDef) -> FeatureResult:
        """Declare and compute a feature. Auto-generates pairwise if team type."""
        self._registry[feature.name] = feature

        if feature.type == FeatureType.TEAM:
            return self._add_team_feature(feature)
        elif feature.type == FeatureType.PAIRWISE:
            return self._add_pairwise_feature(feature)
        elif feature.type == FeatureType.MATCHUP:
            return self._add_matchup_feature(feature)
        elif feature.type == FeatureType.REGIME:
            return self._add_regime_feature(feature)
        else:
            raise ValueError(f"Unknown feature type: {feature.type}")

    def remove(self, name: str) -> None:
        """Remove a feature and its derivatives from registry and cache."""
        feature = self._registry.get(name)
        if feature is None:
            return

        # Find and remove derivatives
        derivatives = [
            n for n, f in self._registry.items()
            if hasattr(f, '_derived_from') and name in getattr(f, '_derived_from', [])
        ]
        # Also check cache for derivatives
        cache_entry = self._cache._entries.get(name)
        if cache_entry:
            derivatives.extend(cache_entry.derivatives)

        for deriv_name in set(derivatives):
            if deriv_name in self._registry:
                del self._registry[deriv_name]

        # Invalidate cache (cascades to derivatives)
        self._cache.invalidate(name)

        # Remove from registry
        if name in self._registry:
            del self._registry[name]

    # ------------------------------------------------------------------
    # Team features
    # ------------------------------------------------------------------

    def _add_team_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a team feature and auto-generate pairwise derivatives."""
        if feature.source is None:
            raise ValueError(
                f"Team feature '{feature.name}' requires a source. "
                f"Set source= to a registered SourceConfig name."
            )

        source_df = self._load_source(feature.source)
        matchup_df = self._load_matchup_data()

        # Extract entity-level column
        if feature.column and feature.column in source_df.columns:
            entity_col = feature.column
        elif feature.name in source_df.columns:
            entity_col = feature.name
        else:
            raise ValueError(
                f"Column '{feature.column or feature.name}' not found in source "
                f"'{feature.source}'. Available: {list(source_df.columns)[:20]}"
            )

        entity_series = source_df[entity_col].astype(float)

        # Compute cache key
        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="team",
            source=feature.source,
            column=entity_col,
        )

        # Cache the entity-level data
        self._cache.put(
            feature.name, cache_key, entity_series,
            feature_type="team", source=feature.source,
        )

        # Auto-generate pairwise
        pairwise_names = []
        if (self.store_config.auto_pairwise
                and feature.pairwise_mode != PairwiseMode.NONE):
            pairwise_names = self._generate_pairwise(
                feature, source_df, matchup_df, entity_col,
            )

        # Update cache entry with derivative names
        entry = self._cache._entries.get(feature.name)
        if entry:
            entry.derivatives = pairwise_names
            self._cache._save_manifest()

        # Compute correlation using the first pairwise derivative if available
        correlation = 0.0
        target_col = self.config.target_column
        if pairwise_names and target_col in matchup_df.columns:
            first_pairwise = pairwise_names[0]
            pair_series = self.compute(first_pairwise)
            try:
                correlation = float(pair_series.corr(matchup_df[target_col].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        stats = _compute_feature_stats(entity_series)
        null_rate = float(entity_series.isna().mean())

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=null_rate,
            stats=stats,
            description=feature.description,
        )

    def _generate_pairwise(
        self,
        feature: FeatureDef,
        source_df: pd.DataFrame,
        matchup_df: pd.DataFrame,
        entity_col: str,
    ) -> list[str]:
        """Generate pairwise features from a team feature."""
        sc = self.store_config
        entity_id = sc.entity_column
        entity_a = sc.entity_a_column
        entity_b = sc.entity_b_column
        period = sc.period_column

        # Check that matchup data has the required columns
        required = {entity_a, entity_b, period}
        if not required.issubset(set(matchup_df.columns)):
            logger.warning(
                "Matchup data missing columns %s — skipping pairwise generation",
                required - set(matchup_df.columns),
            )
            return []

        # Build entity lookup: entity_id + period_id -> value
        if entity_id not in source_df.columns or period not in source_df.columns:
            logger.warning(
                "Source data missing '%s' or '%s' — skipping pairwise generation",
                entity_id, period,
            )
            return []

        # Build lookup dataframe
        lookup = source_df[[entity_id, period, entity_col]].copy()
        lookup = lookup.rename(columns={entity_col: feature.name})

        # Determine modes
        mode = feature.pairwise_mode
        if mode == PairwiseMode.DIFF or mode == PairwiseMode.BOTH:
            methods = ["diff"]
        else:
            methods = []
        if mode == PairwiseMode.RATIO or mode == PairwiseMode.BOTH:
            methods.append("ratio")

        pairwise_names = []
        epsilon = 1e-12

        for method in methods:
            pairwise_name = f"{method}_{feature.name}"

            # Merge entity A values
            a_merge = lookup.rename(columns={
                entity_id: entity_a,
                feature.name: f"_a_{feature.name}",
            })
            merged = matchup_df.merge(a_merge, on=[entity_a, period], how="left")

            # Merge entity B values
            b_merge = lookup.rename(columns={
                entity_id: entity_b,
                feature.name: f"_b_{feature.name}",
            })
            merged = merged.merge(b_merge, on=[entity_b, period], how="left")

            a_col = f"_a_{feature.name}"
            b_col = f"_b_{feature.name}"

            if method == "diff":
                pairwise_series = merged[a_col] - merged[b_col]
            elif method == "ratio":
                pairwise_series = merged[a_col] / (merged[b_col] + epsilon)
            else:
                continue

            pairwise_series = pairwise_series.astype(float)
            pairwise_series.name = pairwise_name

            # Cache
            pair_key = self._cache.compute_key(
                name=pairwise_name,
                feature_type="pairwise",
                source=feature.source,
                column=entity_col,
                pairwise_mode=method,
            )
            self._cache.put(
                pairwise_name, pair_key, pairwise_series,
                feature_type="pairwise",
                derived_from=[feature.name],
            )

            # Register
            pairwise_def = FeatureDef(
                name=pairwise_name,
                type=FeatureType.PAIRWISE,
                source=feature.source,
                category=feature.category,
                description=f"{method} of {feature.name} (auto-derived)",
                nan_strategy=feature.nan_strategy,
                pairwise_mode=PairwiseMode.NONE,
                enabled=feature.enabled,
            )
            # Tag as derived
            pairwise_def._derived_from = [feature.name]  # type: ignore[attr-defined]
            self._registry[pairwise_name] = pairwise_def
            pairwise_names.append(pairwise_name)

        return pairwise_names

    # ------------------------------------------------------------------
    # Pairwise features (custom formula)
    # ------------------------------------------------------------------

    def _add_pairwise_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a custom pairwise feature from a formula."""
        matchup_df = self._load_matchup_data()

        if feature.formula is None:
            raise ValueError(
                f"Pairwise feature '{feature.name}' requires a formula."
            )

        series = _resolve_formula(feature.formula, matchup_df)
        series = series.astype(float)
        series.name = feature.name

        # Cache
        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="pairwise",
            formula=feature.formula,
        )
        self._cache.put(
            feature.name, cache_key, series,
            feature_type="pairwise",
        )

        # Correlation
        correlation = 0.0
        target_col = self.config.target_column
        if target_col in matchup_df.columns:
            try:
                correlation = float(series.corr(matchup_df[target_col].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        stats = _compute_feature_stats(series)
        null_rate = float(series.isna().mean())
        redundant_with = _check_redundancy(series, matchup_df)

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            redundant_with=redundant_with,
            null_rate=null_rate,
            stats=stats,
            description=feature.description,
        )

    # ------------------------------------------------------------------
    # Matchup features
    # ------------------------------------------------------------------

    def _add_matchup_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a matchup-level feature."""
        matchup_df = self._load_matchup_data()

        if feature.column and feature.column in matchup_df.columns:
            series = matchup_df[feature.column].astype(float)
        elif feature.formula:
            series = _resolve_formula(feature.formula, matchup_df)
        else:
            raise ValueError(
                f"Matchup feature '{feature.name}' requires column= or formula=."
            )

        series = series.astype(float)
        series.name = feature.name

        # Cache
        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="matchup",
            column=feature.column,
            formula=feature.formula,
        )
        self._cache.put(
            feature.name, cache_key, series,
            feature_type="matchup",
        )

        # Correlation
        correlation = 0.0
        target_col = self.config.target_column
        if target_col in matchup_df.columns:
            try:
                correlation = float(series.corr(matchup_df[target_col].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        stats = _compute_feature_stats(series)
        null_rate = float(series.isna().mean())

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=null_rate,
            stats=stats,
            description=feature.description,
        )

    # ------------------------------------------------------------------
    # Regime features
    # ------------------------------------------------------------------

    def _add_regime_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a regime feature from a boolean condition."""
        matchup_df = self._load_matchup_data()

        if feature.condition is None:
            raise ValueError(
                f"Regime feature '{feature.name}' requires condition=."
            )

        # Evaluate condition using the formula engine
        series = _resolve_formula(feature.condition, matchup_df)
        # Coerce to 0/1 integer
        series = series.astype(bool).astype(int).astype(float)
        series.name = feature.name

        # Cache
        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="regime",
            condition=feature.condition,
        )
        self._cache.put(
            feature.name, cache_key, series,
            feature_type="regime",
        )

        # Correlation
        correlation = 0.0
        target_col = self.config.target_column
        if target_col in matchup_df.columns:
            try:
                correlation = float(series.corr(matchup_df[target_col].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        # Coverage
        coverage = float(series.mean())
        stats = {"coverage": coverage, "count": float(series.sum())}
        null_rate = float(series.isna().mean())

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=null_rate,
            stats=stats,
            description=feature.description,
        )

    # ------------------------------------------------------------------
    # Computation & resolution
    # ------------------------------------------------------------------

    def compute(self, name: str) -> pd.Series:
        """Compute a single feature. Checks cache first."""
        feature = self._registry.get(name)
        if feature is None:
            raise ValueError(
                f"Feature '{name}' not found in registry. "
                f"Available: {list(self._registry.keys())[:20]}"
            )

        # Check cache
        cache_key = self._compute_key_for(feature)
        cached = self._cache.get(name, cache_key)
        if cached is not None:
            return cached

        # Recompute — re-add the feature
        self.add(feature)
        cached = self._cache.get(name, cache_key)
        if cached is not None:
            return cached

        raise RuntimeError(f"Failed to compute feature '{name}'")

    def _compute_key_for(self, feature: FeatureDef) -> str:
        """Compute cache key for a feature definition."""
        return self._cache.compute_key(
            name=feature.name,
            feature_type=feature.type.value,
            source=feature.source,
            column=feature.column,
            formula=feature.formula,
            condition=feature.condition,
            pairwise_mode=feature.pairwise_mode.value,
        )

    def compute_all(self, names: list[str] | None = None) -> pd.DataFrame:
        """Compute all (or specified) features as a matchup-level DataFrame."""
        if names is None:
            names = [
                f.name for f in self._registry.values()
                if f.enabled and f.type != FeatureType.TEAM  # team features are entity-level
            ]

        series_list = []
        for name in names:
            try:
                s = self.compute(name)
                s.name = name
                series_list.append(s)
            except Exception as exc:
                logger.warning("Failed to compute '%s': %s", name, exc)

        if not series_list:
            return pd.DataFrame()

        return pd.concat(series_list, axis=1)

    def resolve(self, names: list[str]) -> pd.DataFrame:
        """Resolve feature names to a training-ready DataFrame.

        Returns a DataFrame with the matchup-level base data plus
        requested feature columns.
        """
        matchup_df = self._load_matchup_data()
        feature_df = self.compute_all(names)

        if feature_df.empty:
            return matchup_df

        # Align indices
        result = matchup_df.copy()
        for col in feature_df.columns:
            if col not in result.columns:
                result[col] = feature_df[col].values

        return result

    def resolve_sets(self, set_names: list[str]) -> list[str]:
        """Resolve feature set (category) names to column name list."""
        columns = []
        for set_name in set_names:
            for feat in self._registry.values():
                if feat.category == set_name and feat.enabled:
                    # For team features, return the pairwise derivative names
                    if feat.type == FeatureType.TEAM:
                        entry = self._cache._entries.get(feat.name)
                        if entry:
                            columns.extend(entry.derivatives)
                    else:
                        columns.append(feat.name)
        return list(dict.fromkeys(columns))  # dedupe preserving order

    def available(
        self,
        type_filter: FeatureType | None = None,
    ) -> list[FeatureDef]:
        """List all registered features, optionally filtered by type."""
        features = []
        for feat in self._registry.values():
            if type_filter is not None and feat.type != type_filter:
                continue
            if feat.enabled:
                features.append(feat)
        return features

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def refresh(self, sources: list[str] | None = None) -> dict[str, str]:
        """Invalidate and recompute features from specified sources."""
        results = {}
        for name, feat in list(self._registry.items()):
            if sources is not None and feat.source not in sources:
                continue
            self._cache.invalidate(name)
            try:
                self.compute(name)
                results[name] = "refreshed"
            except Exception as exc:
                results[name] = f"error: {exc}"
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_registry(self) -> None:
        """Persist feature_defs back to DataConfig (for YAML serialization)."""
        self.config.feature_defs = dict(self._registry)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/harnessml-runner/tests/test_feature_store.py -v`
Expected: PASS

**Step 5: Run full test suite to verify no regressions**

Run: `uv run pytest packages/harnessml-runner/tests/ -v`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add packages/harnessml-runner/src/harnessml/runner/feature_store.py packages/harnessml-runner/tests/test_feature_store.py
git commit -m "feat(runner): add FeatureStore with team/pairwise/matchup/regime features and auto-derivation"
```

---

### Task 4: Update MCP Tools to Use FeatureStore

**Files:**
- Modify: `packages/harnessml-runner/src/harnessml/runner/config_writer.py` (lines ~467-498, ~332-361)
- Modify: `packages/harnessml-runner/tests/test_project.py` (or create new test)

**Context:** The `add_feature()` MCP tool currently only accepts `(name, formula)` and delegates to `feature_engine.create_feature()`. We need to add `type`, `source`, `column`, `condition`, `pairwise_mode`, and `category` parameters. When `type` is specified, use the `FeatureStore`. When only `formula` is provided (no `type`), fall back to the existing formula path for backward compatibility.

Also update `available_features()` to show feature types when the store has features registered.

**Step 1: Write failing tests**

Add to an existing test file or create a test for the MCP tool wiring:

```python
class TestMCPAddFeatureDeclarative:
    """Test add_feature MCP tool with declarative types."""

    def test_add_feature_backward_compat(self, tmp_path):
        """Formula-only call still works (no type= param)."""
        from harnessml.runner.config_writer import add_feature
        feat_dir = tmp_path / "data" / "features"
        feat_dir.mkdir(parents=True)
        _make_features_parquet(feat_dir / "features.parquet")

        result = add_feature(tmp_path, "x_plus_y", "diff_x + diff_y")
        assert "x_plus_y" in result
        assert "Correlation" in result or "correlation" in result.lower()

    def test_add_feature_with_type_team(self, tmp_path):
        """add_feature with type='team' uses FeatureStore."""
        from harnessml.runner.config_writer import add_feature
        _setup_project_with_sources(tmp_path)

        result = add_feature(
            tmp_path, "adj_em",
            type="team", source="kenpom", column="adj_em",
        )
        assert "adj_em" in result
        assert "diff_adj_em" in result  # auto-pairwise mentioned

    def test_add_feature_with_type_regime(self, tmp_path):
        """add_feature with type='regime' creates boolean feature."""
        from harnessml.runner.config_writer import add_feature
        _setup_project_with_sources(tmp_path)

        result = add_feature(
            tmp_path, "late_season",
            type="regime", condition="day_num > 100",
        )
        assert "late_season" in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/harnessml-runner/tests/test_project.py::TestMCPAddFeatureDeclarative -v`
Expected: FAIL — `add_feature` doesn't accept `type` parameter

**Step 3: Update add_feature in config_writer.py**

Replace the `add_feature` function (lines ~467-483):

```python
def add_feature(
    project_dir: Path,
    name: str,
    formula: str | None = None,
    *,
    type: str | None = None,
    source: str | None = None,
    column: str | None = None,
    condition: str | None = None,
    pairwise_mode: str = "diff",
    category: str = "general",
    description: str = "",
) -> str:
    """Create a new feature — declarative (with type) or formula-based.

    If type is specified, uses the declarative FeatureStore.
    If only formula is given, uses the existing formula engine.
    """
    project_dir = Path(project_dir)

    if type is not None:
        # Declarative path
        from harnessml.runner.feature_store import FeatureStore
        from harnessml.runner.schema import FeatureDef, FeatureType, PairwiseMode
        from harnessml.runner.data_utils import load_data_config

        config = load_data_config(project_dir)
        store = FeatureStore(project_dir, config)

        feature_type = FeatureType(type)
        pw_mode = PairwiseMode(pairwise_mode)

        feature_def = FeatureDef(
            name=name,
            type=feature_type,
            source=source,
            column=column,
            formula=formula,
            condition=condition,
            pairwise_mode=pw_mode,
            category=category,
            description=description,
        )

        result = store.add(feature_def)

        # Format response
        lines = [f"## Added {type} feature: {name}\n"]
        if description:
            lines.append(f"_{description}_\n")

        if feature_type == FeatureType.TEAM:
            # Show auto-generated pairwise features
            pairwise = [f for f in store.available() if f.type == FeatureType.PAIRWISE
                        and hasattr(f, '_derived_from') and name in getattr(f, '_derived_from', [])]
            if not pairwise:
                # Check cache for derivatives
                cache_entry = store._cache._entries.get(name)
                if cache_entry and cache_entry.derivatives:
                    lines.append("**Auto-generated pairwise:**")
                    for deriv in cache_entry.derivatives:
                        deriv_series = store.compute(deriv)
                        target_col = config.target_column
                        matchup_df = store._load_matchup_data()
                        corr = 0.0
                        if target_col in matchup_df.columns:
                            try:
                                corr = float(deriv_series.corr(matchup_df[target_col].astype(float)))
                            except (TypeError, ValueError):
                                pass
                        lines.append(f"- `{deriv}` (r={corr:+.4f})")

        lines.append(f"\n- **Correlation**: {result.correlation:+.4f}")
        lines.append(f"- **Null rate**: {result.null_rate:.1%}")
        if result.stats:
            for k, v in result.stats.items():
                lines.append(f"- **{k.title()}**: {v:.4f}")
        lines.append(f"- **Category**: {category}")

        return "\n".join(lines)

    else:
        # Backward-compatible formula path
        if formula is None:
            raise ValueError("Either type= or formula= must be provided.")
        from harnessml.runner.feature_engine import create_feature

        result = create_feature(
            project_dir=project_dir,
            name=name,
            formula=formula,
            description=description,
        )
        return result.format_summary()
```

**Step 4: Run tests**

Run: `uv run pytest packages/harnessml-runner/tests/test_project.py::TestMCPAddFeatureDeclarative -v`
Expected: PASS

**Step 5: Update available_features to show types**

Update `available_features()` function (lines ~332-361) to check for feature store registrations:

```python
def available_features(
    project_dir: Path,
    prefix: str | None = None,
    type_filter: str | None = None,
) -> str:
    """List available feature columns from the dataset.

    If the project uses the declarative feature store, shows features
    grouped by type. Otherwise falls back to column listing.
    """
    from harnessml.runner.data_utils import get_features_path, load_data_config
    from harnessml.runner.schema import DataConfig

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
        parquet_path = get_features_path(project_dir, config)
    except Exception:
        parquet_path = get_features_path(project_dir, DataConfig())
        config = DataConfig()

    # Check for declarative feature store
    if config.feature_defs:
        from harnessml.runner.feature_store import FeatureStore
        from harnessml.runner.schema import FeatureType

        store = FeatureStore(project_dir, config)
        ft = FeatureType(type_filter) if type_filter else None
        features = store.available(type_filter=ft)

        if not features:
            return "No declarative features registered."

        lines = [f"## Declarative Features ({len(features)})\n"]
        by_type: dict[str, list] = {}
        for f in features:
            by_type.setdefault(f.type.value, []).append(f)

        for ft_name, feats in by_type.items():
            lines.append(f"### {ft_name.title()} ({len(feats)})")
            for f in feats:
                lines.append(f"- `{f.name}` — {f.description or f.category}")
            lines.append("")

        return "\n".join(lines)

    # Fallback: flat column listing
    import pandas as pd

    if not parquet_path.exists():
        return f"**Error**: Data file not found: {parquet_path}"

    df = pd.read_parquet(parquet_path)
    cols = sorted(df.columns)
    if prefix:
        cols = [c for c in cols if c.startswith(prefix)]

    if not cols:
        return "No features found."

    lines = [f"## Available Features ({len(cols)} columns)\n"]
    for col in cols:
        lines.append(f"- `{col}`")
    return "\n".join(lines)
```

**Step 6: Run full test suite**

Run: `uv run pytest packages/harnessml-runner/tests/ -v`
Expected: All tests PASS

**Step 7: Commit**

```bash
git add packages/harnessml-runner/src/harnessml/runner/config_writer.py packages/harnessml-runner/tests/
git commit -m "feat(runner): update add_feature and available_features MCP tools for declarative features"
```

---

### Task 5: Update MCP Server Tool Definitions

**Files:**
- Modify: `packages/harnessml-runner/src/harnessml/runner/mcp_tools.py` (or wherever MCP tool definitions are registered)

**Context:** The MCP server exposes tools to the AI. The `add_feature` tool definition needs updated parameters to include `type`, `source`, `column`, `condition`, `pairwise_mode`, `category`. The `available_features` tool needs a `type_filter` parameter. Check the actual MCP tool registration file for the exact location.

**Step 1: Find the MCP tool registration**

Look for tool registration in `mcp_tools.py`, `server.py`, or wherever the MCP tools are defined. The tool definitions map MCP tool names to `config_writer.py` functions.

**Step 2: Update `add_feature` tool parameters**

Add the new optional parameters to the tool definition:
- `type: str | None` — Feature type: "team", "pairwise", "matchup", "regime"
- `source: str | None` — SourceConfig name (for team features)
- `column: str | None` — Column in source data
- `condition: str | None` — Boolean condition (for regime features)
- `pairwise_mode: str` — "diff", "ratio", "both", "none" (default: "diff")
- `category: str` — Feature category for grouping (default: "general")

Make `formula` optional (was required, now only needed for formula-based features).

**Step 3: Update `available_features` tool parameters**

Add optional `type_filter: str | None` parameter.

**Step 4: Test the MCP tool definitions parse correctly**

Run: `uv run pytest packages/harnessml-runner/tests/ -v -k "mcp or tool"`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/harnessml-runner/src/harnessml/runner/
git commit -m "feat(runner): update MCP tool definitions for declarative feature parameters"
```

---

### Task 6: Wire FeatureStore into Pipeline Runner

**Files:**
- Modify: `packages/harnessml-runner/src/harnessml/runner/pipeline.py` (focus on `_load_data()` and `predict()`)
- Modify: `packages/harnessml-runner/tests/test_mm_integration.py`

**Context:** The pipeline runner currently loads features from a single parquet via `get_features_path()`. When the project has `feature_defs` configured, the runner should use `FeatureStore.resolve()` to compute and assemble features for model training. This also enables `ModelDef.feature_sets` which references category names.

**Key changes:**
1. In `_load_data()`: if `config.data.feature_defs` is non-empty, create a `FeatureStore` and use it
2. In `predict()`: resolve `model.feature_sets` through the store
3. Preserve backward compatibility — projects without `feature_defs` work unchanged

**Step 1: Write failing test**

Add to `test_mm_integration.py` or a new test file:

```python
def test_pipeline_with_feature_store(tmp_path):
    """Pipeline loads features via FeatureStore when feature_defs configured."""
    # Set up project with entity-level source and matchup data
    _setup_project_with_declarative_features(tmp_path)

    from harnessml.runner.pipeline import PipelineRunner
    runner = PipelineRunner(tmp_path)
    runner.load()

    # Verify declared features were computed
    assert "diff_adj_em" in runner._df.columns
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/harnessml-runner/tests/test_mm_integration.py::test_pipeline_with_feature_store -v`
Expected: FAIL

**Step 3: Modify `_load_data()` in pipeline.py**

In the `_load_data()` method, after loading the parquet and before applying injections, add:

```python
# If declarative features are configured, compute them via FeatureStore
if self.config.data.feature_defs:
    from harnessml.runner.feature_store import FeatureStore

    store = FeatureStore(self.project_dir, self.config.data)

    # Determine which features are needed by active models
    needed = set()
    for name, model in self._get_active_models():
        needed.update(model.features)
        if model.feature_sets:
            needed.update(store.resolve_sets(model.feature_sets))

    if needed:
        feature_df = store.compute_all(list(needed))
        for col in feature_df.columns:
            if col not in self._df.columns:
                self._df[col] = feature_df[col].values
```

**Step 4: Run tests**

Run: `uv run pytest packages/harnessml-runner/tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add packages/harnessml-runner/src/harnessml/runner/pipeline.py packages/harnessml-runner/tests/
git commit -m "feat(runner): wire FeatureStore into PipelineRunner for declarative feature loading"
```

---

### Task 7: Integration Test — Full Declarative Feature Workflow

**Files:**
- Create: `packages/harnessml-runner/tests/test_declarative_features_integration.py`

**Context:** End-to-end test that exercises the full workflow: configure sources, add features of all four types via the MCP tool interface, verify auto-pairwise generation, resolve feature sets, and verify caching.

**Step 1: Write integration test**

```python
"""Integration test for declarative feature system end-to-end."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from harnessml.runner.config_writer import add_feature, available_features
from harnessml.runner.feature_store import FeatureStore
from harnessml.runner.schema import (
    DataConfig,
    FeatureDef,
    FeatureStoreConfig,
    FeatureType,
    PairwiseMode,
    SourceConfig,
)


def _setup_full_project(tmp_path: Path) -> Path:
    """Create a full project with sources, matchups, and config."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()

    rng = np.random.default_rng(42)

    # Config
    config_dir = project_dir / "config"
    config_dir.mkdir()
    config = {
        "data": {
            "features_file": "features.parquet",
            "target_column": "result",
            "key_columns": [],
            "time_column": "period_id",
            "sources": {
                "team_stats": {
                    "name": "team_stats",
                    "path": "data/raw/team_stats.parquet",
                },
            },
        },
    }
    (config_dir / "pipeline.yaml").write_text(yaml.dump(config))

    # Entity-level data
    data_dir = project_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    features_dir = data_dir / "features"
    features_dir.mkdir(parents=True)

    n_teams = 30
    entities = []
    for season in [2022, 2023, 2024]:
        for team_id in range(1, n_teams + 1):
            entities.append({
                "entity_id": team_id,
                "period_id": season,
                "adj_em": rng.standard_normal() * 10,
                "adj_tempo": rng.standard_normal() * 5 + 65,
                "adj_oe": rng.standard_normal() * 5 + 105,
                "win_rate": rng.uniform(0.2, 0.9),
            })
    pd.DataFrame(entities).to_parquet(raw_dir / "team_stats.parquet", index=False)

    # Matchup-level data
    n_matchups = 300
    matchups = []
    for i in range(n_matchups):
        season = rng.choice([2022, 2023, 2024])
        team_a = rng.integers(1, n_teams + 1)
        team_b = rng.integers(1, n_teams + 1)
        while team_b == team_a:
            team_b = rng.integers(1, n_teams + 1)
        matchups.append({
            "entity_a_id": int(team_a),
            "entity_b_id": int(team_b),
            "period_id": int(season),
            "result": int(rng.integers(0, 2)),
            "day_num": int(rng.integers(1, 155)),
            "is_neutral": int(rng.integers(0, 2)),
        })
    pd.DataFrame(matchups).to_parquet(features_dir / "features.parquet", index=False)

    return project_dir


class TestFullWorkflow:
    """End-to-end declarative feature workflow."""

    def test_team_feature_with_auto_pairwise(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig(
            sources={
                "team_stats": SourceConfig(
                    name="team_stats",
                    path="data/raw/team_stats.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        # Add team features
        r1 = store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="team_stats", column="adj_em",
            category="efficiency",
        ))
        r2 = store.add(FeatureDef(
            name="adj_tempo", type=FeatureType.TEAM,
            source="team_stats", column="adj_tempo",
            category="pace",
        ))

        # Should have team + auto pairwise
        all_features = store.available()
        names = [f.name for f in all_features]
        assert "adj_em" in names
        assert "diff_adj_em" in names
        assert "adj_tempo" in names
        assert "diff_adj_tempo" in names

    def test_regime_feature(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)

        result = store.add(FeatureDef(
            name="late_season", type=FeatureType.REGIME,
            condition="day_num > 100",
            category="temporal",
        ))

        series = store.compute("late_season")
        assert set(series.dropna().unique()).issubset({0.0, 1.0})

    def test_matchup_feature(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)

        result = store.add(FeatureDef(
            name="neutral_site", type=FeatureType.MATCHUP,
            column="is_neutral",
            category="context",
        ))

        series = store.compute("neutral_site")
        assert len(series) > 0

    def test_formula_pairwise(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig()
        store = FeatureStore(project_dir, config)

        result = store.add(FeatureDef(
            name="day_squared", type=FeatureType.PAIRWISE,
            formula="day_num ** 2",
        ))

        series = store.compute("day_squared")
        assert len(series) > 0

    def test_resolve_multiple_types(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig(
            sources={
                "team_stats": SourceConfig(
                    name="team_stats",
                    path="data/raw/team_stats.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="team_stats", column="adj_em",
            category="efficiency",
        ))
        store.add(FeatureDef(
            name="late_season", type=FeatureType.REGIME,
            condition="day_num > 100",
            category="temporal",
        ))
        store.add(FeatureDef(
            name="neutral_site", type=FeatureType.MATCHUP,
            column="is_neutral",
            category="context",
        ))

        df = store.resolve(["diff_adj_em", "late_season", "neutral_site"])
        assert "diff_adj_em" in df.columns
        assert "late_season" in df.columns
        assert "neutral_site" in df.columns

    def test_resolve_sets_by_category(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig(
            sources={
                "team_stats": SourceConfig(
                    name="team_stats",
                    path="data/raw/team_stats.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="team_stats", column="adj_em",
            category="efficiency",
        ))
        store.add(FeatureDef(
            name="adj_oe", type=FeatureType.TEAM,
            source="team_stats", column="adj_oe",
            category="efficiency",
        ))
        store.add(FeatureDef(
            name="adj_tempo", type=FeatureType.TEAM,
            source="team_stats", column="adj_tempo",
            category="pace",
        ))

        efficiency_cols = store.resolve_sets(["efficiency"])
        assert "diff_adj_em" in efficiency_cols
        assert "diff_adj_oe" in efficiency_cols
        assert "diff_adj_tempo" not in efficiency_cols

    def test_caching_persists_across_instances(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig(
            sources={
                "team_stats": SourceConfig(
                    name="team_stats",
                    path="data/raw/team_stats.parquet",
                ),
            },
        )

        # First instance — add and compute
        store1 = FeatureStore(project_dir, config)
        store1.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="team_stats", column="adj_em",
        ))
        series1 = store1.compute("diff_adj_em")

        # Second instance — should find cache
        config2 = DataConfig(
            sources=config.sources,
            feature_defs=dict(store1._registry),
        )
        store2 = FeatureStore(project_dir, config2)
        series2 = store2.compute("diff_adj_em")

        pd.testing.assert_series_equal(series1, series2)

    def test_remove_cascades(self, tmp_path):
        project_dir = _setup_full_project(tmp_path)
        config = DataConfig(
            sources={
                "team_stats": SourceConfig(
                    name="team_stats",
                    path="data/raw/team_stats.parquet",
                ),
            },
        )
        store = FeatureStore(project_dir, config)

        store.add(FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="team_stats", column="adj_em",
        ))

        assert len(store.available()) >= 2
        store.remove("adj_em")
        assert len(store.available()) == 0
```

**Step 2: Run integration tests**

Run: `uv run pytest packages/harnessml-runner/tests/test_declarative_features_integration.py -v`
Expected: PASS

**Step 3: Run full suite**

Run: `uv run pytest packages/harnessml-runner/tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add packages/harnessml-runner/tests/test_declarative_features_integration.py
git commit -m "test(runner): add integration tests for declarative feature system"
```

---

### Task 8: Final Verification and Cleanup

**Files:**
- Modify: `packages/harnessml-runner/src/harnessml/runner/__init__.py` (if exports needed)
- Review: All modified files

**Step 1: Run the full test suite**

Run: `uv run pytest packages/harnessml-runner/tests/ -v`
Expected: All tests PASS with 0 failures

**Step 2: Run the broader project tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 3: Verify exports**

Ensure `FeatureStore`, `FeatureCache`, `FeatureDef`, `FeatureType`, `PairwiseMode`, `FeatureStoreConfig` are importable:

```python
from harnessml.runner.feature_store import FeatureStore
from harnessml.runner.feature_cache import FeatureCache
from harnessml.runner.schema import FeatureDef, FeatureType, PairwiseMode, FeatureStoreConfig
```

**Step 4: Verify MCP tool works end-to-end**

Test that the HarnessML MCP server can expose the updated tools by checking the tool definitions parse correctly.

**Step 5: Commit any remaining changes**

```bash
git add -A
git commit -m "feat(runner): declarative feature system — complete implementation with tests"
```
