# harness-data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build harness-data, a standalone Python library for declarative data engineering — source ingestion, expression engine, transforms, profiling, and pipeline execution.

**Architecture:** Registry-based source management + first-class expression engine + protocol-driven transform engine + stateless pipeline runner. Every transform step is a single file implementing a protocol. The expression engine provides formula validation, function registry, and safe evaluation. The runner composes sources + transforms into a clean parquet + schema.json.

**Tech Stack:** Python 3.11+, pandas 2.0+, pyarrow 14+, polars 1.0+ (optional backend), pydantic 2.0+, pytest

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Sections 8 (harness-data) + 8.Expression Engine

**Testing note:** Every transform step, source adapter, and public function that has validation logic (raises `ValueError`, `FileNotFoundError`, etc.) MUST have error-path tests alongside the happy-path tests. For each step, test: missing required params, nonexistent column references, empty DataFrames, and type mismatches. The happy-path tests in this plan are the minimum — implementers should add error-path tests inline following the same pattern.

---

## File Structure

```
packages/harness-data/
├── pyproject.toml
├── src/harness/data/
│   ├── __init__.py                    # Public API exports
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── protocol.py               # Source protocol (load, validate, schema)
│   │   ├── registry.py               # SourceRegistry (add, get, list, remove, persist to YAML)
│   │   ├── freshness.py              # FreshnessTracker (staleness detection)
│   │   ├── file.py                   # FileSource (CSV, Parquet, Excel)
│   │   ├── url.py                    # UrlSource (HTTP/HTTPS fetch + parse)
│   │   └── api.py                    # ApiSource (REST with pagination + rate limiting)
│   ├── expressions/
│   │   ├── __init__.py
│   │   ├── engine.py                  # Expression parser + evaluator
│   │   ├── registry.py                # Function registry (register, list, describe, types)
│   │   ├── validator.py               # Validate expression against schema without executing
│   │   └── functions/
│   │       ├── __init__.py            # Auto-registers all function modules
│   │       ├── math.py                # abs, log, sqrt, exp, clip, sign, floor, ceil, round
│   │       ├── stats.py               # zscore, rank_pct
│   │       ├── comparison.py          # where, safe_div, minimum, maximum
│   │       └── null.py                # isna, fillna, coalesce
│   ├── transforms/
│   │   ├── __init__.py
│   │   ├── protocol.py               # TransformStep protocol + StepResult
│   │   ├── engine.py                 # TransformEngine (dispatch, apply_step, run_pipeline)
│   │   └── steps/
│   │       ├── __init__.py            # Auto-discovery of all step modules
│   │       ├── filter.py
│   │       ├── select.py
│   │       ├── derive.py
│   │       ├── cast.py
│   │       ├── fill.py
│   │       ├── sort.py
│   │       ├── head.py
│   │       ├── distinct.py
│   │       ├── rank.py
│   │       ├── isin.py
│   │       ├── null_indicator.py
│   │       ├── join.py
│   │       ├── union.py
│   │       ├── unpivot.py
│   │       ├── aggregate.py
│   │       ├── conditional_agg.py
│   │       ├── rolling.py
│   │       ├── lag.py
│   │       ├── ewm.py
│   │       ├── diff.py
│   │       ├── trend.py
│   │       ├── encode.py
│   │       ├── bin.py
│   │       └── datetime.py
│   ├── profiling/
│   │   ├── __init__.py
│   │   ├── profiler.py               # DataProfiler (column stats, type inference)
│   │   └── validation.py             # SchemaValidator (quality checks)
│   ├── runner.py                      # Stateless pipeline: sources + transforms → parquet + schema
│   └── workspace.py                   # Workspace I/O (read/write sources.yaml, transforms.yaml, data/)
└── tests/
    ├── conftest.py                    # Shared fixtures (sample DataFrames, temp workspace)
    ├── test_sources/
    │   ├── test_protocol.py
    │   ├── test_registry.py
    │   ├── test_freshness.py
    │   ├── test_file.py
    │   ├── test_url.py
    │   └── test_api.py
    ├── test_expressions/
    │   ├── test_engine.py
    │   ├── test_registry.py
    │   ├── test_validator.py
    │   └── test_functions.py
    ├── test_transforms/
    │   ├── test_engine.py
    │   ├── test_filter.py
    │   ├── test_select.py
    │   ├── test_derive.py
    │   ├── ... (one per step)
    │   └── test_datetime.py
    ├── test_profiling/
    │   ├── test_profiler.py
    │   └── test_validation.py
    ├── test_runner.py
    └── test_workspace.py
```

---

### Task 1: Project Scaffolding + Source Protocol

**Files:**
- Create: `packages/harness-data/pyproject.toml`
- Create: `packages/harness-data/src/harness/__init__.py`
- Create: `packages/harness-data/src/harness/data/__init__.py`
- Create: `packages/harness-data/src/harness/data/sources/__init__.py`
- Create: `packages/harness-data/src/harness/data/sources/protocol.py`
- Create: `packages/harness-data/tests/conftest.py`
- Create: `packages/harness-data/tests/test_sources/test_protocol.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "harness-data"
version = "0.1.0"
description = "Declarative data engineering library for the Harness ML platform"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "pyarrow>=14.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
polars = ["polars>=1.0.0"]
all = ["polars>=1.0.0", "openpyxl>=3.1", "requests>=2.31"]
dev = ["pytest>=8.0", "pytest-cov>=4.0"]

[tool.hatch.build.targets.wheel]
packages = ["src/harness"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create package structure with namespace package**

Create `packages/harness-data/src/harness/__init__.py` (empty — namespace package).
Create `packages/harness-data/src/harness/data/__init__.py`:

```python
"""harness-data: Declarative data engineering library."""
```

Create `packages/harness-data/src/harness/data/sources/__init__.py`:

```python
from harness.data.sources.protocol import Source, SourceMetadata
```

- [ ] **Step 3: Write failing test for Source protocol**

Create `packages/harness-data/tests/conftest.py`:

```python
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_df():
    """A simple DataFrame for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [85.0, 92.0, 78.0, 95.0, 88.0],
        "grade": ["B", "A", "C", "A", "B"],
        "enrolled": [True, True, False, True, True],
    })


@pytest.fixture
def numeric_df():
    """DataFrame with numeric columns for transform testing."""
    return pd.DataFrame({
        "entity_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        "period": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "points": [10.0, 15.0, 12.0, 20.0, 18.0, 22.0, 8.0, 9.0, 11.0],
        "rebounds": [5.0, 7.0, 6.0, 10.0, 9.0, 11.0, 3.0, 4.0, 5.0],
        "target": [1, 0, 1, 1, 1, 0, 0, 0, 1],
    })


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "clean").mkdir()
    return tmp_path


@pytest.fixture
def sample_csv(temp_workspace, sample_df):
    """Write sample_df to a CSV in the temp workspace."""
    path = temp_workspace / "data" / "raw" / "sample.csv"
    sample_df.to_csv(path, index=False)
    return path
```

Create `packages/harness-data/tests/test_sources/__init__.py` (empty).
Create `packages/harness-data/tests/test_sources/test_protocol.py`:

```python
from harness.data.sources.protocol import Source, SourceMetadata


class TestSourceProtocol:
    def test_source_metadata_creation(self):
        meta = SourceMetadata(
            name="test_source",
            source_type="file",
            row_count=100,
            columns=["id", "name", "score"],
            column_types={"id": "int64", "name": "object", "score": "float64"},
        )
        assert meta.name == "test_source"
        assert meta.source_type == "file"
        assert meta.row_count == 100
        assert len(meta.columns) == 3

    def test_source_metadata_defaults(self):
        meta = SourceMetadata(name="minimal", source_type="file")
        assert meta.row_count is None
        assert meta.columns == []
        assert meta.column_types == {}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_protocol.py -v`
Expected: FAIL with ImportError

- [ ] **Step 5: Implement Source protocol**

Create `packages/harness-data/src/harness/data/sources/protocol.py`:

```python
"""Source protocol — the contract all source adapters implement."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field
from typing import Any, Protocol, runtime_checkable


class SourceMetadata(BaseModel):
    """Metadata about a loaded source."""

    name: str
    source_type: str  # "file", "url", "api"
    row_count: int | None = None
    columns: list[str] = Field(default_factory=list)
    column_types: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, name: str, source_type: str, df: pd.DataFrame) -> SourceMetadata:
        """Create metadata from a loaded DataFrame."""
        return cls(
            name=name,
            source_type=source_type,
            row_count=len(df),
            columns=list(df.columns),
            column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
        )


class SourceConfig(BaseModel):
    """Configuration for a data source."""

    name: str
    source_type: str = "file"  # "file", "url", "api"
    path: str | None = None
    url: str | None = None
    format: str = "auto"  # "csv", "parquet", "excel", "json", "auto"
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


@runtime_checkable
class Source(Protocol):
    """Protocol that all source adapters implement."""

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Load data from the source and return a DataFrame."""
        ...

    def validate(self, config: SourceConfig) -> list[str]:
        """Validate source config. Returns list of error messages (empty = valid)."""
        ...

    def refresh(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Re-fetch data from the source. Default: delegates to load()."""
        ...
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_protocol.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add packages/harness-data/
git commit -m "feat(harness-data): project scaffolding + source protocol"
```

---

### Task 2: File Source Adapter

**Files:**
- Create: `packages/harness-data/src/harness/data/sources/file.py`
- Create: `packages/harness-data/tests/test_sources/test_file.py`

- [ ] **Step 1: Write failing tests for FileSource**

Create `packages/harness-data/tests/test_sources/test_file.py`:

```python
import pandas as pd
import pytest
from pathlib import Path

from harness.data.sources.file import FileSource
from harness.data.sources.protocol import SourceConfig


class TestFileSource:
    def test_load_csv(self, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path=str(sample_csv))
        df = source.load(config)
        assert len(df) == 5
        assert "score" in df.columns

    def test_load_parquet(self, temp_workspace, sample_df):
        path = temp_workspace / "data" / "raw" / "sample.parquet"
        sample_df.to_parquet(path, index=False)
        source = FileSource()
        config = SourceConfig(name="test", path=str(path))
        df = source.load(config)
        assert len(df) == 5

    def test_load_csv_with_base_dir(self, temp_workspace, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path="data/raw/sample.csv")
        df = source.load(config, base_dir=str(temp_workspace))
        assert len(df) == 5

    def test_auto_detect_format(self, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path=str(sample_csv), format="auto")
        df = source.load(config)
        assert len(df) == 5

    def test_validate_missing_path(self):
        source = FileSource()
        config = SourceConfig(name="test", path=None)
        errors = source.validate(config)
        assert len(errors) > 0
        assert "path" in errors[0].lower()

    def test_validate_nonexistent_file(self):
        source = FileSource()
        config = SourceConfig(name="test", path="/nonexistent/file.csv")
        errors = source.validate(config)
        assert len(errors) > 0

    def test_validate_valid_file(self, sample_csv):
        source = FileSource()
        config = SourceConfig(name="test", path=str(sample_csv))
        errors = source.validate(config)
        assert len(errors) == 0

    def test_implements_source_protocol(self):
        from harness.data.sources.protocol import Source
        assert isinstance(FileSource(), Source)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_file.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement FileSource**

Create `packages/harness-data/src/harness/data/sources/file.py`:

```python
"""File source adapter — loads CSV, Parquet, and Excel files."""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from harness.data.sources.protocol import SourceConfig


class FileSource:
    """Load data from local files (CSV, Parquet, Excel)."""

    LOADERS = {
        "csv": pd.read_csv,
        "parquet": pd.read_parquet,
        "excel": lambda path, **kw: pd.read_excel(path, **kw),
    }

    FORMAT_EXTENSIONS = {
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
        ".xlsx": "excel",
        ".xls": "excel",
    }

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Load data from a file path."""
        path = self._resolve_path(config.path, base_dir)
        fmt = self._detect_format(path, config.format)
        loader = self.LOADERS.get(fmt)
        if loader is None:
            raise ValueError(f"Unsupported format: {fmt}")
        return loader(str(path), **config.params)

    def validate(self, config: SourceConfig) -> list[str]:
        """Validate that the file exists and format is supported."""
        errors = []
        if not config.path:
            errors.append("Source path is required for file sources")
            return errors
        path = Path(config.path)
        if not path.is_absolute() and not path.exists():
            # Relative paths can't be fully validated without base_dir
            pass
        elif path.is_absolute() and not path.exists():
            errors.append(f"File not found: {config.path}")
        fmt = self._detect_format(path, config.format)
        if fmt not in self.LOADERS:
            errors.append(f"Unsupported format: {fmt}")
        return errors

    def refresh(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        """Re-fetch data. For files, this is the same as load (re-read from disk)."""
        return self.load(config, base_dir)

    def _resolve_path(self, path: str | None, base_dir: str | None) -> Path:
        if path is None:
            raise ValueError("Source path is required")
        p = Path(path)
        if not p.is_absolute() and base_dir:
            p = Path(base_dir) / p
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {p}")
        return p

    def _detect_format(self, path: Path, configured_format: str) -> str:
        if configured_format != "auto":
            return configured_format
        suffix = Path(path).suffix.lower()
        fmt = self.FORMAT_EXTENSIONS.get(suffix)
        if fmt is None:
            raise ValueError(f"Cannot auto-detect format for extension: {suffix}")
        return fmt
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_file.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/sources/file.py packages/harness-data/tests/test_sources/test_file.py
git commit -m "feat(harness-data): file source adapter (CSV, Parquet, Excel)"
```

---

### Task 3: Source Registry + Freshness Tracking

**Files:**
- Create: `packages/harness-data/src/harness/data/sources/registry.py`
- Create: `packages/harness-data/src/harness/data/sources/freshness.py`
- Create: `packages/harness-data/tests/test_sources/test_registry.py`
- Create: `packages/harness-data/tests/test_sources/test_freshness.py`

- [ ] **Step 1: Write failing tests for SourceRegistry**

Create `packages/harness-data/tests/test_sources/test_registry.py`:

```python
import pytest
import yaml
from pathlib import Path

from harness.data.sources.registry import SourceRegistry
from harness.data.sources.protocol import SourceConfig


class TestSourceRegistry:
    def test_add_and_get(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        config = SourceConfig(name="games", source_type="file", path="data/raw/games.csv")
        registry.add(config)
        result = registry.get("games")
        assert result is not None
        assert result.name == "games"
        assert result.path == "data/raw/games.csv"

    def test_add_duplicate_raises(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        config = SourceConfig(name="games", source_type="file", path="data/raw/games.csv")
        registry.add(config)
        with pytest.raises(ValueError, match="already exists"):
            registry.add(config)

    def test_add_duplicate_with_overwrite(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        config = SourceConfig(name="games", source_type="file", path="data/raw/games.csv")
        registry.add(config)
        config2 = SourceConfig(name="games", source_type="file", path="data/raw/games_v2.csv")
        registry.add(config2, overwrite=True)
        result = registry.get("games")
        assert result.path == "data/raw/games_v2.csv"

    def test_list_all(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        registry.add(SourceConfig(name="a", source_type="file", path="a.csv"))
        registry.add(SourceConfig(name="b", source_type="file", path="b.csv"))
        sources = registry.list_all()
        assert len(sources) == 2
        assert {s.name for s in sources} == {"a", "b"}

    def test_remove(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        registry.add(SourceConfig(name="games", source_type="file", path="a.csv"))
        registry.remove("games")
        assert registry.get("games") is None

    def test_persistence(self, tmp_path):
        registry1 = SourceRegistry(tmp_path)
        registry1.add(SourceConfig(name="games", source_type="file", path="a.csv"))

        registry2 = SourceRegistry(tmp_path)
        result = registry2.get("games")
        assert result is not None
        assert result.name == "games"

    def test_persists_as_yaml(self, tmp_path):
        registry = SourceRegistry(tmp_path)
        registry.add(SourceConfig(name="games", source_type="file", path="a.csv"))
        yaml_path = tmp_path / "sources.yaml"
        assert yaml_path.exists()
        content = yaml.safe_load(yaml_path.read_text())
        assert "games" in content["sources"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SourceRegistry**

Create `packages/harness-data/src/harness/data/sources/registry.py`:

```python
"""Source registry — persists source declarations to sources.yaml."""

from __future__ import annotations

import yaml
from pathlib import Path

from harness.data.sources.protocol import SourceConfig


class SourceRegistry:
    """Manages source declarations. Persists to sources.yaml in the given directory."""

    def __init__(self, directory: str | Path):
        self._dir = Path(directory)
        self._path = self._dir / "sources.yaml"
        self._sources: dict[str, SourceConfig] = {}
        self._load()

    def add(self, config: SourceConfig, *, overwrite: bool = False) -> None:
        """Register a source. Raises ValueError if name exists and overwrite=False."""
        if config.name in self._sources and not overwrite:
            raise ValueError(f"Source '{config.name}' already exists. Use overwrite=True to replace.")
        self._sources[config.name] = config
        self._save()

    def get(self, name: str) -> SourceConfig | None:
        """Get a source by name, or None if not found."""
        return self._sources.get(name)

    def list_all(self) -> list[SourceConfig]:
        """List all registered sources."""
        return list(self._sources.values())

    def remove(self, name: str) -> None:
        """Remove a source by name."""
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
        self._path.write_text(yaml.dump({"sources": sources}, default_flow_style=False, sort_keys=False))
```

- [ ] **Step 4: Run registry tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_registry.py -v`
Expected: PASS

- [ ] **Step 5: Write failing tests for FreshnessTracker**

Create `packages/harness-data/tests/test_sources/test_freshness.py`:

```python
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from harness.data.sources.freshness import FreshnessTracker


class TestFreshnessTracker:
    def test_record_and_get(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        tracker.record_fetch("games", row_count=500)
        info = tracker.get_info("games")
        assert info is not None
        assert info["row_count"] == 500
        assert "last_fetched" in info

    def test_is_stale_manual(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        tracker.record_fetch("games")
        assert not tracker.is_stale("games", "manual")

    def test_is_stale_daily_fresh(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        tracker.record_fetch("games")
        assert not tracker.is_stale("games", "daily")

    def test_is_stale_unknown_source(self, tmp_path):
        tracker = FreshnessTracker(tmp_path / "freshness.json")
        assert tracker.is_stale("unknown", "daily")

    def test_persistence(self, tmp_path):
        path = tmp_path / "freshness.json"
        tracker1 = FreshnessTracker(path)
        tracker1.record_fetch("games", row_count=100)

        tracker2 = FreshnessTracker(path)
        info = tracker2.get_info("games")
        assert info is not None
        assert info["row_count"] == 100
```

- [ ] **Step 5b: Run freshness tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_freshness.py -v`
Expected: FAIL with ImportError

- [ ] **Step 6: Implement FreshnessTracker**

Create `packages/harness-data/src/harness/data/sources/freshness.py`:

```python
"""Freshness tracking — detect when sources are stale."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


FREQUENCY_DELTAS = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
    "yearly": timedelta(days=365),
}


class FreshnessTracker:
    """Track when sources were last fetched and whether they're stale."""

    def __init__(self, state_file: str | Path):
        self._path = Path(state_file)
        self._state: dict[str, dict] = {}
        self._load()

    def record_fetch(self, source_name: str, row_count: int = 0) -> None:
        """Record that a source was fetched now."""
        self._state[source_name] = {
            "last_fetched": datetime.utcnow().isoformat(),
            "row_count": row_count,
        }
        self._save()

    def is_stale(self, source_name: str, refresh_frequency: str) -> bool:
        """Check if a source is stale based on its refresh frequency."""
        if refresh_frequency == "manual":
            return False
        info = self._state.get(source_name)
        if info is None:
            return True
        delta = FREQUENCY_DELTAS.get(refresh_frequency)
        if delta is None:
            return False
        last = datetime.fromisoformat(info["last_fetched"])
        return datetime.utcnow() - last > delta

    def get_info(self, source_name: str) -> dict | None:
        """Get fetch info for a source."""
        return self._state.get(source_name)

    def check_all(self, sources: list[tuple[str, str]]) -> list[dict]:
        """Check all sources. Takes list of (name, frequency). Returns stale ones."""
        return [
            {"name": name, "frequency": freq, **self._state.get(name, {})}
            for name, freq in sources
            if self.is_stale(name, freq)
        ]

    def _load(self) -> None:
        if self._path.exists():
            self._state = json.loads(self._path.read_text())

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._state, indent=2))
```

- [ ] **Step 7: Run all source tests**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add packages/harness-data/src/harness/data/sources/ packages/harness-data/tests/test_sources/
git commit -m "feat(harness-data): source registry + freshness tracking"
```

---

### Task 4: Expression Engine + Transform Protocol + Engine

This task builds the expression engine (first-class formula evaluation system) and the transform protocol + engine. The expression engine is a standalone subsystem used by the derive step, harness-ml feature resolution, and filter expressions.

**Files:**
- Create: `packages/harness-data/src/harness/data/expressions/__init__.py`
- Create: `packages/harness-data/src/harness/data/expressions/engine.py`
- Create: `packages/harness-data/src/harness/data/expressions/registry.py`
- Create: `packages/harness-data/src/harness/data/expressions/validator.py`
- Create: `packages/harness-data/src/harness/data/expressions/functions/__init__.py`
- Create: `packages/harness-data/src/harness/data/expressions/functions/math.py`
- Create: `packages/harness-data/src/harness/data/expressions/functions/stats.py`
- Create: `packages/harness-data/src/harness/data/expressions/functions/comparison.py`
- Create: `packages/harness-data/src/harness/data/expressions/functions/null.py`
- Create: `packages/harness-data/src/harness/data/transforms/__init__.py`
- Create: `packages/harness-data/src/harness/data/transforms/protocol.py`
- Create: `packages/harness-data/src/harness/data/transforms/engine.py`
- Create: `packages/harness-data/src/harness/data/transforms/steps/__init__.py`
- Create: `packages/harness-data/tests/test_expressions/__init__.py`
- Create: `packages/harness-data/tests/test_expressions/test_engine.py`
- Create: `packages/harness-data/tests/test_expressions/test_registry.py`
- Create: `packages/harness-data/tests/test_expressions/test_validator.py`
- Create: `packages/harness-data/tests/test_expressions/test_functions.py`
- Create: `packages/harness-data/tests/test_transforms/__init__.py`
- Create: `packages/harness-data/tests/test_transforms/test_engine.py`

- [ ] **Step 1: Write failing tests for transform protocol and engine**

Create `packages/harness-data/tests/test_transforms/test_engine.py`:

```python
import pandas as pd
import pytest

from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


class TestTransformEngine:
    def test_register_and_list_steps(self):
        engine = TransformEngine()
        assert "filter" in engine.available_steps()

    def test_apply_unknown_step_raises(self, sample_df):
        engine = TransformEngine()
        config = StepConfig(op="nonexistent_step")
        with pytest.raises(ValueError, match="Unknown"):
            engine.apply_step(sample_df, config)

    def test_run_pipeline_empty_steps(self, sample_df):
        engine = TransformEngine()
        result = engine.run_pipeline(sample_df, [])
        assert len(result) == len(sample_df)

    def test_run_pipeline_chained_steps(self, sample_df):
        engine = TransformEngine()
        steps = [
            StepConfig(op="filter", params={"expr": "score > 80"}),
            StepConfig(op="select", params={"columns": ["name", "score"]}),
        ]
        result = engine.run_pipeline(sample_df, steps)
        assert "name" in result.columns
        assert "grade" not in result.columns
        assert all(result["score"] > 80)
```

- [ ] **Step 2: Write failing tests for expression engine**

Create `packages/harness-data/tests/test_expressions/__init__.py` (empty).
Create `packages/harness-data/tests/test_expressions/test_engine.py`:

```python
import pandas as pd
import pytest
import numpy as np

from harness.data.expressions.engine import ExpressionEngine


class TestExpressionEngine:
    def test_simple_arithmetic(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "score * 2")
        assert result.iloc[0] == 170.0

    def test_column_reference(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "score + 10")
        assert result.iloc[0] == 95.0

    def test_registered_function(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "abs(score - 90)")
        assert all(result >= 0)

    def test_nested_functions(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "abs(zscore(score))")
        assert all(result >= 0)

    def test_rejects_dangerous_builtins(self, sample_df):
        engine = ExpressionEngine()
        with pytest.raises((ValueError, TypeError, KeyError)):
            engine.evaluate(sample_df, "__import__('os').system('ls')")

    def test_rejects_attribute_access(self, sample_df):
        engine = ExpressionEngine()
        with pytest.raises((ValueError, TypeError, KeyError, AttributeError)):
            engine.evaluate(sample_df, "score.__class__.__bases__")
```

Create `packages/harness-data/tests/test_expressions/test_registry.py`:

```python
import pytest
from harness.data.expressions.registry import FunctionRegistry


class TestFunctionRegistry:
    def test_list_functions(self):
        registry = FunctionRegistry()
        registry.load_defaults()
        funcs = registry.list_functions()
        assert "abs" in funcs
        assert "zscore" in funcs
        assert "safe_div" in funcs
        assert "rank_pct" in funcs

    def test_get_function(self):
        registry = FunctionRegistry()
        registry.load_defaults()
        func = registry.get("abs")
        assert func is not None
        assert func.name == "abs"
        assert func.description is not None

    def test_register_custom(self):
        registry = FunctionRegistry()
        import numpy as np
        registry.register("my_func", np.square, description="Square a value")
        assert "my_func" in registry.list_functions()

    def test_get_unknown_returns_none(self):
        registry = FunctionRegistry()
        assert registry.get("nonexistent") is None
```

Create `packages/harness-data/tests/test_expressions/test_validator.py`:

```python
import pytest
from harness.data.expressions.validator import ExpressionValidator


class TestExpressionValidator:
    def test_valid_expression(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score", "rating"], "column_types": {"score": "float64", "rating": "float64"}}
        result = validator.validate("score * 2", schema)
        assert result.is_valid

    def test_missing_column(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score"], "column_types": {"score": "float64"}}
        result = validator.validate("momentum * 2", schema)
        assert not result.is_valid
        assert "momentum" in result.errors[0]
        assert "score" in result.suggestion  # suggest available columns

    def test_unknown_function(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score"], "column_types": {"score": "float64"}}
        result = validator.validate("bad_func(score)", schema)
        assert not result.is_valid
        assert "bad_func" in result.errors[0]

    def test_valid_function(self):
        validator = ExpressionValidator()
        schema = {"columns": ["score"], "column_types": {"score": "float64"}}
        result = validator.validate("abs(score - 50)", schema)
        assert result.is_valid
```

Create `packages/harness-data/tests/test_expressions/test_functions.py`:

```python
import pandas as pd
import numpy as np
import pytest
from harness.data.expressions.engine import ExpressionEngine


class TestMathFunctions:
    def test_abs(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "abs(score - 90)")
        assert all(result >= 0)

    def test_log(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "log(score)")
        assert all(result > 0)

    def test_sqrt(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "sqrt(score)")
        assert all(result > 0)


class TestStatsFunctions:
    def test_zscore(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "zscore(score)")
        assert abs(result.mean()) < 1e-10

    def test_rank_pct(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "rank_pct(score)")
        assert result.max() <= 1.0
        assert result.min() > 0.0


class TestComparisonFunctions:
    def test_safe_div(self, sample_df):
        engine = ExpressionEngine()
        df = sample_df.copy()
        df["zero"] = 0
        result = engine.evaluate(df, "safe_div(score, zero)")
        assert all(result == 0.0)

    def test_where(self, sample_df):
        engine = ExpressionEngine()
        result = engine.evaluate(sample_df, "where(score > 90, 1, 0)")
        assert result.sum() == 2  # 92 and 95
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/ -v`
Expected: FAIL

- [ ] **Step 4: Implement expression engine (registry, functions, validator, engine)**

Create the expression engine package. This is a substantial subsystem — the function registry, built-in functions, expression validator, and evaluation engine. Implementation details are in the spec (Section 8, Expression Engine). Key contracts:

- `ExpressionEngine.evaluate(df, expression) -> pd.Series` — evaluate against a DataFrame
- `FunctionRegistry.register(name, fn, description)` — register a function
- `FunctionRegistry.list_functions() -> list[str]` — list available functions
- `FunctionRegistry.get(name) -> FunctionInfo | None` — get function metadata
- `ExpressionValidator.validate(expression, schema) -> ValidationResult` — validate without executing
- Built-in functions: abs, log, sqrt, exp, clip, sign, floor, ceil, round (math); zscore, rank_pct (stats); where, safe_div, minimum, maximum (comparison); isna, fillna, coalesce (null)
- Safe evaluation only — no raw eval/exec, no attribute access, whitelisted functions only
- Agent-optimized error messages: "Column 'momentum' not found. Available: [score, rating, ...]"

The expression engine is used by the derive transform step and will be used by harness-ml for feature formula resolution.

- [ ] **Step 5: Run expression tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_expressions/ -v`
Expected: ALL PASS

- [ ] **Step 6: Implement transform protocol**

Create `packages/harness-data/src/harness/data/transforms/__init__.py`:

```python
from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig
```

Create `packages/harness-data/src/harness/data/transforms/protocol.py`:

```python
"""Transform step protocol — the contract all transform steps implement."""

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Callable
import pandas as pd


class StepConfig(BaseModel):
    """Configuration for a single transform step."""

    op: str  # Step type name (e.g., "filter", "join", "rolling")
    params: dict[str, Any] = Field(default_factory=dict)


# A step function takes a DataFrame + params dict and returns a DataFrame.
# For steps that need to resolve other sources/views (join, union),
# a resolver callback is passed via params["_resolver"].
StepFunction = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]
```

- [ ] **Step 5: Implement safe_eval**

Create `packages/harness-data/src/harness/data/transforms/safe_eval.py`:

```python
"""Safe formula evaluation — pd.eval with whitelisted functions only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series):
    """Z-score normalization."""
    std = series.std()
    if std == 0:
        return series * 0.0
    return (series - series.mean()) / std


def _rank_pct(series):
    """Percentile rank (0-1)."""
    return series.rank(pct=True)


def _safe_div(a, b):
    """Division that returns 0 when denominator is 0."""
    return np.where(b != 0, a / b, 0.0)


WHITELIST: dict[str, Any] = {
    "abs": np.abs,
    "log": np.log1p,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "clip": np.clip,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round_,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
    "isna": pd.isna,
    "fillna": lambda s, v: s.fillna(v) if hasattr(s, "fillna") else s,
    "zscore": _zscore,
    "rank_pct": _rank_pct,
    "safe_div": _safe_div,
}


def safe_eval(df: pd.DataFrame, expression: str) -> pd.Series:
    """Evaluate an expression against a DataFrame using only whitelisted functions.

    Uses pd.eval with a restricted local namespace. No raw eval/exec.
    """
    local_ns = {**WHITELIST}
    # Add DataFrame columns to namespace
    for col in df.columns:
        local_ns[col] = df[col]

    try:
        result = pd.eval(expression, local_dict=local_ns, engine="python")
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expression}': {e}") from e

    if isinstance(result, pd.DataFrame):
        raise ValueError(f"Expression returned DataFrame, expected Series: '{expression}'")
    return result
```

- [ ] **Step 6: Implement TransformEngine with auto-discovery**

Create `packages/harness-data/src/harness/data/transforms/steps/__init__.py`:

```python
"""Transform steps — auto-discovered by the engine."""
```

Create `packages/harness-data/src/harness/data/transforms/engine.py`:

```python
"""Transform engine — dispatches steps by config, runs pipelines."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from harness.data.transforms.protocol import StepConfig, StepFunction


class TransformEngine:
    """Registry-based transform engine. Auto-discovers step modules."""

    def __init__(self):
        self._steps: dict[str, StepFunction] = {}
        self._discover_steps()

    def apply_step(
        self,
        df: pd.DataFrame,
        config: StepConfig | None = None,
        resolver: Callable[[str], pd.DataFrame] | None = None,
        *,
        step_type: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Apply a single transform step to a DataFrame.

        Can be called two ways:
          engine.apply_step(df, StepConfig(op="rolling", params={...}))
          engine.apply_step(df, step_type="rolling", params={...})
        """
        if config is None and step_type is not None:
            config = StepConfig(op=step_type, params=params or {})
        if config is None:
            raise ValueError("Either config or step_type must be provided")
        step_fn = self._steps.get(config.op)
        if step_fn is None:
            available = ", ".join(sorted(self._steps.keys()))
            raise ValueError(f"Unknown transform step: '{config.op}'. Available: {available}")
        step_params = dict(config.params)
        if resolver is not None:
            step_params["_resolver"] = resolver
        return step_fn(df, step_params)

    def run_pipeline(
        self,
        df: pd.DataFrame,
        steps: list[StepConfig],
        resolver: Callable[[str], pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """Apply a sequence of transform steps to a DataFrame."""
        result = df.copy()
        for step in steps:
            result = self.apply_step(result, step, resolver)
        return result

    def available_steps(self) -> list[str]:
        """List all registered step types."""
        return sorted(self._steps.keys())

    def register(self, name: str, fn: StepFunction) -> None:
        """Manually register a step function."""
        self._steps[name] = fn

    def _discover_steps(self) -> None:
        """Auto-discover step modules from the steps/ package."""
        steps_package = importlib.import_module("harness.data.transforms.steps")
        steps_dir = Path(steps_package.__file__).parent

        for module_info in pkgutil.iter_modules([str(steps_dir)]):
            if module_info.name.startswith("_"):
                continue
            module = importlib.import_module(f"harness.data.transforms.steps.{module_info.name}")
            # Each step module must have a `step` function and a `NAME` constant
            if hasattr(module, "step") and hasattr(module, "NAME"):
                self._steps[module.NAME] = module.step
```

- [ ] **Step 7: Implement the first two steps (filter + select) to make engine tests pass**

Create `packages/harness-data/src/harness/data/transforms/steps/filter.py`:

```python
"""Filter step — keep rows matching an expression."""

from __future__ import annotations

from typing import Any
import pandas as pd

NAME = "filter"


def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Filter rows using a pandas query expression.

    Params:
        expr (str): Pandas query expression (e.g., "score > 80").
    """
    expr = params.get("expr")
    if not expr:
        raise ValueError("filter step requires 'expr' parameter")
    return df.query(expr).reset_index(drop=True)
```

Create `packages/harness-data/src/harness/data/transforms/steps/select.py`:

```python
"""Select step — keep or rename columns."""

from __future__ import annotations

from typing import Any
import pandas as pd

NAME = "select"


def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Select and optionally rename columns.

    Params:
        columns (list[str] | dict[str, str]): Column names to keep,
            or {new_name: old_name} mapping for rename + select.
    """
    columns = params.get("columns")
    if columns is None:
        raise ValueError("select step requires 'columns' parameter")

    if isinstance(columns, dict):
        # {new_name: old_name} — select and rename
        df = df[list(columns.values())]
        df = df.rename(columns={v: k for k, v in columns.items()})
        return df
    elif isinstance(columns, list):
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        return df[columns]
    else:
        raise TypeError(f"columns must be list or dict, got {type(columns)}")
```

- [ ] **Step 8: Run all transform tests**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/ -v`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add packages/harness-data/src/harness/data/transforms/ packages/harness-data/tests/test_transforms/
git commit -m "feat(harness-data): expression engine, transform protocol + engine, filter + select steps"
```

---

### Task 5: Core Transform Steps (derive, cast, fill, sort, head, distinct)

**Files:**
- Create: `packages/harness-data/src/harness/data/transforms/steps/{derive,cast,fill,sort,head,distinct}.py`
- Create: `packages/harness-data/tests/test_transforms/test_steps_core.py`

- [ ] **Step 1: Write failing tests for all 6 steps**

Create `packages/harness-data/tests/test_transforms/test_steps_core.py`:

```python
import pandas as pd
import numpy as np
import pytest

from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


@pytest.fixture
def engine():
    return TransformEngine()


class TestDeriveStep:
    def test_simple_arithmetic(self, engine, sample_df):
        config = StepConfig(op="derive", params={"columns": {"double_score": "score * 2"}})
        result = engine.apply_step(sample_df, config)
        assert "double_score" in result.columns
        assert result["double_score"].iloc[0] == 170.0

    def test_multiple_columns(self, engine, sample_df):
        config = StepConfig(op="derive", params={
            "columns": {"double": "score * 2", "half": "score / 2"}
        })
        result = engine.apply_step(sample_df, config)
        assert "double" in result.columns
        assert "half" in result.columns


class TestCastStep:
    def test_cast_to_int(self, engine, sample_df):
        config = StepConfig(op="cast", params={"columns": {"score": "int"}})
        result = engine.apply_step(sample_df, config)
        assert result["score"].dtype == np.int64 or result["score"].dtype == int

    def test_cast_to_str(self, engine, sample_df):
        config = StepConfig(op="cast", params={"columns": {"id": "str"}})
        result = engine.apply_step(sample_df, config)
        assert result["id"].dtype == object


class TestFillStep:
    def test_fill_with_value(self, engine):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, "x", "y"]})
        config = StepConfig(op="fill", params={"columns": {"a": 0.0}})
        result = engine.apply_step(df, config)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == 0.0

    def test_fill_with_strategy(self, engine):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        config = StepConfig(op="fill", params={"strategy": "median"})
        result = engine.apply_step(df, config)
        assert result["a"].isna().sum() == 0
        assert result["a"].iloc[1] == 2.0


class TestSortStep:
    def test_sort_ascending(self, engine, sample_df):
        config = StepConfig(op="sort", params={"by": "score", "ascending": True})
        result = engine.apply_step(sample_df, config)
        assert result["score"].iloc[0] == 78.0

    def test_sort_descending(self, engine, sample_df):
        config = StepConfig(op="sort", params={"by": "score", "ascending": False})
        result = engine.apply_step(sample_df, config)
        assert result["score"].iloc[0] == 95.0


class TestHeadStep:
    def test_head_n(self, engine, sample_df):
        config = StepConfig(op="head", params={"n": 3})
        result = engine.apply_step(sample_df, config)
        assert len(result) == 3

    def test_head_with_sort(self, engine, sample_df):
        config = StepConfig(op="head", params={"n": 2, "order_by": "score", "ascending": False})
        result = engine.apply_step(sample_df, config)
        assert len(result) == 2
        assert result["score"].iloc[0] == 95.0


class TestDistinctStep:
    def test_distinct_all(self, engine):
        df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})
        config = StepConfig(op="distinct", params={})
        result = engine.apply_step(df, config)
        assert len(result) == 2

    def test_distinct_subset(self, engine):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "y", "x"]})
        config = StepConfig(op="distinct", params={"columns": ["a"]})
        result = engine.apply_step(df, config)
        assert len(result) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_core.py -v`
Expected: FAIL

- [ ] **Step 3: Implement all 6 steps**

Create each step file following the same `NAME` + `step(df, params)` pattern. Each step is a self-contained module.

`packages/harness-data/src/harness/data/transforms/steps/derive.py`:
```python
"""Derive step — create new columns from expressions."""
from __future__ import annotations
from typing import Any
import pandas as pd
from harness.data.expressions.engine import ExpressionEngine

NAME = "derive"

_engine = ExpressionEngine()

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns")
    if not columns or not isinstance(columns, dict):
        raise ValueError("derive step requires 'columns' dict of {name: expression}")
    result = df.copy()
    for col_name, expr in columns.items():
        result[col_name] = _engine.evaluate(result, expr)
    return result
```

`packages/harness-data/src/harness/data/transforms/steps/cast.py`:
```python
"""Cast step — change column types."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "cast"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns")
    if not columns or not isinstance(columns, dict):
        raise ValueError("cast step requires 'columns' dict of {column: type}")
    result = df.copy()
    for col, dtype in columns.items():
        if col not in result.columns:
            raise ValueError(f"Column not found: {col}")
        result[col] = result[col].astype(dtype)
    return result
```

`packages/harness-data/src/harness/data/transforms/steps/fill.py`:
```python
"""Fill step — handle null values."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "fill"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    columns = params.get("columns")  # {col: value} for per-column fills
    strategy = params.get("strategy")  # "median", "mean", "zero", "mode", "ffill"

    if columns and isinstance(columns, dict):
        for col, value in columns.items():
            if col in result.columns:
                result[col] = result[col].fillna(value)
    elif strategy:
        numeric_cols = result.select_dtypes(include="number").columns
        if strategy == "median":
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
        elif strategy == "mean":
            result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        elif strategy == "zero":
            result[numeric_cols] = result[numeric_cols].fillna(0)
        elif strategy == "mode":
            for col in result.columns:
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val.iloc[0])
        elif strategy == "ffill":
            result = result.ffill()
        else:
            raise ValueError(f"Unknown fill strategy: {strategy}")
    return result
```

`packages/harness-data/src/harness/data/transforms/steps/sort.py`:
```python
"""Sort step — order rows."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "sort"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    by = params.get("by")
    if not by:
        raise ValueError("sort step requires 'by' parameter")
    ascending = params.get("ascending", True)
    if isinstance(by, str):
        by = [by]
    return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/head.py`:
```python
"""Head step — take first/last N rows, optionally per group."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "head"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    n = params.get("n", 10)
    order_by = params.get("order_by")
    ascending = params.get("ascending", True)
    keys = params.get("keys")

    result = df
    if order_by:
        if isinstance(order_by, str):
            order_by = [order_by]
        result = result.sort_values(by=order_by, ascending=ascending)

    if keys:
        return result.groupby(keys, sort=False).head(n).reset_index(drop=True)
    return result.head(n).reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/distinct.py`:
```python
"""Distinct step — deduplicate rows."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "distinct"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns")
    keep = params.get("keep", "first")
    subset = columns if columns else None
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_core.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/transforms/steps/ packages/harness-data/tests/test_transforms/test_steps_core.py
git commit -m "feat(harness-data): core transform steps (derive, cast, fill, sort, head, distinct)"
```

---

### Task 6: Row Operation Steps (rank, isin, null_indicator)

**Files:**
- Create: `packages/harness-data/src/harness/data/transforms/steps/{rank,isin,null_indicator}.py`
- Create: `packages/harness-data/tests/test_transforms/test_steps_row_ops.py`

- [ ] **Step 1: Write failing tests**

Create `packages/harness-data/tests/test_transforms/test_steps_row_ops.py`:

```python
import pandas as pd
import numpy as np
import pytest

from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


@pytest.fixture
def engine():
    return TransformEngine()


class TestRankStep:
    def test_rank_column(self, engine, sample_df):
        config = StepConfig(op="rank", params={
            "columns": {"score_rank": "score"},
            "ascending": False,
        })
        result = engine.apply_step(sample_df, config)
        assert "score_rank" in result.columns
        # Highest score (95) should have rank 1
        best_idx = result["score"].idxmax()
        assert result.loc[best_idx, "score_rank"] == 1.0

    def test_rank_with_pct(self, engine, sample_df):
        config = StepConfig(op="rank", params={
            "columns": {"score_pct": "score"},
            "pct": True,
        })
        result = engine.apply_step(sample_df, config)
        assert result["score_pct"].max() <= 1.0


class TestIsInStep:
    def test_isin_filter(self, engine, sample_df):
        config = StepConfig(op="isin", params={
            "column": "grade",
            "values": ["A"],
        })
        result = engine.apply_step(sample_df, config)
        assert all(result["grade"] == "A")

    def test_isin_negate(self, engine, sample_df):
        config = StepConfig(op="isin", params={
            "column": "grade",
            "values": ["A"],
            "negate": True,
        })
        result = engine.apply_step(sample_df, config)
        assert "A" not in result["grade"].values


class TestNullIndicatorStep:
    def test_null_indicators(self, engine):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, "x", None]})
        config = StepConfig(op="null_indicator", params={"columns": ["a", "b"]})
        result = engine.apply_step(df, config)
        assert "missing_a" in result.columns
        assert "missing_b" in result.columns
        assert result["missing_a"].iloc[1] == 1
        assert result["missing_b"].iloc[1] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_row_ops.py -v`
Expected: FAIL

- [ ] **Step 3: Implement steps**

`packages/harness-data/src/harness/data/transforms/steps/rank.py`:
```python
"""Rank step — add rank columns."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "rank"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns")
    if not columns or not isinstance(columns, dict):
        raise ValueError("rank step requires 'columns' dict of {new_col: source_col}")
    keys = params.get("keys")
    method = params.get("method", "average")
    ascending = params.get("ascending", True)
    pct = params.get("pct", False)

    result = df.copy()
    for new_col, source_col in columns.items():
        if source_col not in result.columns:
            raise ValueError(f"Column not found: {source_col}")
        if keys:
            result[new_col] = result.groupby(keys)[source_col].rank(
                method=method, ascending=ascending, pct=pct
            )
        else:
            result[new_col] = result[source_col].rank(
                method=method, ascending=ascending, pct=pct
            )
    return result
```

`packages/harness-data/src/harness/data/transforms/steps/isin.py`:
```python
"""IsIn step — filter by column values."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "isin"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params.get("column")
    values = params.get("values")
    negate = params.get("negate", False)
    if not column or values is None:
        raise ValueError("isin step requires 'column' and 'values' parameters")
    mask = df[column].isin(values)
    if negate:
        mask = ~mask
    return df[mask].reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/null_indicator.py`:
```python
"""Null indicator step — binary missing value flags."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "null_indicator"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    columns = params.get("columns")
    prefix = params.get("prefix", "missing_")
    if not columns:
        raise ValueError("null_indicator step requires 'columns' parameter")
    result = df.copy()
    for col in columns:
        if col not in result.columns:
            raise ValueError(f"Column not found: {col}")
        result[f"{prefix}{col}"] = result[col].isna().astype(int)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_row_ops.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/transforms/steps/ packages/harness-data/tests/test_transforms/
git commit -m "feat(harness-data): row operation steps (rank, isin, null_indicator)"
```

---

### Task 7: Windowed Operation Steps (rolling, lag, ewm, diff, trend)

**Files:**
- Create: `packages/harness-data/src/harness/data/transforms/steps/{rolling,lag,ewm,diff,trend}.py`
- Create: `packages/harness-data/tests/test_transforms/test_steps_windowed.py`

- [ ] **Step 1: Write failing tests**

Create `packages/harness-data/tests/test_transforms/test_steps_windowed.py`:

```python
import pandas as pd
import numpy as np
import pytest

from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


@pytest.fixture
def engine():
    return TransformEngine()


class TestRollingStep:
    def test_rolling_mean(self, engine, numeric_df):
        config = StepConfig(op="rolling", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "window": 2,
            "aggs": {"pts_mean_2": "points:mean"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_mean_2" in result.columns
        # First row per group should be NaN (window=2, min_periods=1 returns value)
        assert not result["pts_mean_2"].isna().all()

    def test_rolling_sum(self, engine, numeric_df):
        config = StepConfig(op="rolling", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "window": 2,
            "aggs": {"pts_sum_2": "points:sum"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_sum_2" in result.columns


class TestLagStep:
    def test_lag_1(self, engine, numeric_df):
        config = StepConfig(op="lag", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "columns": {"prev_points": "points:1"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "prev_points" in result.columns
        # First row per group should be NaN
        first_rows = result.groupby("entity_id").first()
        assert first_rows["prev_points"].isna().all()


class TestEwmStep:
    def test_ewm_mean(self, engine, numeric_df):
        config = StepConfig(op="ewm", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "span": 2,
            "aggs": {"pts_ewm": "points:mean"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_ewm" in result.columns
        assert not result["pts_ewm"].isna().all()


class TestDiffStep:
    def test_diff_1(self, engine, numeric_df):
        config = StepConfig(op="diff", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "columns": {"pts_diff": "points:1"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_diff" in result.columns

    def test_pct_change(self, engine, numeric_df):
        config = StepConfig(op="diff", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "columns": {"pts_pct": "points:1"},
            "pct": True,
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_pct" in result.columns


class TestTrendStep:
    def test_trend_slope(self, engine, numeric_df):
        config = StepConfig(op="trend", params={
            "keys": ["entity_id"],
            "order_by": "period",
            "window": 3,
            "columns": {"pts_trend": "points"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_trend" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_windowed.py -v`
Expected: FAIL

- [ ] **Step 3: Implement all 5 windowed steps**

`packages/harness-data/src/harness/data/transforms/steps/rolling.py`:
```python
"""Rolling step — windowed aggregations partitioned by keys."""
from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np

NAME = "rolling"

BUILTIN_AGGS = {"mean", "std", "sum", "min", "max", "count", "median"}

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    window = params.get("window")
    aggs = params.get("aggs", {})
    min_periods = params.get("min_periods", 1)

    if not window or not aggs:
        raise ValueError("rolling step requires 'window' and 'aggs' parameters")

    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])

    for new_col, spec in aggs.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Rolling agg spec must be 'col:func', got: {spec}")
        source_col, func = parts

        if keys:
            grouped = result.groupby(keys, sort=False)[source_col]
            rolling_obj = grouped.rolling(window=window, min_periods=min_periods)
        else:
            rolling_obj = result[source_col].rolling(window=window, min_periods=min_periods)

        if func in BUILTIN_AGGS:
            values = getattr(rolling_obj, func)()
        elif func == "slope":
            values = _rolling_slope(result, keys, source_col, window, min_periods)
        else:
            raise ValueError(f"Unknown rolling function: {func}")

        if keys:
            values = values.reset_index(level=list(range(len(keys))), drop=True)
        result[new_col] = values

    return result.sort_index().reset_index(drop=True)


def _rolling_slope(df, keys, col, window, min_periods):
    """OLS slope over rolling window."""
    def slope(arr):
        if len(arr) < min_periods:
            return np.nan
        x = np.arange(len(arr), dtype=float)
        y = arr.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x, y = x[mask], y[mask]
        return np.polyfit(x, y, 1)[0]

    if keys:
        return df.groupby(keys, sort=False)[col].transform(
            lambda s: s.rolling(window, min_periods=min_periods).apply(slope, raw=False)
        )
    return df[col].rolling(window, min_periods=min_periods).apply(slope, raw=False)
```

`packages/harness-data/src/harness/data/transforms/steps/lag.py`:
```python
"""Lag step — shift values within groups."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "lag"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    columns = params.get("columns", {})

    if not columns:
        raise ValueError("lag step requires 'columns' dict of {new_col: 'source_col:periods'}")

    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])

    for new_col, spec in columns.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"Lag spec must be 'col:periods', got: {spec}")
        source_col, periods = parts[0], int(parts[1])

        if keys:
            result[new_col] = result.groupby(keys, sort=False)[source_col].shift(periods)
        else:
            result[new_col] = result[source_col].shift(periods)

    return result.sort_index().reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/ewm.py`:
```python
"""EWM step — exponentially weighted moving statistics."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "ewm"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    span = params.get("span")
    aggs = params.get("aggs", {})

    if not span or not aggs:
        raise ValueError("ewm step requires 'span' and 'aggs' parameters")

    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])

    for new_col, spec in aggs.items():
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"EWM spec must be 'col:stat', got: {spec}")
        source_col, stat = parts

        if keys:
            grouped = result.groupby(keys, sort=False)[source_col]
            ewm_obj = grouped.transform(lambda s: getattr(s.ewm(span=span), stat)())
        else:
            ewm_obj = getattr(result[source_col].ewm(span=span), stat)()

        result[new_col] = ewm_obj

    return result.sort_index().reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/diff.py`:
```python
"""Diff step — differences or percent change within groups."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "diff"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    columns = params.get("columns", {})
    pct = params.get("pct", False)

    if not columns:
        raise ValueError("diff step requires 'columns' dict")

    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])

    for new_col, spec in columns.items():
        parts = spec.split(":")
        source_col = parts[0]
        periods = int(parts[1]) if len(parts) > 1 else 1

        if pct:
            if keys:
                result[new_col] = result.groupby(keys, sort=False)[source_col].pct_change(periods)
            else:
                result[new_col] = result[source_col].pct_change(periods)
        else:
            if keys:
                result[new_col] = result.groupby(keys, sort=False)[source_col].diff(periods)
            else:
                result[new_col] = result[source_col].diff(periods)

    return result.sort_index().reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/trend.py`:
```python
"""Trend step — OLS slope over rolling window."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

NAME = "trend"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys", [])
    order_by = params.get("order_by")
    window = params.get("window")
    columns = params.get("columns", {})

    if not window or not columns:
        raise ValueError("trend step requires 'window' and 'columns' parameters")

    result = df.copy()
    if order_by:
        result = result.sort_values(by=keys + [order_by])

    def slope(arr):
        y = arr.values.astype(float)
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x = np.arange(len(y), dtype=float)[mask]
        y = y[mask]
        return np.polyfit(x, y, 1)[0]

    for new_col, source_col in columns.items():
        if keys:
            result[new_col] = result.groupby(keys, sort=False)[source_col].transform(
                lambda s: s.rolling(window, min_periods=2).apply(slope, raw=False)
            )
        else:
            result[new_col] = result[source_col].rolling(window, min_periods=2).apply(slope, raw=False)

    return result.sort_index().reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_windowed.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/transforms/steps/ packages/harness-data/tests/test_transforms/
git commit -m "feat(harness-data): windowed transform steps (rolling, lag, ewm, diff, trend)"
```

---

### Task 8: Reshaping Steps (join, union, unpivot, aggregate, conditional_agg)

**Files:**
- Create: `packages/harness-data/src/harness/data/transforms/steps/{join,union,unpivot,aggregate,conditional_agg}.py`
- Create: `packages/harness-data/tests/test_transforms/test_steps_reshape.py`

- [ ] **Step 1: Write failing tests**

Create `packages/harness-data/tests/test_transforms/test_steps_reshape.py`:

```python
import pandas as pd
import pytest

from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


@pytest.fixture
def engine():
    return TransformEngine()


@pytest.fixture
def left_df():
    return pd.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})


@pytest.fixture
def right_df():
    return pd.DataFrame({"id": [1, 2, 4], "score": [90, 85, 70]})


class TestJoinStep:
    def test_left_join(self, engine, left_df, right_df):
        resolver = lambda name: right_df
        config = StepConfig(op="join", params={
            "other": "scores",
            "on": ["id"],
            "how": "left",
        })
        result = engine.apply_step(left_df, config, resolver=resolver)
        assert len(result) == 3
        assert "score" in result.columns

    def test_inner_join(self, engine, left_df, right_df):
        resolver = lambda name: right_df
        config = StepConfig(op="join", params={
            "other": "scores",
            "on": ["id"],
            "how": "inner",
        })
        result = engine.apply_step(left_df, config, resolver=resolver)
        assert len(result) == 2


class TestUnionStep:
    def test_union(self, engine):
        df1 = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df2 = pd.DataFrame({"id": [3, 4], "val": [30, 40]})
        resolver = lambda name: df2
        config = StepConfig(op="union", params={"other": "df2"})
        result = engine.apply_step(df1, config, resolver=resolver)
        assert len(result) == 4


class TestUnpivotStep:
    def test_unpivot(self, engine):
        df = pd.DataFrame({
            "id": [1, 2],
            "score_a": [90, 80],
            "score_b": [85, 75],
        })
        config = StepConfig(op="unpivot", params={
            "id_columns": ["id"],
            "unpivot_columns": {"score": ["score_a", "score_b"]},
        })
        result = engine.apply_step(df, config)
        assert len(result) == 4
        assert "score" in result.columns


class TestAggregateStep:
    def test_group_by_mean(self, engine, numeric_df):
        config = StepConfig(op="aggregate", params={
            "keys": ["entity_id"],
            "aggs": {"points": "mean", "rebounds": "sum"},
        })
        result = engine.apply_step(numeric_df, config)
        assert len(result) == 3  # 3 unique entity_ids
        assert "points_mean" in result.columns or "points" in result.columns


class TestConditionalAggStep:
    def test_conditional_agg(self, engine, numeric_df):
        config = StepConfig(op="conditional_agg", params={
            "keys": ["entity_id"],
            "aggs": {"pts_when_win": "points:mean:target == 1"},
        })
        result = engine.apply_step(numeric_df, config)
        assert "pts_when_win" in result.columns
        assert len(result) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_reshape.py -v`
Expected: FAIL

- [ ] **Step 3: Implement all 5 reshape steps**

`packages/harness-data/src/harness/data/transforms/steps/join.py`:
```python
"""Join step — merge with another source/view."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "join"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    other_name = params.get("other")
    on = params.get("on")
    how = params.get("how", "left")
    select = params.get("select")
    prefix = params.get("prefix")
    resolver = params.get("_resolver")

    if not other_name or not on:
        raise ValueError("join step requires 'other' and 'on' parameters")
    if resolver is None:
        raise ValueError("join step requires a resolver to load the other source/view")

    other_df = resolver(other_name)

    if select:
        keep_cols = list(set(select + (on if isinstance(on, list) else list(on.keys()))))
        other_df = other_df[[c for c in keep_cols if c in other_df.columns]]

    if isinstance(on, dict):
        result = df.merge(other_df, left_on=list(on.keys()), right_on=list(on.values()), how=how)
    else:
        result = df.merge(other_df, on=on, how=how)

    if prefix:
        new_cols = [c for c in result.columns if c not in df.columns and c not in (on if isinstance(on, list) else list(on.keys()))]
        result = result.rename(columns={c: f"{prefix}{c}" for c in new_cols})

    return result
```

`packages/harness-data/src/harness/data/transforms/steps/union.py`:
```python
"""Union step — vertical concat with another source/view."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "union"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    other_name = params.get("other")
    resolver = params.get("_resolver")
    if not other_name:
        raise ValueError("union step requires 'other' parameter")
    if resolver is None:
        raise ValueError("union step requires a resolver")
    other_df = resolver(other_name)
    return pd.concat([df, other_df], ignore_index=True)
```

`packages/harness-data/src/harness/data/transforms/steps/unpivot.py`:
```python
"""Unpivot step — melt wide columns to long format."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "unpivot"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    id_columns = params.get("id_columns", [])
    unpivot_columns = params.get("unpivot_columns", {})

    if not unpivot_columns:
        raise ValueError("unpivot step requires 'unpivot_columns' dict of {value_name: [col_list]}")

    result = df.copy()
    for value_name, cols in unpivot_columns.items():
        result = pd.melt(
            result,
            id_vars=[c for c in result.columns if c not in cols],
            value_vars=cols,
            var_name=f"{value_name}_source",
            value_name=value_name,
        )
    return result.reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/aggregate.py`:
```python
"""Aggregate step — group by keys and aggregate."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "aggregate"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys")
    aggs = params.get("aggs")

    if not keys or not aggs:
        raise ValueError("aggregate step requires 'keys' and 'aggs' parameters")

    agg_dict = {}
    for col, funcs in aggs.items():
        if isinstance(funcs, str):
            funcs = [funcs]
        agg_dict[col] = funcs

    result = df.groupby(keys, as_index=False).agg(agg_dict)

    # Flatten MultiIndex columns
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            f"{col}_{func}" if func else col
            for col, func in result.columns
        ]

    return result.reset_index(drop=True)
```

`packages/harness-data/src/harness/data/transforms/steps/conditional_agg.py`:
```python
"""Conditional aggregation step — aggregate with per-agg filters."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "conditional_agg"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    keys = params.get("keys")
    aggs = params.get("aggs", {})

    if not keys or not aggs:
        raise ValueError("conditional_agg step requires 'keys' and 'aggs'")

    result_parts = []
    for new_col, spec in aggs.items():
        parts = spec.split(":")
        if len(parts) == 2:
            source_col, func = parts
            condition = None
        elif len(parts) == 3:
            source_col, func, condition = parts
        else:
            raise ValueError(f"Spec must be 'col:func' or 'col:func:condition', got: {spec}")

        subset = df
        if condition:
            subset = df.query(condition.strip())

        agg_result = subset.groupby(keys, as_index=False)[source_col].agg(func)
        agg_result = agg_result.rename(columns={source_col: new_col})
        result_parts.append(agg_result)

    result = result_parts[0]
    for part in result_parts[1:]:
        result = result.merge(part, on=keys, how="outer")

    return result.reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_reshape.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/transforms/steps/ packages/harness-data/tests/test_transforms/
git commit -m "feat(harness-data): reshape transform steps (join, union, unpivot, aggregate, conditional_agg)"
```

---

### Task 9: Encoding Steps (encode, bin, datetime)

**Files:**
- Create: `packages/harness-data/src/harness/data/transforms/steps/{encode,bin,datetime}.py`
- Create: `packages/harness-data/tests/test_transforms/test_steps_encoding.py`

- [ ] **Step 1: Write failing tests**

Create `packages/harness-data/tests/test_transforms/test_steps_encoding.py`:

```python
import pandas as pd
import pytest

from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


@pytest.fixture
def engine():
    return TransformEngine()


class TestEncodeStep:
    def test_frequency_encoding(self, engine, sample_df):
        config = StepConfig(op="encode", params={
            "column": "grade",
            "method": "frequency",
        })
        result = engine.apply_step(sample_df, config)
        assert "grade_encoded" in result.columns
        assert result["grade_encoded"].dtype in ["float64", "int64"]

    def test_ordinal_encoding(self, engine, sample_df):
        config = StepConfig(op="encode", params={
            "column": "grade",
            "method": "ordinal",
        })
        result = engine.apply_step(sample_df, config)
        assert "grade_encoded" in result.columns


class TestBinStep:
    def test_quantile_binning(self, engine, sample_df):
        config = StepConfig(op="bin", params={
            "column": "score",
            "method": "quantile",
            "n_bins": 3,
        })
        result = engine.apply_step(sample_df, config)
        assert "score_binned" in result.columns

    def test_uniform_binning(self, engine, sample_df):
        config = StepConfig(op="bin", params={
            "column": "score",
            "method": "uniform",
            "n_bins": 3,
        })
        result = engine.apply_step(sample_df, config)
        assert "score_binned" in result.columns


class TestDatetimeStep:
    def test_extract_components(self, engine):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-15", "2024-06-20", "2024-12-25"]),
        })
        config = StepConfig(op="datetime", params={
            "column": "timestamp",
            "extract": ["year", "month", "dayofweek"],
        })
        result = engine.apply_step(df, config)
        assert "timestamp_year" in result.columns
        assert "timestamp_month" in result.columns
        assert "timestamp_dayofweek" in result.columns
        assert result["timestamp_year"].iloc[0] == 2024
        assert result["timestamp_month"].iloc[0] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_encoding.py -v`
Expected: FAIL

- [ ] **Step 3: Implement all 3 encoding steps**

`packages/harness-data/src/harness/data/transforms/steps/encode.py`:
```python
"""Encode step — categorical encoding."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "encode"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params.get("column")
    method = params.get("method", "frequency")
    output = params.get("output", f"{column}_encoded")

    if not column:
        raise ValueError("encode step requires 'column' parameter")

    result = df.copy()

    if method == "frequency":
        freq = result[column].value_counts(normalize=True)
        result[output] = result[column].map(freq)
    elif method == "ordinal":
        categories = sorted(result[column].dropna().unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        result[output] = result[column].map(mapping)
    else:
        raise ValueError(f"Unknown encode method: {method}")

    return result
```

`packages/harness-data/src/harness/data/transforms/steps/bin.py`:
```python
"""Bin step — discretize continuous columns."""
from __future__ import annotations
from typing import Any
import pandas as pd

NAME = "bin"

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params.get("column")
    method = params.get("method", "quantile")
    n_bins = params.get("n_bins", 5)
    output = params.get("output", f"{column}_binned")
    boundaries = params.get("boundaries")

    if not column:
        raise ValueError("bin step requires 'column' parameter")

    result = df.copy()

    if method == "quantile":
        result[output] = pd.qcut(result[column], q=n_bins, labels=False, duplicates="drop")
    elif method == "uniform":
        result[output] = pd.cut(result[column], bins=n_bins, labels=False)
    elif method == "custom" and boundaries:
        result[output] = pd.cut(result[column], bins=boundaries, labels=False)
    else:
        raise ValueError(f"Unknown bin method: {method}")

    return result
```

`packages/harness-data/src/harness/data/transforms/steps/datetime.py`:
```python
"""Datetime step — extract calendar components."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

NAME = "datetime"

EXTRACTORS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    "dayofweek": lambda s: s.dt.dayofweek,
    "hour": lambda s: s.dt.hour,
    "quarter": lambda s: s.dt.quarter,
    "weekofyear": lambda s: s.dt.isocalendar().week.astype(int),
}

def step(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    column = params.get("column")
    extract = params.get("extract", [])
    cyclical = params.get("cyclical", [])

    if not column:
        raise ValueError("datetime step requires 'column' parameter")

    result = df.copy()
    col_dt = pd.to_datetime(result[column])

    for component in extract:
        extractor = EXTRACTORS.get(component)
        if extractor is None:
            raise ValueError(f"Unknown datetime component: {component}")
        result[f"{column}_{component}"] = extractor(col_dt)

    for component in cyclical:
        extractor = EXTRACTORS.get(component)
        if extractor is None:
            raise ValueError(f"Unknown datetime component: {component}")
        values = extractor(col_dt).astype(float)
        max_val = {"month": 12, "dayofweek": 7, "hour": 24, "quarter": 4}.get(component, values.max())
        result[f"{column}_{component}_sin"] = np.sin(2 * np.pi * values / max_val)
        result[f"{column}_{component}_cos"] = np.cos(2 * np.pi * values / max_val)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_transforms/test_steps_encoding.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/transforms/steps/ packages/harness-data/tests/test_transforms/
git commit -m "feat(harness-data): encoding transform steps (encode, bin, datetime)"
```

---

### Task 10: Profiler + Validation

**Files:**
- Create: `packages/harness-data/src/harness/data/profiling/__init__.py`
- Create: `packages/harness-data/src/harness/data/profiling/profiler.py`
- Create: `packages/harness-data/src/harness/data/profiling/validation.py`
- Create: `packages/harness-data/tests/test_profiling/test_profiler.py`
- Create: `packages/harness-data/tests/test_profiling/test_validation.py`

- [ ] **Step 1: Write failing tests for profiler**

Create `packages/harness-data/tests/test_profiling/__init__.py` (empty).
Create `packages/harness-data/tests/test_profiling/test_profiler.py`:

```python
import pandas as pd
import pytest

from harness.data.profiling.profiler import DataProfiler, ColumnProfile, DataProfile


class TestDataProfiler:
    def test_profile_basic(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        assert isinstance(profile, DataProfile)
        assert profile.row_count == 5
        assert profile.column_count == 5
        assert len(profile.columns) == 5

    def test_column_profile_numeric(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        score_col = next(c for c in profile.columns if c.name == "score")
        assert score_col.dtype == "float64"
        assert score_col.null_count == 0
        assert score_col.mean is not None
        assert score_col.inferred_type == "numeric"

    def test_column_profile_categorical(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        grade_col = next(c for c in profile.columns if c.name == "grade")
        assert grade_col.inferred_type == "categorical"
        assert grade_col.n_unique == 3

    def test_column_profile_boolean(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        enrolled_col = next(c for c in profile.columns if c.name == "enrolled")
        assert enrolled_col.inferred_type in ("boolean", "binary")

    def test_high_null_detection(self):
        df = pd.DataFrame({
            "a": [1.0, None, None, None, 5.0],
            "b": [1, 2, 3, 4, 5],
        })
        profiler = DataProfiler()
        profile = profiler.profile(df, high_null_threshold=50.0)
        assert len(profile.high_null_columns) == 1
        assert profile.high_null_columns[0].name == "a"

    def test_zero_variance_detection(self):
        df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]})
        profiler = DataProfiler()
        profile = profiler.profile(df)
        assert "a" in profile.zero_variance_columns
```

- [ ] **Step 2: Write failing tests for validation**

Create `packages/harness-data/tests/test_profiling/test_validation.py`:

```python
import pandas as pd
import pytest

from harness.data.profiling.validation import SchemaValidator, ValidationResult


class TestSchemaValidator:
    def test_validate_required_columns(self, sample_df):
        validator = SchemaValidator(required_columns=["id", "score"])
        result = validator.validate(sample_df)
        assert result.is_valid

    def test_validate_missing_required_column(self, sample_df):
        validator = SchemaValidator(required_columns=["id", "nonexistent"])
        result = validator.validate(sample_df)
        assert not result.is_valid
        assert "nonexistent" in result.errors[0]

    def test_validate_column_types(self, sample_df):
        validator = SchemaValidator(column_types={"score": "float64", "id": "int64"})
        result = validator.validate(sample_df)
        assert result.is_valid

    def test_validate_no_nulls(self, sample_df):
        validator = SchemaValidator(no_null_columns=["id", "score"])
        result = validator.validate(sample_df)
        assert result.is_valid

    def test_validate_nulls_found(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        validator = SchemaValidator(no_null_columns=["a"])
        result = validator.validate(df)
        assert not result.is_valid

    def test_validate_min_rows(self, sample_df):
        validator = SchemaValidator(min_rows=3)
        result = validator.validate(sample_df)
        assert result.is_valid

    def test_validate_too_few_rows(self, sample_df):
        validator = SchemaValidator(min_rows=100)
        result = validator.validate(sample_df)
        assert not result.is_valid
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_profiling/ -v`
Expected: FAIL

- [ ] **Step 4: Implement DataProfiler**

Create `packages/harness-data/src/harness/data/profiling/__init__.py`:
```python
from harness.data.profiling.profiler import DataProfiler, DataProfile, ColumnProfile
from harness.data.profiling.validation import SchemaValidator, ValidationResult
```

Create `packages/harness-data/src/harness/data/profiling/profiler.py`:

```python
"""Data profiler — column stats, type inference, quality indicators."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Profile for a single column."""
    name: str
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    inferred_type: str = "unknown"


class DataProfile(BaseModel):
    """Profile for an entire dataset."""
    row_count: int
    column_count: int
    columns: list[ColumnProfile] = Field(default_factory=list)
    high_null_columns: list[ColumnProfile] = Field(default_factory=list)
    zero_variance_columns: list[str] = Field(default_factory=list)


class DataProfiler:
    """Profile a DataFrame — compute column stats and infer types."""

    def profile(self, df: pd.DataFrame, high_null_threshold: float = 50.0) -> DataProfile:
        columns = []
        high_null = []
        zero_var = []

        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0.0
            n_unique = int(series.nunique())

            cp = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_count=null_count,
                null_pct=round(null_pct, 2),
                n_unique=n_unique,
            )

            if pd.api.types.is_numeric_dtype(series):
                non_null = series.dropna()
                if len(non_null) > 0:
                    cp.mean = round(float(non_null.mean()), 4)
                    cp.std = round(float(non_null.std()), 4)
                    cp.min = float(non_null.min())
                    cp.max = float(non_null.max())
                    cp.median = float(non_null.median())

            cp.inferred_type = self._infer_type(series, n_unique)
            columns.append(cp)

            if null_pct >= high_null_threshold:
                high_null.append(cp)

            if pd.api.types.is_numeric_dtype(series) and n_unique <= 1 and null_count < len(df):
                zero_var.append(col)

        return DataProfile(
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            high_null_columns=high_null,
            zero_variance_columns=zero_var,
        )

    def _infer_type(self, series: pd.Series, n_unique: int) -> str:
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if pd.api.types.is_numeric_dtype(series):
            if n_unique == 2:
                return "binary"
            if n_unique <= 20:
                return "categorical"
            return "numeric"
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                return "empty"
            unique_ratio = n_unique / len(non_null) if len(non_null) > 0 else 0
            if unique_ratio > 0.9:
                return "id"
            if n_unique <= 50:
                return "categorical"
            return "high_cardinality"
        return "unknown"
```

- [ ] **Step 5: Implement SchemaValidator**

Create `packages/harness-data/src/harness/data/profiling/validation.py`:

```python
"""Schema validation — data quality checks."""
from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    """Result of a validation check."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SchemaValidator:
    """Validate a DataFrame against schema expectations."""

    def __init__(
        self,
        required_columns: list[str] | None = None,
        column_types: dict[str, str] | None = None,
        no_null_columns: list[str] | None = None,
        min_rows: int | None = None,
    ):
        self.required_columns = required_columns or []
        self.column_types = column_types or {}
        self.no_null_columns = no_null_columns or []
        self.min_rows = min_rows

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors = []
        warnings = []

        for col in self.required_columns:
            if col not in df.columns:
                errors.append(f"Required column missing: '{col}'")

        for col, expected_type in self.column_types.items():
            if col in df.columns and str(df[col].dtype) != expected_type:
                warnings.append(
                    f"Column '{col}' has type {df[col].dtype}, expected {expected_type}"
                )

        for col in self.no_null_columns:
            if col in df.columns and df[col].isna().any():
                null_count = int(df[col].isna().sum())
                errors.append(f"Column '{col}' has {null_count} null values")

        if self.min_rows is not None and len(df) < self.min_rows:
            errors.append(f"Dataset has {len(df)} rows, minimum required is {self.min_rows}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
```

- [ ] **Step 6: Run profiling tests**

Run: `cd packages/harness-data && python -m pytest tests/test_profiling/ -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add packages/harness-data/src/harness/data/profiling/ packages/harness-data/tests/test_profiling/
git commit -m "feat(harness-data): profiler (column stats, type inference) + schema validator"
```

---

### Task 11: Pipeline Runner

**Files:**
- Create: `packages/harness-data/src/harness/data/runner.py`
- Create: `packages/harness-data/tests/test_runner.py`

- [ ] **Step 1: Write failing tests for the pipeline runner**

Create `packages/harness-data/tests/test_runner.py`:

```python
import json
import pandas as pd
import pytest
from pathlib import Path

from harness.data.runner import PipelineRunner, PipelineResult


class TestPipelineRunner:
    def test_run_single_source_no_transforms(self, temp_workspace, sample_csv):
        runner = PipelineRunner(temp_workspace)
        sources = [{"name": "sample", "source_type": "file", "path": str(sample_csv)}]
        result = runner.run(sources=sources, transforms=[])
        assert isinstance(result, PipelineResult)
        assert result.row_count > 0
        assert (temp_workspace / "data" / "clean" / "dataset.parquet").exists()
        assert (temp_workspace / "data" / "clean" / "schema.json").exists()

    def test_run_with_transforms(self, temp_workspace, sample_csv):
        runner = PipelineRunner(temp_workspace)
        sources = [{"name": "sample", "source_type": "file", "path": str(sample_csv)}]
        transforms = [
            {"op": "filter", "params": {"expr": "score > 80"}},
            {"op": "select", "params": {"columns": ["name", "score"]}},
        ]
        result = runner.run(sources=sources, transforms=transforms)
        assert result.row_count == 4  # 4 students with score > 80

    def test_schema_json_contents(self, temp_workspace, sample_csv):
        runner = PipelineRunner(temp_workspace)
        sources = [{"name": "sample", "source_type": "file", "path": str(sample_csv)}]
        runner.run(sources=sources, transforms=[])
        schema_path = temp_workspace / "data" / "clean" / "schema.json"
        schema = json.loads(schema_path.read_text())
        assert "columns" in schema
        assert "row_count" in schema
        assert "column_types" in schema

    def test_run_returns_error_on_bad_source(self, temp_workspace):
        runner = PipelineRunner(temp_workspace)
        sources = [{"name": "bad", "source_type": "file", "path": "/nonexistent.csv"}]
        with pytest.raises(FileNotFoundError):
            runner.run(sources=sources, transforms=[])

    def test_idempotent_output(self, temp_workspace, sample_csv):
        runner = PipelineRunner(temp_workspace)
        sources = [{"name": "sample", "source_type": "file", "path": str(sample_csv)}]
        result1 = runner.run(sources=sources, transforms=[])
        result2 = runner.run(sources=sources, transforms=[])
        assert result1.row_count == result2.row_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_runner.py -v`
Expected: FAIL

- [ ] **Step 3: Implement PipelineRunner**

Create `packages/harness-data/src/harness/data/runner.py`:

```python
"""Pipeline runner — stateless: sources + transforms → parquet + schema.json."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from harness.data.sources.protocol import SourceConfig
from harness.data.sources.file import FileSource
from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig


class PipelineResult(BaseModel):
    """Result of a pipeline run."""
    row_count: int
    column_count: int
    columns: list[str] = Field(default_factory=list)
    output_path: str = ""
    schema_path: str = ""
    data_hash: str = ""


SOURCE_ADAPTERS = {
    "file": FileSource(),
}


class PipelineRunner:
    """Stateless pipeline: load sources → apply transforms → write parquet + schema."""

    def __init__(self, workspace_dir: str | Path):
        self._workspace = Path(workspace_dir)
        self._engine = TransformEngine()

    def run(
        self,
        sources: list[dict],
        transforms: list[dict],
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            sources: List of source config dicts.
            transforms: List of transform step config dicts.

        Returns:
            PipelineResult with output details.
        """
        # Load and merge sources
        dfs = []
        for source_dict in sources:
            config = SourceConfig(**source_dict)
            adapter = SOURCE_ADAPTERS.get(config.source_type)
            if adapter is None:
                raise ValueError(f"Unknown source type: {config.source_type}")
            df = adapter.load(config, base_dir=str(self._workspace))
            dfs.append(df)

        if not dfs:
            raise ValueError("No sources provided")

        # Merge all sources (left join on common columns, or concat if no overlap)
        combined = dfs[0]
        for df in dfs[1:]:
            common = list(set(combined.columns) & set(df.columns))
            if common:
                combined = combined.merge(df, on=common, how="left")
            else:
                combined = pd.concat([combined, df], axis=1)

        # Apply transforms
        steps = [StepConfig(**t) for t in transforms]
        result_df = self._engine.run_pipeline(combined, steps)

        # Write outputs
        clean_dir = self._workspace / "data" / "clean"
        clean_dir.mkdir(parents=True, exist_ok=True)

        output_path = clean_dir / "dataset.parquet"
        result_df.to_parquet(output_path, index=False)

        # Compute data hash
        data_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

        # Write schema
        schema = {
            "row_count": len(result_df),
            "column_count": len(result_df.columns),
            "columns": list(result_df.columns),
            "column_types": {col: str(dtype) for col, dtype in result_df.dtypes.items()},
            "data_hash": data_hash,
        }
        schema_path = clean_dir / "schema.json"
        schema_path.write_text(json.dumps(schema, indent=2))

        return PipelineResult(
            row_count=len(result_df),
            column_count=len(result_df.columns),
            columns=list(result_df.columns),
            output_path=str(output_path),
            schema_path=str(schema_path),
            data_hash=data_hash,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_runner.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/src/harness/data/runner.py packages/harness-data/tests/test_runner.py
git commit -m "feat(harness-data): pipeline runner (sources + transforms → parquet + schema)"
```

---

### Task 12: Workspace Integration

**Files:**
- Create: `packages/harness-data/src/harness/data/workspace.py`
- Create: `packages/harness-data/tests/test_workspace.py`

- [ ] **Step 1: Write failing tests for workspace integration**

Create `packages/harness-data/tests/test_workspace.py`:

```python
import json
import yaml
import pandas as pd
import pytest
from pathlib import Path

from harness.data.workspace import DataWorkspace


class TestDataWorkspace:
    def test_init_creates_structure(self, tmp_path):
        ws = DataWorkspace(tmp_path)
        ws.init()
        assert (tmp_path / "data").is_dir()
        assert (tmp_path / "data" / "raw").is_dir()
        assert (tmp_path / "data" / "clean").is_dir()
        assert (tmp_path / "data" / "sources.yaml").exists()
        assert (tmp_path / "data" / "transforms.yaml").exists()

    def test_add_source(self, tmp_path, sample_df):
        ws = DataWorkspace(tmp_path)
        ws.init()
        csv_path = tmp_path / "data" / "raw" / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        ws.add_source("test_data", str(csv_path))
        sources = ws.list_sources()
        assert len(sources) == 1
        assert sources[0].name == "test_data"

    def test_add_transform(self, tmp_path):
        ws = DataWorkspace(tmp_path)
        ws.init()
        ws.add_transform({"op": "filter", "params": {"expr": "score > 80"}})
        transforms = ws.load_transforms()
        assert len(transforms) == 1
        assert transforms[0]["op"] == "filter"

    def test_run_pipeline(self, tmp_path, sample_df):
        ws = DataWorkspace(tmp_path)
        ws.init()
        csv_path = tmp_path / "data" / "raw" / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        ws.add_source("test_data", str(csv_path))
        result = ws.run_pipeline()
        assert result.row_count == 5
        assert (tmp_path / "data" / "clean" / "dataset.parquet").exists()

    def test_load_clean_data(self, tmp_path, sample_df):
        ws = DataWorkspace(tmp_path)
        ws.init()
        csv_path = tmp_path / "data" / "raw" / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        ws.add_source("test_data", str(csv_path))
        ws.run_pipeline()
        df = ws.load_clean_data()
        assert len(df) == 5

    def test_load_schema(self, tmp_path, sample_df):
        ws = DataWorkspace(tmp_path)
        ws.init()
        csv_path = tmp_path / "data" / "raw" / "test.csv"
        sample_df.to_csv(csv_path, index=False)
        ws.add_source("test_data", str(csv_path))
        ws.run_pipeline()
        schema = ws.load_schema()
        assert schema["row_count"] == 5
        assert "columns" in schema
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_workspace.py -v`
Expected: FAIL

- [ ] **Step 3: Implement DataWorkspace**

Create `packages/harness-data/src/harness/data/workspace.py`:

```python
"""Workspace integration — read/write sources.yaml, transforms.yaml, data/."""
from __future__ import annotations

import json
import yaml
from pathlib import Path

import pandas as pd

from harness.data.sources.protocol import SourceConfig
from harness.data.sources.registry import SourceRegistry
from harness.data.runner import PipelineRunner, PipelineResult


class DataWorkspace:
    """Manages the data layer of a harness workspace."""

    def __init__(self, workspace_dir: str | Path):
        self._root = Path(workspace_dir)
        self._data_dir = self._root / "data"

    def init(self) -> None:
        """Initialize the workspace data directory structure."""
        (self._data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self._data_dir / "clean").mkdir(parents=True, exist_ok=True)

        sources_path = self._data_dir / "sources.yaml"
        if not sources_path.exists():
            sources_path.write_text(yaml.dump({"sources": {}}, default_flow_style=False))

        transforms_path = self._data_dir / "transforms.yaml"
        if not transforms_path.exists():
            transforms_path.write_text(yaml.dump({"steps": []}, default_flow_style=False))

    def add_source(self, name: str, path: str, **kwargs) -> None:
        """Register a new data source."""
        registry = SourceRegistry(self._data_dir)
        config = SourceConfig(name=name, source_type="file", path=path, **kwargs)
        registry.add(config)

    def list_sources(self) -> list[SourceConfig]:
        """List all registered sources."""
        registry = SourceRegistry(self._data_dir)
        return registry.list_all()

    def add_transform(self, step: dict) -> None:
        """Add a transform step to the pipeline."""
        transforms_path = self._data_dir / "transforms.yaml"
        content = yaml.safe_load(transforms_path.read_text()) or {}
        steps = content.get("steps", [])
        steps.append(step)
        content["steps"] = steps
        transforms_path.write_text(yaml.dump(content, default_flow_style=False, sort_keys=False))

    def load_transforms(self) -> list[dict]:
        """Load transform steps from transforms.yaml."""
        transforms_path = self._data_dir / "transforms.yaml"
        if not transforms_path.exists():
            return []
        content = yaml.safe_load(transforms_path.read_text()) or {}
        return content.get("steps", [])

    def run_pipeline(self) -> PipelineResult:
        """Execute the full data pipeline: sources + transforms → clean dataset."""
        sources = [s.model_dump() for s in self.list_sources()]
        transforms = self.load_transforms()
        runner = PipelineRunner(self._root)
        return runner.run(sources=sources, transforms=transforms)

    def load_clean_data(self) -> pd.DataFrame:
        """Load the clean dataset."""
        path = self._data_dir / "clean" / "dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Clean dataset not found. Run the pipeline first: {path}")
        return pd.read_parquet(path)

    def load_schema(self) -> dict:
        """Load the dataset schema."""
        path = self._data_dir / "clean" / "schema.json"
        if not path.exists():
            raise FileNotFoundError(f"Schema not found. Run the pipeline first: {path}")
        return json.loads(path.read_text())
```

- [ ] **Step 4: Update __init__.py with public API exports**

Update `packages/harness-data/src/harness/data/__init__.py`:

```python
"""harness-data: Declarative data engineering library."""

from harness.data.workspace import DataWorkspace
from harness.data.runner import PipelineRunner, PipelineResult
from harness.data.expressions.engine import ExpressionEngine
from harness.data.expressions.registry import FunctionRegistry
from harness.data.expressions.validator import ExpressionValidator
from harness.data.transforms.engine import TransformEngine
from harness.data.transforms.protocol import StepConfig
from harness.data.profiling.profiler import DataProfiler, DataProfile
from harness.data.profiling.validation import SchemaValidator, ValidationResult
from harness.data.sources.protocol import Source, SourceConfig, SourceMetadata
from harness.data.sources.registry import SourceRegistry
```

- [ ] **Step 5: Run workspace tests**

Run: `cd packages/harness-data && python -m pytest tests/test_workspace.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `cd packages/harness-data && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add packages/harness-data/
git commit -m "feat(harness-data): workspace integration + public API exports"
```

---

### Task 13: URL + API Source Adapters

**Files:**
- Create: `packages/harness-data/src/harness/data/sources/url.py`
- Create: `packages/harness-data/src/harness/data/sources/api.py`
- Create: `packages/harness-data/tests/test_sources/test_url.py`
- Create: `packages/harness-data/tests/test_sources/test_api.py`

- [ ] **Step 1: Write tests for UrlSource (using mocked HTTP)**

Create `packages/harness-data/tests/test_sources/test_url.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

from harness.data.sources.url import UrlSource
from harness.data.sources.protocol import SourceConfig


class TestUrlSource:
    def test_validate_missing_url(self):
        source = UrlSource()
        config = SourceConfig(name="test", source_type="url")
        errors = source.validate(config)
        assert len(errors) > 0

    def test_validate_valid_url(self):
        source = UrlSource()
        config = SourceConfig(name="test", source_type="url", url="https://example.com/data.csv")
        errors = source.validate(config)
        assert len(errors) == 0

    @patch("harness.data.sources.url.requests")
    def test_load_csv_from_url(self, mock_requests):
        csv_content = "id,name,score\n1,Alice,85\n2,Bob,92\n"
        mock_response = MagicMock()
        mock_response.text = csv_content
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response

        source = UrlSource()
        config = SourceConfig(name="test", source_type="url", url="https://example.com/data.csv")
        df = source.load(config)
        assert len(df) == 2
        assert "score" in df.columns

    def test_implements_source_protocol(self):
        from harness.data.sources.protocol import Source
        assert isinstance(UrlSource(), Source)
```

- [ ] **Step 2: Implement UrlSource**

Create `packages/harness-data/src/harness/data/sources/url.py`:

```python
"""URL source adapter — fetch data from HTTP/HTTPS endpoints."""
from __future__ import annotations

from io import StringIO, BytesIO
from typing import Any

import pandas as pd

from harness.data.sources.protocol import SourceConfig

try:
    import requests
except ImportError:
    requests = None


class UrlSource:
    """Load data from a URL (CSV, JSON, Parquet)."""

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        if requests is None:
            raise ImportError("requests is required for URL sources: pip install requests")
        if not config.url:
            raise ValueError("URL source requires 'url' parameter")

        headers = config.params.get("headers", {})
        response = requests.get(config.url, headers=headers, timeout=30)
        response.raise_for_status()

        fmt = config.format
        if fmt == "auto":
            if config.url.endswith(".parquet") or config.url.endswith(".pq"):
                fmt = "parquet"
            elif config.url.endswith(".json"):
                fmt = "json"
            else:
                fmt = "csv"

        if fmt == "csv":
            return pd.read_csv(StringIO(response.text), **{k: v for k, v in config.params.items() if k != "headers"})
        elif fmt == "json":
            return pd.read_json(StringIO(response.text))
        elif fmt == "parquet":
            return pd.read_parquet(BytesIO(response.content))
        else:
            raise ValueError(f"Unsupported format for URL source: {fmt}")

    def validate(self, config: SourceConfig) -> list[str]:
        errors = []
        if not config.url:
            errors.append("URL source requires 'url' parameter")
        return errors
```

- [ ] **Step 3: Run UrlSource tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_url.py -v`
Expected: FAIL with ImportError

- [ ] **Step 4: Implement UrlSource** (code follows in Step 2's implementation block above)

- [ ] **Step 5: Run UrlSource tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_url.py -v`
Expected: ALL PASS

- [ ] **Step 6: Write failing tests for ApiSource**

Create `packages/harness-data/tests/test_sources/test_api.py`:

```python
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from harness.data.sources.api import ApiSource
from harness.data.sources.protocol import SourceConfig


class TestApiSource:
    def test_validate_missing_url(self):
        source = ApiSource()
        config = SourceConfig(name="test", source_type="api")
        errors = source.validate(config)
        assert len(errors) > 0

    @patch("harness.data.sources.api.requests")
    def test_load_single_page(self, mock_requests):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = mock_response

        source = ApiSource()
        config = SourceConfig(
            name="test",
            source_type="api",
            url="https://api.example.com/data",
            params={"records_key": "data"},
        )
        df = source.load(config)
        assert len(df) == 2

    def test_implements_source_protocol(self):
        from harness.data.sources.protocol import Source
        assert isinstance(ApiSource(), Source)
```

Create `packages/harness-data/src/harness/data/sources/api.py`:

```python
"""API source adapter — REST APIs with pagination and rate limiting."""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from harness.data.sources.protocol import SourceConfig

try:
    import requests
except ImportError:
    requests = None


class ApiSource:
    """Load data from a REST API."""

    def load(self, config: SourceConfig, base_dir: str | None = None) -> pd.DataFrame:
        if requests is None:
            raise ImportError("requests is required for API sources: pip install requests")
        if not config.url:
            raise ValueError("API source requires 'url' parameter")

        headers = config.params.get("headers", {})
        records_key = config.params.get("records_key")
        rate_limit = config.params.get("rate_limit", 0.0)

        response = requests.get(config.url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()

        if records_key:
            data = data[records_key]

        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        else:
            raise ValueError(f"Unexpected API response type: {type(data)}")

    def validate(self, config: SourceConfig) -> list[str]:
        errors = []
        if not config.url:
            errors.append("API source requires 'url' parameter")
        return errors
```

- [ ] **Step 7: Run ApiSource tests to verify they fail**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_api.py -v`
Expected: FAIL with ImportError

- [ ] **Step 8: Implement ApiSource** (code follows in Step 6's block above)

- [ ] **Step 9: Run ApiSource tests to verify they pass**

Run: `cd packages/harness-data && python -m pytest tests/test_sources/test_api.py -v`
Expected: ALL PASS

- [ ] **Step 10: Register adapters in runner and run all tests**

Update the `SOURCE_ADAPTERS` dict in `packages/harness-data/src/harness/data/runner.py`:

```python
from harness.data.sources.url import UrlSource
from harness.data.sources.api import ApiSource

SOURCE_ADAPTERS = {
    "file": FileSource(),
    "url": UrlSource(),
    "api": ApiSource(),
}
```

Run: `cd packages/harness-data && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add packages/harness-data/
git commit -m "feat(harness-data): URL + API source adapters"
```

---

### Task 14: Final Integration Test + Package Verification

**Files:**
- Create: `packages/harness-data/tests/test_integration.py`

- [ ] **Step 1: Write integration test exercising the full pipeline**

Create `packages/harness-data/tests/test_integration.py`:

```python
"""End-to-end integration test: workspace init → add sources → add transforms → run → profile → validate."""

import json
import pandas as pd
import pytest
from pathlib import Path

from harness.data.workspace import DataWorkspace
from harness.data.profiling.profiler import DataProfiler
from harness.data.profiling.validation import SchemaValidator


class TestEndToEnd:
    def test_full_pipeline(self, tmp_path):
        # 1. Create test data
        games = pd.DataFrame({
            "game_id": range(1, 11),
            "team_a_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
            "team_b_id": [2, 3, 1, 3, 1, 2, 2, 3, 1, 3],
            "score_a": [85, 72, 90, 88, 76, 95, 82, 79, 91, 87],
            "score_b": [78, 80, 85, 91, 70, 88, 84, 75, 89, 90],
            "season": [2023] * 5 + [2024] * 5,
        })
        csv_path = tmp_path / "data" / "raw" / "games.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        games.to_csv(csv_path, index=False)

        # 2. Init workspace
        ws = DataWorkspace(tmp_path)
        ws.init()

        # 3. Add source
        ws.add_source("games", str(csv_path))

        # 4. Add transforms
        ws.add_transform({
            "op": "derive",
            "params": {"columns": {
                "score_diff": "score_a - score_b",
                "total_score": "score_a + score_b",
                "home_win": "where(score_a > score_b, 1, 0)",
            }},
        })
        ws.add_transform({
            "op": "filter",
            "params": {"expr": "total_score > 0"},
        })

        # 5. Run pipeline
        result = ws.run_pipeline()
        assert result.row_count == 10
        assert "score_diff" in result.columns
        assert "home_win" in result.columns

        # 6. Load and verify clean data
        df = ws.load_clean_data()
        assert len(df) == 10
        assert "total_score" in df.columns

        # 7. Profile the dataset
        profiler = DataProfiler()
        profile = profiler.profile(df)
        assert profile.row_count == 10
        assert profile.column_count == 9  # 6 original + 3 derived

        # 8. Validate schema
        validator = SchemaValidator(
            required_columns=["game_id", "score_a", "score_b", "home_win"],
            min_rows=5,
        )
        validation = validator.validate(df)
        assert validation.is_valid

        # 9. Verify schema.json
        schema = ws.load_schema()
        assert schema["row_count"] == 10
        assert "data_hash" in schema

    def test_transform_engine_used_as_library(self):
        """Verify harness-ml can use the transform engine as a library."""
        from harness.data.transforms.engine import TransformEngine
        from harness.data.transforms.protocol import StepConfig

        engine = TransformEngine()
        df = pd.DataFrame({
            "entity_id": [1, 1, 1, 2, 2, 2],
            "period": [1, 2, 3, 1, 2, 3],
            "points": [10.0, 15.0, 12.0, 20.0, 18.0, 22.0],
        })

        # Use rolling transform (what harness-ml would call for feature computation)
        result = engine.apply_step(df, StepConfig(
            op="rolling",
            params={
                "keys": ["entity_id"],
                "order_by": "period",
                "window": 2,
                "aggs": {"pts_rolling_mean_2": "points:mean"},
            },
        ))
        assert "pts_rolling_mean_2" in result.columns
        assert not result["pts_rolling_mean_2"].isna().all()
```

- [ ] **Step 2: Run integration tests**

Run: `cd packages/harness-data && python -m pytest tests/test_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite with coverage**

Run: `cd packages/harness-data && python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 4: Verify package installs cleanly**

Run: `cd packages/harness-data && pip install -e ".[dev]" && python -c "from harness.data import DataWorkspace, TransformEngine, DataProfiler; print('harness-data OK')"`
Expected: `harness-data OK`

- [ ] **Step 5: Final commit**

```bash
git add packages/harness-data/
git commit -m "feat(harness-data): integration tests + package verification"
```
