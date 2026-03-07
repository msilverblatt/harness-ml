# Data Layer Improvements Design

## Problem

The data loading/cleaning/alignment layer has structural inconsistencies that prevent
it from working generically end-to-end:

1. **Filename split**: MCP tooling uses `features.parquet` (via DataConfig), pipeline
   runner hardcodes `matchup_features.parquet`. A user who builds features via MCP
   tools and runs a backtest gets a FileNotFoundError.
2. **Missing schema fields**: `team_features_path` used in pipeline.py but absent
   from DataConfig.
3. **Disconnected source registry**: `SourceRegistry`/`RefreshOrchestrator` in
   harnessml-data exist but aren't wired to the ingest pipeline.
4. **Domain-specific hardcodes**: Profiler, pipeline runner, and join key detection
   all have March Madness-specific column names and logic baked in.
5. **Path hardcodes**: `config_writer.py`, `data_ingest.py`, and `stage_guards.py`
   bypass `data_utils.get_features_path()` in multiple places.
6. **Untracked dependency**: `data_utils.py` is imported everywhere but not committed.

## Approach

**Config-Driven Data Pipeline** (Approach A): DataConfig becomes the full pipeline
specification. Sources, cleaning rules, and join strategies are all declared in YAML.
A single `refresh()` runs the full chain. The AI-facing surface is purely config --
"add this data, make these features, run this model."

Two phases: fix inconsistencies first, then build the pipeline.

---

## Phase 1: Fix Inconsistencies

### 1. Unify feature file path

`PipelineRunner._load_data()` and `stage_guards.py` read `config.data.features_file`
via `data_utils.get_features_path()` instead of hardcoding `matchup_features.parquet`.

Affected files:
- `pipeline.py` (~3 occurrences)
- `stage_guards.py` (~3 occurrences)
- `cli.py` (~1 occurrence)

### 2. Add `team_features_path` to DataConfig

Optional field, defaults to `None`. Pipeline reads it from config instead of assuming
it exists as an attribute.

### 3. Fix path hardcodes

Route all feature store path resolution through `get_features_path()`:
- `config_writer.py` (lines 279, 302, 391)
- `data_ingest.py` `fill_nulls()`/`drop_duplicates()`/`rename_columns()` override branches

### 4. Make profiler config-aware

`profile_dataset()` accepts DataConfig fields (`target_column`, `key_columns`,
`time_column`) and uses them instead of hardcoded column name lookups (`TeamAWon`,
`Season`, `margin`, `TeamA`/`TeamB` prefixes).

### 5. Fix join key detection

`_detect_join_keys()` uses `DataConfig.target_column` and `exclude_columns` as
negative filters instead of the hardcoded skip list (`"result"`, `"target"`,
`"label"`, `"y"`, `"prediction"`, `"pred"`).

### 6. Fix correlation preview target

Use `config.target_column` consistently in `_compute_correlation_preview()` and
call sites, not `"result"` as fallback.

### 7. Commit `data_utils.py`

Already imported by `data_ingest.py`, `feature_engine.py`, and
`transformation_tester.py` but untracked in git.

---

## Phase 2: Config-Driven Data Pipeline

### Schema: ColumnCleaningRule and SourceConfig

```python
class ColumnCleaningRule(BaseModel):
    """Per-column cleaning configuration."""
    null_strategy: Literal["median", "mode", "zero", "drop", "ffill", "constant"] = "median"
    null_fill_value: Any | None = None
    coerce_numeric: bool = False
    clip_outliers: tuple[float, float] | None = None  # (lower_pct, upper_pct)
    log_transform: bool = False
    normalize: Literal["none", "zscore", "minmax"] = "none"

class SourceConfig(BaseModel):
    """A declared data source in the pipeline."""
    name: str
    path: str | None = None
    format: Literal["csv", "parquet", "excel", "auto"] = "auto"
    join_on: list[str] | None = None
    columns: dict[str, ColumnCleaningRule] | None = None
    default_cleaning: ColumnCleaningRule = ColumnCleaningRule()
    temporal_safety: Literal["pre_tournament", "post_tournament", "mixed", "unknown"] = "unknown"
    enabled: bool = True
```

### Extended DataConfig

```python
class DataConfig(BaseModel):
    # ... existing fields ...
    sources: dict[str, SourceConfig] = {}
    default_cleaning: ColumnCleaningRule = ColumnCleaningRule()
```

Cleaning rule cascade: column-level > source-level > global-level > smart defaults.

### DataPipeline orchestrator

New module `data_pipeline.py`:

```python
class DataPipeline:
    def __init__(self, project_dir: Path, config: DataConfig): ...

    def refresh(self, sources: list[str] | None = None) -> PipelineResult:
        """Full pipeline: read -> clean -> merge -> feature store."""

    def add_source(self, name: str, path: str, **kwargs) -> SourceConfig:
        """Register a new source and run initial ingest."""

    def remove_source(self, name: str) -> None:
        """Remove source and optionally its columns."""

    def clean(self, source: str | None = None) -> CleanResult:
        """Re-apply cleaning rules without re-reading source files."""
```

MCP `add_dataset` maps to `DataPipeline.add_source()`.

### Generic profiler with plugin hooks

```python
class ProfilerPlugin(Protocol):
    name: str
    def analyze(self, df: DataFrame, config: DataConfig) -> str: ...

class DataProfiler:
    def __init__(self, config: DataConfig, plugins: list[ProfilerPlugin] = []): ...
    def profile(self, df: DataFrame) -> DataProfile: ...
```

Core profiler is generic (uses DataConfig fields). Domain modules register plugins
for specialized analysis (team balance, season coverage, etc.).

### Pipeline runner integration

- `_load_data()` reads `get_features_path()` -- one path, from config
- Column normalization (TeamAWon -> result) moves to a migration/alias layer in
  DataConfig, not hardcoded in the runner
- `InjectionDef` and `InteractionDef` unchanged (already config-driven)

---

## Migration

Existing March Madness project needs:
- `features_file: matchup_features.parquet` added to its `data:` config
- Column aliases declared in DataConfig for backward compat
- Profiler domain logic extracted to a sports profiler plugin
