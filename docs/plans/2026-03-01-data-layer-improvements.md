# Data Layer Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all hardcoded paths, domain-specific logic, and missing schema fields in the data layer so it works generically end-to-end, then build a config-driven data pipeline with per-column cleaning rules and a pluggable profiler.

**Architecture:** Phase 1 makes the existing code consistent (DataConfig is the single source of truth for paths and column names). Phase 2 adds `SourceConfig`, `ColumnCleaningRule`, `DataPipeline` orchestrator, and a generic profiler with plugin hooks.

**Tech Stack:** Python 3.11+, Pydantic v2, pandas, pyarrow, uv, pytest

---

## Phase 1: Fix Inconsistencies

### Task 1: Add `team_features_path` to DataConfig and commit `data_utils.py`

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/schema.py:56-72`
- Modify: `packages/easyml-runner/tests/test_schema.py:441-476`
- Stage: `packages/easyml-runner/src/easyml/runner/data_utils.py` (untracked)

**Step 1: Write the failing test**

In `packages/easyml-runner/tests/test_schema.py`, add to class `TestDataConfigExtensions`:

```python
def test_team_features_path_default(self):
    d = DataConfig()
    assert d.team_features_path is None

def test_team_features_path_set(self):
    d = DataConfig(team_features_path="data/features/team_features.parquet")
    assert d.team_features_path == "data/features/team_features.parquet"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_schema.py::TestDataConfigExtensions::test_team_features_path_default -v`
Expected: FAIL — `DataConfig` has no field `team_features_path`

**Step 3: Add `team_features_path` to DataConfig**

In `packages/easyml-runner/src/easyml/runner/schema.py`, add to `DataConfig` after line 71 (`exclude_columns`):

```python
    team_features_path: str | None = None   # path to team-level features parquet
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/easyml-runner/tests/test_schema.py::TestDataConfigExtensions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/schema.py packages/easyml-runner/tests/test_schema.py packages/easyml-runner/src/easyml/runner/data_utils.py
git commit -m "feat(schema): add team_features_path to DataConfig, commit data_utils"
```

---

### Task 2: Make `PipelineGuards` use `DataConfig.features_file`

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/stage_guards.py:41-81`
- Modify: `packages/easyml-runner/tests/test_stage_guards.py`

**Step 1: Write/update the failing test**

Update the test fixture helper to use `features_file` from config. Add a new test class in `packages/easyml-runner/tests/test_stage_guards.py`:

```python
class TestConfigDrivenFeatureFile:
    """Guards use DataConfig.features_file instead of hardcoded name."""

    def test_custom_features_file(self, tmp_path):
        """Guard finds parquet using config's features_file."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "my_data.parquet")

        data_config = DataConfig(
            raw_dir="raw",
            processed_dir="proc",
            features_dir=str(features_dir),
            features_file="my_data.parquet",
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_train()  # Should not raise

    def test_default_features_file(self, tmp_path):
        """Guard uses 'features.parquet' as default."""
        features_dir = tmp_path / "features"
        features_dir.mkdir()
        df = pd.DataFrame({"a": range(20)})
        df.to_parquet(features_dir / "features.parquet")

        data_config = DataConfig(
            raw_dir="raw",
            processed_dir="proc",
            features_dir=str(features_dir),
        )
        guards = PipelineGuards(data_config, tmp_path)
        guards.guard_train()  # Should not raise
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_stage_guards.py::TestConfigDrivenFeatureFile -v`
Expected: FAIL — guards still look for `matchup_features.parquet`

**Step 3: Update `PipelineGuards` to use config**

In `packages/easyml-runner/src/easyml/runner/stage_guards.py`, change `_resolve_features_path` to return the full file path (not just dir), and update all guard methods:

```python
def _resolve_features_path(self) -> Path:
    """Resolve the features parquet file path from config."""
    features_dir = Path(self.data_config.features_dir)
    if not features_dir.is_absolute():
        features_dir = self.project_dir / features_dir
    return features_dir / self.data_config.features_file

def guard_train(self) -> None:
    if not self.enabled:
        return
    features_path = self._resolve_features_path()
    guard = StageGuard(name="train_ready", requires=[str(features_path)])
    guard.check()

def guard_predict(self, models_dir: Path | None = None) -> None:
    if not self.enabled:
        return
    features_path = self._resolve_features_path()
    requires = [str(features_path)]
    if models_dir is not None:
        requires.append(str(models_dir))
    guard = StageGuard(name="predict_ready", requires=requires)
    guard.check()

def guard_backtest(self) -> None:
    if not self.enabled:
        return
    features_path = self._resolve_features_path()
    guard = StageGuard(
        name="backtest_ready",
        requires=[str(features_path)],
        min_rows=10,
    )
    guard.check()
```

**Step 4: Update existing tests to use `features_file` in config**

Update all existing test fixtures that create `matchup_features.parquet` to instead use the default `features.parquet` (since that's now the DataConfig default). Change every occurrence of `df.to_parquet(features_dir / "matchup_features.parquet")` to `df.to_parquet(features_dir / "features.parquet")`.

**Step 5: Run all stage guard tests**

Run: `uv run pytest packages/easyml-runner/tests/test_stage_guards.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/stage_guards.py packages/easyml-runner/tests/test_stage_guards.py
git commit -m "fix(stage-guards): use DataConfig.features_file instead of hardcoded filename"
```

---

### Task 3: Make `PipelineRunner._load_data()` use DataConfig

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/pipeline.py:290-327`

**Step 1: Update `_load_data()` to use `data_utils.get_features_path()`**

Replace the hardcoded path logic in `_load_data()` (lines 290-306):

```python
def _load_data(self) -> None:
    """Read features parquet and normalize column names.

    Also loads team features if any model has
    ``provides_level="team"``.
    """
    from easyml.runner.data_utils import get_features_path

    parquet_path = get_features_path(self.project_dir, self.config.data)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {parquet_path}. "
            f"Configured: features_dir={self.config.data.features_dir}, "
            f"features_file={self.config.data.features_file}"
        )

    self._df = pd.read_parquet(parquet_path)
    self._normalize_columns()

    # Load team features if any model has provides_level="team"
    has_team_providers = any(
        m.provides and m.provides_level == "team"
        for m in self.config.models.values()
    )
    if has_team_providers and self.config.data.team_features_path:
        team_path = Path(self.config.data.team_features_path)
        if not team_path.is_absolute():
            team_path = self.project_dir / team_path
        if team_path.exists():
            self._team_df = pd.read_parquet(team_path)
            logger.info("Loaded team features: %s", team_path)
        else:
            logger.warning(
                "Team features path configured but not found: %s",
                team_path,
            )

    # Apply feature injections if configured
    if self.config.injections:
        self._apply_injections()

    # Apply interaction features if configured
    if self.config.interactions:
        from easyml.runner.matchups import compute_interactions
        self._df = compute_interactions(self._df, self.config.interactions)
```

**Step 2: Run existing pipeline tests**

Run: `uv run pytest packages/easyml-runner/tests/test_pipeline.py -v`
Expected: PASS (tests should still work since they create their own data)

**Step 3: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/pipeline.py
git commit -m "fix(pipeline): use DataConfig.features_file instead of hardcoded path"
```

---

### Task 4: Fix CLI hardcoded path

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/cli.py:163-164`

**Step 1: Update CLI to use `get_features_path()`**

Replace lines 163-164 in `cli.py`:

```python
    from easyml.runner.data_utils import get_features_path
    parquet_path = get_features_path(Path.cwd(), result.config.data)
```

**Step 2: Run CLI tests**

Run: `uv run pytest packages/easyml-runner/tests/test_cli.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/cli.py
git commit -m "fix(cli): use DataConfig for features path resolution"
```

---

### Task 5: Fix path hardcodes in `config_writer.py`

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/config_writer.py:270-302,380-391`

**Step 1: Replace hardcoded fallback paths with `get_features_path()`**

In all three functions (`profile_data`, `available_features`, `discover_features`), the `except` branch falls back to a hardcoded path. Change each to:

```python
    try:
        config = load_data_config(project_dir)
        parquet_path = get_features_path(project_dir, config)
    except Exception:
        parquet_path = get_features_path(project_dir, DataConfig())
```

This uses `DataConfig()` defaults (`data/features/features.parquet`) instead of repeating the path literal.

Add the import at the top of each function or the module:

```python
from easyml.runner.schema import DataConfig
```

**Step 2: Run config_writer tests**

Run: `uv run pytest packages/easyml-runner/tests/test_config_writer.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/config_writer.py
git commit -m "fix(config-writer): use DataConfig defaults for fallback paths"
```

---

### Task 6: Fix path hardcodes in `data_ingest.py` granular tools

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/data_ingest.py:596-710`

**Step 1: Fix `fill_nulls`, `drop_duplicates`, `rename_columns`**

In each function, the `features_dir` override branch hardcodes `"features.parquet"`. Change each to load config and use `features_file`:

For `fill_nulls` (lines 599-606):
```python
    if features_dir is not None:
        try:
            config = load_data_config(project_dir)
            parquet_path = Path(features_dir) / config.features_file
        except Exception:
            parquet_path = Path(features_dir) / "features.parquet"
    else:
        try:
            config = load_data_config(project_dir)
            parquet_path = get_features_path(project_dir, config)
        except Exception:
            parquet_path = project_dir / "data" / "features" / "features.parquet"
```

Apply the same pattern to `drop_duplicates` (lines 659-666) and `rename_columns` (lines 703-710).

**Step 2: Run data ingest tests**

Run: `uv run pytest packages/easyml-runner/tests/test_data_ingest.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/data_ingest.py
git commit -m "fix(data-ingest): use config.features_file in granular tools"
```

---

### Task 7: Fix `_detect_join_keys()` to use config-aware negative filter

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/data_ingest.py:40-76`
- Modify: `packages/easyml-runner/tests/test_data_ingest.py`

**Step 1: Write failing test**

Add to `packages/easyml-runner/tests/test_data_ingest.py`:

```python
class TestDetectJoinKeysConfigAware:
    """_detect_join_keys uses config columns as negative filter."""

    def test_excludes_target_column(self):
        """Target column should not be used as a join key."""
        from easyml.runner.data_ingest import _detect_join_keys

        new_df = pd.DataFrame({"season": [2020], "outcome": [1], "feat": [0.5]})
        existing_df = pd.DataFrame({"season": [2020], "outcome": [0], "diff_x": [1.0]})

        # Without exclude_cols, "outcome" would be a join key candidate
        keys = _detect_join_keys(new_df, existing_df)
        assert "outcome" in keys  # Old behavior

        # With exclude_cols, "outcome" is filtered out
        keys = _detect_join_keys(new_df, existing_df, exclude_cols=["outcome"])
        assert "outcome" not in keys
        assert "season" in keys
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_data_ingest.py::TestDetectJoinKeysConfigAware -v`
Expected: FAIL — `_detect_join_keys` doesn't accept `exclude_cols`

**Step 3: Add `exclude_cols` parameter to `_detect_join_keys()`**

```python
def _detect_join_keys(
    new_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    key_columns: list[str] | None = None,
    exclude_cols: list[str] | None = None,
) -> list[str] | None:
    """Auto-detect join keys between new data and existing features."""
    new_cols = set(new_df.columns)
    existing_cols = set(existing_df.columns)

    # Strategy 1: use configured key_columns
    if key_columns:
        overlap = [k for k in key_columns if k in new_cols and k in existing_cols]
        if overlap:
            return overlap

    # Strategy 2: find all overlapping columns
    common = sorted(new_cols & existing_cols)
    if not common:
        return None

    # Build exclusion set from config
    skip = set()
    if exclude_cols:
        skip.update(c.lower() for c in exclude_cols)

    key_candidates = []
    for col in common:
        if col.lower() in skip:
            continue
        key_candidates.append(col)

    return key_candidates if key_candidates else None
```

**Step 4: Update call site in `ingest_dataset()` to pass exclude_cols**

In `ingest_dataset()` around line 395, update the call:

```python
    if join_on is None:
        # Build exclusion list from config
        exclude_cols = [target_col]
        try:
            config = load_data_config(project_dir)
            exclude_cols.extend(config.exclude_columns)
        except Exception:
            pass
        join_on = _detect_join_keys(
            new_df, existing_df,
            key_columns=configured_keys,
            exclude_cols=exclude_cols,
        )
```

**Step 5: Run tests**

Run: `uv run pytest packages/easyml-runner/tests/test_data_ingest.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/data_ingest.py packages/easyml-runner/tests/test_data_ingest.py
git commit -m "fix(data-ingest): config-aware join key detection excludes target and excluded columns"
```

---

### Task 8: Fix correlation preview to use `config.target_column`

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/data_ingest.py:146-151`

**Step 1: Update `_compute_correlation_preview` signature**

Remove the default value for `target_col` — callers must pass it explicitly:

```python
def _compute_correlation_preview(
    df: pd.DataFrame,
    new_columns: list[str],
    target_col: str,
    top_n: int = 5,
) -> list[tuple[str, float]]:
```

This is already called correctly in `ingest_dataset()` which resolves `target_col` from config. The signature change just removes the misleading default.

**Step 2: Run tests**

Run: `uv run pytest packages/easyml-runner/tests/test_data_ingest.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/data_ingest.py
git commit -m "fix(data-ingest): remove hardcoded target_col default from correlation preview"
```

---

### Task 9: Make profiler config-aware (generic)

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/data_profiler.py:161-246`
- Modify: `packages/easyml-runner/tests/test_data_profiler.py`
- Modify: `packages/easyml-runner/src/easyml/runner/config_writer.py` (pass config to profiler)
- Modify: `packages/easyml-runner/src/easyml/runner/cli.py` (pass config to profiler)

**Step 1: Write failing test for config-aware profiler**

Add to `packages/easyml-runner/tests/test_data_profiler.py`:

```python
from easyml.runner.schema import DataConfig


class TestConfigAwareProfiler:
    """Profiler uses DataConfig fields instead of hardcoded column names."""

    def test_custom_target_column(self, tmp_path):
        """Profiler detects label from config.target_column."""
        df = pd.DataFrame({
            "date": [2020, 2021, 2022],
            "won": [1.0, 0.0, 1.0],
            "score_diff": [5.0, -3.0, 12.0],
            "feat_a": [1.0, 2.0, 3.0],
        })
        path = tmp_path / "features.parquet"
        df.to_parquet(path)

        config = DataConfig(
            target_column="won",
            time_column="date",
            key_columns=[],
            exclude_columns=["score_diff"],
        )
        profile = profile_dataset(path, config=config)
        assert profile.label_column == "won"
        assert profile.time_column == "date"

    def test_custom_key_columns_excluded(self, tmp_path):
        """Key columns and exclude columns are not profiled as features."""
        df = pd.DataFrame({
            "game_id": [1, 2, 3],
            "season": [2020, 2021, 2022],
            "result": [1.0, 0.0, 1.0],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [4.0, 5.0, 6.0],
        })
        path = tmp_path / "features.parquet"
        df.to_parquet(path)

        config = DataConfig(
            target_column="result",
            time_column="season",
            key_columns=["game_id"],
        )
        profile = profile_dataset(path, config=config)
        all_feature_names = [c.name for c in profile.feature_columns]
        assert "game_id" not in all_feature_names
        assert "season" not in all_feature_names
        assert "result" not in all_feature_names
        assert "feat_a" in all_feature_names
        assert "feat_b" in all_feature_names

    def test_backward_compat_no_config(self, sample_parquet):
        """Without config, profiler still works using column name heuristics."""
        profile = profile_dataset(sample_parquet)
        assert profile.n_rows == 200
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_data_profiler.py::TestConfigAwareProfiler -v`
Expected: FAIL — `profile_dataset` doesn't accept `config` param

**Step 3: Update `profile_dataset()` to accept optional DataConfig**

Refactor `data_profiler.py`:

1. Change `DataProfile` to use `feature_columns` instead of `diff_columns` + `non_diff_columns` (with backward compat properties).
2. Update `profile_dataset()` signature:

```python
def profile_dataset(
    path: str | Path,
    high_null_threshold: float = 50.0,
    config: DataConfig | None = None,
) -> DataProfile:
```

When `config` is provided:
- Use `config.target_column` instead of searching `["result", "TeamAWon"]`
- Use `config.time_column` instead of searching `["Season", "season"]`
- Use `config.key_columns` + `config.exclude_columns` to determine skip set
- Classify features generically (no `diff_` prefix assumption)

When `config` is None:
- Fall back to current heuristic behavior for backward compat

Add `time_column` field to `DataProfile`:

```python
@dataclass
class DataProfile:
    # ... existing fields ...
    time_column: str | None = None
    feature_columns: list[ColumnProfile] = field(default_factory=list)
```

Keep `diff_columns` and `non_diff_columns` as computed properties for backward compat:

```python
    @property
    def diff_columns(self) -> list[ColumnProfile]:
        return [c for c in self.feature_columns if c.name.startswith("diff_")]

    @property
    def non_diff_columns(self) -> list[ColumnProfile]:
        return [c for c in self.feature_columns if not c.name.startswith("diff_")]
```

Remove the hardcoded `id_patterns` list and the `s >= 2015 and s != 2020` backtest logic from `format_summary()`. Instead, just show all time periods.

**Step 4: Update callers to pass config**

In `config_writer.py` `profile_data()`:
```python
    profile = profile_dataset(parquet_path, config=config)
```

In `cli.py`:
```python
    profile = profile_dataset(parquet_path, config=result.config.data)
```

**Step 5: Run all profiler tests**

Run: `uv run pytest packages/easyml-runner/tests/test_data_profiler.py -v`
Expected: ALL PASS

**Step 6: Run full test suite to check nothing broke**

Run: `uv run pytest packages/easyml-runner/tests/ -v`
Expected: ALL PASS

**Step 7: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/data_profiler.py packages/easyml-runner/tests/test_data_profiler.py packages/easyml-runner/src/easyml/runner/config_writer.py packages/easyml-runner/src/easyml/runner/cli.py
git commit -m "feat(profiler): make profiler config-aware, remove domain-specific hardcodes"
```

---

### Task 10: Phase 1 integration test

**Files:**
- Modify: `packages/easyml-runner/tests/test_data_ingest.py`

**Step 1: Write an end-to-end test that exercises the full path**

```python
class TestEndToEndConfigDrivenPath:
    """Full path: DataConfig -> ingest -> profiler -> guards all use same config."""

    def test_custom_features_file_works_e2e(self, tmp_path):
        """Ingest, profile, and guard check all work with custom features_file."""
        # Setup project structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n"
            "  features_dir: data/features\n"
            "  features_file: my_dataset.parquet\n"
            "  target_column: outcome\n"
            "  key_columns: [id, period]\n"
            "  time_column: period\n"
        )
        features_dir = tmp_path / "data" / "features"
        features_dir.mkdir(parents=True)

        # Create initial data
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "period": [2020] * 5 + [2021] * 5,
            "outcome": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })
        df.to_parquet(features_dir / "my_dataset.parquet", index=False)

        # Verify profiler works
        from easyml.runner.data_profiler import profile_dataset
        from easyml.runner.data_utils import load_data_config, get_features_path

        config = load_data_config(tmp_path)
        assert config.features_file == "my_dataset.parquet"
        assert config.target_column == "outcome"

        parquet_path = get_features_path(tmp_path, config)
        assert parquet_path.exists()

        profile = profile_dataset(parquet_path, config=config)
        assert profile.n_rows == 10
        assert profile.label_column == "outcome"

        # Verify guards work
        from easyml.runner.stage_guards import PipelineGuards
        guards = PipelineGuards(config, tmp_path)
        guards.guard_train()  # Should not raise
        guards.guard_backtest()  # Should not raise
```

**Step 2: Run the integration test**

Run: `uv run pytest packages/easyml-runner/tests/test_data_ingest.py::TestEndToEndConfigDrivenPath -v`
Expected: PASS

**Step 3: Commit**

```bash
git add packages/easyml-runner/tests/test_data_ingest.py
git commit -m "test: add e2e integration test for config-driven data path"
```

---

## Phase 2: Config-Driven Data Pipeline

### Task 11: Add `ColumnCleaningRule` and `SourceConfig` schemas

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/schema.py`
- Modify: `packages/easyml-runner/tests/test_schema.py`

**Step 1: Write failing tests**

Add to `packages/easyml-runner/tests/test_schema.py`:

```python
from easyml.runner.schema import ColumnCleaningRule, SourceConfig


class TestColumnCleaningRule:
    """ColumnCleaningRule schema and defaults."""

    def test_defaults(self):
        rule = ColumnCleaningRule()
        assert rule.null_strategy == "median"
        assert rule.null_fill_value is None
        assert rule.coerce_numeric is False
        assert rule.clip_outliers is None
        assert rule.log_transform is False
        assert rule.normalize == "none"

    def test_custom_values(self):
        rule = ColumnCleaningRule(
            null_strategy="constant",
            null_fill_value=0,
            coerce_numeric=True,
            clip_outliers=(1.0, 99.0),
            log_transform=True,
            normalize="zscore",
        )
        assert rule.null_strategy == "constant"
        assert rule.null_fill_value == 0
        assert rule.clip_outliers == (1.0, 99.0)

    def test_invalid_null_strategy(self):
        with pytest.raises(ValidationError):
            ColumnCleaningRule(null_strategy="invalid")

    def test_invalid_normalize(self):
        with pytest.raises(ValidationError):
            ColumnCleaningRule(normalize="invalid")


class TestSourceConfig:
    """SourceConfig schema and defaults."""

    def test_defaults(self):
        src = SourceConfig(name="test_source")
        assert src.path is None
        assert src.format == "auto"
        assert src.join_on is None
        assert src.columns is None
        assert src.temporal_safety == "unknown"
        assert src.enabled is True

    def test_with_columns(self):
        src = SourceConfig(
            name="kenpom",
            path="data/raw/kenpom.csv",
            format="csv",
            join_on=["team_id", "season"],
            columns={
                "adj_oe": ColumnCleaningRule(null_strategy="zero"),
                "adj_de": ColumnCleaningRule(coerce_numeric=True),
            },
            temporal_safety="pre_tournament",
        )
        assert src.columns["adj_oe"].null_strategy == "zero"
        assert src.columns["adj_de"].coerce_numeric is True

    def test_invalid_format(self):
        with pytest.raises(ValidationError):
            SourceConfig(name="bad", format="jsonl")

    def test_invalid_temporal_safety(self):
        with pytest.raises(ValidationError):
            SourceConfig(name="bad", temporal_safety="future")


class TestDataConfigSources:
    """DataConfig with sources section."""

    def test_data_config_with_sources(self):
        d = DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.csv",
                    temporal_safety="pre_tournament",
                ),
            },
            default_cleaning=ColumnCleaningRule(null_strategy="zero"),
        )
        assert "kenpom" in d.sources
        assert d.default_cleaning.null_strategy == "zero"

    def test_data_config_sources_default_empty(self):
        d = DataConfig()
        assert d.sources == {}
        assert d.default_cleaning.null_strategy == "median"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_schema.py::TestColumnCleaningRule -v`
Expected: FAIL — classes don't exist yet

**Step 3: Implement the schemas**

Add to `packages/easyml-runner/src/easyml/runner/schema.py`, before `DataConfig`:

```python
class ColumnCleaningRule(BaseModel):
    """Per-column cleaning configuration with cascade support."""

    null_strategy: Literal["median", "mode", "zero", "drop", "ffill", "constant"] = "median"
    null_fill_value: Any | None = None
    coerce_numeric: bool = False
    clip_outliers: tuple[float, float] | None = None
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

Add to `DataConfig`:

```python
    sources: dict[str, SourceConfig] = {}
    default_cleaning: ColumnCleaningRule = ColumnCleaningRule()
```

**Step 4: Run tests**

Run: `uv run pytest packages/easyml-runner/tests/test_schema.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/schema.py packages/easyml-runner/tests/test_schema.py
git commit -m "feat(schema): add ColumnCleaningRule and SourceConfig schemas"
```

---

### Task 12: Build `DataPipeline` orchestrator

**Files:**
- Create: `packages/easyml-runner/src/easyml/runner/data_pipeline.py`
- Create: `packages/easyml-runner/tests/test_data_pipeline.py`

**Step 1: Write failing tests**

Create `packages/easyml-runner/tests/test_data_pipeline.py`:

```python
"""Tests for DataPipeline orchestrator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.runner.schema import ColumnCleaningRule, DataConfig, SourceConfig


class TestDataPipelineRefresh:
    """DataPipeline.refresh() processes declared sources."""

    @pytest.fixture
    def project_with_source(self, tmp_path):
        """Project dir with a CSV source and pipeline config."""
        # Create source data
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        df = pd.DataFrame({
            "team_id": [1, 2, 3],
            "season": [2020, 2020, 2020],
            "adj_oe": [110.5, 105.2, np.nan],
            "adj_de": ["98.1", "102.3", "95.0"],  # strings that should be coerced
        })
        df.to_csv(raw_dir / "kenpom.csv", index=False)

        # Write config
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config = {
            "data": {
                "features_dir": "data/features",
                "features_file": "features.parquet",
                "target_column": "outcome",
                "key_columns": ["team_id", "season"],
                "sources": {
                    "kenpom": {
                        "name": "kenpom",
                        "path": "data/raw/kenpom.csv",
                        "format": "csv",
                        "join_on": ["team_id", "season"],
                        "temporal_safety": "pre_tournament",
                        "columns": {
                            "adj_oe": {"null_strategy": "zero"},
                            "adj_de": {"coerce_numeric": True},
                        },
                    },
                },
            },
        }
        (config_dir / "pipeline.yaml").write_text(yaml.dump(config))
        return tmp_path

    def test_refresh_bootstrap(self, project_with_source):
        """First refresh bootstraps the feature store."""
        from easyml.runner.data_pipeline import DataPipeline
        from easyml.runner.data_utils import load_data_config

        config = load_data_config(project_with_source)
        pipeline = DataPipeline(project_with_source, config)
        result = pipeline.refresh()

        assert result.sources_processed == 1
        features_path = project_with_source / "data" / "features" / "features.parquet"
        assert features_path.exists()

        df = pd.read_parquet(features_path)
        assert len(df) == 3
        # adj_oe null should be filled with zero (per column rule)
        assert df["adj_oe"].isna().sum() == 0
        # adj_de should be numeric (coerced)
        assert pd.api.types.is_numeric_dtype(df["adj_de"])

    def test_refresh_specific_source(self, project_with_source):
        """Can refresh a specific source by name."""
        from easyml.runner.data_pipeline import DataPipeline
        from easyml.runner.data_utils import load_data_config

        config = load_data_config(project_with_source)
        pipeline = DataPipeline(project_with_source, config)
        result = pipeline.refresh(sources=["kenpom"])
        assert result.sources_processed == 1

    def test_refresh_nonexistent_source_error(self, project_with_source):
        """Requesting a source that doesn't exist raises an error."""
        from easyml.runner.data_pipeline import DataPipeline
        from easyml.runner.data_utils import load_data_config

        config = load_data_config(project_with_source)
        pipeline = DataPipeline(project_with_source, config)
        with pytest.raises(ValueError, match="not_real"):
            pipeline.refresh(sources=["not_real"])


class TestCleaningRuleCascade:
    """Cleaning rules cascade: column > source > global."""

    def test_column_overrides_source_default(self, tmp_path):
        """Column-level rule overrides source-level default."""
        from easyml.runner.data_pipeline import resolve_cleaning_rule

        source = SourceConfig(
            name="test",
            default_cleaning=ColumnCleaningRule(null_strategy="median"),
            columns={"col_a": ColumnCleaningRule(null_strategy="zero")},
        )
        global_default = ColumnCleaningRule(null_strategy="mode")

        rule = resolve_cleaning_rule("col_a", source, global_default)
        assert rule.null_strategy == "zero"

    def test_source_default_overrides_global(self, tmp_path):
        """Source default used when no column rule exists."""
        from easyml.runner.data_pipeline import resolve_cleaning_rule

        source = SourceConfig(
            name="test",
            default_cleaning=ColumnCleaningRule(null_strategy="median"),
        )
        global_default = ColumnCleaningRule(null_strategy="mode")

        rule = resolve_cleaning_rule("col_b", source, global_default)
        assert rule.null_strategy == "median"

    def test_global_used_when_no_source_override(self, tmp_path):
        """Global default used when source has no override."""
        from easyml.runner.data_pipeline import resolve_cleaning_rule

        source = SourceConfig(name="test")
        global_default = ColumnCleaningRule(null_strategy="mode")

        rule = resolve_cleaning_rule("col_c", source, global_default)
        assert rule.null_strategy == "mode"


class TestAddSource:
    """DataPipeline.add_source() registers and ingests."""

    def test_add_source(self, tmp_path):
        """add_source registers in config and runs initial ingest."""
        from easyml.runner.data_pipeline import DataPipeline

        # Setup minimal project
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "pipeline.yaml").write_text(
            "data:\n  features_dir: data/features\n  target_column: y\n"
        )

        # Create data file
        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        df = pd.DataFrame({"id": [1, 2], "y": [0, 1], "feat": [1.0, 2.0]})
        df.to_csv(raw_dir / "new_source.csv", index=False)

        config = DataConfig(target_column="y")
        pipeline = DataPipeline(tmp_path, config)
        source_config = pipeline.add_source("new_data", "data/raw/new_source.csv")

        assert source_config.name == "new_data"
        features_path = tmp_path / "data" / "features" / "features.parquet"
        assert features_path.exists()


class TestRemoveSource:
    """DataPipeline.remove_source() removes tracked source."""

    def test_remove_source(self, tmp_path):
        """remove_source removes source from config."""
        from easyml.runner.data_pipeline import DataPipeline

        config = DataConfig(
            sources={
                "old": SourceConfig(name="old", path="data/raw/old.csv"),
            },
        )
        pipeline = DataPipeline(tmp_path, config)
        pipeline.remove_source("old")
        assert "old" not in pipeline.config.sources
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_data_pipeline.py -v`
Expected: FAIL — module doesn't exist

**Step 3: Implement `DataPipeline`**

Create `packages/easyml-runner/src/easyml/runner/data_pipeline.py`:

```python
"""Config-driven data pipeline orchestrator.

Reads SourceConfig declarations from DataConfig, executes the full
source -> clean -> merge -> feature store chain.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from easyml.runner.data_utils import get_features_path
from easyml.runner.schema import ColumnCleaningRule, DataConfig, SourceConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a pipeline refresh."""

    sources_processed: int = 0
    sources_skipped: int = 0
    errors: dict[str, str] = field(default_factory=dict)
    columns_added: dict[str, list[str]] = field(default_factory=dict)


def resolve_cleaning_rule(
    column: str,
    source: SourceConfig,
    global_default: ColumnCleaningRule,
) -> ColumnCleaningRule:
    """Resolve the cleaning rule for a column via cascade.

    Priority: column-level > source-level > global-level.
    """
    if source.columns and column in source.columns:
        return source.columns[column]
    if source.default_cleaning != ColumnCleaningRule():
        return source.default_cleaning
    return global_default


def apply_cleaning_rule(series: pd.Series, rule: ColumnCleaningRule) -> pd.Series:
    """Apply a cleaning rule to a pandas Series."""
    s = series.copy()

    # Coerce numeric
    if rule.coerce_numeric:
        s = pd.to_numeric(s, errors="coerce")

    # Clip outliers
    if rule.clip_outliers is not None:
        lower_pct, upper_pct = rule.clip_outliers
        lower = s.quantile(lower_pct / 100)
        upper = s.quantile(upper_pct / 100)
        s = s.clip(lower, upper)

    # Log transform
    if rule.log_transform and pd.api.types.is_numeric_dtype(s):
        s = np.log1p(s.clip(lower=0))

    # Fill nulls
    null_count = s.isna().sum()
    if null_count > 0:
        if rule.null_strategy == "median" and pd.api.types.is_numeric_dtype(s):
            s = s.fillna(s.median())
        elif rule.null_strategy == "mode":
            modes = s.mode()
            if len(modes) > 0:
                s = s.fillna(modes.iloc[0])
        elif rule.null_strategy == "zero":
            s = s.fillna(0)
        elif rule.null_strategy == "ffill":
            s = s.ffill()
        elif rule.null_strategy == "constant":
            if rule.null_fill_value is not None:
                s = s.fillna(rule.null_fill_value)
        elif rule.null_strategy == "drop":
            pass  # handled at DataFrame level by caller

    # Normalize
    if rule.normalize == "zscore" and pd.api.types.is_numeric_dtype(s):
        std = s.std()
        if std > 0:
            s = (s - s.mean()) / std
    elif rule.normalize == "minmax" and pd.api.types.is_numeric_dtype(s):
        smin, smax = s.min(), s.max()
        if smax > smin:
            s = (s - smin) / (smax - smin)

    return s


def _read_source(project_dir: Path, source: SourceConfig) -> pd.DataFrame:
    """Read a source file based on its config."""
    if source.path is None:
        raise ValueError(f"Source '{source.name}' has no path configured")

    path = Path(source.path)
    if not path.is_absolute():
        path = project_dir / path

    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    fmt = source.format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            fmt = "parquet"
        elif suffix == ".csv":
            fmt = "csv"
        elif suffix in (".xlsx", ".xls"):
            fmt = "excel"
        else:
            raise ValueError(f"Cannot auto-detect format for {path}")

    if fmt == "parquet":
        return pd.read_parquet(path)
    elif fmt == "csv":
        return pd.read_csv(path)
    elif fmt == "excel":
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


class DataPipeline:
    """Config-driven data pipeline orchestrator."""

    def __init__(self, project_dir: Path, config: DataConfig) -> None:
        self.project_dir = Path(project_dir)
        self.config = config

    def refresh(self, sources: list[str] | None = None) -> PipelineResult:
        """Run the full pipeline for all (or specified) sources."""
        result = PipelineResult()

        # Determine which sources to process
        if sources is not None:
            for name in sources:
                if name not in self.config.sources:
                    raise ValueError(
                        f"Source '{name}' not found in config. "
                        f"Available: {list(self.config.sources.keys())}"
                    )
            to_process = {k: v for k, v in self.config.sources.items() if k in sources}
        else:
            to_process = {k: v for k, v in self.config.sources.items() if v.enabled}

        features_path = get_features_path(self.project_dir, self.config)
        features_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing features if they exist
        existing_df = None
        if features_path.exists():
            existing_df = pd.read_parquet(features_path)

        for name, source in to_process.items():
            try:
                df = _read_source(self.project_dir, source)
                df = self._apply_cleaning(df, source)

                if existing_df is None:
                    existing_df = df
                    result.columns_added[name] = list(df.columns)
                else:
                    existing_df, new_cols = self._merge(existing_df, df, source)
                    result.columns_added[name] = new_cols

                result.sources_processed += 1
            except Exception as exc:
                result.errors[name] = str(exc)
                logger.error("Failed to process source '%s': %s", name, exc)

        if existing_df is not None:
            existing_df.to_parquet(features_path, index=False)

        return result

    def add_source(self, name: str, path: str, **kwargs) -> SourceConfig:
        """Register a new source and run initial ingest."""
        source = SourceConfig(name=name, path=path, **kwargs)
        self.config.sources[name] = source
        self.refresh(sources=[name])
        return source

    def remove_source(self, name: str) -> None:
        """Remove a source from config."""
        self.config.sources.pop(name, None)

    def _apply_cleaning(self, df: pd.DataFrame, source: SourceConfig) -> pd.DataFrame:
        """Apply cleaning rules to all columns in a DataFrame."""
        df = df.copy()
        drop_cols = []

        for col in df.columns:
            rule = resolve_cleaning_rule(col, source, self.config.default_cleaning)
            df[col] = apply_cleaning_rule(df[col], rule)
            if rule.null_strategy == "drop" and df[col].isna().any():
                drop_cols.append(col)

        if drop_cols:
            df = df.dropna(subset=drop_cols)

        # Drop exact duplicates
        df = df.drop_duplicates()

        return df

    def _merge(
        self,
        existing: pd.DataFrame,
        new: pd.DataFrame,
        source: SourceConfig,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Merge new data into existing features. Returns (merged, new_columns)."""
        join_on = source.join_on
        if join_on is None:
            # Auto-detect from config key_columns
            common = sorted(set(new.columns) & set(existing.columns))
            if self.config.key_columns:
                join_on = [k for k in self.config.key_columns if k in common]
            if not join_on:
                join_on = common
            if not join_on:
                raise ValueError(
                    f"Cannot auto-detect join keys for source '{source.name}'"
                )

        existing_cols = set(existing.columns)
        new_columns = [c for c in new.columns if c not in join_on and c not in existing_cols]

        if not new_columns:
            return existing, []

        merge_cols = join_on + new_columns
        merge_df = new[merge_cols].drop_duplicates(subset=join_on, keep="first")
        merged = existing.merge(merge_df, on=join_on, how="left")

        return merged, new_columns
```

**Step 4: Run tests**

Run: `uv run pytest packages/easyml-runner/tests/test_data_pipeline.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/data_pipeline.py packages/easyml-runner/tests/test_data_pipeline.py
git commit -m "feat(data-pipeline): add config-driven DataPipeline orchestrator"
```

---

### Task 13: Add profiler plugin system

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/data_profiler.py`
- Modify: `packages/easyml-runner/tests/test_data_profiler.py`

**Step 1: Write failing test**

Add to `packages/easyml-runner/tests/test_data_profiler.py`:

```python
class TestProfilerPlugins:
    """Profiler accepts plugins for domain-specific analysis."""

    def test_plugin_output_included(self, tmp_path):
        """Plugin analyze() output appears in the profile."""
        df = pd.DataFrame({
            "season": [2020, 2021],
            "result": [1.0, 0.0],
            "feat_a": [1.0, 2.0],
        })
        path = tmp_path / "features.parquet"
        df.to_parquet(path)

        class TestPlugin:
            name = "test_domain"
            def analyze(self, df, config):
                return "## Test Domain\nCustom analysis here."

        config = DataConfig(target_column="result", time_column="season")
        profile = profile_dataset(path, config=config, plugins=[TestPlugin()])
        assert "test_domain" in [s.name for s in profile.plugin_sections]

    def test_no_plugins_by_default(self, sample_parquet):
        """Without plugins, no plugin sections."""
        profile = profile_dataset(sample_parquet)
        assert profile.plugin_sections == []
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/easyml-runner/tests/test_data_profiler.py::TestProfilerPlugins -v`
Expected: FAIL

**Step 3: Add plugin support to profiler**

Add to `DataProfile`:

```python
@dataclass
class PluginSection:
    """Output from a profiler plugin."""
    name: str
    content: str
```

In `DataProfile`, add:

```python
    plugin_sections: list[PluginSection] = field(default_factory=list)
```

Update `profile_dataset` signature:

```python
def profile_dataset(
    path: str | Path,
    high_null_threshold: float = 50.0,
    config: DataConfig | None = None,
    plugins: list | None = None,
) -> DataProfile:
```

At the end of `profile_dataset`, after all standard profiling:

```python
    # Run plugins
    if plugins:
        for plugin in plugins:
            try:
                content = plugin.analyze(df, config)
                profile.plugin_sections.append(PluginSection(name=plugin.name, content=content))
            except Exception as exc:
                logger.warning("Plugin '%s' failed: %s", getattr(plugin, 'name', '?'), exc)
```

Update `format_summary` to include plugin output:

```python
        if self.plugin_sections:
            for section in self.plugin_sections:
                lines.append(f"\n{section.content}")
```

**Step 4: Run all profiler tests**

Run: `uv run pytest packages/easyml-runner/tests/test_data_profiler.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/data_profiler.py packages/easyml-runner/tests/test_data_profiler.py
git commit -m "feat(profiler): add plugin system for domain-specific profiling"
```

---

### Task 14: Wire DataPipeline into MCP tools

**Files:**
- Modify: `packages/easyml-runner/src/easyml/runner/config_writer.py`
- Modify: `packages/easyml-runner/tests/test_config_writer.py`

**Step 1: Update `add_dataset` to use DataPipeline when sources are configured**

In `config_writer.py`, update `add_dataset`:

```python
def add_dataset(
    project_dir: Path,
    data_path: str,
    *,
    join_on: list[str] | None = None,
    prefix: str | None = None,
    features_dir: str | None = None,
    auto_clean: bool = True,
) -> str:
    """Add a new dataset by merging into the features parquet.

    If DataConfig has sources configured, uses DataPipeline.
    Otherwise falls back to direct ingest.
    """
    from easyml.runner.data_utils import load_data_config

    project_dir = Path(project_dir)
    try:
        config = load_data_config(project_dir)
    except Exception:
        config = None

    # Use DataPipeline if config supports it
    if config is not None and config.sources:
        from easyml.runner.data_pipeline import DataPipeline
        pipeline = DataPipeline(project_dir, config)
        name = Path(data_path).stem
        source = pipeline.add_source(name, data_path, join_on=join_on)
        result = pipeline.refresh(sources=[name])
        cols = result.columns_added.get(name, [])
        return f"## Ingested: {name}\n- Columns added: {len(cols)}\n- Source registered in pipeline config"

    # Fallback to direct ingest
    from easyml.runner.data_ingest import ingest_dataset
    result = ingest_dataset(
        project_dir=project_dir,
        data_path=data_path,
        join_on=join_on,
        prefix=prefix,
        features_dir=features_dir,
        auto_clean=auto_clean,
    )
    return result.format_summary()
```

**Step 2: Run config_writer tests**

Run: `uv run pytest packages/easyml-runner/tests/test_config_writer.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add packages/easyml-runner/src/easyml/runner/config_writer.py packages/easyml-runner/tests/test_config_writer.py
git commit -m "feat(config-writer): wire DataPipeline into add_dataset MCP tool"
```

---

### Task 15: Update `mm_pipeline.yaml` fixture and run full test suite

**Files:**
- Modify: `packages/easyml-runner/tests/fixtures/mm_pipeline.yaml`
- Run: full test suite

**Step 1: Add `features_file` to the fixture**

In `packages/easyml-runner/tests/fixtures/mm_pipeline.yaml`, add under `data:`:

```yaml
  features_file: matchup_features.parquet
```

This ensures the March Madness project continues to work with its existing filename.

**Step 2: Run the full runner test suite**

Run: `uv run pytest packages/easyml-runner/tests/ -v`
Expected: ALL PASS

**Step 3: Run integration tests if they exist**

Run: `uv run pytest packages/easyml-runner/tests/test_mm_integration.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add packages/easyml-runner/tests/fixtures/mm_pipeline.yaml
git commit -m "fix(fixtures): add features_file to mm_pipeline.yaml for backward compat"
```

---

### Task 16: Final verification and cleanup commit

**Step 1: Run the entire project test suite**

Run: `uv run pytest packages/easyml-runner/tests/ -v --tb=short`
Expected: ALL PASS

**Step 2: Check for any remaining hardcoded references**

Run: `grep -r "matchup_features" packages/easyml-runner/src/`
Expected: Zero matches (all moved to config)

**Step 3: Commit any stragglers**

```bash
git add -A
git status
# Only commit if there are meaningful changes
git commit -m "chore: final cleanup for data layer improvements"
```
