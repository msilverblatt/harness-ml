# EasyML Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract a reusable AI-driven ML framework from the March Madness pipeline into 7 composable packages that let an AI agent operate ML pipelines safely through config alone.

**Architecture:** Monorepo with 7 Python packages sharing a contracts library (`easyml-schemas`). Each package is independently installable. Uses uv workspaces for development. Packages depend on each other only through `easyml-schemas`. The MCP guardrails layer (`easyml-guardrails`) sits on top and auto-generates inspection tools from the other packages.

**Tech Stack:** Python 3.11+, uv (package manager + workspaces), Pydantic v2, OmegaConf, FastMCP, pytest, numpy, scikit-learn. Optional: xgboost, catboost, lightgbm, torch.

**Reference codebase:** `~/mm` — the March Madness pipeline. Many implementations can be adapted from existing code there. Key source files are called out per task.

---

## Phase 0: Repository Setup

### Task 0.1: Create monorepo with uv workspaces

**Files:**
- Create: `~/easyml/pyproject.toml` (workspace root)
- Create: `~/easyml/packages/easyml-schemas/pyproject.toml`
- Create: `~/easyml/packages/easyml-config/pyproject.toml`
- Create: `~/easyml/packages/easyml-features/pyproject.toml`
- Create: `~/easyml/packages/easyml-models/pyproject.toml`
- Create: `~/easyml/packages/easyml-data/pyproject.toml`
- Create: `~/easyml/packages/easyml-experiments/pyproject.toml`
- Create: `~/easyml/packages/easyml-guardrails/pyproject.toml`

**Step 1: Create the directory structure**

```
easyml/
├── pyproject.toml                 # workspace root
├── packages/
│   ├── easyml-schemas/
│   │   ├── pyproject.toml
│   │   ├── src/easyml/schemas/
│   │   │   └── __init__.py
│   │   └── tests/
│   ├── easyml-config/
│   │   ├── pyproject.toml
│   │   ├── src/easyml/config/
│   │   │   └── __init__.py
│   │   └── tests/
│   ├── easyml-features/
│   │   ├── pyproject.toml
│   │   ├── src/easyml/features/
│   │   │   └── __init__.py
│   │   └── tests/
│   ├── easyml-models/
│   │   ├── pyproject.toml
│   │   ├── src/easyml/models/
│   │   │   └── __init__.py
│   │   └── tests/
│   ├── easyml-data/
│   │   ├── pyproject.toml
│   │   ├── src/easyml/data/
│   │   │   └── __init__.py
│   │   └── tests/
│   ├── easyml-experiments/
│   │   ├── pyproject.toml
│   │   ├── src/easyml/experiments/
│   │   │   └── __init__.py
│   │   └── tests/
│   └── easyml-guardrails/
│       ├── pyproject.toml
│       ├── src/easyml/guardrails/
│       │   └── __init__.py
│       └── tests/
└── tests/                         # integration tests
    └── __init__.py
```

**Step 2: Write workspace root pyproject.toml**

```toml
[project]
name = "easyml"
version = "0.1.0"
description = "AI-driven ML framework with guardrails"
requires-python = ">=3.11"

[tool.uv.workspace]
members = ["packages/*"]

[tool.pytest.ini_options]
testpaths = ["packages/*/tests", "tests"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]
```

**Step 3: Write each package's pyproject.toml**

Each package follows this pattern (example for easyml-schemas):

```toml
[project]
name = "easyml-schemas"
version = "0.1.0"
description = "Shared contracts and metrics for EasyML"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.24",
    "scikit-learn>=1.3",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/easyml"]
```

Dependencies per package:
- `easyml-schemas`: pydantic, numpy, scikit-learn
- `easyml-config`: easyml-schemas, omegaconf, pyyaml
- `easyml-features`: easyml-schemas
- `easyml-models`: easyml-schemas, numpy, scipy (optional extras for xgboost, catboost, lightgbm, torch)
- `easyml-data`: easyml-schemas
- `easyml-experiments`: easyml-schemas, easyml-config
- `easyml-guardrails`: easyml-schemas, easyml-experiments, fastmcp

**Step 4: Initialize git and install**

```bash
cd ~/easyml
git init
uv sync
uv run pytest  # should collect 0 tests, pass
```

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: initialize easyml monorepo with 7 package stubs"
```

---

## Phase 1: `easyml-schemas` — Shared Contracts + Metrics

**Reference:** `~/mm/src/mm/utils/config.py` (schema shapes), `~/mm/tools/mm_tools/guardrails.py` (GuardrailViolation), `~/mm/src/mm/models/calibration.py` (calibration table logic)

### Task 1.1: Core data schemas

**Files:**
- Create: `packages/easyml-schemas/src/easyml/schemas/core.py`
- Test: `packages/easyml-schemas/tests/test_core.py`

**Step 1: Write failing tests**

```python
# packages/easyml-schemas/tests/test_core.py
import pytest
from easyml.schemas.core import (
    FeatureMeta, TemporalFilter, ModelConfig, EnsembleConfig,
    StageConfig, ArtifactDecl, PipelineConfig, RunManifest,
    ExperimentResult, GuardrailViolation, Fold,
)

def test_feature_meta_basic():
    fm = FeatureMeta(
        name="scoring_margin",
        category="offense",
        level="team",
        output_columns=["scoring_margin", "scoring_margin_std"],
    )
    assert fm.name == "scoring_margin"
    assert fm.nan_strategy == "median"  # default

def test_feature_meta_temporal_filter():
    fm = FeatureMeta(
        name="test",
        category="test",
        level="team",
        output_columns=["x"],
        temporal_filter=TemporalFilter(exclude_event_types=["tournament"]),
    )
    assert fm.temporal_filter.exclude_event_types == ["tournament"]

def test_feature_meta_tainted_columns():
    fm = FeatureMeta(
        name="test",
        category="test",
        level="team",
        output_columns=["x"],
        tainted_columns=["kp_adj_o", "kp_sos"],
    )
    assert "kp_adj_o" in fm.tainted_columns

def test_model_config():
    mc = ModelConfig(
        name="xgb_core",
        type="xgboost",
        mode="classifier",
        features=["diff_seed_num", "diff_adj_em"],
        params={"max_depth": 3, "learning_rate": 0.05},
    )
    assert mc.type == "xgboost"
    assert mc.mode == "classifier"

def test_ensemble_config():
    ec = EnsembleConfig(
        method="stacked",
        meta_learner_params={"C": 2.5},
    )
    assert ec.method == "stacked"

def test_artifact_decl():
    ad = ArtifactDecl(name="base_features", type="features", path="data/features/")
    assert ad.type == "features"

def test_stage_config():
    sc = StageConfig(
        script="pipelines/featurize.py",
        consumes=["raw_data"],
        produces=[ArtifactDecl(name="features", type="features", path="data/features/")],
    )
    assert sc.consumes == ["raw_data"]

def test_run_manifest():
    rm = RunManifest(
        run_id="20260228_143021",
        created_at="2026-02-28T14:30:21Z",
        labels=["current"],
        stage="train",
    )
    assert "current" in rm.labels

def test_experiment_result():
    er = ExperimentResult(
        experiment_id="exp-055-test",
        baseline_metrics={"brier": 0.1752},
        result_metrics={"brier": 0.1748},
        delta={"brier": -0.0004},
        verdict="keep",
        models_trained=["xgb_core"],
    )
    assert er.delta["brier"] == -0.0004

def test_guardrail_violation():
    gv = GuardrailViolation(
        blocked=True,
        rule="sanity_check",
        message="Sanity check failed",
        source="scripts/sanity_check.py",
        override_hint="Re-call with human_override=true",
    )
    assert gv.blocked is True
    assert gv.to_dict()["rule"] == "sanity_check"

def test_fold():
    import numpy as np
    f = Fold(
        fold_id=2018,
        train_idx=np.array([0, 1, 2]),
        test_idx=np.array([3, 4]),
        calibration_idx=None,
    )
    assert f.fold_id == 2018
    assert len(f.train_idx) == 3
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/easyml && uv run pytest packages/easyml-schemas/tests/test_core.py -v
```

Expected: FAIL (import errors)

**Step 3: Implement schemas**

All schemas are Pydantic `BaseModel` classes. `TemporalFilter` is a nested model. `GuardrailViolation` has a `to_dict()` method. `Fold` uses numpy arrays (with `model_config = ConfigDict(arbitrary_types_allowed=True)`).

Reference `~/mm/tools/mm_tools/guardrails.py` lines 10-20 for `GuardrailViolation` shape.

**Step 4: Run tests, verify pass**

```bash
cd ~/easyml && uv run pytest packages/easyml-schemas/tests/test_core.py -v
```

**Step 5: Commit**

```bash
git add packages/easyml-schemas/ && git commit -m "feat(schemas): add core data schemas"
```

### Task 1.2: Source metadata schema

**Files:**
- Modify: `packages/easyml-schemas/src/easyml/schemas/core.py`
- Test: `packages/easyml-schemas/tests/test_core.py`

Add `SourceMeta` schema with `temporal_safety` and `leakage_notes` fields.

```python
def test_source_meta():
    sm = SourceMeta(
        name="kenpom_archive",
        category="external",
        outputs=["data/external/kenpom/"],
        temporal_safety="pre_tournament",
        leakage_notes="Archive endpoint returns ratings as of Selection Sunday",
    )
    assert sm.temporal_safety == "pre_tournament"
```

**Commit:** `feat(schemas): add SourceMeta with leakage metadata`

### Task 1.3: Built-in metrics — probability metrics

**Files:**
- Create: `packages/easyml-schemas/src/easyml/schemas/metrics.py`
- Test: `packages/easyml-schemas/tests/test_metrics.py`

**Step 1: Write failing tests**

```python
# packages/easyml-schemas/tests/test_metrics.py
import numpy as np
from easyml.schemas.metrics import brier_score, log_loss, ece, calibration_table, accuracy

def test_brier_score_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == 0.0

def test_brier_score_worst():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_prob) == 1.0

def test_brier_score_mid():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    assert abs(brier_score(y_true, y_prob) - 0.25) < 1e-10

def test_log_loss_basic():
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])
    result = log_loss(y_true, y_prob)
    assert 0 < result < 1

def test_accuracy_basic():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.3, 0.6, 0.4])
    assert accuracy(y_true, y_prob, threshold=0.5) == 1.0

def test_accuracy_custom_threshold():
    y_true = np.array([1, 0])
    y_prob = np.array([0.6, 0.4])
    assert accuracy(y_true, y_prob, threshold=0.7) == 0.5

def test_ece_perfect_calibration():
    # If predicted probability matches actual frequency, ECE should be near 0
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 30% positive
    y_prob = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    result = ece(y_true, y_prob, n_bins=5)
    assert result < 0.05

def test_calibration_table_structure():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.6, 0.4, 0.85, 0.15])
    table = calibration_table(y_true, y_prob, n_bins=5)
    assert isinstance(table, list)
    assert all("bin_start" in row and "mean_predicted" in row and "actual_accuracy" in row for row in table)
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement**

- `brier_score`: wrap `sklearn.metrics.brier_score_loss`
- `log_loss`: wrap `sklearn.metrics.log_loss`
- `accuracy`: threshold + `sklearn.metrics.accuracy_score`
- `ece`: custom implementation (~30 LOC) — bin predictions, compare mean predicted vs actual frequency per bin, weighted average
- `calibration_table`: custom (~40 LOC) — returns list of dicts with bin boundaries, mean predicted, actual accuracy, count

**Step 4: Run tests, verify pass**

**Step 5: Commit**

```bash
git add packages/easyml-schemas/ && git commit -m "feat(schemas): add probability and classification metrics"
```

### Task 1.4: Built-in metrics — regression + ensemble diagnostics

**Files:**
- Modify: `packages/easyml-schemas/src/easyml/schemas/metrics.py`
- Test: `packages/easyml-schemas/tests/test_metrics.py`

Add: `rmse`, `mae`, `r_squared`, `auc_roc`, `f1_score`, `model_correlations`, `model_audit`.

```python
def test_model_correlations():
    preds = {
        "model_a": np.array([0.5, 0.6, 0.7, 0.8]),
        "model_b": np.array([0.5, 0.6, 0.7, 0.8]),  # identical = correlation 1.0
        "model_c": np.array([0.8, 0.7, 0.6, 0.5]),  # inverse = correlation -1.0
    }
    corr = model_correlations(preds)
    assert abs(corr["model_a|model_b"] - 1.0) < 1e-10
    assert abs(corr["model_a|model_c"] - (-1.0)) < 1e-10

def test_model_audit():
    preds = {
        "model_a": np.array([0.9, 0.1, 0.8, 0.2]),
        "model_b": np.array([0.5, 0.5, 0.5, 0.5]),
    }
    y_true = np.array([1, 0, 1, 0])
    audit = model_audit(preds, y_true, metrics=["brier", "accuracy"])
    assert audit["model_a"]["accuracy"] == 1.0
    assert audit["model_b"]["accuracy"] == 0.5  # coin flip at 0.5 threshold
```

**Commit:** `feat(schemas): add regression metrics and ensemble diagnostics`

### Task 1.5: Package exports and __init__.py

**Files:**
- Modify: `packages/easyml-schemas/src/easyml/schemas/__init__.py`

Re-export all schemas and metrics from the top-level package. Verify with:

```python
from easyml.schemas import FeatureMeta, ModelConfig, brier_score, ece
```

**Commit:** `feat(schemas): finalize package exports`

---

## Phase 2: `easyml-config` — Split Config Resolution

**Reference:** `~/mm/src/mm/utils/config.py` (resolve_config, _deep_merge)

### Task 2.1: Deep merge with OmegaConf

**Files:**
- Create: `packages/easyml-config/src/easyml/config/merge.py`
- Test: `packages/easyml-config/tests/test_merge.py`

**Step 1: Write failing tests**

```python
# packages/easyml-config/tests/test_merge.py
from easyml.config.merge import deep_merge

def test_deep_merge_simple():
    base = {"a": 1, "b": 2}
    overlay = {"b": 3, "c": 4}
    result = deep_merge(base, overlay)
    assert result == {"a": 1, "b": 3, "c": 4}

def test_deep_merge_nested():
    base = {"models": {"xgb": {"params": {"depth": 3, "lr": 0.05}}}}
    overlay = {"models": {"xgb": {"params": {"depth": 5}}}}
    result = deep_merge(base, overlay)
    assert result["models"]["xgb"]["params"]["depth"] == 5
    assert result["models"]["xgb"]["params"]["lr"] == 0.05  # preserved

def test_deep_merge_list_replaces():
    base = {"features": ["a", "b", "c"]}
    overlay = {"features": ["x", "y"]}
    result = deep_merge(base, overlay)
    assert result["features"] == ["x", "y"]  # replaced, not appended
```

**Step 3: Implement using `OmegaConf.merge()`**

Reference `~/mm/src/mm/utils/config.py` `_deep_merge()` for semantics, but use OmegaConf under the hood.

**Commit:** `feat(config): add deep merge via OmegaConf`

### Task 2.2: File loading with variant resolution

**Files:**
- Create: `packages/easyml-config/src/easyml/config/loader.py`
- Test: `packages/easyml-config/tests/test_loader.py`

Tests should use `tmp_path` fixture to create temporary YAML files.

```python
def test_variant_resolution(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("data_dir: data/m/")
    (tmp_path / "pipeline_w.yaml").write_text("data_dir: data/w/")

    result = load_config_file(tmp_path, "pipeline.yaml", variant="w")
    assert result["data_dir"] == "data/w/"

def test_variant_fallback(tmp_path):
    (tmp_path / "pipeline.yaml").write_text("data_dir: data/m/")
    # No pipeline_w.yaml — should fall back to pipeline.yaml
    result = load_config_file(tmp_path, "pipeline.yaml", variant="w")
    assert result["data_dir"] == "data/m/"
```

**Commit:** `feat(config): add file loading with variant suffix resolution`

### Task 2.3: resolve_config — the main entry point

**Files:**
- Create: `packages/easyml-config/src/easyml/config/resolver.py`
- Test: `packages/easyml-config/tests/test_resolver.py`

```python
def test_resolve_config_full(tmp_path):
    # Create split config files
    (tmp_path / "pipeline.yaml").write_text("backtest_folds: [2020, 2021, 2022]")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "production.yaml").write_text("""
models:
  xgb_core:
    type: xgboost
    features: [feat_a]
    params:
      max_depth: 3
""")
    (tmp_path / "ensemble.yaml").write_text("method: stacked\nmeta_learner_params:\n  C: 2.5")

    config = resolve_config(
        config_dir=tmp_path,
        file_map={
            "pipeline": "pipeline.yaml",
            "models": ["models/production.yaml"],
            "ensemble": "ensemble.yaml",
        },
    )
    assert config["models"]["xgb_core"]["type"] == "xgboost"
    assert config["ensemble"]["method"] == "stacked"
    assert config["backtest_folds"] == [2020, 2021, 2022]

def test_resolve_config_with_overlay(tmp_path):
    # ... setup base config ...
    config = resolve_config(
        config_dir=tmp_path,
        file_map={"models": ["models/production.yaml"]},
        overlay={"models": {"xgb_core": {"params": {"max_depth": 5}}}},
    )
    assert config["models"]["xgb_core"]["params"]["max_depth"] == 5
```

Reference `~/mm/src/mm/utils/config.py` `resolve_config()` for the merge logic.

**Commit:** `feat(config): add resolve_config entry point with overlay support`

### Task 2.4: Package exports

**Commit:** `feat(config): finalize package exports`

---

## Phase 3: `easyml-features` — Registry, Discovery, Resolution, Building

**Reference:** `~/mm/src/mm/features/registry.py`, `~/mm/src/mm/features/builder.py`, `~/mm/src/mm/features/manifest.py`, `~/mm/src/mm/models/feature_selection.py`

### Task 3.1: Feature registry — decorator registration

**Files:**
- Create: `packages/easyml-features/src/easyml/features/registry.py`
- Test: `packages/easyml-features/tests/test_registry.py`

```python
def test_register_feature():
    registry = FeatureRegistry()

    @registry.register(
        name="test_feature",
        category="offense",
        level="team",
        output_columns=["col_a", "col_b"],
    )
    def compute_test(df, config):
        return df

    assert "test_feature" in registry
    meta = registry.get_metadata("test_feature")
    assert meta.category == "offense"
    assert meta.output_columns == ["col_a", "col_b"]

def test_register_with_temporal_filter():
    registry = FeatureRegistry()

    @registry.register(
        name="safe_feature",
        category="stats",
        level="team",
        output_columns=["x"],
        temporal_filter=TemporalFilter(exclude_event_types=["tournament"]),
    )
    def compute(df, config):
        return df

    meta = registry.get_metadata("safe_feature")
    assert "tournament" in meta.temporal_filter.exclude_event_types

def test_source_hash_changes_on_code_change():
    registry = FeatureRegistry()

    @registry.register(name="v1", category="t", level="t", output_columns=["x"])
    def compute_v1(df, config):
        return df

    hash1 = registry.source_hash("v1")
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 hex

def test_list_features():
    registry = FeatureRegistry()
    # register 3 features across 2 categories
    for name, cat in [("a", "off"), ("b", "off"), ("c", "def")]:
        @registry.register(name=name, category=cat, level="team", output_columns=["x"])
        def compute(df, config):
            return df

    assert len(registry.list_features()) == 3
    assert len(registry.list_features(category="off")) == 2
```

Reference `~/mm/src/mm/features/registry.py` for the `_FeatureDefinition` dataclass and `FeatureRegistry` class. Source hashing uses `inspect.getsource()` → SHA256.

**Commit:** `feat(features): add feature registry with decorator registration and source hashing`

### Task 3.2: Auto-discovery via pkgutil

**Files:**
- Modify: `packages/easyml-features/src/easyml/features/registry.py`
- Test: `packages/easyml-features/tests/test_registry.py`

```python
def test_discover_modules(tmp_path):
    # Create a temporary Python package with feature modules
    # registry.discover() should import all modules, triggering @register decorators
    ...
```

Reference `~/mm/src/mm/features/builder.py` `discover_feature_modules()` for the pkgutil pattern.

**Commit:** `feat(features): add auto-discovery via pkgutil`

### Task 3.3: Feature resolver

**Files:**
- Create: `packages/easyml-features/src/easyml/features/resolver.py`
- Test: `packages/easyml-features/tests/test_resolver.py`

```python
def test_resolve_explicit_columns():
    resolver = FeatureResolver(registry=registry)
    columns = resolver.resolve(
        ["scoring_margin", "win_pct"],
        available_columns=["scoring_margin", "win_pct", "ppg", "seed_num"],
    )
    assert columns == ["scoring_margin", "win_pct"]

def test_resolve_missing_column_raises():
    resolver = FeatureResolver(registry=registry)
    with pytest.raises(ValueError, match="not found"):
        resolver.resolve(["nonexistent"], available_columns=["a", "b"])

def test_resolve_categories():
    # registry has features with categories, resolver maps category -> columns
    ...
```

Reference `~/mm/src/mm/models/feature_selection.py` `get_feature_columns()`.

**Commit:** `feat(features): add feature resolver for model config -> DataFrame columns`

### Task 3.4: Feature builder with incremental caching

**Files:**
- Create: `packages/easyml-features/src/easyml/features/builder.py`
- Create: `packages/easyml-features/src/easyml/features/manifest.py`
- Test: `packages/easyml-features/tests/test_builder.py`

```python
def test_build_computes_and_caches(tmp_path):
    registry = FeatureRegistry()
    call_count = {"n": 0}

    @registry.register(name="test", category="t", level="team", output_columns=["x"])
    def compute(df, config):
        call_count["n"] += 1
        df["x"] = 1.0
        return df[["entity_id", "period_id", "x"]]

    builder = FeatureBuilder(registry=registry, cache_dir=tmp_path / "cache", manifest_path=tmp_path / "manifest.json")

    # First build — should compute
    result1 = builder.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 1

    # Second build — should use cache (source hash unchanged)
    result2 = builder.build_all(raw_data=sample_df, config={})
    assert call_count["n"] == 1  # NOT called again
```

Reference `~/mm/src/mm/features/builder.py` and `~/mm/src/mm/features/manifest.py`.

**Commit:** `feat(features): add feature builder with manifest-based incremental caching`

### Task 3.5: Pairwise feature builder

**Files:**
- Create: `packages/easyml-features/src/easyml/features/pairwise.py`
- Test: `packages/easyml-features/tests/test_pairwise.py`

```python
def test_pairwise_diff():
    entity_df = pd.DataFrame({
        "entity_id": [1, 2, 3],
        "period_id": [2025, 2025, 2025],
        "scoring_margin": [10.0, 5.0, -3.0],
    })
    matchups = pd.DataFrame({
        "entity_a_id": [1, 2],
        "entity_b_id": [2, 3],
        "period_id": [2025, 2025],
    })
    builder = PairwiseFeatureBuilder(methods=["diff"])
    result = builder.build(entity_df, matchups)
    assert "diff_scoring_margin" in result.columns
    assert result.iloc[0]["diff_scoring_margin"] == 5.0  # 10.0 - 5.0
```

Reference `~/mm/src/mm/features/matchup.py`.

**Commit:** `feat(features): add pairwise feature builder`

### Task 3.6: Package exports

**Commit:** `feat(features): finalize package exports`

---

## Phase 4: `easyml-models` — The Big Package

**Reference:** `~/mm/src/mm/models/` (all files), `~/mm/src/mm/models/meta_learner.py`, `~/mm/src/mm/models/calibration.py`, `~/mm/src/mm/utils/run_manager.py`

This is the largest package (~2500 LOC). Break into sub-tasks by component.

### Task 4.1: BaseModel protocol + ModelRegistry

**Files:**
- Create: `packages/easyml-models/src/easyml/models/base.py`
- Create: `packages/easyml-models/src/easyml/models/registry.py`
- Test: `packages/easyml-models/tests/test_registry.py`

```python
def test_register_and_create():
    registry = ModelRegistry()
    registry.register("mock", MockModel)
    model = registry.create("mock", params={"depth": 3})
    assert isinstance(model, MockModel)

def test_create_from_config():
    registry = ModelRegistry()
    registry.register("mock", MockModel)
    config = ModelConfig(name="test", type="mock", mode="classifier", features=["x"], params={"depth": 3})
    model = registry.create_from_config(config)
    assert isinstance(model, MockModel)

def test_with_defaults_loads_builtins():
    registry = ModelRegistry.with_defaults()
    assert "logistic_regression" in registry
    # xgboost only if installed
```

`BaseModel` is an ABC with `fit`, `predict_proba`, `predict_margin`, `save`, `load`, `is_regression` property.

Reference `~/mm/src/mm/models/model_registry.py`.

**Commit:** `feat(models): add BaseModel protocol and ModelRegistry`

### Task 4.2: Built-in model wrappers — LogisticRegression + ElasticNet

Start with the two that have no heavy dependencies (pure sklearn).

**Files:**
- Create: `packages/easyml-models/src/easyml/models/wrappers/logistic.py`
- Create: `packages/easyml-models/src/easyml/models/wrappers/elastic_net.py`
- Test: `packages/easyml-models/tests/test_wrappers_sklearn.py`

```python
def test_logistic_regression_fit_predict():
    model = LogisticRegressionModel(params={"C": 1.0})
    X = np.random.randn(100, 3)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (100,)
    assert all(0 <= p <= 1 for p in probs)
    assert model.is_regression is False

def test_logistic_regression_save_load(tmp_path):
    model = LogisticRegressionModel(params={"C": 1.0})
    X = np.random.randn(50, 2)
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    model.save(tmp_path / "model")
    loaded = LogisticRegressionModel.load(tmp_path / "model")
    np.testing.assert_array_almost_equal(model.predict_proba(X), loaded.predict_proba(X))
```

Reference `~/mm/src/mm/models/logreg_model.py`.

**Commit:** `feat(models): add logistic regression and elastic net wrappers`

### Task 4.3: Built-in model wrappers — XGBoost (optional extra)

**Files:**
- Create: `packages/easyml-models/src/easyml/models/wrappers/xgboost.py`
- Test: `packages/easyml-models/tests/test_wrappers_xgb.py`

Tests should be marked `@pytest.mark.skipif(not HAS_XGBOOST, ...)`.

Both classifier and regressor modes. Regressor includes CDF margin→probability conversion.

Reference `~/mm/src/mm/models/xgb_model.py`.

**Commit:** `feat(models): add XGBoost wrapper (classifier + regressor)`

### Task 4.4: Built-in model wrappers — CatBoost, LightGBM, RandomForest

Same pattern as 4.3. Each is ~50-80 LOC.

Reference `~/mm/src/mm/models/catboost_model.py`, `lgbm_model.py`, `rf_model.py`.

**Commit:** `feat(models): add CatBoost, LightGBM, RandomForest wrappers`

### Task 4.5: Built-in model wrappers — MLP + TabNet (torch extras)

Multi-seed averaging support. `@pytest.mark.skipif(not HAS_TORCH, ...)`.

Reference `~/mm/src/mm/models/mlp_model.py`, `tabnet_model.py`.

**Commit:** `feat(models): add MLP and TabNet wrappers with multi-seed averaging`

### Task 4.6: Temporal cross-validation strategies

**Files:**
- Create: `packages/easyml-models/src/easyml/models/cv.py`
- Test: `packages/easyml-models/tests/test_cv.py`

```python
def test_loso_basic():
    fold_ids = np.array([2015, 2015, 2016, 2016, 2017, 2017])
    loso = LeaveOneSeasonOut(min_train_folds=1)
    folds = loso.split(None, fold_ids=fold_ids)
    assert len(folds) == 3  # one per season

    # Check temporal ordering
    for fold in folds:
        train_seasons = set(fold_ids[fold.train_idx])
        test_season = set(fold_ids[fold.test_idx])
        assert max(train_seasons) < min(test_season)

def test_loso_min_train_folds():
    fold_ids = np.array([2015, 2016, 2017])
    loso = LeaveOneSeasonOut(min_train_folds=2)
    folds = loso.split(None, fold_ids=fold_ids)
    assert len(folds) == 1  # only 2017 has 2+ training folds

def test_expanding_window():
    fold_ids = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    ew = ExpandingWindow(min_train_size=1)
    folds = ew.split(None, fold_ids=fold_ids)
    assert len(folds) == 3  # folds 2, 3, 4 (fold 1 has no training data)

def test_sliding_window():
    fold_ids = np.array([1, 2, 3, 4, 5])
    sw = SlidingWindow(window_size=2)
    folds = sw.split(None, fold_ids=fold_ids)
    # For fold 5: train on [3, 4]
    fold_5 = [f for f in folds if f.fold_id == 5][0]
    train_ids = set(fold_ids[fold_5.train_idx])
    assert train_ids == {3, 4}

def test_nested_cv():
    fold_ids = np.array([2015]*20 + [2016]*20 + [2017]*20)
    nested = NestedCV(
        outer=LeaveOneSeasonOut(min_train_folds=1),
        inner_calibration_fraction=0.25,
    )
    folds = nested.split(None, fold_ids=fold_ids)
    for fold in folds:
        assert fold.calibration_idx is not None
        # calibration indices are subset of training indices
        assert set(fold.calibration_idx).issubset(set(fold.train_idx))
        # calibration and remaining training don't overlap after carve-out
        remaining_train = set(fold.train_idx) - set(fold.calibration_idx)
        assert len(remaining_train) > 0

def test_temporal_ordering_enforced():
    """All strategies must enforce train fold IDs < test fold ID."""
    fold_ids = np.array([2015, 2016, 2017, 2018])
    for strategy in [LeaveOneSeasonOut(), ExpandingWindow(), SlidingWindow(window_size=2)]:
        folds = strategy.split(None, fold_ids=fold_ids)
        for fold in folds:
            max_train = max(fold_ids[fold.train_idx])
            min_test = min(fold_ids[fold.test_idx])
            assert max_train < min_test, f"{strategy.__class__.__name__} violated temporal ordering"
```

Reference `~/mm/src/mm/models/meta_learner.py` for the LOSO loop pattern. The `PurgedKFold` embargo logic can reference the `tscv` library's algorithm (~20 LOC).

**Commit:** `feat(models): add 5 temporal CV strategies with temporal ordering enforcement`

### Task 4.7: Fingerprint caching

**Files:**
- Create: `packages/easyml-models/src/easyml/models/fingerprint.py`
- Test: `packages/easyml-models/tests/test_fingerprint.py`

```python
def test_fingerprint_match(tmp_path):
    fp = Fingerprint.compute({"type": "xgb", "depth": 3}, "hash123", 1000.0)
    fp.save(tmp_path / ".fingerprint")
    assert fp.matches(tmp_path / ".fingerprint")

def test_fingerprint_mismatch_on_config_change(tmp_path):
    fp1 = Fingerprint.compute({"type": "xgb", "depth": 3}, "hash123", 1000.0)
    fp1.save(tmp_path / ".fingerprint")
    fp2 = Fingerprint.compute({"type": "xgb", "depth": 5}, "hash123", 1000.0)
    assert not fp2.matches(tmp_path / ".fingerprint")
```

Reference `~/mm/src/mm/models/fingerprint.py` (~35 LOC).

**Commit:** `feat(models): add fingerprint-based cache invalidation`

### Task 4.8: Calibrators — Spline, Platt, Isotonic

**Files:**
- Create: `packages/easyml-models/src/easyml/models/calibration.py`
- Test: `packages/easyml-models/tests/test_calibration.py`

```python
def test_spline_calibrator_improves_ece():
    # Generate miscalibrated predictions
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true + np.random.randn(200) * 0.3, 0.01, 0.99)

    cal = SplineCalibrator(n_bins=10, prob_max=0.985)
    cal.fit(y_true, y_raw)
    y_cal = cal.transform(y_raw)

    ece_before = ece(y_true, y_raw, n_bins=10)
    ece_after = ece(y_true, y_cal, n_bins=10)
    assert ece_after <= ece_before

def test_spline_calibrator_save_load(tmp_path):
    ...

def test_platt_calibrator():
    ...

def test_isotonic_calibrator():
    ...
```

Reference `~/mm/src/mm/models/calibration.py` for the `SplineCalibrator` PCHIP implementation (~85 LOC).

**Commit:** `feat(models): add Spline, Platt, Isotonic calibrators`

### Task 4.9: StackedEnsemble — meta-learner with per-fold pre-calibration

**Files:**
- Create: `packages/easyml-models/src/easyml/models/ensemble.py`
- Test: `packages/easyml-models/tests/test_ensemble.py`

```python
def test_stacked_ensemble_basic():
    ensemble = StackedEnsemble(
        method="stacked",
        meta_learner_type="logistic",
        meta_learner_params={"C": 1.0},
    )
    preds = {
        "model_a": np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3]),
        "model_b": np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.4]),
    }
    y = np.array([1, 0, 1, 0, 1, 0])
    folds = np.array([1, 1, 2, 2, 3, 3])

    ensemble.fit(preds, y, cv=LeaveOneSeasonOut(), fold_ids=folds)
    final = ensemble.predict(preds)
    assert final.shape == (6,)
    assert all(0 < p < 1 for p in final)

    coeffs = ensemble.coefficients()
    assert "model_a" in coeffs
    assert "model_b" in coeffs

def test_stacked_ensemble_pre_calibration():
    ensemble = StackedEnsemble(
        method="stacked",
        meta_learner_type="logistic",
        meta_learner_params={"C": 1.0},
        pre_calibrate={"model_a": SplineCalibrator(n_bins=5)},
    )
    # Pre-calibration should happen per-fold inside CV, not globally
    ...

def test_ensemble_temporal_ordering_assertion():
    """Ensemble.fit() should assert temporal ordering on folds."""
    ...
```

Reference `~/mm/src/mm/models/meta_learner.py` and `~/mm/src/mm/models/stacking.py`.

**Commit:** `feat(models): add StackedEnsemble with per-fold pre-calibration`

### Task 4.10: EnsemblePostprocessor chain

**Files:**
- Create: `packages/easyml-models/src/easyml/models/postprocessing.py`
- Test: `packages/easyml-models/tests/test_postprocessing.py`

```python
def test_postprocessor_chain():
    chain = EnsemblePostprocessor(steps=[
        ("clip", ProbabilityClipping(floor=0.05, ceiling=0.95)),
        ("temperature", TemperatureScaling(T=1.5)),
    ])
    probs = np.array([0.01, 0.5, 0.99])
    result = chain.apply(probs)
    assert all(0.05 <= p <= 0.95 for p in result)
```

Reference `~/mm/src/mm/ensemble/postprocessing.py`.

**Commit:** `feat(models): add extensible ensemble post-processing chain`

### Task 4.11: TrainOrchestrator

**Files:**
- Create: `packages/easyml-models/src/easyml/models/orchestrator.py`
- Test: `packages/easyml-models/tests/test_orchestrator.py`

```python
def test_orchestrator_trains_all_models(tmp_path):
    registry = ModelRegistry()
    registry.register("mock", MockModel)
    config = {
        "models": {
            "model_a": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}},
            "model_b": {"type": "mock", "mode": "classifier", "features": ["x"], "params": {}},
        }
    }
    orch = TrainOrchestrator(model_registry=registry, config=config)
    results = orch.train_all(data=sample_df, output_dir=tmp_path)
    assert "model_a" in results
    assert "model_b" in results
    assert (tmp_path / "model_a").exists()

def test_orchestrator_skips_cached(tmp_path):
    # Train once, then train again — second call should skip
    ...

def test_orchestrator_saves_feature_medians(tmp_path):
    ...

def test_orchestrator_failure_policy_skip(tmp_path):
    # One model fails, others still train
    ...
```

Reference `~/mm/src/mm/models/train.py` `train_all_models()`.

**Commit:** `feat(models): add TrainOrchestrator with fingerprint caching and failure policy`

### Task 4.12: BacktestRunner

**Files:**
- Create: `packages/easyml-models/src/easyml/models/backtest.py`
- Test: `packages/easyml-models/tests/test_backtest.py`

```python
def test_backtest_runner_basic():
    runner = BacktestRunner(
        ensemble=ensemble,
        cv=LeaveOneSeasonOut(),
        metrics=["brier", "accuracy"],
    )
    result = runner.run(per_fold_predictions=preds_by_fold, actuals=actuals_by_fold)
    assert "brier" in result.pooled_metrics
    assert len(result.per_fold_metrics) > 0
```

Reference `~/mm/src/mm/backtest/runner.py`.

**Commit:** `feat(models): add BacktestRunner with two-pass architecture`

### Task 4.13: RunManager — versioned output management

**Files:**
- Create: `packages/easyml-models/src/easyml/models/run_manager.py`
- Test: `packages/easyml-models/tests/test_run_manager.py`

```python
def test_run_manager_lifecycle(tmp_path):
    rm = RunManager(base_dir=tmp_path)
    run1 = rm.new_run()
    assert run1.exists()
    rm.promote(run1.name)
    assert rm.get_latest() == run1

    run2 = rm.new_run()
    rm.promote(run2.name)
    assert rm.get_latest() == run2
    assert len(rm.list_runs()) == 2
```

Reference `~/mm/src/mm/utils/run_manager.py` (~100 LOC).

**Commit:** `feat(models): add RunManager with manifest and symlink promotion`

### Task 4.14: TrackingCallback protocol

**Files:**
- Create: `packages/easyml-models/src/easyml/models/tracking.py`
- Test: `packages/easyml-models/tests/test_tracking.py`

```python
def test_tracking_callback_called():
    tracker = MockTracker()
    orch = TrainOrchestrator(..., callbacks=[tracker])
    orch.train_all(...)
    assert tracker.on_model_trained_called
```

**Commit:** `feat(models): add TrackingCallback protocol for MLflow/W&B integration`

### Task 4.15: Package exports

**Commit:** `feat(models): finalize package exports`

---

## Phase 5: `easyml-data` — Source Registry, Stage Guards, DVC Integration

**Reference:** `~/mm/src/mm/data/source_registry.py`, `~/mm/src/mm/utils/stage_guards.py`

### Task 5.1: Data source registry with leakage metadata

**Files:**
- Create: `packages/easyml-data/src/easyml/data/sources.py`
- Test: `packages/easyml-data/tests/test_sources.py`

```python
def test_register_source():
    sources = SourceRegistry()

    @sources.register(
        name="test_source",
        category="external",
        outputs=["data/external/test/"],
        temporal_safety="pre_tournament",
        leakage_notes="Safe — pre-tournament snapshot",
    )
    def scrape(output_dir, config):
        pass

    assert "test_source" in sources
    meta = sources.get_metadata("test_source")
    assert meta.temporal_safety == "pre_tournament"

def test_freshness_check(tmp_path):
    sources = SourceRegistry()
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b\n1,2")

    @sources.register(
        name="test",
        category="external",
        outputs=[str(tmp_path)],
        freshness_check=lambda path: (time.time() - path.stat().st_mtime) < 3600,
    )
    def scrape(output_dir, config):
        pass

    result = sources.check_freshness("test")
    assert result["fresh"] is True
```

Reference `~/mm/src/mm/data/source_registry.py`.

**Commit:** `feat(data): add source registry with leakage metadata and freshness checks`

### Task 5.2: Typed artifact declarations + DVC config generator

**Files:**
- Create: `packages/easyml-data/src/easyml/data/artifacts.py`
- Create: `packages/easyml-data/src/easyml/data/dvc_generator.py`
- Test: `packages/easyml-data/tests/test_artifacts.py`

```python
def test_generate_dvc_yaml():
    artifacts = {
        "ingest": StageConfig(
            script="pipelines/ingest.py",
            consumes=[],
            produces=[ArtifactDecl(name="raw_data", type="data", path="data/processed/")],
        ),
        "featurize": StageConfig(
            script="pipelines/featurize.py",
            consumes=["raw_data"],
            produces=[ArtifactDecl(name="features", type="features", path="data/features/")],
        ),
    }
    dvc_yaml = generate_dvc_yaml(artifacts)
    assert "ingest" in dvc_yaml["stages"]
    assert "data/processed/" in dvc_yaml["stages"]["ingest"]["outs"]
    assert "data/processed/" in dvc_yaml["stages"]["featurize"]["deps"]
```

**Commit:** `feat(data): add typed artifacts and DVC config generator`

### Task 5.3: Stage guards with staleness detection

**Files:**
- Create: `packages/easyml-data/src/easyml/data/guards.py`
- Test: `packages/easyml-data/tests/test_guards.py`

```python
def test_guard_passes(tmp_path):
    data_file = tmp_path / "features.parquet"
    pd.DataFrame({"x": range(1000)}).to_parquet(data_file)

    guard = StageGuard(
        name="features_exist",
        requires=[str(data_file)],
        min_rows=500,
    )
    guard.check()  # should not raise

def test_guard_fails_missing_file(tmp_path):
    guard = StageGuard(name="test", requires=[str(tmp_path / "nonexistent.parquet")])
    with pytest.raises(GuardrailViolation):
        guard.check()

def test_guard_stale_artifact(tmp_path):
    data_file = tmp_path / "features.parquet"
    data_file.write_bytes(b"old data")
    config_file = tmp_path / "config.yaml"
    config_file.write_text("new config")
    # Make data_file older than config_file
    import os
    os.utime(data_file, (1000, 1000))

    guard = StageGuard(
        name="test",
        requires=[str(data_file)],
        stale_if_older_than=[str(config_file)],
    )
    with pytest.raises(GuardrailViolation, match="stale"):
        guard.check()
```

Reference `~/mm/src/mm/utils/stage_guards.py`.

**Commit:** `feat(data): add stage guards with staleness detection`

### Task 5.4: Refresh orchestrator

**Files:**
- Create: `packages/easyml-data/src/easyml/data/refresh.py`
- Test: `packages/easyml-data/tests/test_refresh.py`

**Commit:** `feat(data): add failure-tolerant refresh orchestrator`

### Task 5.5: Package exports

**Commit:** `feat(data): finalize package exports`

---

## Phase 6: `easyml-experiments` — Experiment Protocol Enforcement

**Reference:** `~/mm/src/mm/experiment/runner.py`, `~/mm/src/mm/experiment/overlay.py`, `~/mm/pipelines/experiment.py`, `~/mm/tools/mm_tools/guardrails.py`

### Task 6.1: Experiment creation + naming validation

**Files:**
- Create: `packages/easyml-experiments/src/easyml/experiments/manager.py`
- Test: `packages/easyml-experiments/tests/test_manager.py`

```python
def test_create_experiment(tmp_path):
    mgr = ExperimentManager(
        experiments_dir=tmp_path,
        baseline_metrics={"brier": 0.175},
        naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    )
    exp = mgr.create("exp-001-test")
    assert (tmp_path / "exp-001-test").exists()
    assert (tmp_path / "exp-001-test" / "overlay.yaml").exists()

def test_create_experiment_bad_name(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path, naming_pattern=r"exp-\d{3}-[a-z0-9-]+$")
    with pytest.raises(GuardrailViolation, match="naming"):
        mgr.create("bad_name")
```

**Commit:** `feat(experiments): add experiment creation with naming validation`

### Task 6.2: Change detection + single-variable enforcement

**Files:**
- Modify: `packages/easyml-experiments/src/easyml/experiments/manager.py`
- Test: `packages/easyml-experiments/tests/test_manager.py`

```python
def test_detect_changes_single():
    production = {"models": {"xgb": {"params": {"depth": 3}}}}
    overlay = {"models": {"xgb": {"params": {"depth": 5}}}}
    changes = mgr.detect_changes(production, overlay)
    assert changes.changed_models == ["xgb"]
    assert len(changes.new_models) == 0

def test_detect_changes_multi_blocked():
    overlay = {
        "models": {"xgb": {"params": {"depth": 5}}, "cat": {"params": {"lr": 0.01}}},
    }
    changes = mgr.detect_changes(production, overlay)
    assert changes.total_changes > 1
```

Reference `~/mm/src/mm/experiment/overlay.py` `detect_changes()`.

**Commit:** `feat(experiments): add change detection and single-variable enforcement`

### Task 6.3: Do-not-retry registry

**Files:**
- Modify: `packages/easyml-experiments/src/easyml/experiments/manager.py`
- Test: `packages/easyml-experiments/tests/test_do_not_retry.py`

```python
def test_do_not_retry_blocks(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.add_do_not_retry(pattern="temperature scaling", reference="EXP-002", reason="hurts")

    with pytest.raises(GuardrailViolation, match="temperature scaling"):
        mgr.check_do_not_retry("Let's try temperature scaling T=1.5")

def test_do_not_retry_allows_unmatched(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path)
    mgr.add_do_not_retry(pattern="temperature", reference="EXP-002", reason="hurts")
    mgr.check_do_not_retry("Adding a new XGBoost model")  # should not raise
```

Reference `~/mm/tools/mm_tools/guardrails.py` `check_do_not_retry()`.

**Commit:** `feat(experiments): add do-not-retry registry`

### Task 6.4: Experiment logging + mandatory logging enforcement

**Files:**
- Modify: `packages/easyml-experiments/src/easyml/experiments/manager.py`
- Create: `packages/easyml-experiments/src/easyml/experiments/logger.py`
- Test: `packages/easyml-experiments/tests/test_logging.py`

```python
def test_log_experiment(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path, log_path=tmp_path / "LOG.md")
    mgr.create("exp-001-test")
    mgr.log(experiment_id="exp-001-test", hypothesis="test", changes="test", verdict="revert")
    assert "exp-001-test" in (tmp_path / "LOG.md").read_text()

def test_mandatory_logging_blocks_next(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path, log_path=tmp_path / "LOG.md")
    mgr.create("exp-001-test")
    # Don't log — next create should be blocked
    with pytest.raises(GuardrailViolation, match="not been logged"):
        mgr.create("exp-002-next")
```

**Commit:** `feat(experiments): add experiment logging with mandatory enforcement`

### Task 6.5: Atomic promote with rollback

**Files:**
- Modify: `packages/easyml-experiments/src/easyml/experiments/manager.py`
- Test: `packages/easyml-experiments/tests/test_promote.py`

```python
def test_promote_requires_logged_verdict(tmp_path):
    mgr = ExperimentManager(experiments_dir=tmp_path, log_path=tmp_path / "LOG.md")
    mgr.create("exp-001-test")
    with pytest.raises(GuardrailViolation, match="verdict"):
        mgr.promote("exp-001-test", model_name="xgb", production_config_path=tmp_path / "prod.yaml")

def test_promote_atomic_write(tmp_path):
    # Setup: create experiment, log it as "keep"
    # Promote should create backup, write atomically
    ...
```

Reference `~/mm/pipelines/experiment.py` `cmd_promote()`.

**Commit:** `feat(experiments): add atomic promote with backup and rollback`

### Task 6.6: Package exports

**Commit:** `feat(experiments): finalize package exports`

---

## Phase 7: `easyml-guardrails` — MCP Server Generation + AI Safety

**Reference:** `~/mm/tools/mcp_server.py`, `~/mm/tools/mm_tools/helpers.py`, `~/mm/tools/mm_tools/guardrails.py`, `~/mm/tools/mm_tools/inspection.py`

### Task 7.1: Guardrail base classes

**Files:**
- Create: `packages/easyml-guardrails/src/easyml/guardrails/base.py`
- Test: `packages/easyml-guardrails/tests/test_guardrails.py`

```python
def test_guardrail_check():
    g = Guardrail.naming_convention(pattern=r"exp-\d{3}-[a-z]+$")
    g.check(context={"experiment_id": "exp-001-test"})  # should not raise
    with pytest.raises(GuardrailViolation):
        g.check(context={"experiment_id": "bad"})

def test_guardrail_human_override():
    g = Guardrail.naming_convention(pattern=r"exp-\d{3}-[a-z]+$")
    g.check(context={"experiment_id": "bad"}, human_override=True)  # should not raise

def test_non_overridable_guardrail():
    g = Guardrail.feature_leakage(denylist=["leaky_col"])
    with pytest.raises(GuardrailViolation):
        g.check(context={"model_features": ["leaky_col"]}, human_override=True)  # still raises!
```

**Commit:** `feat(guardrails): add guardrail base classes with override support`

### Task 7.2: Full guardrail inventory

**Files:**
- Create: `packages/easyml-guardrails/src/easyml/guardrails/inventory.py`
- Test: `packages/easyml-guardrails/tests/test_inventory.py`

Implement all 11 guardrails:

| Guardrail | Test |
|-----------|------|
| `sanity_check` | Runs script, blocks on failure |
| `naming_convention` | Regex validation |
| `do_not_retry` | Delegates to ExperimentManager |
| `single_variable` | Delegates to ExperimentManager.detect_changes |
| `feature_leakage` | Checks model features against denylist (non-overridable) |
| `config_protection` | Runs `git diff` on production config |
| `critical_paths` | Blocks delete/overwrite on protected dirs (non-overridable) |
| `rate_limit` | Timestamp-based cooldown |
| `experiment_logged` | Delegates to ExperimentManager.has_unlogged |
| `feature_staleness` | Compares feature manifest hashes to source code |
| `temporal_ordering` | (Enforced in CV, not a standalone guardrail) |

**Commit:** `feat(guardrails): implement full guardrail inventory`

### Task 7.3: Pipeline command execution

**Files:**
- Create: `packages/easyml-guardrails/src/easyml/guardrails/execution.py`
- Test: `packages/easyml-guardrails/tests/test_execution.py`

```python
def test_run_pipeline_command():
    result = run_pipeline_command(["echo", "hello"], tool_name="test")
    assert result["status"] == "success"
    assert result["log_path"].exists()

def test_run_pipeline_command_timeout():
    result = run_pipeline_command(["sleep", "10"], tool_name="test", timeout=1)
    assert result["status"] == "timeout"
```

Reference `~/mm/tools/mm_tools/helpers.py` `run_pipeline_command()`.

**Commit:** `feat(guardrails): add pipeline command execution with logging`

### Task 7.4: PipelineServer base class + auto-generated inspection tools

**Files:**
- Create: `packages/easyml-guardrails/src/easyml/guardrails/server.py`
- Test: `packages/easyml-guardrails/tests/test_server.py`

```python
@pytest.mark.asyncio
async def test_server_registers_tools():
    server = TestMLServer(...)
    tools = await server.mcp.list_tools()
    tool_names = {t.name for t in tools}
    # Auto-generated inspection tools
    assert "show_config" in tool_names
    assert "list_features" in tool_names
    assert "list_models" in tool_names
    assert "list_experiments" in tool_names
    assert "list_runs" in tool_names

@pytest.mark.asyncio
async def test_guardrail_blocks_tool():
    # Configure a guardrail that always blocks
    # Call a tool — should get GuardrailViolation response
    ...
```

This is the core MCP server generator. It introspects the other packages' registries and auto-generates read-only inspection tools.

Reference `~/mm/tools/mcp_server.py` and `~/mm/tools/mm_tools/inspection.py`.

**Commit:** `feat(guardrails): add PipelineServer with auto-generated inspection tools`

### Task 7.5: Structured audit log

**Files:**
- Create: `packages/easyml-guardrails/src/easyml/guardrails/audit.py`
- Test: `packages/easyml-guardrails/tests/test_audit.py`

```python
def test_audit_log_records_invocation(tmp_path):
    logger = AuditLogger(log_path=tmp_path / "audit.jsonl")
    logger.log_invocation(
        tool="train_models",
        args={"run_id": None},
        guardrails_passed=True,
        result_status="success",
        duration_s=342.1,
    )
    lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    entry = json.loads(lines[0])
    assert entry["tool"] == "train_models"
```

**Commit:** `feat(guardrails): add structured JSON audit log`

### Task 7.6: Package exports

**Commit:** `feat(guardrails): finalize package exports`

---

## Phase 8: Integration Tests

### Task 8.1: End-to-end test with mock data

**Files:**
- Create: `tests/test_integration.py`

Write an integration test that:
1. Creates a `resolve_config` with a mock config directory
2. Registers 3 features via `FeatureRegistry`
3. Builds features via `FeatureBuilder`
4. Registers 2 model types (mock classifiers)
5. Trains via `TrainOrchestrator`
6. Runs backtest via `BacktestRunner`
7. Checks metrics via `brier_score`, `accuracy`
8. Creates and logs an experiment via `ExperimentManager`

This validates the full pipeline works end-to-end across all 7 packages.

**Commit:** `test: add end-to-end integration test across all packages`

### Task 8.2: Test DVC config generation from typed artifacts

**Commit:** `test: validate DVC config generation from artifact declarations`

---

## Phase 9: Documentation + Cleanup

### Task 9.1: Package READMEs

One README per package with installation, quick start, and API reference.

**Commit:** `docs: add per-package READMEs`

### Task 9.2: Root README with architecture diagram

**Commit:** `docs: add root README with architecture and quick start`

### Task 9.3: CLAUDE.md for the easyml repo

Write a CLAUDE.md that tells the AI agent how to use the framework for a new project.

**Commit:** `docs: add CLAUDE.md for AI agent operation`

---

## Task Summary

| Phase | Package | Tasks | Est. LOC |
|-------|---------|-------|----------|
| 0 | Repo setup | 1 task | ~100 (config) |
| 1 | easyml-schemas | 5 tasks | ~500 |
| 2 | easyml-config | 4 tasks | ~300 |
| 3 | easyml-features | 6 tasks | ~700 |
| 4 | easyml-models | 15 tasks | ~2500 |
| 5 | easyml-data | 5 tasks | ~400 |
| 6 | easyml-experiments | 6 tasks | ~600 |
| 7 | easyml-guardrails | 6 tasks | ~800 |
| 8 | Integration tests | 2 tasks | ~200 |
| 9 | Documentation | 3 tasks | ~300 |
| **Total** | **7 packages** | **53 tasks** | **~5800 + tests** |

## Dependency Order (Must Build In This Sequence)

```
Phase 0 (repo) → Phase 1 (schemas) → Phase 2 (config) → Phase 3 (features)
                                                                    ↓
Phase 5 (data) ←──────────────────── Phase 4 (models) ─────────────┘
       ↓
Phase 6 (experiments) → Phase 7 (guardrails) → Phase 8 (integration) → Phase 9 (docs)
```

Phases 4 and 5 can be parallelized after Phase 3 completes. Phase 6 depends on Phase 2 (config-overlay). Phase 7 depends on Phase 6.
