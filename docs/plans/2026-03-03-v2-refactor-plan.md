# HarnessML v2 Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate 8 packages into 3 (core, plugin, sports), extract domain-specific code, add general-purpose feature engineering, comprehensive metrics, data source management, and MCP optimizations.

**Architecture:** harness-core (general engine), harness-plugin (MCP thin dispatcher with hot-reload), harness-sports (optional matchup domain plugin). Core exposes extension hooks; plugins register implementations.

**Tech Stack:** Python 3.11+, uv workspace, Pydantic v2, OmegaConf, FastMCP (async), Pandera (optional), shap (optional), matplotlib (optional)

**Design doc:** `docs/plans/2026-03-03-v2-refactor-design.md`

**Migration strategy:** Break-and-fix. mm-women will be broken during refactor and fixed at the end via the sports plugin.

---

## Phase 1: Package Consolidation

**Goal:** Merge 5 packages (schemas, config, guardrails, models, experiments) into harness-core alongside the runner. Delete 3 dead packages (features, data, config). All existing tests pass under new import paths.

### Task 1.1: Create harness-core package skeleton

**Files:**
- Create: `packages/harness-core/pyproject.toml`
- Create: `packages/harness-core/src/harnessml/core/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "harness-core"
version = "0.1.0"
description = "General-purpose agentic ML framework"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "numpy>=1.24",
    "pandas>=2.0",
    "pyarrow>=14.0",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "omegaconf>=2.3",
    "pyyaml>=6.0",
    "click>=8.0",
]

[project.optional-dependencies]
xgboost = ["xgboost>=1.7"]
catboost = ["catboost>=1.2"]
lightgbm = ["lightgbm>=4.0"]
neural = ["torch>=2.0", "pytorch-tabnet>=4.0"]
explore = ["optuna>=3.0"]
shap = ["shap>=0.42"]
viz = ["matplotlib>=3.7"]
quality = ["pandera>=0.17"]
all = [
    "harness-core[xgboost,catboost,lightgbm,neural,explore,shap,viz,quality]",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/harnessml"]
```

**Step 2: Create namespace __init__.py**

```python
# packages/harness-core/src/harnessml/core/__init__.py
"""HarnessML Core — general-purpose agentic ML framework."""
```

**Step 3: Create subdirectory structure**

Run:
```bash
mkdir -p packages/harness-core/src/harnessml/core/{schemas,config,guardrails,models,models/wrappers,runner,runner/sources,feature_eng}
mkdir -p packages/harness-core/tests
```

**Step 4: Commit**

```bash
git add packages/harness-core/
git commit -m "feat: create harness-core package skeleton"
```

---

### Task 1.2: Move harnessml-schemas into harness-core/schemas

**Files:**
- Move: `packages/harnessml-schemas/src/harnessml/schemas/core.py` → `packages/harness-core/src/harnessml/core/schemas/contracts.py`
- Move: `packages/harnessml-schemas/src/harnessml/schemas/metrics.py` → `packages/harness-core/src/harnessml/core/schemas/metrics.py`
- Create: `packages/harness-core/src/harnessml/core/schemas/__init__.py`
- Move: `packages/harnessml-schemas/tests/` → `packages/harness-core/tests/schemas/`

**Step 1: Copy files**

```bash
cp packages/harnessml-schemas/src/harnessml/schemas/core.py packages/harness-core/src/harnessml/core/schemas/contracts.py
cp packages/harnessml-schemas/src/harnessml/schemas/metrics.py packages/harness-core/src/harnessml/core/schemas/metrics.py
cp -r packages/harnessml-schemas/tests/ packages/harness-core/tests/schemas/
```

**Step 2: Create __init__.py that re-exports everything**

```python
# packages/harness-core/src/harnessml/core/schemas/__init__.py
"""Shared Pydantic contracts and metric functions."""
from harnessml.core.schemas.contracts import *  # noqa: F401,F403
from harnessml.core.schemas.metrics import *    # noqa: F401,F403
```

**Step 3: Create backward-compat shim at old import path**

To avoid breaking everything at once, create a shim that re-exports from the new location. This will be removed at the end of Phase 1.

```python
# packages/harnessml-schemas/src/harnessml/schemas/__init__.py (TEMPORARY)
# Backward-compat shim — will be removed after all imports are updated
from harnessml.core.schemas import *  # noqa: F401,F403
```

**Step 4: Run existing tests**

```bash
uv run pytest packages/harness-core/tests/schemas/ -v
```

Expected: PASS (tests use the types, which are now in the new location)

**Step 5: Commit**

```bash
git add packages/harness-core/src/harnessml/core/schemas/ packages/harness-core/tests/schemas/
git commit -m "feat: move harnessml-schemas into harness-core/schemas"
```

---

### Task 1.3: Move harnessml-config into harness-core/config

**Files:**
- Move: `packages/harnessml-config/src/harnessml/config/merge.py` → `packages/harness-core/src/harnessml/core/config/merge.py`
- Move: `packages/harnessml-config/src/harnessml/config/loader.py` → `packages/harness-core/src/harnessml/core/config/loader.py`
- Move: `packages/harnessml-config/src/harnessml/config/resolver.py` → `packages/harness-core/src/harnessml/core/config/resolver.py`
- Create: `packages/harness-core/src/harnessml/core/config/__init__.py`
- Move tests

**Follow the same pattern as Task 1.2:** Copy files, create __init__.py with re-exports, add backward-compat shim, run tests, commit.

Internal imports within config (e.g., resolver.py imports from merge.py) must be updated to use `harnessml.core.config.merge` instead of `harnessml.config.merge`.

---

### Task 1.4: Move harnessml-models into harness-core/models

**Files:**
- Move all files from `packages/harnessml-models/src/harnessml/models/` → `packages/harness-core/src/harnessml/core/models/`
- Move `packages/harnessml-models/src/harnessml/models/wrappers/` → `packages/harness-core/src/harnessml/core/models/wrappers/`
- Move tests

**Key imports to update inside the models package:**
- `wrappers/*.py`: `from harnessml.models.base import BaseModel` → `from harnessml.core.models.base import BaseModel`
- `registry.py`: `from harnessml.models.base` → `from harnessml.core.models.base`
- `cv.py`: `from harnessml.schemas.core import Fold` → `from harnessml.core.schemas.contracts import Fold`

Add backward-compat shim at old path. Run tests. Commit.

---

### Task 1.5: Move harnessml-experiments into harness-core (inline into runner)

The experiments package is 3 files (~200 LOC). Rather than creating a separate `experiments/` directory, merge its functionality into the runner since that's where experiments are actually orchestrated.

**Files:**
- `packages/harnessml-experiments/src/harnessml/experiments/manager.py` → `packages/harness-core/src/harnessml/core/runner/experiment.py` (merge with existing `experiment.py` in runner)

**Note:** The runner already has `experiment.py` with CLI-level experiment logic. The experiments package has `ExperimentManager` with create/detect/log/promote. These should be unified into one file.

**Step 1: Read both files and merge** — the runner's experiment.py likely already duplicates most of the ExperimentManager logic. Keep the most complete version.

**Step 2: Update imports from `harnessml.experiments` → `harnessml.core.runner.experiment`**

**Step 3: Add backward-compat shim. Run tests. Commit.**

---

### Task 1.6: Move harnessml-guardrails into harness-core/guardrails

**Files:**
- Move all files from `packages/harnessml-guardrails/src/harnessml/guardrails/` → `packages/harness-core/src/harnessml/core/guardrails/`
- Move tests

**Key imports to update:**
- `base.py`: `from harnessml.schemas.core import GuardrailViolation` → `from harnessml.core.schemas.contracts import GuardrailViolation`
- `inventory.py`: update schema imports

Add backward-compat shim. Run tests. Commit.

---

### Task 1.7: Move harnessml-runner into harness-core/runner

This is the largest move. All files from `packages/harnessml-runner/src/harnessml/runner/` go to `packages/harness-core/src/harnessml/core/runner/`.

**Step 1: Copy all source files**

```bash
cp -r packages/harnessml-runner/src/harnessml/runner/* packages/harness-core/src/harnessml/core/runner/
cp -r packages/harnessml-runner/tests/* packages/harness-core/tests/runner/
```

**Step 2: Mass update imports across ALL files in harness-core/runner/**

Find and replace (in all `.py` files under `packages/harness-core/src/`):
- `from harnessml.schemas` → `from harnessml.core.schemas`
- `from harnessml.config` → `from harnessml.core.config`
- `from harnessml.models` → `from harnessml.core.models`
- `from harnessml.features` → `from harnessml.core.runner` (features package is dead, any remaining refs point to feature_store)
- `from harnessml.data` → `from harnessml.core.runner` (data package is dead)
- `from harnessml.experiments` → `from harnessml.core.runner.experiment`
- `from harnessml.guardrails` → `from harnessml.core.guardrails`
- `from harnessml.runner` → `from harnessml.core.runner`
- `import harnessml.runner` → `import harnessml.core.runner`

**Step 3: Update harness-core pyproject.toml** — remove workspace dependencies on old packages, all code is now local.

**Step 4: Run full test suite**

```bash
uv run pytest packages/harness-core/tests/ -v
```

Fix any remaining import issues until all tests pass.

**Step 5: Commit**

```bash
git commit -m "feat: move harnessml-runner into harness-core/runner"
```

---

### Task 1.8: Update harness-plugin imports

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/mcp_server.py`
- Modify: `packages/harness-plugin/pyproject.toml`

**Step 1: Update pyproject.toml dependency**

```toml
dependencies = ["harness-core", "mcp>=1.0"]
```

**Step 2: Update all imports in mcp_server.py**

The MCP server currently does `from harnessml.runner import config_writer as cw` inside each tool function. Update to:
```python
from harnessml.core.runner import config_writer as cw
```

**Step 3: Run MCP server smoke test**

```bash
uv run python -c "from harnessml.plugin.mcp_server import mcp; print('MCP server loads OK')"
```

**Step 4: Commit**

---

### Task 1.9: Remove backward-compat shims and delete old packages

**Step 1: Remove shims** from:
- `packages/harnessml-schemas/src/harnessml/schemas/__init__.py`
- `packages/harnessml-config/src/harnessml/config/__init__.py`
- `packages/harnessml-models/src/harnessml/models/__init__.py`
- `packages/harnessml-experiments/src/harnessml/experiments/__init__.py`
- `packages/harnessml-guardrails/src/harnessml/guardrails/__init__.py`
- `packages/harnessml-runner/src/harnessml/runner/__init__.py`

**Step 2: Delete old packages**

```bash
rm -rf packages/harnessml-schemas
rm -rf packages/harnessml-config
rm -rf packages/harnessml-features
rm -rf packages/harnessml-data
rm -rf packages/harnessml-models
rm -rf packages/harnessml-experiments
rm -rf packages/harnessml-guardrails
rm -rf packages/harnessml-runner
```

**Step 3: Update root pyproject.toml workspace** to only include harness-core, harness-plugin, (and later harness-sports).

**Step 4: Run full test suite one final time**

```bash
uv run pytest -v
```

All tests must pass with only harness-core and harness-plugin.

**Step 5: Commit**

```bash
git commit -m "chore: remove old packages, consolidation complete"
```

---

## Phase 2: MCP Architecture

**Goal:** Restructure MCP server into thin async dispatcher with hot-reloadable handlers, add batch actions, progress reporting, response modes, and input validation.

### Task 2.1: Create handler dispatch structure

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/__init__.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/models.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/data.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/features.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/experiments.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/config.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py`
- Create: `packages/harness-plugin/src/harnessml/plugin/handlers/_validation.py`

**Step 1: Create _validation.py with shared utilities**

```python
from difflib import get_close_matches

def validate_enum(value: str, valid: set[str], param_name: str) -> str | None:
    if value in valid:
        return None
    closest = get_close_matches(value, list(valid), n=1, cutoff=0.6)
    msg = f"**Error**: Invalid `{param_name}` '{value}'. Valid: {', '.join(sorted(valid))}"
    if closest:
        msg += f"\n\nDid you mean **{closest[0]}**?"
    return msg

def validate_required(value, param_name: str) -> str | None:
    if value is None or value == "":
        return f"**Error**: `{param_name}` is required."
    return None
```

**Step 2: Extract manage_models logic into handlers/models.py**

Move the `if action == "add": ... elif action == "update": ...` chain from mcp_server.py into a dispatch dict in handlers/models.py. Each action becomes its own function.

**Step 3: Repeat for all 6 handlers** — data.py, features.py, experiments.py, config.py, pipeline.py.

**Step 4: Rewrite mcp_server.py as thin dispatcher**

```python
import importlib
import os
from mcp.server.fastmcp import FastMCP, Context

mcp = FastMCP("harnessml")
_DEV_MODE = os.environ.get("HARNESS_DEV", "0") == "1"

def _load_handler(module_name: str):
    mod = importlib.import_module(f"harnessml.plugin.handlers.{module_name}")
    if _DEV_MODE:
        importlib.reload(mod)
    return mod

@mcp.tool()
async def manage_models(action: str, ctx: Context, ...) -> str:
    return _load_handler("models").dispatch(action, ctx=ctx, ...)
```

**Step 5: Run MCP smoke test. Commit.**

---

### Task 2.2: Convert tools to async with progress reporting

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/mcp_server.py` (all tool defs → `async def`)
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py` (add progress to backtest)
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/experiments.py` (add progress to explore)

**Step 1:** Change all 6 tool functions from `def` to `async def` and add `ctx: Context` parameter.

**Step 2:** In pipeline handler's `_handle_run_backtest`, add progress calls at fold boundaries. The handler wraps the synchronous `PipelineRunner.backtest()` — inject progress callback via a new optional parameter on the runner method, or wrap the loop in the handler.

**Step 3:** In experiments handler's `_handle_explore`, add progress per trial.

**Step 4: Commit.**

---

### Task 2.3: Add batch actions

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/models.py` — add `add_batch`, `update_batch`, `remove_batch`, `clone`
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/data.py` — add `add_sources_batch`, `fill_nulls_batch`, `add_views_batch`
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py` — add `compare_runs`
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/experiments.py` — add `compare`

For each batch action:
1. Accept a JSON array of items
2. Iterate through, collecting results and errors
3. Return summary: `"Added 5 models. 1 error: ..."`

**These are all handler-only changes — no MCP restart needed** since they're new action strings within existing tools.

**Commit after each handler is updated.**

---

### Task 2.4: Add response modes

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/mcp_server.py` — add `detail: str | None = None` to pipeline, configure, experiments tool signatures
- Modify: handlers to respect `detail="summary"|"full"|"metrics"`

**This changes the MCP contract (new parameter) — requires restart.**

For `pipeline(action="show_run", detail="summary")`: return 5-line metrics table.
For `configure(action="show", section="models")`: return only the models YAML block.

**Commit.**

---

### Task 2.5: Add cross-parameter hints and actionable errors

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/_validation.py` — add hint generators
- Modify: each handler — add hint collection after successful operations and context to error messages

**Commit.**

---

## Phase 3: Domain Extraction

**Goal:** Create harness-sports, move `generate_pairwise_matchups()`, rename seed→prior everywhere, parameterize hardcoded column detection. Fix mm-women.

### Task 3.1: Create harness-sports package

**Files:**
- Create: `packages/harness-sports/pyproject.toml`
- Create: `packages/harness-sports/src/harnessml/sports/__init__.py`
- Create: `packages/harness-sports/src/harnessml/sports/matchups.py`
- Create: `packages/harness-sports/src/harnessml/sports/hooks.py`
- Create: `packages/harness-sports/tests/`

**Step 1:** Move `generate_pairwise_matchups()` (lines 76-190 of runner/matchups.py) to `harness-sports/matchups.py`. Keep `compute_interactions()` and `predict_all_matchups()` in core (they're generic).

**Step 2:** Create `hooks.py` that registers sports-specific behavior with core extension points (defined in Task 3.3).

**Step 3:** Commit.

---

### Task 3.2: Rename seed→prior throughout core

**Files (all in `packages/harness-core/src/harnessml/core/runner/`):**
- Modify: `schema.py` lines 455-456: `seed_compression` → `prior_compression`, `seed_compression_threshold` → `prior_compression_threshold`
- Modify: `meta_learner.py` lines 32, 39, 57, 80, 133, 156, 168, 223, 232, 265: `seed_diffs` → `prior_diffs`
- Modify: `postprocessing.py` lines 64-70, 108-113: `seed_compression` → `prior_compression`
- Modify: `config_writer.py`: update any references to seed_compression in YAML writing
- Modify: MCP handler: update parameter names

**This is a global find-and-replace** with these substitutions:
- `seed_compression` → `prior_compression`
- `seed_compression_threshold` → `prior_compression_threshold`
- `seed_diffs` → `prior_diffs`

**Step 1:** Run find-and-replace across all `.py` files in harness-core.
**Step 2:** Update the mm-women pipeline.yaml config to use new names.
**Step 3:** Run tests. Fix any breakage.
**Step 4:** Commit.

---

### Task 3.3: Define extension hooks in core

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/feature_store.py` — add `_expansion_hooks` class variable and hook registration
- Modify: `packages/harness-core/src/harnessml/core/runner/pipeline.py` — parameterize `_inject_entity()` lines 168-171 to remove hardcoded "TeamA"/"TeamB", add `_provider_injection_hooks`
- Modify: `packages/harness-core/src/harnessml/core/runner/training.py` — add `_pre_training_hooks`
- Modify: `packages/harness-core/src/harnessml/core/runner/postprocessing.py` — add `_post_prediction_hooks`

**Step 1:** In `pipeline.py:_inject_entity()`, replace the hardcoded fallback chain at lines 168-171 with a configurable list of column name candidates read from config. Default to generic names (`["entity_a", "group_a"]`). The sports plugin will add `["team_a", "TeamA"]` to this list via hook.

**Step 2:** Add hook registration pattern (as described in the design doc Section 3) to feature_store, pipeline, training, and postprocessing.

**Step 3:** In `harness-sports/hooks.py`, register the sports-specific implementations.

**Step 4:** Run tests. Commit.

---

### Task 3.4: Fix mm-women project

**Files:**
- Modify: `projects/womens-tournament/config/pipeline.yaml` (or wherever mm-women config lives)
- Ensure harness-sports is installed as dependency

**Step 1:** Update pipeline.yaml to use renamed config keys (prior_compression instead of seed_compression).
**Step 2:** Ensure sports plugin is installed and hooks are registered at project load time.
**Step 3:** Run mm-women backtest end-to-end. Verify Brier/accuracy match previous results.
**Step 4:** Commit.

---

## Phase 4: Feature Engineering

**Goal:** Add 8 view steps, 15 formula functions, 10 rolling aggregations, enhanced auto-search.

### Task 4.1: Add formula functions

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/feature_engine.py` — extend `_SAFE_FUNCTIONS`
- Create: `packages/harness-core/tests/runner/test_formula_functions.py`

**Step 1: Write tests** for each new function (reciprocal, exp, sin_cycle, cos_cycle, zscore, minmax, rank_pct, winsorize, safe_div, pct_of_total, maximum, minimum, where, isnull, expm1, power).

**Step 2: Add functions to `_SAFE_FUNCTIONS` dict.**

**Step 3: Run tests. Commit.**

---

### Task 4.2: Add rolling aggregation functions

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/view_executor.py` — extend rolling step handler
- Create: `packages/harness-core/tests/runner/test_rolling_aggs.py`

**Step 1: Write tests** for: median, skew, kurt, slope, ema, range, cv, pct_change, first, last.

**Step 2: Implement** each as a recognized aggregation in the rolling step executor. `slope` requires a small OLS helper (~15 lines of numpy).

**Step 3: Run tests. Commit.**

---

### Task 4.3: Add lag step

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/schema.py` — add LagStep to TransformStep union
- Modify: `packages/harness-core/src/harnessml/core/runner/view_executor.py` — implement lag executor
- Create: `packages/harness-core/tests/runner/test_lag_step.py`

**Schema:**
```python
class LagStep(BaseModel):
    op: Literal["lag"] = "lag"
    keys: list[str]           # group columns
    order_by: str             # sort column
    columns: dict[str, str]   # {new_col: "source_col:lag_periods"}
```

**TDD: Write failing test → implement → verify → commit.**

---

### Task 4.4-4.10: Add remaining view steps

Repeat the TDD pattern from Task 4.3 for each:
- **4.4:** `ewm` step (exponentially weighted moving stats)
- **4.5:** `diff` step (first/second differences, pct_change)
- **4.6:** `trend` step (OLS slope over window)
- **4.7:** `encode` step (target_loo, target_temporal, frequency, ordinal)
- **4.8:** `bin` step (quantile, uniform, custom, kmeans)
- **4.9:** `datetime` step (calendar extraction + cyclical sin/cos)
- **4.10:** `null_indicator` step (binary null flags)

Each task: add Pydantic model to schema.py, implement executor in view_executor.py, write tests, commit.

---

### Task 4.11: Enhance auto-feature search

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/transformation_tester.py`
- Modify: handler for `manage_features(action="auto_search")`

Add `search_types` parameter: `["interactions", "lags", "rolling"]`. For each type, systematically generate candidates and rank by marginal lift (correlation or residual-based).

**TDD. Commit.**

---

## Phase 5: Metrics & Diagnostics

**Goal:** Comprehensive metrics for all ML task types, enhanced calibration, SHAP, visualization.

### Task 5.1: Implement MetricRegistry with task-type dispatch

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/schemas/metrics.py`
- Create: `packages/harness-core/tests/schemas/test_metric_registry.py`

Refactor existing metric functions into a `MetricRegistry` class that auto-selects metrics by task type. Existing metric functions stay as-is; the registry wraps them.

**TDD. Commit.**

---

### Task 5.2: Add binary classification metrics

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/schemas/metrics.py`
- Create: `packages/harness-core/tests/schemas/test_binary_metrics.py`

Add: precision, recall, mcc, pr_auc, specificity, confusion_matrix, cohen_kappa.

Register all with `MetricRegistry.register("binary", ...)`.

**TDD. Commit.**

---

### Task 5.3: Add multiclass metrics

Add: macro/micro/weighted f1/precision/recall, NxN confusion_matrix, per_class_report, mcc_multiclass, cohen_kappa.

Register with `MetricRegistry.register("multiclass", ...)`.

**TDD. Commit.**

---

### Task 5.4: Add regression metric extras

Add: mape, median_ae, explained_variance, mean_bias, quantile_loss.

**TDD. Commit.**

---

### Task 5.5: Add ranking metrics

Add: ndcg_at_k, map, mrr, precision_at_k, recall_at_k, spearman_rank_correlation.

**TDD. Commit.**

---

### Task 5.6: Add survival metrics

Add: concordance_index, time_dependent_brier, cumulative_incidence_auc.

**TDD. Commit.**

---

### Task 5.7: Add probabilistic metrics

Add: crps, pit_histogram_data, sharpness, coverage_at_level.

**TDD. Commit.**

---

### Task 5.8: Enhanced calibration

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/calibration.py`
- Create: `packages/harness-core/tests/runner/test_calibration_enhanced.py`

Add: BetaCalibrator, reliability_diagram_data, hosmer_lemeshow_test, calibration_slope_intercept, calibration_decomposition, bootstrap_ci, per_class_calibration.

**TDD. Commit per calibrator/diagnostic.**

---

### Task 5.9: SHAP integration

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/diagnostics.py`
- Create: `packages/harness-core/tests/runner/test_shap.py`

Add `compute_shap_values()` with optional `shap` import. Methods: auto/tree/kernel/linear.

**TDD. Commit.**

---

### Task 5.10: ROC/PR curve data + permutation importance

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/diagnostics.py`

Add: `roc_curve_data()`, `pr_curve_data()`, `permutation_importance()`.

All return structured dicts/arrays (not plots).

**TDD. Commit.**

---

### Task 5.11: Optional matplotlib rendering

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/viz.py`

When `matplotlib` is installed, render: roc_curve.png, pr_curve.png, calibration.png, confusion_matrix.png, shap_summary.png, feature_importance.png.

Uses structured data from diagnostics.py as input. All rendering in one module.

Exposed via `pipeline(action="diagnostics", render="png", output_dir="./plots/")`.

**TDD. Commit.**

---

### Task 5.12: Wire progress reporting into backtest and exploration

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/pipeline.py` — accept optional progress callback
- Modify: `packages/harness-core/src/harnessml/core/runner/exploration.py` — accept optional progress callback
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py` — pass `ctx.report_progress` as callback
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/experiments.py` — same

The core runner methods accept an optional `on_progress: Callable[[int, int, str], None]` parameter. The MCP handlers wrap `ctx.report_progress` into this callback.

**Commit.**

---

## Phase 6: Data Source Management

**Goal:** Source registry with freshness tracking, schema validation, built-in adapters, MCP actions.

### Task 6.1: Create source registry

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/sources/registry.py`
- Create: `packages/harness-core/src/harnessml/core/runner/sources/__init__.py`
- Create: `packages/harness-core/tests/runner/sources/test_registry.py`

Implement `SourceDef` dataclass and `SourceRegistry` with topological ordering (reuse pattern from mm's source_registry.py).

Sources stored in `config/sources.yaml`. State in `config/sources_state.json`.

**TDD. Commit.**

---

### Task 6.2: Implement freshness tracker

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/sources/freshness.py`
- Create: `packages/harness-core/tests/runner/sources/test_freshness.py`

`FreshnessTracker`: reads sources_state.json, compares timestamps to refresh_frequency, returns stale sources.

**TDD. Commit.**

---

### Task 6.3: Implement schema validation

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/sources/validation.py`
- Create: `packages/harness-core/tests/runner/sources/test_validation.py`

Optional Pandera integration. `validate_source(source_def, df)` returns list of violations.

**TDD. Commit.**

---

### Task 6.4: Implement built-in adapters

**Files:**
- Create: `packages/harness-core/src/harnessml/core/runner/sources/adapters.py`
- Create: `packages/harness-core/tests/runner/sources/test_adapters.py`

Four adapters: `FileAdapter`, `UrlAdapter`, `ApiAdapter`, `ComputedAdapter`.

**TDD. Commit per adapter.**

---

### Task 6.5: Wire MCP actions

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/data.py`

Add actions: `add_source`, `add_sources_batch`, `check_freshness`, `refresh`, `refresh_all`, `validate_source`.

`refresh_all` uses topological ordering + progress reporting.

**TDD. Commit.**

---

## Summary

| Phase | Tasks | Depends On | Parallelizable With |
|-------|-------|------------|---------------------|
| 1. Package Consolidation | 1.1-1.9 | None | Nothing (critical path) |
| 2. MCP Architecture | 2.1-2.5 | Phase 1 | Phases 3, 4, 5 |
| 3. Domain Extraction | 3.1-3.4 | Phase 1 | Phases 2, 4, 5 |
| 4. Feature Engineering | 4.1-4.11 | Phase 1 | Phases 2, 3, 5 |
| 5. Metrics & Diagnostics | 5.1-5.12 | Phase 1 | Phases 2, 3, 4 |
| 6. Data Source Management | 6.1-6.5 | Phase 2 | Phases 3, 4, 5 |

**Total tasks: 46**
