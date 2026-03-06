# Port Missing Functionality from Main to V2

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Port all unstaged functionality from main into the v2 refactor before merging, ensuring everything is configurable (not hardcoded).

**Architecture:** Each task ports a discrete unit of functionality. Model wrappers gain optional config params (normalize, batch_norm, early_stopping, etc.). Training gains configurable CDF conversion. Feature diversity is a new optional module. All new behavior is opt-in via config — no defaults change existing behavior.

**Tech Stack:** Python 3.11+, PyTorch, scikit-learn, XGBoost, CatBoost, LightGBM, pytest

**Key Principle:** This is a GENERIC ML framework. Every new capability is a configurable option. Nothing is hardcoded. No domain-specific assumptions.

---

## Batch 1: Model Wrappers (independent, parallelizable)

### Task 1: RandomForest param filtering

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/models/wrappers/random_forest.py`
- Test: `packages/easyml-core/tests/models/test_random_forest.py`

**Step 1: Write failing test**

```python
# In test_random_forest.py — add test
def test_invalid_params_filtered(caplog):
    """RF silently filters params meant for boosting models."""
    import logging
    with caplog.at_level(logging.WARNING):
        model = RandomForestModel(
            n_estimators=10,
            learning_rate=0.1,       # XGBoost param
            colsample_bytree=0.8,    # XGBoost param
            max_depth=3,             # valid RF param
        )
    assert "learning_rate" in caplog.text
    # Should still work with valid params
    assert model.model.max_depth == 3
```

**Step 2:** Run test, verify FAIL

**Step 3: Implement**

Add `_INVALID_RF_PARAMS` frozenset at module level with known boosting-only params (`learning_rate`, `colsample_bytree`, `subsample`, `reg_alpha`, `reg_lambda`, `gamma`, `min_child_weight`, `scale_pos_weight`, `early_stopping_rounds`, `num_leaves`, `bagging_fraction`, `feature_fraction`). Filter them in `__init__` with a warning log.

**Step 4:** Run test, verify PASS

**Step 5:** Commit: `feat: add param filtering to RandomForest wrapper`

---

### Task 2: CatBoost eval_set + early stopping

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/models/wrappers/catboost.py`
- Test: `packages/easyml-core/tests/models/test_catboost.py`

**Step 1: Write failing test**

```python
def test_fit_with_eval_set():
    """CatBoost accepts eval_set for early stopping."""
    model = CatBoostModel(iterations=50, early_stopping_rounds=10)
    X_train = pd.DataFrame({"a": range(80), "b": range(80)})
    y_train = pd.Series([0]*40 + [1]*40)
    X_val = pd.DataFrame({"a": range(20), "b": range(20)})
    y_val = pd.Series([0]*10 + [1]*10)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    probs = model.predict_proba(X_train)
    assert len(probs) == 80

def test_fit_without_eval_set_strips_early_stopping():
    """CatBoost drops early_stopping_rounds when no eval_set."""
    model = CatBoostModel(iterations=20, early_stopping_rounds=5)
    X = pd.DataFrame({"a": range(50), "b": range(50)})
    y = pd.Series([0]*25 + [1]*25)
    model.fit(X, y)  # should not error
    probs = model.predict_proba(X)
    assert len(probs) == 50
```

**Step 2:** Run test, verify FAIL

**Step 3: Implement**

Update `fit()` to accept `eval_set=None` kwarg. When eval_set provided, use CatBoost Pool objects and pass eval_set. When no eval_set, strip `early_stopping_rounds` from params before model init. Set `verbose=0` when using eval_set.

**Step 4:** Run test, verify PASS

**Step 5:** Commit: `feat: add eval_set support to CatBoost wrapper`

---

### Task 3: MLP enhancements (normalize, batch_norm, early_stopping, weight_decay)

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/models/wrappers/mlp.py`
- Test: `packages/easyml-core/tests/models/test_mlp.py`

**Step 1: Write failing tests**

```python
def test_mlp_normalize():
    """MLP with normalize=True standardizes features."""
    model = MLPModel(hidden_dims=[16], epochs=5, normalize=True)
    X = pd.DataFrame({"a": [100, 200, 300] * 20, "b": [0.01, 0.02, 0.03] * 20})
    y = pd.Series([0, 1, 1] * 20)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 60
    # Means/stds saved in meta
    meta = model.get_meta()
    assert "feature_means" in meta
    assert "feature_stds" in meta

def test_mlp_batch_norm():
    """MLP with batch_norm=True adds BatchNorm layers."""
    model = MLPModel(hidden_dims=[16, 8], epochs=5, batch_norm=True)
    X = pd.DataFrame({"a": range(50), "b": range(50)})
    y = pd.Series([0]*25 + [1]*25)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 50

def test_mlp_weight_decay():
    """MLP accepts weight_decay param."""
    model = MLPModel(hidden_dims=[16], epochs=5, weight_decay=1e-4)
    X = pd.DataFrame({"a": range(50), "b": range(50)})
    y = pd.Series([0]*25 + [1]*25)
    model.fit(X, y)  # should not error

def test_mlp_early_stopping():
    """MLP with early_stopping_rounds stops when val loss plateaus."""
    model = MLPModel(hidden_dims=[16], epochs=200, early_stopping_rounds=5)
    X = pd.DataFrame({"a": range(80), "b": range(80)})
    y = pd.Series([0]*40 + [1]*40)
    X_val = pd.DataFrame({"a": range(20), "b": range(20)})
    y_val = pd.Series([0]*10 + [1]*10)
    model.fit(X, y, eval_set=[(X_val, y_val)])
    # Should have stopped before 200 epochs
    probs = model.predict_proba(X)
    assert len(probs) == 80

def test_mlp_seed_stride():
    """Different n_seeds produce different models."""
    model = MLPModel(hidden_dims=[8], epochs=5, n_seeds=3, seed=42, seed_stride=100)
    X = pd.DataFrame({"a": range(50), "b": range(50)})
    y = pd.Series([0]*25 + [1]*25)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert len(probs) == 50
```

**Step 2:** Run tests, verify FAIL

**Step 3: Implement**

Port from main's mlp.py:
- Add `normalize`, `batch_norm`, `weight_decay`, `early_stopping_rounds`, `seed_stride` params to `__init__`
- `_build_mlp()`: insert `nn.BatchNorm1d` after each Linear when `batch_norm=True`
- `fit()`: z-score standardize when `normalize=True`, store means/stds; accept `eval_set` kwarg; implement early stopping patience loop tracking best val loss; add `weight_decay` to optimizer
- `predict_proba()`: apply stored normalization before forward pass
- `get_meta()`/`load_meta()`: persist `feature_means`, `feature_stds`, `batch_norm`
- Per-seed: use `seed + i * seed_stride` for each seed iteration

All params are optional with defaults that preserve current behavior (normalize=False, batch_norm=False, weight_decay=0, early_stopping_rounds=None, seed_stride=1).

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add normalize, batch_norm, early_stopping, weight_decay to MLP wrapper`

---

### Task 4: TabNet enhancements (normalize, val_fraction, scheduler, eval_set)

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/models/wrappers/tabnet.py`
- Test: `packages/easyml-core/tests/models/test_tabnet.py`

**Step 1: Write failing tests**

```python
def test_tabnet_normalize():
    """TabNet with normalize=True standardizes features."""
    tabnet = pytest.importorskip("pytorch_tabnet")
    model = TabNetModel(max_epochs=5, normalize=True)
    X = pd.DataFrame({"a": [100, 200, 300]*20, "b": [0.01, 0.02, 0.03]*20})
    y = pd.Series([0, 1, 1]*20)
    model.fit(X, y)
    meta = model.get_meta()
    assert "feature_means" in meta

def test_tabnet_val_fraction():
    """TabNet with val_fraction auto-carves validation data."""
    tabnet = pytest.importorskip("pytorch_tabnet")
    model = TabNetModel(max_epochs=5, val_fraction=0.2)
    X = pd.DataFrame({"a": range(100), "b": range(100)})
    y = pd.Series([0]*50 + [1]*50)
    model.fit(X, y)  # should not error — internally splits 80/20

def test_tabnet_learning_rate():
    """TabNet accepts learning_rate as optimizer param."""
    tabnet = pytest.importorskip("pytorch_tabnet")
    model = TabNetModel(max_epochs=5, learning_rate=0.01)
    X = pd.DataFrame({"a": range(50), "b": range(50)})
    y = pd.Series([0]*25 + [1]*25)
    model.fit(X, y)

def test_tabnet_scheduler():
    """TabNet with scheduler params uses StepLR."""
    tabnet = pytest.importorskip("pytorch_tabnet")
    model = TabNetModel(max_epochs=5, scheduler_step_size=2, scheduler_gamma=0.9)
    X = pd.DataFrame({"a": range(50), "b": range(50)})
    y = pd.Series([0]*25 + [1]*25)
    model.fit(X, y)

def test_tabnet_eval_set():
    """TabNet accepts eval_set for validation."""
    tabnet = pytest.importorskip("pytorch_tabnet")
    model = TabNetModel(max_epochs=10)
    X = pd.DataFrame({"a": range(80), "b": range(80)})
    y = pd.Series([0]*40 + [1]*40)
    X_val = pd.DataFrame({"a": range(20), "b": range(20)})
    y_val = pd.Series([0]*10 + [1]*10)
    model.fit(X, y, eval_set=[(X_val, y_val)])
    probs = model.predict_proba(X)
    assert len(probs) == 80
```

**Step 2:** Run tests, verify FAIL

**Step 3: Implement**

Port from main's tabnet.py:
- Add `_WRAPPER_KEYS` frozenset and `_PARAM_RENAMES` dict at module level
- Add params: `normalize`, `val_fraction`, `learning_rate`, `scheduler_step_size`, `scheduler_gamma`, `seed_stride`
- `_split_params()`: separate wrapper keys from TabNet keys, apply renames, inject learning_rate into optimizer_params
- `fit()`: standardize when normalize=True; accept `eval_set` kwarg; when no eval_set and val_fraction set, auto-split; build scheduler_params dict from step_size/gamma
- `predict_proba()`: apply stored normalization
- `get_meta()`/`load_meta()`: persist feature_means, feature_stds

All optional, defaults preserve current behavior.

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add normalize, val_fraction, scheduler, eval_set to TabNet wrapper`

---

### Task 5: ModelRegistry kwargs forwarding

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/models/registry.py`
- Test: `packages/easyml-core/tests/models/test_registry.py`

**Step 1: Write failing test**

```python
def test_create_with_kwargs():
    """Registry.create() forwards accepted kwargs to constructor."""
    registry = ModelRegistry()
    registry.register("test_model", DummyModel)
    model = registry.create("test_model", {"n_estimators": 10}, mode="classifier")
    assert model is not None

def test_create_filters_unknown_kwargs():
    """Registry.create() silently drops kwargs the constructor doesn't accept."""
    registry = ModelRegistry()
    registry.register("test_model", DummyModel)
    # DummyModel doesn't accept cdf_scale
    model = registry.create("test_model", {"n_estimators": 10}, cdf_scale=15.0)
    assert model is not None
```

**Step 2:** Run test, verify FAIL

**Step 3: Implement**

Update `create()` to accept `**kwargs`. Use `inspect.signature()` to get the constructor's parameters. Forward only kwargs that match parameter names (or if constructor has `**kwargs`). Log a debug message for dropped kwargs.

**Step 4:** Run test, verify PASS

**Step 5:** Commit: `feat: add inspect-based kwargs forwarding to ModelRegistry.create()`

---

## Batch 2: Training Pipeline (depends on Batch 1 Task 5)

### Task 6: Training — sigmoid CDF, post-training scale, generic _create_model

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/runner/training.py`
- Test: `packages/easyml-core/tests/runner/test_training.py`

**Step 1: Write failing tests**

```python
def test_sigmoid_cdf_conversion():
    """_sigmoid produces valid probabilities."""
    from easyml.core.runner.training import _sigmoid
    import numpy as np
    margins = np.array([-10, -1, 0, 1, 10])
    probs = _sigmoid(margins, scale=5.0)
    assert all(0 < p < 1 for p in probs)
    assert probs[2] == pytest.approx(0.5)  # margin=0 -> prob=0.5
    assert probs[0] < probs[4]  # monotonic

def test_fit_cdf_scale_after_training():
    """Post-training CDF scale fitting optimizes on predicted margins."""
    from easyml.core.runner.training import _fit_cdf_scale_after_training
    import numpy as np
    margins = np.array([-5, -2, 0, 2, 5, -3, 1, 4, -1, 3])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 1, 0, 1])
    scale = _fit_cdf_scale_after_training(margins, y)
    assert scale > 0

def test_nan_median_fallback():
    """All-NaN columns get median fallback of 0.0."""
    # Tested implicitly through train_single_model with NaN columns
    pass
```

**Step 2:** Run tests, verify FAIL

**Step 3: Implement**

Port from main's training.py:
- Add `_sigmoid(x, scale)` function: `1 / (1 + exp(-x / scale))` with clipping
- Add `_fit_cdf_scale_after_training(margins, y_true)`: optimize `log_scale` to minimize Brier score using `scipy.optimize.minimize_scalar` with logistic CDF, bounds based on margin std
- Update `_apply_cdf_conversion()` to use `_sigmoid` instead of `norm.cdf`
- Move CDF scale fitting to post-training: call `_fit_cdf_scale_after_training` using model's predicted margins
- Update NaN median fallback: `np.where(np.isnan(col_medians), 0.0, col_medians)`
- Update `_create_model()` to use `registry.create(name, params, **kwargs)` generically instead of special-casing XGBoost

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: sigmoid CDF conversion, post-training scale fitting, generic model creation`

---

## Batch 3: Config Writer + MCP (independent from Batch 1/2)

### Task 7: Config writer — cdf_scale, zero_fill_features, prior_feature, spline params

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/runner/config_writer.py`
- Test: `packages/easyml-core/tests/runner/test_config_writer.py`

**Step 1: Write failing tests**

```python
def test_add_model_with_cdf_scale(tmp_pipeline):
    """add_model accepts cdf_scale option."""
    result = add_model(tmp_pipeline, "margin_model", "xgboost",
                       mode="regressor", cdf_scale=15.0)
    assert "cdf_scale" in result
    cfg = _load_models(tmp_pipeline)
    assert cfg["margin_model"]["cdf_scale"] == 15.0

def test_add_model_with_zero_fill(tmp_pipeline):
    """add_model accepts zero_fill_features option."""
    result = add_model(tmp_pipeline, "test_model", "xgboost",
                       zero_fill_features=["feat_a", "feat_b"])
    cfg = _load_models(tmp_pipeline)
    assert cfg["test_model"]["zero_fill_features"] == ["feat_a", "feat_b"]

def test_update_model_cdf_scale(tmp_pipeline):
    """update_model can set cdf_scale."""
    add_model(tmp_pipeline, "m1", "xgboost")
    update_model(tmp_pipeline, "m1", cdf_scale=10.0)
    cfg = _load_models(tmp_pipeline)
    assert cfg["m1"]["cdf_scale"] == 10.0

def test_configure_ensemble_prior_feature(tmp_pipeline):
    """configure_ensemble accepts prior_feature."""
    result = configure_ensemble(tmp_pipeline, prior_feature="seed_diff")
    assert "prior_feature" in result

def test_configure_ensemble_spline_params(tmp_pipeline):
    """configure_ensemble accepts spline_prob_max and spline_n_bins."""
    result = configure_ensemble(tmp_pipeline,
                                calibration="spline",
                                spline_prob_max=0.95,
                                spline_n_bins=15)
    assert "spline" in result.lower()
```

**Step 2:** Run tests, verify FAIL

**Step 3: Implement**

- `add_model()`: add `cdf_scale: float | None = None`, `zero_fill_features: list[str] | None = None` params. Set on model_def if provided.
- `update_model()`: same two params. Merge into existing model_def.
- `configure_ensemble()`: add `prior_feature: str | None = None`, `spline_prob_max: float | None = None`, `spline_n_bins: int | None = None`. Write to ensemble.yaml.

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add cdf_scale, zero_fill_features, prior_feature, spline params to config writer`

---

### Task 8: MCP server + handler updates

**Files:**
- Modify: `packages/easyml-plugin/src/easyml/plugin/mcp_server.py`
- Modify: `packages/easyml-plugin/src/easyml/plugin/handlers/models.py`
- Modify: `packages/easyml-plugin/src/easyml/plugin/handlers/config.py`
- Modify: `packages/easyml-plugin/src/easyml/plugin/handlers/features.py`

**Step 1: Update MCP server tool signatures**

In `manage_models()`:
- Change `active: bool = True` and `include_in_ensemble: bool = True` to `bool | None = None`
- Add `cdf_scale: float | None = None`
- Add `zero_fill_features: list[str] | None = None`
- Pass through to handler

In `configure()`:
- Add `prior_feature: str | None = None`
- Add `spline_prob_max: float | None = None`
- Add `spline_n_bins: int | None = None`
- Pass through to handler

In `manage_features()`:
- Add `"diversity"` to action docstring

Update all docstrings to document new params.

**Step 2: Update handlers**

- `handlers/models.py`: pass `cdf_scale`, `zero_fill_features` to `config_writer.add_model()` and `update_model()`. Handle `active`/`include_in_ensemble` None vs explicit bool.
- `handlers/config.py`: pass `prior_feature`, `spline_prob_max`, `spline_n_bins` to `configure_ensemble()`.
- `handlers/features.py`: add `_handle_diversity()` action that calls `feature_diversity.format_diversity_report()`.

**Step 3:** Verify no import errors: `uv run python -c "from easyml.plugin.mcp_server import mcp"`

**Step 4:** Commit: `feat: update MCP server and handlers with new config options`

---

## Batch 4: Feature Diversity (independent)

### Task 9: Feature diversity module

**Files:**
- Create: `packages/easyml-core/src/easyml/core/runner/feature_diversity.py`
- Create: `packages/easyml-core/tests/runner/test_feature_diversity.py`

**Step 1: Write failing tests**

Port and adapt from main's `test_feature_diversity.py` (27 tests):
- `TestOverlapMatrix`: disjoint/identical/partial overlap, symmetry, diagonal=1, inactive model filtering
- `TestDiversityScore`: perfect/zero/partial scores, single model, empty, inactive
- `TestFindRedundant`: threshold filtering, sorting, shared features
- `TestSuggestRemoval`: already diverse, suggestions reduce overlap
- `TestFormatReport`: contains score/matrix/status

All tests use plain dicts for model configs (no domain assumptions):
```python
models = {
    "model_a": {"features": ["f1", "f2", "f3"], "active": True},
    "model_b": {"features": ["f4", "f5", "f6"], "active": True},
}
```

**Step 2:** Run tests, verify FAIL

**Step 3: Implement**

Port `feature_diversity.py` from main, updating import paths to `easyml.core.runner`:
- `compute_overlap_matrix(models)` — Jaccard similarity matrix
- `compute_diversity_score(models)` — scalar 0-1 score
- `find_redundant_features(models, threshold=0.8)` — pairs above threshold
- `suggest_removal(models, target_score=0.7)` — greedy removal suggestions
- `format_diversity_report(models)` — markdown report

**Step 4:** Run tests, verify PASS

**Step 5:** Commit: `feat: add feature diversity analysis module`

---

### Task 10: Feature diversity guardrail

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/guardrails/inventory.py`
- Test: `packages/easyml-core/tests/guardrails/test_inventory.py`

**Step 1: Write failing test**

```python
def test_feature_diversity_guardrail_pass():
    """Diversity guardrail passes when models have diverse features."""
    g = FeatureDiversityGuardrail()
    ctx = {
        "models": {
            "m1": {"features": ["a", "b"], "active": True},
            "m2": {"features": ["c", "d"], "active": True},
        }
    }
    result = g.check(ctx)
    assert result.passed

def test_feature_diversity_guardrail_fail():
    """Diversity guardrail fails when models have identical features."""
    g = FeatureDiversityGuardrail()
    ctx = {
        "models": {
            "m1": {"features": ["a", "b", "c"], "active": True},
            "m2": {"features": ["a", "b", "c"], "active": True},
        }
    }
    result = g.check(ctx)
    assert not result.passed
```

**Step 2:** Run test, verify FAIL

**Step 3: Implement**

Add `FeatureDiversityGuardrail` to inventory.py:
- Overridable guardrail (not critical)
- Configurable `min_diversity_score` threshold (default 0.5)
- Uses `compute_diversity_score()` from feature_diversity module
- Actionable error message suggesting `manage_features(action='diversity')`
- Update guardrail count in docstring (11 -> 12)

**Step 4:** Run test, verify PASS

**Step 5:** Commit: `feat: add FeatureDiversityGuardrail (12th guardrail)`

---

## Batch 5: OMP fix + final verification

### Task 11: OMP_NUM_THREADS fix

**Files:**
- Modify: `packages/easyml-core/src/easyml/core/runner/pipeline.py` (top of file)

**Step 1: Implement**

Add at the very top of `pipeline.py`, before any imports that trigger OpenMP:
```python
import os
import platform
if platform.system() == "Darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
```

This prevents OpenMP deadlocks between LightGBM and PyTorch on macOS. Uses `setdefault` so users can override.

**Step 2:** Verify tests still pass: `uv run pytest packages/easyml-core/tests/ -q`

**Step 3:** Commit: `fix: prevent OpenMP deadlock on macOS`

---

### Task 12: Full test suite + backtest verification

**Step 1:** Run full test suite
```bash
uv run pytest packages/easyml-core/tests/ -q
```
Expected: all pass, 0 failures

**Step 2:** Run women's MM backtest
```python
from easyml.core.runner.pipeline import PipelineRunner
runner = PipelineRunner(project_dir='.', config_dir='config')
runner.load()
result = runner.backtest()
print(result['report'])
```
Expected: Brier ~0.14, no errors

**Step 3:** Final commit if any fixups needed

---

## Task Dependency Graph

```
Batch 1 (parallel):
  Task 1: RF param filtering
  Task 2: CatBoost eval_set
  Task 3: MLP enhancements
  Task 4: TabNet enhancements
  Task 5: Registry kwargs

Batch 2 (depends on Task 5):
  Task 6: Training sigmoid CDF + generic _create_model

Batch 3 (parallel, independent):
  Task 7: Config writer params
  Task 8: MCP server + handlers (depends on Task 7 + Task 9)

Batch 4 (parallel, independent):
  Task 9: Feature diversity module
  Task 10: Diversity guardrail (depends on Task 9)

Batch 5 (after all):
  Task 11: OMP fix
  Task 12: Final verification
```

## Execution Order (respecting dependencies)

**Wave 1** (all parallel): Tasks 1, 2, 3, 4, 5, 7, 9
**Wave 2** (after wave 1): Tasks 6, 8, 10
**Wave 3** (after wave 2): Tasks 11, 12
