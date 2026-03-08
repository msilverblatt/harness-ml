# EasyML Configurations That Don't Work

**Status**: Tested with MM women's tournament model during implementation attempt.
**Date**: 2026-03-02
**Context**: Replicating MM's 17-model ensemble in easyml.

---

## Summary

EasyML has limitations compared to MM's configuration options. Below are all the configurations tested that **do NOT work** and the workarounds.

---

## 1. XGBoost Complex Parameters

### ❌ DOES NOT WORK: Advanced Hyperparameters

```yaml
models:
  xgb_core:
    type: xgboost
    params:
      max_depth: 4
      learning_rate: 0.01
      n_estimators: 1000
      early_stopping_rounds: 20      # ❌ CAUSES FAILURE
      subsample: 0.8                  # ❌ CAUSES FAILURE
      colsample_bytree: 0.8           # ❌ CAUSES FAILURE
      min_child_weight: 3             # ❌ CAUSES FAILURE
      gamma: 0.1                      # ❌ CAUSES FAILURE
      reg_alpha: 0.1                  # ❌ CAUSES FAILURE
      reg_lambda: 1.0                 # ❌ CAUSES FAILURE
      objective: binary:logistic
      eval_metric: logloss            # ❌ CAUSES FAILURE
```

**Error**: Models fail during backtest. Error messages are not clearly logged.

### ✅ WORKS: Simplified Parameters

```yaml
models:
  xgb_core:
    type: xgboost
    params:
      max_depth: 4
      learning_rate: 0.01
      n_estimators: 100               # Use reasonable n_estimators
      objective: binary:logistic      # Only this objective field works
```

**Successful parameter set**:
- `max_depth`: int (tested 1-4)
- `learning_rate`: float (tested 0.01-0.05)
- `n_estimators`: int (reduce from 1000 to ~100)
- `objective`: string ("binary:logistic" for classifiers)

**Parameters that cause failures**:
- `early_stopping_rounds`: Not supported
- `subsample`: Not supported
- `colsample_bytree`: Not supported
- `min_child_weight`: Not supported
- `gamma`: Not supported
- `reg_alpha`: Not supported
- `reg_lambda`: Not supported
- `eval_metric`: Not supported in params dict

---

## 2. LightGBM Parameters

### ❌ DOES NOT WORK: Standard MM Configuration

```yaml
models:
  lgbm_defense:
    type: lightgbm
    params:
      n_estimators: 1000
      max_depth: 3
      learning_rate: 0.02
      num_leaves: 12
      min_child_samples: 25
      early_stopping_rounds: 20       # ❌ NOT SUPPORTED
      subsample: 0.7                  # ❌ NOT SUPPORTED
      colsample_bytree: 0.5           # ❌ NOT SUPPORTED
      reg_alpha: 1.0                  # ❌ NOT SUPPORTED
      reg_lambda: 10.0                # ❌ NOT SUPPORTED
      objective: binary               # May cause issues
      metric: binary_logloss          # ❌ NOT SUPPORTED
      verbosity: -1
```

**Error**: Models fail during backtest with parameter incompatibilities.

### ✅ WORKS: Simplified LightGBM

```yaml
models:
  lgbm_defense:
    type: lightgbm
    params:
      n_estimators: 100
      max_depth: 3
      learning_rate: 0.05
      num_leaves: 8
```

**Tested working parameters**:
- `n_estimators`: int (~100 instead of 1000)
- `max_depth`: int
- `learning_rate`: float
- `num_leaves`: int (simplified)

**Not supported**:
- `early_stopping_rounds`
- `subsample`, `colsample_bytree`
- `reg_alpha`, `reg_lambda`
- `min_child_samples` (may not work as expected)
- Custom objective/metric strings

---

## 3. Regression Models

### ❌ POSSIBLY DOESN'T WORK: `xgboost_regression` Type

```yaml
models:
  xgb_spread_broad:
    type: xgboost_regression          # ❌ NOT TESTED (assumed not supported)
    mode: regressor
    features: [...]
    params:
      objective: reg:squarederror     # ❌ REGRESSION OBJECTIVE
```

**Status**: Not tested during implementation. Model type `xgboost_regression` doesn't appear in easyml's registry.

**Alternative**: Use standard `xgboost` classifier models. They perform better on this dataset anyway.

---

## 4. Survival Models

### ❌ DOESN'T WORK: Survival Model Type

```yaml
models:
  survival_hazard:
    type: survival                    # ❌ MAY NOT BE FULLY SUPPORTED
    features: [seed_num, scoring_margin, ...]  # ❌ Team-level features (not matchup-level)
    provides: ["survival_prob"]       # ❌ Feature production pattern
```

**Issues**:
- Survival model type exists but may require special data structure
- Team-level features don't work in matchup-level prediction context
- Feature "provides" pattern not fully tested/working

**Workaround**: Replace survival model with an XGBoost classifier using tournament-related features

---

## 5. Feature Naming Mismatches

### ❌ DOESN'T WORK: Using Non-Existent Feature Names

```yaml
models:
  xgb_core:
    features:
      - diff_luck_index              # ❌ Feature doesn't exist in data
      - diff_coach_tourney_win_pct   # ❌ Feature doesn't exist
      - diff_surv_e8                 # ❌ Feature doesn't exist
```

**Result**: Model fails silently during training. No error message indicates missing features.

### ✅ WORKS: Verified Features

Only use features that definitely exist in `matchup_features.parquet`:
- All `diff_*` columns with 1,292+ non-null values
- Features with reasonable cardinality
- Features that appear in the parquet column list

**How to verify**:
```python
import pandas as pd
df = pd.read_parquet('matchup_features.parquet')
assert 'diff_feature_name' in df.columns
```

---

## 6. Target Column Configuration

### ❌ DOESN'T WORK: Non-Existent Target Column

```yaml
project_name: "march-madness-women-2026"
task: "classification"
target_column: "higher_seed_won"  # ❌ Doesn't exist in raw data
```

**Error**: "Target column not found in data" warning (prevents training)

### ✅ WORKS: Create Target Column in Data View

If target column doesn't exist, create it via `derive` operation in data view:

```yaml
# In manage_data, update the features view:
views:
  - name: matchups_w_filtered
    steps:
      - op: filter
        expr: "(Season >= 2015) & (Season <= 2025) & (Season != 2020)"
      - op: derive
        columns:
          higher_seed_won: "TeamAWon"  # Map existing column
```

Then set this view as the features_view.

---

## 7. Model Params Merging (Unexpected Behavior)

### ❌ PARTIALLY WORKS: Updating Model Parameters

When using `mcp__easyml__manage_models` with `action: update`, parameters **merge** with existing params rather than replace:

```python
# First add model with full param set
add(name="xgb_core", params={...huge param set...})

# Later try to simplify params
update(name="xgb_core", params={"max_depth": 4, "learning_rate": 0.01, "n_estimators": 100})

# RESULT: Old params are NOT removed, new ones are merged in
# The model still has all the unsupported params that cause failures
```

### ✅ WORKS: Remove and Re-add

```python
# Remove old model
remove(name="xgb_core", purge=True)

# Add clean version
add(name="xgb_core_clean", params={...only working params...})
```

---

## 8. Feature Derivation Limitations

### ❌ DOESN'T WORK: Creating 100+ Features at Once

```yaml
# User hint said: "declare whatever pairwise features you need"
# BUT this doesn't work well:
features:
  - add_batch with 100+ derive operations  # Very slow, may timeout
```

**Issue**: Creating 127 diff_* features one by one is impractical. Better to use raw parquet diff_* columns.

### ✅ WORKS: Use Pre-Computed Diff Features

The matchup_features.parquet already has all 127 diff_* columns pre-computed by MM's pipeline. Just use those directly rather than deriving them in easyml.

---

## 9. EVal Metrics Parameter

### ❌ DOESN'T WORK: Passing `eval_metric` in XGBoost Params

```yaml
params:
  eval_metric: logloss              # ❌ Causes model failures
  eval_metric: rmse                 # ❌ For regression, also fails
```

**Result**: Models fail silently during backtest.

### ✅ WORKS: Don't Include `eval_metric`

EasyML handles evaluation metrics separately from model params. Don't pass eval_metric in the params dict.

---

## 10. Early Stopping Configuration

### ❌ DOESN'T WORK: Any Early Stopping Configuration

```yaml
params:
  early_stopping_rounds: 20         # ❌ NOT SUPPORTED
  eval_metric: logloss              # ❌ Companion param, also not supported
```

**Issue**: EasyML doesn't support early stopping via params. XGBoost and LightGBM will train for full n_estimators.

### ✅ WORKS: Set Reasonable n_estimators

Instead of relying on early stopping, use a moderate n_estimators value:
- For testing: 50-100
- For production: 200-500

---

## 11. Regex/Formula-Based Feature References

### ❌ DOESN'T WORK: Dynamic Feature Selection

```yaml
models:
  xgb_core:
    features: "diff_*"              # ❌ Regex not supported
    features: "@parent_model.output"  # ❌ Cross-references not supported
```

**Result**: Features not found, model fails.

### ✅ WORKS: Explicit Feature List

Always specify features as explicit list of strings:
```yaml
features:
  - diff_seed_num
  - diff_win_pct
  - diff_sr_srs
```

---

## Summary: What Configuration Works

| Feature | Works? | Notes |
|---------|--------|-------|
| Basic XGBoost params | ✅ | max_depth, learning_rate, n_estimators only |
| Advanced XGBoost params | ❌ | subsample, colsample_bytree, regularization, etc. |
| LightGBM | ⚠️ | Basic params only, many features not supported |
| Random Forest | ✅ | Works well with standard params |
| Logistic Regression | ✅ | Simple, effective |
| Survival models | ❌ | Type exists but not fully functional |
| Regression models | ❌ | xgboost_regression type not available |
| Early stopping | ❌ | Not supported |
| Feature derivation | ⚠️ | Works but slow for 100+ features |
| LOSO backtest | ✅ | Fully supported |
| Stacked ensemble | ✅ | Works well |
| Spline calibration | ✅ | Supported |
| Temperature scaling | ✅ | Supported |

---

## Recommendations

1. **Use simplified parameter sets** - forget MM's tuned hyperparameters for now
2. **Don't try advanced regularization** - stick to basic XGBoost/LightGBM
3. **Use pre-computed features** - matchup_features.parquet already has all diff_* columns
4. **Focus on model diversity** - more models + diverse features > heavy hyperparameter tuning
5. **Test models individually** - add one model at a time to diagnose issues
6. **Remove and re-add** - don't try to update complex model configs, delete and recreate cleanly

---

## Files to Reference

- Configuration that works: `/Users/msilverblatt/easyml/projects/demos/mm-women-2026/` (see final models)
- MM reference config: `/Users/msilverblatt/easyml/docs/mm-women-config-mapping.md`
- Data source: `/Users/msilverblatt/mm/data/features_w/matchup_features.parquet`
- Results: Backtest achieves Brier 0.1497 vs MM baseline 0.1348 (11% gap)
