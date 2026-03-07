# Multiclass Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the full pipeline (training → prediction → ensemble → metrics → diagnostics → reporting) work for multiclass classification tasks (3+ classes).

**Architecture:** The pipeline currently assumes binary classification everywhere — 1D probability arrays, binary metrics (brier, ECE), binary-specific ensemble (logistic meta-learner), and matchup-style reporting. We need a clean multiclass branch at each stage, detected via `config.data.task == "multiclass"`, that handles probability matrices, multiclass metrics, and multiclass-aware ensembling.

**Tech Stack:** numpy, pandas, scikit-learn (LogisticRegression with multinomial), existing MetricRegistry multiclass metrics.

**Prep work already done (committed):**
- Model wrappers return full `(n_samples, n_classes)` probability matrix when >2 classes
- Hardcoded `"result"` column references replaced with `config.data.target_column`
- Empty features default to all available numeric columns
- Better error messages on backtest failure

---

### Task 1: Pipeline prediction storage for multiclass

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/pipeline.py`
- Test: `packages/harness-core/tests/runner/test_pipeline_multiclass.py`

**Problem:** Line ~1192 does `preds_df[f"prob_{model_name}"] = probs`. For multiclass, `probs` is shape `(n_samples, n_classes)` — can't assign a 2D array to a single DataFrame column.

**Design:**
- Add a helper `_is_multiclass(self)` that returns `self.config.data.task == "multiclass"`
- When multiclass:
  - Store per-class columns: `prob_{model}_c0`, `prob_{model}_c1`, `prob_{model}_c2`, etc.
  - Do NOT store a `prob_{model}` column (avoid ambiguity)
- When binary: keep existing behavior (single `prob_{model}` column)
- Update prediction cache storage (line ~1184) similarly — store multiclass predictions as separate columns or as a numpy array

**Also update:**
- `_generate_predictions_for_fold` (line ~1025-1033 path): same multiclass storage
- The `prob_cols` detection throughout: `[c for c in df.columns if c.startswith("prob_")]` needs to distinguish per-model base columns from per-class columns. Convention: `prob_{model}_c{i}` for per-class, `prob_{model}` for binary.

**Step 1:** Write failing test — create a mock 3-class pipeline scenario, verify predictions stored correctly.

**Step 2:** Implement the multiclass branch in `_generate_predictions_for_fold`.

**Step 3:** Run tests.

---

### Task 2: Multiclass ensemble (average method)

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/pipeline.py`
- Test: `packages/harness-core/tests/runner/test_pipeline_multiclass.py`

**Problem:** The ensemble path (lines ~878-891) does `prob_cols.mean(axis=1)` which averages across all `prob_*` columns. For multiclass, we need to average per-class probabilities across models, then produce `prob_ensemble_c0`, `prob_ensemble_c1`, etc.

**Design:**
- When multiclass + average ensemble:
  - For each class `c`, find all `prob_{model}_c{c}` columns → average them → store as `prob_ensemble_c{c}`
  - Also store `prob_ensemble` as the argmax class (for accuracy metrics)
  - Detect n_classes from the column names
- When multiclass + stacked ensemble:
  - Train a multinomial logistic regression meta-learner
  - Input features: all `prob_{model}_c{i}` columns (flattened)
  - Output: per-class ensemble probabilities
  - This is Task 3 (separate)

**Step 1:** Write failing test for average ensemble with 3 classes.

**Step 2:** Implement the multiclass average ensemble branch.

**Step 3:** Run tests.

---

### Task 3: Multiclass stacked meta-learner

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/meta_learner.py`
- Modify: `packages/harness-core/src/harnessml/core/runner/pipeline.py`
- Test: `packages/harness-core/tests/runner/test_meta_learner_multiclass.py`

**Problem:** `train_meta_learner_loso` uses binary logistic regression. For multiclass, needs multinomial.

**Design:**
- Add `n_classes` parameter to `train_meta_learner_loso` (default 2 for backward compat)
- When `n_classes > 2`:
  - Input features: per-class probabilities from all models, flattened
  - Use `LogisticRegression(multi_class="multinomial", solver="lbfgs")`
  - Output: per-class calibrated probabilities
  - Skip binary-specific calibration (Platt, isotonic, spline) — those are binary-only
  - Return coefficients per class for reporting
- Pipeline `_apply_stacked_ensemble`: detect multiclass, pass n_classes, store per-class ensemble probs
- Pipeline `_train_meta_for_fold`: same multiclass handling

**Step 1:** Write failing test for multinomial meta-learner.

**Step 2:** Implement multiclass branch in meta-learner.

**Step 3:** Wire into pipeline.

**Step 4:** Run tests.

---

### Task 4: Multiclass metrics routing

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/pipeline.py` (`_compute_backtest_metrics`)
- Modify: `packages/harness-core/src/harnessml/core/runner/diagnostics.py`
- Test: `packages/harness-core/tests/runner/test_pipeline_multiclass.py`

**Problem:** `_compute_backtest_metrics` (line ~1336-1338) builds `per_fold_data` with `"y"` and `"preds"` assuming binary (1D arrays). `BacktestRunner` and diagnostics functions compute binary metrics (brier, ECE).

**Design:**
- `_compute_backtest_metrics`: when multiclass:
  - `"y"` stays the same (integer class labels)
  - `"preds"` should include per-class probability arrays
  - Use MetricRegistry multiclass metrics: `accuracy`, `log_loss`, `f1_macro`, `f1_weighted`
  - Skip binary-only metrics: `brier`, `ece`, `auc_roc` (single-class AUC doesn't apply)
- `evaluate_fold_predictions` in diagnostics.py: when multiclass:
  - Compute accuracy from argmax of per-class probabilities vs true labels
  - Compute log_loss using full probability matrix
  - Skip brier_score and ECE (binary-only)
- `compute_pooled_metrics`: same multiclass branch

**Step 1:** Write failing test — multiclass metrics computed correctly.

**Step 2:** Implement multiclass branches.

**Step 3:** Run tests.

---

### Task 5: Multiclass diagnostics and reporting

**Files:**
- Modify: `packages/harness-core/src/harnessml/core/runner/diagnostics.py`
- Modify: `packages/harness-core/src/harnessml/core/runner/reporting.py`
- Modify: `packages/harness-core/src/harnessml/core/runner/config_writer.py` (`show_diagnostics`)
- Test: `packages/harness-core/tests/runner/test_pipeline_multiclass.py`

**Problem:** `build_pick_log` assumes binary matchup (A vs B). Calibration curves assume binary. `show_diagnostics` assumes binary prob columns.

**Design:**
- `build_pick_log`: skip entirely for multiclass (it's matchup-specific). Guard with `if task != "multiclass"`.
- `build_diagnostics_report`: when multiclass:
  - Show per-class accuracy breakdown (confusion matrix style)
  - Show per-model accuracy and log_loss (skip brier, ECE)
  - Show classification report (precision/recall/f1 per class)
- `show_diagnostics` in config_writer: detect task type from config, pass to diagnostics functions
- Calibration curves: skip for multiclass (or do per-class calibration later as enhancement)

**Step 1:** Write failing test — diagnostics for multiclass pipeline.

**Step 2:** Implement multiclass guards and alternative reports.

**Step 3:** Run tests.

---

### Task 6: End-to-end integration test

**Files:**
- Create: `packages/harness-core/tests/runner/test_multiclass_e2e.py`

**Design:** Full pipeline test with synthetic 3-class data:
1. Create a DataFrame with 1000 rows, 10 numeric features, 3-class target, 5 folds
2. Write pipeline.yaml with `task: multiclass`
3. Add 2 models (xgboost, logistic) with no explicit features (test auto-detection)
4. Run backtest via `run_backtest()`
5. Verify:
   - Result has `status: success`
   - Metrics include `accuracy`, `log_loss` (not `brier`)
   - Per-fold breakdown exists
   - Models trained list is correct
   - No crashes or KeyErrors

**Step 1:** Write the full e2e test.

**Step 2:** Run it — it should pass if Tasks 1-5 are done correctly.

**Step 3:** Fix any integration issues.

---

## Dependency Order

```
Task 1 (storage) → Task 2 (average ensemble) → Task 4 (metrics)
                 → Task 3 (stacked ensemble)  → Task 5 (diagnostics)
                                                → Task 6 (e2e test)
```

Tasks 1 must be done first. Tasks 2 and 3 can be parallel. Task 4 depends on 1+2. Task 5 depends on 4. Task 6 depends on all.

## What this does NOT cover (future work)

- Multiclass calibration (per-class Platt/isotonic)
- Multiclass SHAP explanations
- Multiclass prediction inspection (inspect_predictions)
- Multiclass feature discovery
- MLP multiclass (requires architecture change from single sigmoid output to softmax)
- Ordinal classification (ordered multiclass)
