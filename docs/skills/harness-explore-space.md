# Skill: Explore the Model Space

**Priority: HIGHEST** — This is the most important skill in HarnessML. It prevents the single most common failure mode in ML projects: rushing to hyperparameter tuning after trying 2-3 models and leaving massive performance gains on the table.

**Applies to:** Any ML task type (binary classification, multiclass, regression, ranking, survival, probabilistic forecasting).

---

## Why This Exists

Agents naturally gravitate toward tuning what they already have rather than exploring what they haven't tried. This leads to:
- Ensembles dominated by a single model family (e.g., 3 XGBoost variants)
- Missed diversity that would reduce ensemble variance
- Premature optimization on hyperparameters when feature engineering would yield 10x the gain
- Dismissing model types after a single failed configuration

This skill enforces a **phased workflow with hard gates** between phases. You may not advance to a later phase until the gate conditions of the current phase are met.

---

## Phase 1: EDA & Feature Discovery

**Goal:** Understand the data and feature landscape before touching any models.

### Steps

1. **Inspect the data** — understand shape, types, missing values, target distribution:
   ```
   data(action="inspect")
   data(action="list_features")
   ```

2. **Run feature discovery** — get correlations between all available features and the target:
   ```
   features(action="discover")
   ```

3. **Run auto-search** — find candidate interaction, lag, and rolling features systematically:
   ```
   features(action="auto_search", search_types=["interactions", "lags", "rolling"])
   ```

4. **Document findings** — before any modeling, write down:
   - How many features are available
   - Which features have the strongest signal
   - What the target distribution looks like
   - Any obvious data quality issues (high nulls, leakage risks, class imbalance)

### Gate: Phase 1 -> Phase 2
- Feature discovery has been run and results reviewed
- Auto-search has been run
- Data quality issues are documented

---

## Phase 2: Baseline & Model Diversity

**Goal:** Establish a baseline, then systematically try EVERY available model type before settling on an ensemble. You need genuine diversity — not 5 gradient boosted tree variants.

### Steps

1. **Pick the right CV strategy** before any modeling — let the framework suggest one based on your data:
   ```
   configure(action="suggest_cv")
   ```

2. **Establish a baseline** — start with the simplest model that makes sense:
   ```
   models(action="add", model_type="logistic_regression", name="lr_baseline")
   pipeline(action="run_backtest")
   ```
   Record baseline metrics. This is your floor.

3. **Systematically try every model family:**

   **Linear models:**
   ```
   models(action="add", model_type="logistic_regression", ...)
   models(action="add", model_type="elastic_net", ...)
   ```

   **Gradient boosted trees (try ALL three — they have different strengths):**
   ```
   models(action="add", model_type="xgboost", ...)
   models(action="add", model_type="lightgbm", ...)
   models(action="add", model_type="catboost", ...)
   ```

   **Bagging:**
   ```
   models(action="add", model_type="random_forest", ...)
   ```

   **Neural networks (evaluate feasibility based on dataset size):**
   ```
   models(action="add", model_type="mlp", ...)
   models(action="add", model_type="tabnet", ...)
   ```

4. **Give each model type a fair shot** — at least 2-3 configurations before dismissing any model type. Vary:
   - Feature subsets (not every model needs every feature)
   - Key structural parameters (depth, regularization strength)
   - Do NOT do full hyperparameter sweeps yet — just enough to evaluate the model family

5. **Check diversity regularly:**
   ```
   features(action="diversity")
   ```
   If two models have >0.95 prediction correlation, they are not adding diversity. Differentiate them with different feature sets or replace one.

6. **Target: 4-6 diverse models in the ensemble** before any hyperparameter tuning.

### Gate: Phase 2 -> Phase 3
- At least 4 distinct model types have been tried
- Each dismissed model type got at least 2-3 configurations
- Diversity analysis has been run
- Baseline metrics are recorded for comparison

---

## Phase 3: Feature Engineering Deep Dive

**Goal:** Improve the feature set using domain knowledge and systematic experimentation.

### Steps

1. **Snapshot before big changes** — save a snapshot of your current config and features so you can restore if feature engineering goes sideways:
   ```
   data(action="snapshot", name="pre_feature_engineering")
   ```

2. **Domain research** — use the harness-domain-research skill to generate hypothesis-driven features based on domain expertise for the problem at hand.

3. **Test features individually** — use single-variable experiments to isolate the impact of each new feature:
   ```
   experiments(action="create", name="test_feature_X", changes={"features": {"add": [...]}})
   experiments(action="run", name="test_feature_X")
   ```

4. **Systematic interaction discovery:**
   ```
   features(action="auto_search", features=[...], search_types=["interactions"])
   ```

5. **Try different feature sets per model** — this is a key source of genuine ensemble diversity. A linear model may benefit from binned features that a tree model does not need.

6. **Re-run diversity analysis** after feature changes:
   ```
   features(action="diversity")
   ```

### Gate: Phase 3 -> Phase 4
- Domain-driven feature hypotheses have been explored
- Feature experiments have been run and results logged
- Diversity analysis confirms models are still diverse

---

## Phase 4: Hyperparameter Tuning

**ONLY begin this phase after Phases 1-3 are complete.**

### Hard Prerequisites
- 4+ model types have been attempted
- Feature discovery and engineering are done
- You have a stable ensemble with diverse models

### Steps

1. **Tune one model at a time** — use Bayesian exploration:
   ```
   experiments(action="explore", search_space="{ ... }")
   ```

2. **Check feature importance** — use builtin importances for a fast check (no SHAP dependency needed), or SHAP for richer explanations:
   ```
   pipeline(action="explain", method="builtin")
   pipeline(action="explain", method="shap")
   ```

3. **Compare tuned vs. untuned** — measure the actual gain from tuning. If the gain is marginal, stop tuning that model and move on.

4. **Watch for diminishing returns** — the first round of tuning gives the biggest gains. A second round rarely justifies the time.

5. **Re-check diversity after tuning** — aggressive tuning can cause models to converge in behavior:
   ```
   features(action="diversity")
   ```

---

## Phase 5: Ensemble Optimization

**Goal:** Squeeze the last bit of performance from the ensemble composition itself.

### Steps

1. **Calibration experiments** — try different calibration methods and evaluate:
   ```
   experiments(action="create", name="calibration_test", changes={"calibration": "isotonic"})
   ```

2. **Temperature tuning** — ensemble-level adjustments only.

3. **LOMO (Leave-One-Model-Out) analysis** — for each model in the ensemble, measure the impact of removing it. Drop models that hurt or add nothing:
   ```
   pipeline(action="diagnostics")
   ```

4. **Final comparison:**
   ```
   pipeline(action="compare_runs")
   ```

---

## Progress Checklist

Track this checklist throughout the workflow. Do not skip items.

```
[ ] Phase 1: Feature discovery completed
[ ] Phase 1: Auto-search completed
[ ] Phase 2: Baseline established (metrics: ___)
[ ] Phase 2: Linear models tried (logistic_regression, elastic_net)
[ ] Phase 2: Tree models tried (xgboost, lightgbm, catboost)
[ ] Phase 2: Bagging tried (random_forest)
[ ] Phase 2: Neural nets evaluated (mlp, tabnet)
[ ] Phase 2: 4+ model types in ensemble
[ ] Phase 2: Diversity analysis run
[ ] Phase 3: Domain research completed
[ ] Phase 3: Feature engineering experiments run
[ ] Phase 4: Hyperparameter tuning (per model)
[ ] Phase 5: Calibration optimized
[ ] Phase 5: Ensemble finalized
```

---

## Red Flags — If You Think This, STOP

These thoughts are signals that you are about to skip critical exploration:

| Thought | Why It Is Wrong |
|---------|-----------------|
| "Let me just tune the XGBoost and call it done" | You have not tried all model types. Tune AFTER you have diversity. |
| "CatBoost regressed, let's skip it" | One bad config does not condemn a model family. Try at least 3 configurations with different feature sets and parameters before dismissing. |
| "We have 3 models, let's start tuning" | 3 is not enough. Try more model types — you need genuine diversity across model families. |
| "The baseline is already good enough" | Good enough for what? You have not explored the space. The baseline exists to measure improvement, not to be the final answer. |
| "Neural nets won't work on this dataset" | Maybe, but prove it. Try at least 2 configurations. Small datasets can still benefit from MLP with proper regularization. |
| "These two XGBoost variants give us diversity" | Two models from the same family are not diverse. Check prediction correlations. |
| "Feature engineering can wait, let me get the models right first" | Features drive more performance gain than hyperparameters in almost every case. Do not defer feature work. |

---

## Key Principles

1. **Breadth before depth.** Try all model families before tuning any single one.
2. **Features before hyperparameters.** A good feature is worth more than a tuned parameter.
3. **Diversity is the goal.** An ensemble of 4 diverse models beats an ensemble of 8 correlated ones.
4. **Give every model type a fair trial.** Minimum 2-3 configurations before dismissal.
5. **Measure everything.** Log results with `experiments(action="log_result")` and compare with `experiments(action="compare")`.
6. **Gates are non-negotiable.** Do not advance phases until gate conditions are met.
