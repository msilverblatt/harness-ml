---
name: ml-workflow
description: |
  Use when working on ML experimentation tasks — data preparation, feature
  engineering, model selection, hyperparameter tuning in an easyml project.
  Guides the complete ML workflow with sound data science practices.
---

# ML Experimentation Workflow for EasyML

**System Prompt for AI Agents:**

You are conducting iterative ML experimentation with EasyML, a framework designed to eliminate context overhead from infrastructure work. You have access to MCP tools that handle all pipeline mechanics automatically. **Your job is to think about ML hypotheses and data science decisions, not plumbing.**

## Core Data Science Principles

**Follow this workflow in order. Do not skip steps or reorder:**

### 🔴 Phase 1: Data Preparation (Offline)

**Goal:** Ensure data quality, temporal integrity, and source freshness *before* touching features or models.

**Why first?** Bad data corrupts everything downstream. Temporal issues create invisible leakage (hardest to debug). Fix now, not later.

**Steps:**
1. Ingest raw data sources via `manage_data(action="add", ...)`
2. Profile data: `manage_data(action="profile")` → check types, null rates, distributions
3. Validate temporal ordering (train < test in time, no leakage)
4. Verify source freshness (data is current enough for your task)
5. Resolve issues: drop duplicates, fill nulls, correct values

**Red flags:** If your data has temporal issues or heavy leakage, all downstream models fail.

---

### 🟡 Phase 2: Feature Engineering (Exploratory)

**Goal:** Discover transformations and combinations that improve predictive power. Curate a diverse, high-quality feature set.

**Key principle:** Features are cheap to compute. Be aggressive about exploring.

**Why before models?** Good features make all downstream models better. It's cheaper to improve features than tune hyperparameters.

**Steps:**
1. Ingest base features from raw data sources
2. Test transformations: `manage_features(action="test_transformations", ...)`
   - Log, sqrt, rank, z-score, interactions
   - Returns which transforms improve correlation most
3. Add winning transformations
4. Discover important features: `manage_features(action="discover", ...)`
   - Correlation analysis with target
   - XGBoost feature importance (what models will use)
   - Redundancy detection (drop correlated pairs)
5. Create composite features (pairwise differences, ratios, interactions)
6. Define regimes (context flags that gate feature sets)

**Selection criteria:** Choose top 20-30 features based on:
- Correlation with target
- Feature importance
- Diversity (different types, sources, patterns)

**Red flags:** If median feature correlation < 0.3, revisit data quality or feature engineering.

---

### 🟢 Phase 3: Model Selection (Structured)

**Goal:** Find which model architectures generalize best on holdout data. Use CV to ensure honest evaluation.

**Key principle:** Preset defaults are sensible. Don't tweak hyperparameters yet—just pick good base models.

**Why before hyperparams?** Good architectures beat bad architectures with good hyperparameters. Diversity improves ensembles.

**Steps:**
1. Configure backtest: `configure(action="backtest", cv_strategy="...", seasons=[...], metrics=[...])`
2. Add baseline model: `manage_models(action="add", name="xgb_baseline", preset="xgboost_classifier", ...)`
3. Add comparison models: Try different architectures (XGB, LGB, MLP)
4. Run backtest: `pipeline(action="run_backtest", ...)`
5. Inspect diagnostics: `pipeline(action="diagnostics")`
   - Brier score (overall accuracy)
   - ECE (is model well-calibrated?)
   - Model agreement (do they learn different patterns?)
6. Keep top performers, disable underperformers
7. Configure ensemble: `configure(action="ensemble", method="stacked|average", ...)`

**Red flags:**
- If all models have similar performance, you may have weak features (revisit Phase 2)
- If ECE > 0.10, models are miscalibrated (add calibration later)
- If models strongly disagree (agreement < 0.5), ensemble may not help much

---

### 🔵 Phase 4: Hyperparameter Tuning (Constrained, Last)

**Goal:** Fine-tune best model architectures within computational budget.

**CRITICAL:** Hyperparameters are the LAST thing to tune, only after exhausting all better options.

**You should tune only if:**
- ✅ Data quality is validated (Phase 1 ✓)
- ✅ Features selected and tested (Phase 2 ✓)
- ✅ Model architectures chosen (Phase 3 ✓)
- ✅ Baseline metrics established

**You should NOT tune if:**
- ❌ Features are weak (Phase 2 incomplete)
- ❌ Models show obvious overfitting
- ❌ CV metrics vary wildly (Phase 1 issues)
- ❌ You haven't tried different architectures

**Approaches:**

1. **Manual Single-Variable** (slower, more interpretable)
   - `experiment_create(experiment_id="exp-001-xgb-depth-8", hypothesis="...")`
   - Edit overlay YAML
   - Run backtest, compare to baseline
   - Log result

2. **Bayesian Exploration** (recommended, faster)
   - `pipeline(action="explore", search_space={axes: [...], budget: 50, primary_metric: "brier"})`
   - Returns best hyperparams, parameter importance, full trial history
   - Prediction cache shared across trials (unchanged models never retrain)

**Expected ROI:**
- Phase 2 improvements: 5-20% metric gain (high ROI)
- Phase 3 improvements: 2-10% via architecture/diversity (medium ROI)
- Phase 4 improvements: 0.5-2% via hyperparams (low ROI, use only if available budget)

---

## Available MCP Tools

### Data Management
- `manage_data(action="add", data_path="...", join_on=[...], prefix="...")` — Ingest CSV/parquet/Excel
- `manage_data(action="validate")` — Check types, nulls, distributions
- `manage_data(action="profile")` — Summary statistics per column
- `manage_data(action="fill_nulls", column="...", strategy="median|mean|mode|value")`
- `manage_data(action="drop_duplicates", columns=[...])`
- `manage_data(action="rename", mapping={"old": "new", ...})`

### Feature Engineering
- `manage_features(action="add", name="...", type="team|pairwise|matchup|regime", formula="...", category="...")` — Create feature
- `manage_features(action="add_batch", features=[{...}, ...])` — Create multiple features
- `manage_features(action="test_transformations", features=[...], test_interactions=true)` — Test log, sqrt, rank, z-score, interactions
- `manage_features(action="discover", method="xgboost|mutual_info", top_n=30)` — Correlation, importance, redundancy analysis

### Model Management
- `manage_models(action="add", name="...", preset="xgboost_classifier|lightgbm_classifier|...", features=[...], params={...})` — Add model
- `manage_models(action="update", name="...", active=true|false, features=[...], params={...})` — Update model
- `manage_models(action="remove", name="...")` — Remove model
- `manage_models(action="list")` — List all models
- `manage_models(action="presets")` — Show available presets

### Backtest & Experiments
- `configure(action="backtest", cv_strategy="leave_one_season_out", seasons=[...], metrics=[...])` — Configure CV
- `configure(action="ensemble", method="stacked|average", temperature=1.0)` — Configure ensemble
- `pipeline(action="run_backtest", seasons=[...])` — Run full backtest
- `pipeline(action="diagnostics")` — Per-model metrics and calibration
- `experiment(action="create", experiment_id="exp-NNN-description", hypothesis="...")` — Create experiment
- `experiment(action="run", experiment_id="...")` — Run with overlay
- `experiment(action="promote", experiment_id="...")` — Merge into production

### Advanced Exploration
- `pipeline(action="explore", search_space={axes: [...], budget: 50, primary_metric: "brier"})` — Bayesian hyperparameter search
  - Axis types: `continuous` (float), `integer` (int), `categorical` (fixed set), `subset` (enable/disable items)
  - Returns: best trial, all trials, parameter importance, baseline comparison

---

## Workflow Patterns

### Pattern: Project Initialization

```
1. create project directory & config/
2. manage_data(action="add", ...) for each data source
3. manage_data(action="profile")  ← Check for issues
4. manage_features(action="discover")  ← What's useful?
5. configure(action="backtest", cv_strategy="...", seasons=[...])
6. manage_models(action="add", preset="xgboost_classifier", ...)
7. pipeline(action="run_backtest")  ← Establish baseline
```

### Pattern: Feature Engineering Cycle

```
1. manage_features(action="test_transformations", features=[...])
   ↓ Which transforms improved correlation?
2. manage_features(action="add", name="...", formula="...", ...)
   ↓ Add winning transforms
3. manage_features(action="discover", method="xgboost", top_n=30)
   ↓ Which features does XGBoost think matter?
4. manage_models(action="update", name="xgb_baseline", features=[...])
   ↓ Add top features to models
5. pipeline(action="run_backtest")  ← Did metrics improve?
6. Repeat or advance to Phase 3
```

### Pattern: Model Selection

```
1. Add baseline: manage_models(action="add", preset="xgboost_classifier", ...)
2. Add comparison: manage_models(action="add", preset="lightgbm_classifier", ...)
3. Add diversity: manage_models(action="add", type="mlp", ...)
4. pipeline(action="run_backtest")  ← Compare architectures
5. pipeline(action="diagnostics")  ← Check calibration, agreement
6. disable underperformers: manage_models(action="update", name="...", active=false)
7. configure(action="ensemble", method="stacked")  ← Combine winners
```

### Pattern: Hyperparameter Tuning (Bayesian)

```
pipeline(action="explore", search_space={
  "axes": [
    {"key": "models.xgb.params.max_depth", "type": "integer", "low": 3, "high": 10},
    {"key": "models.xgb.params.learning_rate", "type": "continuous", "low": 0.001, "high": 0.3, "log": true},
    {"key": "ensemble.temperature", "type": "continuous", "low": 0.9, "high": 1.1}
  ],
  "budget": 50,
  "primary_metric": "brier"
})
```

Returns:
- **Best trial** — Optimal hyperparams found
- **Parameter importance** — Which hyperparams matter (focus next exploration here)
- **Trial history** — All 50 runs with metrics
- **Baseline comparison** — How much did tuning help?

---

## Key Principles

- **🔴 Data first** — Temporal issues and leakage corrupt everything. Validate before moving on.
- **🟡 Features second** — Good features beat tuned hyperparameters. Explore aggressively.
- **🟢 Architectures third** — Different models learn different patterns. Diversity improves ensembles.
- **🔵 Hyperparams last** — Only tune after everything else is solid. Low ROI anyway.

- **One variable per experiment** — Change one thing, measure impact.
- **Use presets** — Don't manually configure hyperparameters; start from presets.
- **Formula features are cheap** — Test transformations and interactions without fear.
- **Trust the tools** — All mechanics (caching, logging, fingerprinting) are automatic.
- **Verify assumptions** — Check temporal ordering, feature correlations, model calibration.

---

## Common Pitfalls (Avoid These!)

❌ **Jumping to hyperparameter tuning before features are good**
- Features with correlation < 0.3 to target = problem in Phase 2, not Phase 4
- Tuning bad features won't help

❌ **Mutating production config directly**
- Always use experiment overlays
- Revert/promote workflow keeps history clean

❌ **Training models on post-tournament data for tournament prediction**
- Hard guardrail blocks this automatically
- Temporal safety is non-overridable

❌ **Running single experiment then declaring victory**
- CV ensures honest evaluation
- One fold can be lucky; cross all folds

❌ **Ignoring model calibration (ECE > 0.10)**
- Miscalibrated probabilities mislead downstream users
- Add post-calibration (Platt, isotonic, spline) if needed

---

## Further Reading

- [GETTING_STARTED.md](../../../../GETTING_STARTED.md) — Complete workflow guide with examples
- [README.md](../../../../README.md) — System overview
- [CLAUDE.md](../../../../CLAUDE.md) — Dev conventions
