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

### Phase 1: Data Preparation (Offline)

**Goal:** Ensure data quality, temporal integrity, and source freshness *before* touching features or models.

**Why first?** Bad data corrupts everything downstream. Temporal issues create invisible leakage (hardest to debug). Fix now, not later.

**Steps:**
1. Register raw data sources via `data(action="add_source", name="...", data_path="...")`
2. Ingest into feature store: `data(action="add", data_path="...", join_on=[...], prefix="...")`
3. Profile data: `data(action="profile")` — check types, null rates, distributions
4. Check freshness: `data(action="check_freshness")` — verify sources are current
5. Validate sources: `data(action="validate_source", name="...")` — schema checks
6. Resolve issues: drop duplicates, fill nulls (single or batch), rename columns

**Red flags:** If your data has temporal issues or heavy leakage, all downstream models fail.

---

### Phase 2: Feature Engineering (Exploratory)

**Goal:** Discover transformations and combinations that improve predictive power. Curate a diverse, high-quality feature set.

**Key principle:** Features are cheap to compute. Be aggressive about exploring.

**Why before models?** Good features make all downstream models better. It's cheaper to improve features than tune hyperparameters.

**Steps:**
1. Ingest base features from raw data sources
2. Test transformations: `features(action="test_transformations", ...)`
   - Log, sqrt, rank, z-score, interactions
   - Returns which transforms improve correlation most
3. Auto-search for features: `features(action="auto_search", features=[...], search_types=["interactions","lags","rolling"])`
   - Automatically discovers interactions, lag features, and rolling aggregations
4. Add winning transformations
5. Discover important features: `features(action="discover", ...)`
   - Correlation analysis with target
   - XGBoost feature importance (what models will use)
   - Redundancy detection (drop correlated pairs)
6. Check diversity: `features(action="diversity")` — ensure models use different feature sets
7. Create composite features (pairwise differences, ratios, interactions)
8. Define regimes (context flags that gate feature sets)

**Available formula functions:** abs, log, sqrt, cbrt, clip, log1p, sign, square, reciprocal, exp, expm1, power, sin_cycle, cos_cycle, zscore, minmax, rank_pct, winsorize, maximum, minimum, where, isnull, safe_div, pct_of_total

**Selection criteria:** Choose top 20-30 features based on:
- Correlation with target
- Feature importance
- Diversity (different types, sources, patterns)

**Red flags:** If median feature correlation < 0.3, revisit data quality or feature engineering.

---

### Phase 3: Model Selection (Structured)

**Goal:** Find which model architectures generalize best on holdout data. Use CV to ensure honest evaluation.

**Key principle:** Preset defaults are sensible. Don't tweak hyperparameters yet—just pick good base models.

**Why before hyperparams?** Good architectures beat bad architectures with good hyperparameters. Diversity improves ensembles.

**Steps:**
1. Configure backtest: `configure(action="backtest", cv_strategy="...", seasons=[...], metrics=[...])`
2. Add baseline model: `models(action="add", name="xgb_baseline", preset="xgboost_classifier", ...)`
3. Add comparison models: Try different architectures (XGB, LGB, MLP)
4. Clone and tweak: `models(action="clone", name="xgb_baseline", ...)` — clone with overrides
5. Run backtest: `pipeline(action="run_backtest", ...)`
6. Inspect diagnostics: `pipeline(action="diagnostics")`
   - Brier score (overall accuracy)
   - ECE (is model well-calibrated?)
   - Model agreement (do they learn different patterns?)
7. Compare runs: `pipeline(action="compare_runs", run_ids=["run-001", "run-002"])`
8. Keep top performers, disable underperformers
9. Configure ensemble: `configure(action="ensemble", method="stacked|average", ...)`
   - Calibration options: spline, isotonic, platt, beta, none
   - Per-model pre-calibration: `pre_calibration={"model_name": "platt"}`

**Red flags:**
- If all models have similar performance, you may have weak features (revisit Phase 2)
- If ECE > 0.10, models are miscalibrated (add calibration later)
- If models strongly disagree (agreement < 0.5), ensemble may not help much

---

### Phase 4: Hyperparameter Tuning (Constrained, Last)

**Goal:** Fine-tune best model architectures within computational budget.

**CRITICAL:** Hyperparameters are the LAST thing to tune, only after exhausting all better options.

**You should tune only if:**
- Data quality is validated (Phase 1)
- Features selected and tested (Phase 2)
- Model architectures chosen (Phase 3)
- Baseline metrics established

**You should NOT tune if:**
- Features are weak (Phase 2 incomplete)
- Models show obvious overfitting
- CV metrics vary wildly (Phase 1 issues)
- You haven't tried different architectures

**Approaches:**

1. **Manual Single-Variable** (slower, more interpretable)
   - `experiments(action="create", description="...", hypothesis="...")`
   - `experiments(action="write_overlay", experiment_id="...", overlay={...})`
   - `experiments(action="run", experiment_id="...")`
   - Compare to baseline

2. **Bayesian Exploration** (recommended, faster)
   - `experiments(action="explore", search_space={axes: [...], budget: 50, primary_metric: "brier"})`
   - Returns best hyperparams, parameter importance, full trial history
   - Prediction cache shared across trials (unchanged models never retrain)

3. **Quick Run** (one-shot experiment)
   - `experiments(action="quick_run", description="...", overlay={...})`
   - Creates, configures, and runs in one call

**Expected ROI:**
- Phase 2 improvements: 5-20% metric gain (high ROI)
- Phase 3 improvements: 2-10% via architecture/diversity (medium ROI)
- Phase 4 improvements: 0.5-2% via hyperparams (low ROI, use only if available budget)

---

## Available MCP Tools

### Data Management (`data`)

**Ingestion & Profiling:**
- `action="add"` — Ingest CSV/parquet/Excel into feature store. Params: `data_path`, `join_on`, `prefix`, `auto_clean`
- `action="validate"` — Preview dataset without ingesting. Params: `data_path`
- `action="profile"` — Summary statistics per column. Optional: `category`
- `action="status"` — Quick overview (row/col count, target distribution, time range)
- `action="list_features"` — List available feature columns. Optional: `prefix`

**Data Cleaning:**
- `action="fill_nulls"` — Fill nulls in one column. Params: `column`, `strategy` (median/mean/mode/zero/value), `value`
- `action="fill_nulls_batch"` — Fill nulls in multiple columns at once
- `action="drop_duplicates"` — Remove duplicates. Optional: `columns` (subset)
- `action="rename"` — Rename columns. Params: `mapping` (JSON `{"old": "new"}`)

**Source Registry:**
- `action="add_source"` — Register a raw data source. Params: `name`, `data_path`, `format`
- `action="add_sources_batch"` — Register multiple sources at once
- `action="list_sources"` — List all registered sources
- `action="check_freshness"` — Check source staleness against frequency expectations
- `action="refresh"` — Re-fetch a specific source. Params: `name`
- `action="refresh_all"` — Re-fetch all sources
- `action="validate_source"` — Run schema validation on a source. Params: `name`

**Views (Transform Chains):**
- `action="add_view"` — Declare a view. Params: `name`, `source`, `steps` (JSON array), `description`
- `action="add_views_batch"` — Declare multiple views at once
- `action="update_view"` — Update existing view. Params: `name`, `source`, `steps`, `description`
- `action="remove_view"` — Remove a view. Params: `name`
- `action="list_views"` — List all views with descriptions
- `action="preview_view"` — Materialize and show first N rows. Params: `name`, `n_rows`
- `action="set_features_view"` — Set which view becomes prediction table. Params: `name`
- `action="view_dag"` — Show view dependency graph

**Available view step ops:** filter, select, derive, group_by, join, union, unpivot, sort, head, rolling, cast, distinct, rank, isin, cond_agg, lag, ewm, diff, trend, encode, bin, datetime, null_indicator

### Feature Engineering (`features`)
- `action="add"` — Create a feature. Params: `name`, `type` (team/pairwise/matchup/regime), `formula`, `source`, `column`, `condition`, `pairwise_mode`, `category`, `description`
- `action="add_batch"` — Create multiple features with topological ordering. Params: `features` (JSON array)
- `action="test_transformations"` — Test math transforms. Params: `features` (column names), `test_interactions`
- `action="discover"` — Run feature discovery. Params: `method` (xgboost/mutual_info), `top_n`
- `action="diversity"` — Analyze feature diversity across models
- `action="auto_search"` — Auto-search for features. Params: `features`, `search_types` (interactions/lags/rolling), `top_n`

### Model Management (`models`)
- `action="add"` — Add a model. Params: `name`, `preset` or `model_type`, `features`, `params`, `mode`, `prediction_type`, `cdf_scale`, `zero_fill_features`
- `action="update"` — Update model config. Same params as add (merges)
- `action="remove"` — Disable model. Params: `name`, `purge` (permanent delete)
- `action="clone"` — Clone model with overrides. Params: `name`, plus any override params
- `action="list"` — List all models with type, status, feature count
- `action="presets"` — Show available model presets
- `action="add_batch"` — Add multiple models. Params: `items` (JSON array)
- `action="update_batch"` — Update multiple models. Params: `items`
- `action="remove_batch"` — Remove multiple models. Params: `items`

### Configuration (`configure`)
- `action="init"` — Initialize new project. Params: `project_name`, `task`, `target_column`, `key_columns`, `time_column`
- `action="ensemble"` — Update ensemble. Params: `method` (stacked/average), `temperature`, `exclude_models`, `calibration` (spline/isotonic/platt/beta/none), `pre_calibration`, `prior_feature`, `spline_prob_max`, `spline_n_bins`
- `action="backtest"` — Update backtest. Params: `cv_strategy`, `seasons`, `metrics`, `min_train_folds`
- `action="show"` — Show full config. Optional: `section`, `detail`
- `action="check_guardrails"` — Run safety guardrails (leakage, naming, model config)
- `action="exclude_columns"` — Manage excluded columns. Params: `add_columns`, `remove_columns`
- `action="set_denylist"` — Manage feature leakage denylist. Params: `add_columns`, `remove_columns`

### Experiments (`experiments`)
- `action="create"` — Create experiment. Params: `description`, `hypothesis`
- `action="write_overlay"` — Write overlay YAML. Params: `experiment_id`, `overlay` (JSON, supports dot-notation keys)
- `action="run"` — Run experiment backtest. Params: `experiment_id`, `primary_metric`, `variant`
- `action="promote"` — Promote experiment to production. Params: `experiment_id`, `primary_metric`
- `action="quick_run"` — One-shot create+configure+run. Params: `description`, `overlay`, `hypothesis`, `primary_metric`
- `action="explore"` — Bayesian search. Params: `search_space` (JSON with axes, budget, primary_metric)
- `action="promote_trial"` — Promote exploration trial. Params: `experiment_id`, `trial`, `primary_metric`, `hypothesis`
- `action="compare"` — Compare two experiments. Params: `experiment_ids` (list of 2)

### Pipeline (`pipeline`)
- `action="run_backtest"` — Run full backtest. Optional: `experiment_id`, `variant`
- `action="predict"` — Generate predictions. Params: `season`, `run_id`, `variant`
- `action="diagnostics"` — Per-model metrics, calibration, SHAP. Optional: `run_id`, `detail`
- `action="list_runs"` — List all pipeline runs
- `action="show_run"` — Show run results. Optional: `run_id`, `detail`
- `action="compare_runs"` — Compare two runs. Params: `run_ids` (list of 2)

---

## Workflow Patterns

### Pattern: Project Initialization

```
1. configure(action="init", project_name="...", task="binary", target_column="...")
2. data(action="add_source", name="...", data_path="...")  # register sources
3. data(action="add", data_path="...")  # ingest into feature store
4. data(action="profile")  # check for issues
5. data(action="check_freshness")  # verify data is current
6. features(action="discover")  # what's useful?
7. configure(action="backtest", cv_strategy="...", seasons=[...])
8. models(action="add", preset="xgboost_classifier", ...)
9. pipeline(action="run_backtest")  # establish baseline
```

### Pattern: Feature Engineering Cycle

```
1. features(action="test_transformations", features=[...])
   -> Which transforms improved correlation?
2. features(action="auto_search", features=[...], search_types=["interactions","lags","rolling"])
   -> Automated discovery of interactions, lags, rolling features
3. features(action="add", name="...", formula="...", ...)
   -> Add winning transforms
4. features(action="discover", method="xgboost", top_n=30)
   -> Which features does XGBoost think matter?
5. features(action="diversity")
   -> Are models using diverse feature sets?
6. models(action="update", name="xgb_baseline", features=[...])
   -> Add top features to models
7. pipeline(action="run_backtest")  # did metrics improve?
8. Repeat or advance to Phase 3
```

### Pattern: Model Selection

```
1. Add baseline: models(action="add", preset="xgboost_classifier", ...)
2. Add comparison: models(action="add", preset="lightgbm_classifier", ...)
3. Clone variant: models(action="clone", name="xgb_baseline", ...)
4. pipeline(action="run_backtest")  # compare architectures
5. pipeline(action="diagnostics")  # check calibration, agreement
6. pipeline(action="compare_runs", run_ids=["run-001", "run-002"])  # compare runs
7. Disable underperformers: models(action="update", name="...", active=false)
8. configure(action="ensemble", method="stacked", calibration="spline")
```

### Pattern: Hyperparameter Tuning (Bayesian)

```
experiments(action="explore", search_space={
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

- **Data first** — Temporal issues and leakage corrupt everything. Validate before moving on.
- **Features second** — Good features beat tuned hyperparameters. Explore aggressively.
- **Architectures third** — Different models learn different patterns. Diversity improves ensembles.
- **Hyperparams last** — Only tune after everything else is solid. Low ROI anyway.

- **One variable per experiment** — Change one thing, measure impact.
- **Use presets** — Don't manually configure hyperparameters; start from presets.
- **Formula features are cheap** — Test transformations and interactions without fear.
- **Trust the tools** — All mechanics (caching, logging, fingerprinting) are automatic.
- **Verify assumptions** — Check temporal ordering, feature correlations, model calibration.

---

## Common Pitfalls (Avoid These!)

**Jumping to hyperparameter tuning before features are good**
- Features with correlation < 0.3 to target = problem in Phase 2, not Phase 4
- Tuning bad features won't help

**Mutating production config directly**
- Always use experiment overlays
- Revert/promote workflow keeps history clean

**Training models on post-tournament data for tournament prediction**
- Hard guardrail blocks this automatically
- Temporal safety is non-overridable

**Running single experiment then declaring victory**
- CV ensures honest evaluation
- One fold can be lucky; cross all folds

**Ignoring model calibration (ECE > 0.10)**
- Miscalibrated probabilities mislead downstream users
- Add post-calibration (platt, isotonic, spline, beta) if needed

---

## Further Reading

- [GETTING_STARTED.md](../../../../GETTING_STARTED.md) — Complete workflow guide with examples
- [README.md](../../../../README.md) — System overview
- [CLAUDE.md](../../../../CLAUDE.md) — Dev conventions
