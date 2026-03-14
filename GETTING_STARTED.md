# HarnessML Getting Started Guide

**For AI agents conducting iterative ML experimentation without context overhead.**

This guide covers the complete HarnessML workflow, from project initialization through Bayesian hyperparameter exploration, with emphasis on sound data science practices.

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Data Science Workflow](#data-science-workflow)
3. [Project Setup](#project-setup)
4. [Phase 1: Data Preparation](#phase-1-data-preparation)
5. [Phase 2: Feature Engineering](#phase-2-feature-engineering)
6. [Phase 3: Model Selection](#phase-3-model-selection)
7. [Phase 4: Hyperparameter Tuning](#phase-4-hyperparameter-tuning)
8. [Advanced: Bayesian Exploration](#advanced-bayesian-exploration)
9. [Guardrails & Safety](#guardrails--safety)

---

## Core Philosophy

HarnessML is built specifically for **iterative AI-driven ML** without context overhead. The system automates:

- **File I/O & Caching** — Fingerprinting ensures deterministic results; prediction caching prevents unnecessary retraining
- **Experiment Tracking** — Every run is logged with metrics, hypothesis, and changes; nothing gets lost
- **Data Hygiene** — Guardrails enforce temporal integrity and prevent data leakage automatically
- **Configuration Management** — YAML config is the single source of truth; overlays enable isolated experimentation

This means **you focus on data science decisions**, not infrastructure:

> ✅ "Should we try PCA on these features?" (data science)
> ❌ "Let me write a data pipeline and manage caching..." (infrastructure)

---

## Data Science Workflow

**The proper order is critical. Do not skip steps or reorder.**

### Stage 1: Data Preparation (Offline)

**Goal:** Ensure data quality, temporal integrity, and source freshness before touching features or models.

**Steps:**
1. Ingest raw data sources
2. Validate data types, null rates, value distributions
3. Check temporal ordering (no time travel in training)
4. Verify freshness (sources are current enough)
5. Detect and resolve obvious errors (duplicates, bad joins)

**Why this order?**
- Bad data corrupts everything downstream
- Temporal issues create invisible leakage (hardest to debug)
- Better to fix now than hunt for leakage bugs later

**Tools:**
- `data(action="add", data_path=...)` — Ingest sources
- `data(action="validate")` — Type/null/distribution checks
- `data(action="profile")` — Summary statistics per column
- `data(action="drop_duplicates")`
- `data(action="fill_nulls", strategy="median")`

---

### Stage 2: Feature Engineering (Exploratory)

**Goal:** Discover which transformations and combinations improve predictive power. Curate a diverse feature set.

**Key principle:** Features are cheap to compute. Be aggressive about exploring.

**Steps:**

1. **Ingest base features** — Load raw data columns as features
   ```yaml
   features:
     seed:
       type: team
       source: ratings
       column: seed
   ```

2. **Test transformations** — Log, square root, rank, z-score, interactions
   ```
   features(action="test_transformations",
                   features=["seed", "rating"],
                   test_interactions=true)
   ```
   - Returns which transforms improve correlation most
   - Automatically suggests winner for each feature

3. **Add winning transformations** — Only keep what improved correlation
   ```yaml
   features:
     log_seed:
       type: team
       formula: "log(seed + 1)"  # Address zero values
   ```

4. **Discover important features** — Use correlation and importance analysis
   ```
   features(action="discover",
                   method="xgboost",  # or "mutual_info"
                   top_n=30)
   ```
   - Shows correlation with target
   - XGBoost feature importance (what the models will use)
   - Mutual information (feature-target dependency)
   - Redundancy detection (correlated feature pairs)

5. **Create composite features** — Pairwise differences, ratios, interactions
   ```yaml
   features:
     diff_rating:
       type: pairwise
       formula: "home_rating - away_rating"

     home_court_advantage:
       type: pairwise
       formula: "diff_rating + 3.5"  # Add domain knowledge
   ```

6. **Define regimes** — Context flags that enable/disable feature sets
   ```yaml
   features:
     tournament_time:
       type: regime
       condition: "season_stage == 'tournament'"
   ```

**Why this order?**
- Understanding what features matter saves hyperparameter tuning time
- Transformations are fast; there's no cost to exploring
- Better features make all downstream models better
- Prevents "garbage in, garbage out" in model training

---

### Stage 3: Model Selection (Structured)

**Goal:** Find which model architectures generalize best on holdout data. Use CV to ensure honest evaluation.

**Key principle:** Preset defaults are sensible. Don't tweak hyperparameters yet—just pick good base models.

**Steps:**

1. **Configure backtest** — Define CV strategy, seasons, metrics
   ```
   configure(action="backtest",
             cv_strategy="leave_one_season_out",
             seasons=[2015, 2016, 2017, 2018, 2019, 2020],
             metrics=["brier", "accuracy", "log_loss"])
   ```

   **CV Strategies:**
   - `leave_one_season_out` — For each season, train on all previous, test on held-out season (default)
   - `expanding_window` — For each fold, use all prior rows (if no season boundaries)
   - `sliding_window` — Fixed train/test window sizes
   - `purged_kfold` — K-fold with temporal purging for overlapping data

2. **Add baseline model** — Simple, proven approach
   ```
   models(action="add",
                 name="xgb_baseline",
                 preset="xgboost_classifier",
                 features=["diff_rating", "diff_seed"])
   ```

3. **Add comparison models** — Test different architectures
   ```
   models(action="add", name="lgb_v1", preset="lightgbm_classifier", ...)
   models(action="add", name="mlp_v1", type="mlp", ...)
   ```

4. **Run backtest** — Get honest CV metrics on all models
   ```
   pipeline(action="run_backtest", seasons=[2015, 2016, 2017, 2018, 2019, 2020])
   ```

   Returns per-model metrics (brier, accuracy, log_loss, ece, etc.) across all folds.

5. **Inspect diagnostics** — Check for overfitting, miscalibration, disagreement
   ```
   pipeline(action="diagnostics")
   ```

   - **Brier score** — Overall probabilistic accuracy
   - **ECE** — Expected calibration error (are predicted probs honest?)
   - **Log loss** — Penalty for confident wrong predictions
   - **Model agreement** — Do models agree? (if agreement < 0.5, they're learning different patterns)

6. **Keep top performers** — Disable models with worse metrics
   ```
   models(action="update",
                 name="mlp_v1",
                 active=false)
   ```

7. **Configure ensemble** — Combine remaining models
   ```
   configure(action="ensemble",
             method="stacked",  # or "average"
             temperature=1.0)
   ```

**Why this order?**
- CV ensures honest evaluation (no leakage via tuning on test set)
- Different architectures learn different patterns (diversity helps)
- Good base models beat bad models with good hyperparameters
- Ensemble improves diversity; now it's time to optimize

---

### Stage 4: Hyperparameter Tuning (Constrained)

**Goal:** Fine-tune the best model architectures within computational budget.

**Key principle:** Hyperparameters are the LAST thing to tune, only after exhausting better options.

**When tuning makes sense:**
- You've selected a base model with good architecture (Stage 3)
- You have diverse features with proven predictive power (Stage 2)
- Data preparation is complete and validated (Stage 1)
- You have remaining model accuracy headroom to target

**When tuning does NOT make sense:**
- Features are weak (improve Stage 2 first)
- You haven't tried different architectures (improve Stage 3 first)
- You have data quality issues (fix Stage 1 first)
- You're overfitting (add regularization, reduce complexity instead)

#### Approach 1: Manual Single-Variable Experiments

Create one experiment per hypothesis:

```
experiment_create(experiment_id="exp-001-xgb-max-depth",
                  hypothesis="Deeper trees (depth=8) should improve fit")
```

Edit overlay:
```yaml
models:
  xgb_baseline:
    params:
      max_depth: 8  # Changed from 4
```

Run and compare to baseline:
```
pipeline(action="run_backtest")
```

Log result:
```
experiment_create(...) → log(hypothesis, changes, verdict)
```

**Pros:**
- Understand single-variable impact
- Build intuition about which params matter
- Reversible (overlays don't mutate production config)

**Cons:**
- Slow (one experiment per param = many runs)
- Low coverage (huge hyperparameter space)

#### Approach 2: Bayesian Exploration (Recommended)

Define a search space and let Optuna explore intelligently:

```
pipeline(action="explore",
         search_space={
           "axes": [
             {"key": "models.xgb_baseline.params.max_depth", "type": "integer", "low": 2, "high": 12},
             {"key": "models.xgb_baseline.params.learning_rate", "type": "continuous", "low": 0.001, "high": 0.3, "log": true},
             {"key": "ensemble.temperature", "type": "continuous", "low": 0.8, "high": 1.2},
           ],
           "budget": 50,
           "primary_metric": "brier"
         })
```

Returns:
- **Best trial** — Optimal hyperparams found
- **All trials** — Full history with metrics
- **Parameter importance** — Which hyperparams actually matter
- **Baseline comparison** — How much did tuning help?

**Advantages over manual:**
- Covers hyperparameter space systematically
- Prediction cache shared across trials (unchanged models never retrain)
- Optuna TPE sampler gets smarter as it learns
- Parameter importance tells you which params to focus on
- Single call returns full report

---

## Project Setup

### Step 1: Initialize Project

```bash
# Via CLI
harnessml init my-project
cd my-project

# Or manually
mkdir my-project && cd my-project
mkdir -p config data/{raw,processed,features,cache} models logs experiments
```

### Step 2: Create Base Config

```yaml
# config/pipeline.yaml
data:
  features_path: data/features/matchup_features.parquet
  task: classification
  target_column: result        # 0/1 (home win / away win)
  key_columns: [home_id, away_id, season]
  time_column: season

backtest:
  cv_strategy: leave_one_season_out
  seasons: [2015, 2016, 2017, 2018, 2019, 2020]
  metrics: [brier, accuracy, log_loss, ece]
  min_train_folds: 1

ensemble:
  method: stacked
  meta_learner_type: logistic
  post_calibration: null
  temperature: 1.0
```

### Step 3: Verify Setup

```bash
harnessml validate --config config/pipeline.yaml
```

---

## Phase 1: Data Preparation

### 1.1 Ingest Data Sources

```bash
# Raw tournament schedule
data(action="add",
            data_path="raw/schedule.csv",
            join_on=["home_id", "away_id", "season"],
            prefix="schedule_")

# Team statistics
data(action="add",
            data_path="raw/team_stats.csv",
            join_on=["team_id", "season"],
            prefix="team_")

# External ratings (KenPom, BartorVik, etc.)
data(action="add",
            data_path="raw/kenpom.csv",
            join_on=["team_id", "season"],
            prefix="kenpom_")
```

### 1.2 Validate Data Quality

```bash
# Profile the merged feature set
data(action="profile")
```

Returns:
- Column types
- Null rates per column
- Value distributions
- Duplicate counts

Look for:
- ✅ Numeric columns where expected
- ✅ Zero nulls in key columns (id, season, target)
- ✅ Target column has balanced classes
- ✅ No obvious duplicates

### 1.3 Check Temporal Integrity

In your data processing:
- Ensure `schedule_year` is the prediction target
- Ensure `team_stats_*` are lagged (prior season, prior N games)
- Ensure `kenpom_*` ratings are from pre-tournament only
- Avoid any post-tournament data leakage

```python
# Example: Lag team stats to prevent leakage
team_stats_2019 = get_prior_season_stats(2018)  # Previous season
```

### 1.4 Resolve Issues

If nulls are found:
```bash
data(action="fill_nulls",
            column="kenpom_rating",
            strategy="median")
```

If duplicates:
```bash
data(action="drop_duplicates",
            columns=["home_id", "away_id", "season"])
```

---

## Phase 2: Feature Engineering

### 2.1 Ingest Base Features

```yaml
# config/features.yaml
features:
  seed:
    type: team
    source: schedule
    column: seed

  rating:
    type: team
    source: kenpom
    column: adjusted_em

  wins_pct:
    type: team
    source: team_stats
    column: win_percentage
```

### 2.2 Test Transformations

```bash
features(action="test_transformations",
                features=["seed", "rating", "wins_pct"],
                test_interactions=true)
```

Output:
```
Feature: seed
├── log (correlation 0.35)
├── sqrt (correlation 0.32)
├── rank (correlation 0.28)
└── z-score (correlation 0.30)
→ RECOMMENDED: log (best correlation gain)

Feature: rating
├── log (correlation 0.52)
├── sqrt (correlation 0.48)
└── z-score (correlation 0.50)
→ RECOMMENDED: z-score (better for downstream models)

Interactions:
├── seed × rating (correlation 0.45)
└── seed × wins_pct (correlation 0.28)
→ RECOMMENDED: seed × rating
```

Add the winners:

```yaml
features:
  log_seed:
    type: team
    formula: "log(seed + 1)"  # +1 for zero-safe

  z_rating:
    type: team
    formula: "standardize(rating)"  # Zero mean, unit variance

  seed_rating_product:
    type: team
    formula: "seed * rating"
```

### 2.3 Discover Important Features

```bash
features(action="discover",
                method="xgboost",
                top_n=30)
```

Output:
```
Feature Analysis (Top 30)
─────────────────────────────────────
Feature              Correlation  Importance  Category
─────────────────────────────────────
diff_rating               0.68      0.42     pairwise
diff_seed                 0.65      0.38     pairwise
diff_wins_pct             0.52      0.25     pairwise
home_court_advantage      0.12      0.08     matchup
tournament_time           0.31      0.18     regime
log_seed                  0.45      0.15     team
z_rating                  0.63      0.35     team

Redundancy Detected (>0.95 correlation)
─────────────────────────────────────
seed ↔ log_seed (0.98)  → Keep log_seed, drop seed
```

Select top 20-30 features based on:
1. Correlation with target (simple indicator)
2. XGBoost importance (what models will use)
3. Diversity (pick from different feature types and sources)

Example selection:
```yaml
features:
  # Keep these
  diff_rating: ...
  diff_seed: ...
  diff_wins_pct: ...
  home_court_advantage: ...
  tournament_time: ...
  z_rating: ...
  # Remove redundant features
  seed: null  # Redundant with log_seed
```

---

## Phase 3: Model Selection

### 3.1 Configure Backtest

```bash
configure(action="backtest",
          cv_strategy="leave_one_season_out",
          seasons=[2015, 2016, 2017, 2018, 2019, 2020],
          metrics=["brier", "accuracy", "log_loss", "ece"])
```

### 3.2 Add Baseline Model

```bash
models(action="add",
              name="xgb_baseline",
              preset="xgboost_classifier",
              features=["diff_rating", "diff_seed", "diff_wins_pct", "home_court_advantage"])
```

Presets include sensible defaults:
- **xgboost_classifier** — max_depth=5, learning_rate=0.05, n_estimators=100
- **lightgbm_classifier** — num_leaves=31, learning_rate=0.05, n_estimators=100
- **catboost_classifier** — depth=6, learning_rate=0.05, iterations=100
- **mlp** — (3 hidden layers, Adam optimizer, batch norm, dropout)

### 3.3 Add Diverse Models

```bash
models(action="add",
              name="lgb_v1",
              preset="lightgbm_classifier",
              features=["diff_rating", "diff_seed", "diff_wins_pct", "home_court_advantage"])

models(action="add",
              name="mlp_v1",
              type="mlp",
              features=["diff_rating", "diff_seed", "diff_wins_pct", "home_court_advantage", "tournament_time"],
              params={"hidden_sizes": [64, 32, 16], "dropout": 0.2})
```

### 3.4 Run Backtest

```bash
pipeline(action="run_backtest",
         seasons=[2015, 2016, 2017, 2018, 2019, 2020])
```

### 3.5 Inspect Results

```bash
pipeline(action="diagnostics")
```

Output:
```
Model Diagnostics
─────────────────────────────────────
Model              Brier    Accuracy  Log Loss   ECE
─────────────────────────────────────
xgb_baseline       0.187    0.712     0.521      0.042
lgb_v1             0.184    0.718     0.509      0.038
mlp_v1             0.191    0.705     0.535      0.055

Ensemble (avg)     0.180    0.725     0.495      0.032

Notes:
─────────────────────────────────────
- All models well-calibrated (ECE < 0.06)
- LGB slightly better than XGB (Brier 0.184 vs 0.187)
- MLP is overconstrained (ECE 0.055)
- Ensemble improves via diversity
```

### 3.6 Select Best Models

Disable underperformers:

```bash
models(action="update",
              name="mlp_v1",
              active=false)
```

### 3.7 Configure Ensemble

```bash
configure(action="ensemble",
          method="stacked",
          meta_learner_type="logistic")
```

**Method choices:**
- `average` — Simple arithmetic mean of predicted probabilities (fast, stable)
- `stacked` — Train meta-learner on hold-out fold predictions (slower, can improve if models disagree)

---

## Phase 4: Hyperparameter Tuning

### 4.1 When to Tune

You're ready to tune if:
- ✅ Data quality is validated (Phase 1 complete)
- ✅ Features are selected and tested (Phase 2 complete)
- ✅ Model architectures are chosen (Phase 3 complete)
- ✅ Baseline metrics establish what to beat

You're NOT ready if:
- ❌ Features are weak (Correlation < 0.4 with target)
- ❌ Models show obvious overfitting (ECE > 0.10)
- ❌ CV metrics vary wildly (suggesting data issues)

### 4.2 Single-Variable Approach (Manual)

For one hypothesis at a time:

```bash
experiment_create(experiment_id="exp-001-xgb-depth-8",
                  hypothesis="Increasing max_depth from 5 to 8 should improve fit")
```

Edit overlay:
```yaml
models:
  xgb_baseline:
    params:
      max_depth: 8
```

Run:
```bash
pipeline(action="run_backtest")
```

Compare to baseline metrics. Log result:

```
Result: Brier improved 0.187 → 0.184 (revert on test set, too risky)
```

### 4.3 Bayesian Exploration (Recommended)

Define search space over promising hyperparameters:

```bash
pipeline(action="explore",
         search_space={
           "axes": [
             {
               "key": "models.xgb_baseline.params.max_depth",
               "type": "integer",
               "low": 3,
               "high": 10
             },
             {
               "key": "models.xgb_baseline.params.learning_rate",
               "type": "continuous",
               "low": 0.001,
               "high": 0.3,
               "log": true
             },
             {
               "key": "models.xgb_baseline.params.subsample",
               "type": "continuous",
               "low": 0.5,
               "high": 1.0
             },
             {
               "key": "ensemble.temperature",
               "type": "continuous",
               "low": 0.9,
               "high": 1.1
             }
           ],
           "budget": 50,
           "primary_metric": "brier"
         })
```

Output:
```
Bayesian Exploration Results (50 trials)
─────────────────────────────────────────────

Baseline (Trial 0)
  Brier: 0.1870
  Config: max_depth=5, lr=0.05, subsample=1.0, temp=1.0

Best Trial (Trial 23)
  Brier: 0.1809  ← 0.61% improvement
  Config: max_depth=7, lr=0.032, subsample=0.8, temp=0.95

Parameter Importance
  max_depth: 0.42      ← Matters most
  learning_rate: 0.38
  subsample: 0.15
  temperature: 0.05    ← Minimal impact

Trials by Metric
┌─────────────────────────────────────────┐
│ Brier Score Distribution (50 trials)    │
│                                         │
│ 0.175 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░    │
│ 0.180 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│ 0.185 │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ 0.190 │ ░░░░░░░░░░░░░░░░░░░░░░░░░    │
└─────────────────────────────────────────┘

Recommendation:
  Promoting best trial improves Brier by 0.61%
  This is modest but consistent across CV folds
  Worth deploying if model lift is in problem domain
```

---

## Advanced: Bayesian Exploration

### Exploration Axis Types

```python
# Continuous (float) — real-valued hyperparameter
AxisDef(
  key="models.xgb_baseline.params.learning_rate",
  type="continuous",
  low=0.001,
  high=0.3,
  log=true  # Log scale (exponential range)
)

# Integer — discrete integer hyperparameter
AxisDef(
  key="models.xgb_baseline.params.max_depth",
  type="integer",
  low=2,
  high=15
)

# Categorical — choose from fixed set
AxisDef(
  key="ensemble.method",
  type="categorical",
  values=["stacked", "average"]
)

# Subset — enable/disable items from a list
AxisDef(
  key="models.active",
  type="subset",
  candidates=["xgb_baseline", "lgb_v1", "catboost_v1"],
  min_size=1  # At least 1 model must be active
)
```

### Shared Prediction Cache

Critical feature: **Prediction cache is shared across trials.**

This means:
- Trial 1 trains XGB with depth=5, stores predictions
- Trial 2 trains XGB with depth=6, stores predictions
- Trial 3 runs with LGB (unchanged) → **reuses Trial 1's LGB predictions**

Result: **Massive speedup for multi-trial exploration** (often 5-10x faster).

### Parameter Importance

Optuna computes parameter importance via fANOVA (Functional ANOVA):
- Measures how much each parameter affects the objective
- Higher = more impact on final metric
- Guides further exploration

Example interpretation:
```
learning_rate: 0.38  → Tuning this helps most
max_depth: 0.42      → This matters slightly more
subsample: 0.15      → Minor impact
temperature: 0.05    → Barely matters, skip from future exploration
```

---

## Guardrails & Safety

### Automatic Guardrails (Always Enabled)

**Hard guardrails (cannot override):**

1. **Feature Leakage Detection**
   - Blocks features from "post_tournament" sources at train time
   - Example: Don't use actual tournament results to predict tournament results

2. **Temporal Ordering**
   - Ensures CV folds maintain train < test in time
   - Prevents data leakage via time travel

3. **Critical Path**
   - Prevents accidental removal of required models/features

**Advisory guardrails (overridable):**

- NamingConvention (experiment IDs match `exp-NNN-*`)
- DoNotRetry (skip known failed patterns)
- ConfigProtection (guard production YAML)
- FeatureStaleness (warn if features are >N days old)
- SingleVariable (only one variable per experiment)

### Audit Logging

Every tool invocation is logged:

```json
{
  "timestamp": "2026-03-01T14:23:45Z",
  "tool": "experiments",
  "action": "run",
  "args": {"experiment_id": "exp-001-xgb-depth"},
  "guardrails_passed": true,
  "result_status": "success",
  "duration_s": 45.2,
  "human_override": false
}
```

Query logs:
```python
logs = logger.query(tool="experiments", status="success")
```

---

## Common Patterns

### Pattern 1: Feature Ablation

Test whether each feature matters:

```bash
# Baseline with all features
models(add, name="baseline", features=[...all 20 features...])

# Ablation 1: Remove feature X
models(add, name="ablation_no_x", features=[...19 features...])

# Compare metrics
# If metrics drop significantly, feature X matters
```

### Pattern 2: Model Ensemble Diversity

Test ensemble value:

```bash
# Single best model
configure(ensemble=null)
pipeline(run_backtest)  # Brier: 0.187

# Simple average of top 3
configure(ensemble=average)
pipeline(run_backtest)  # Brier: 0.181 (improvement from diversity)

# Stacked ensemble
configure(ensemble=stacked)
pipeline(run_backtest)  # Brier: 0.179 (better)
```

### Pattern 3: Calibration Verification

Check if models are well-calibrated:

```bash
pipeline(action="diagnostics")
# Look at ECE (Expected Calibration Error)
# If ECE > 0.05, add post-calibration:

configure(action="ensemble",
          post_calibration="platt")  # or "isotonic"
pipeline(run_backtest)
```

### Pattern 4: Do-Not-Repeat Safety

Prevent accidental retraining:

```bash
# After discovering a config fails
experiment_manager.add_do_not_retry(
  pattern="models.xgb.params.max_depth == 15",
  reason="OOMs the system, known failure"
)

# Future experiments matching pattern will be blocked
```

---

## Troubleshooting

### Problem: Features have low correlation (< 0.3)

**Diagnosis:** Data quality or feature engineering issues

**Solution:**
1. Check temporal leakage (are you using future data?)
2. Verify raw data is correct (check source freshness)
3. Try different transformations (log, rank, interactions)
4. Add domain-specific features (hand-crafted based on problem knowledge)

### Problem: CV metrics vary wildly across folds

**Diagnosis:** Data distribution is not stationary, or temporal issues exist

**Solution:**
1. Check for time series trends (is recent data different?)
2. Verify train/test temporal separation (no leakage)
3. Check for class imbalance per fold
4. Use stratified CV if applicable

### Problem: Hyperparameter tuning doesn't help

**Diagnosis:** Better tuning targets may exist elsewhere

**Solution:**
1. Go back to Phase 2: Add more diverse features
2. Go back to Phase 3: Try different model architectures
3. Verify features aren't stale or outdated
4. Ensure baseline model is properly trained (not underfitting)

### Problem: Ensemble doesn't improve over single model

**Diagnosis:** Models are too similar (not diverse enough)

**Solution:**
1. Add different model types (XGB + LGB + MLP)
2. Add feature subsets (different features per model)
3. Check model disagreement (if < 0.5, they're learning different things)
4. Increase ensemble diversity rather than tuning individuals

---

## Key Takeaways

1. **Follow the workflow** — Data → Features → Models → Hyperparams. Don't skip or reorder.
2. **Features first** — Good features beat tuned hyperparameters every time.
3. **Temporal integrity matters** — One data leak ruins everything; hard guardrails enforce this.
4. **Leverage caching** — Unchanged models never retrain; exploration is fast.
5. **One variable per experiment** — Change one thing, measure impact, commit/revert.
6. **Trust the tools** — Config, overlays, fingerprinting, logging—all automatic.
7. **Explore boldly** — Features are cheap; test transformations, interactions, combinations.
8. **Tune last** — Only after data, features, and architectures are solid.

---

## Further Reading

- [README.md](README.md) — High-level overview
- [CLAUDE.md](CLAUDE.md) — Developer conventions for HarnessML
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture and design decisions
- [packages/harness-plugin/skills/ml-workflow/SKILL.md](packages/harness-plugin/skills/ml-workflow/SKILL.md) — MCP tool guide
