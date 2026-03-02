# easyml-runner

YAML-driven ML orchestration layer. The only consumer of the 7 EasyML library packages. Orchestrates the complete data science workflow: data prep → feature engineering → model selection → hyperparameter tuning.

**Designed for iterative AI-driven ML without context overhead.**

## Install

```bash
pip install easyml-runner
```

## Quick Start

```bash
# Scaffold a new project structure
easyml init my-project && cd my-project

# Validate YAML configuration
easyml validate

# Phase 1: Data Preparation
easyml data ingest --path raw/schedule.csv
easyml data validate
easyml data profile

# Phase 2: Feature Engineering
easyml feature add --name diff_seed --type pairwise --formula "home_seed - away_seed"
easyml feature discover --method xgboost --top_n 30
easyml feature test_transform --features seed,rating,elo

# Phase 3: Model Selection
easyml model add xgb_baseline --preset xgboost_classifier
easyml model add lgb_v1 --preset lightgbm_classifier
easyml configure backtest --cv_strategy leave_one_season_out --seasons 2015 2016 2017
easyml run backtest

# Phase 4: Hyperparameter Tuning
easyml explore --search_space '{"axes": [...], "budget": 50, "primary_metric": "brier"}'

# Experiment management
easyml experiment create exp-001-test-hyperparams
easyml experiment run exp-001
easyml experiment promote exp-001

# Start MCP server for AI agents
easyml serve
```

## The Four Phases (In Order)

### 🔴 Phase 1: Data Preparation

Ensure data quality, temporal integrity, source freshness **before touching models**.

- Ingest raw data sources
- Validate types, nulls, distributions
- Check temporal ordering (no leakage)
- Resolve issues (duplicates, bad joins, stale data)

**Tools:** `manage_data(action="add|validate|profile|fill_nulls|drop_duplicates|rename")`

**Why first?** Bad data corrupts all downstream models. Temporal issues create invisible leakage.

---

### 🟡 Phase 2: Feature Engineering

Discover transformations and combinations that improve predictive power.

- Test transformations (log, sqrt, rank, z-score, interactions)
- Discover important features (correlation, XGBoost importance, redundancy)
- Create composite features (pairwise diffs, ratios, interactions)
- Define regimes (context flags that gate feature sets)
- Curate top 20-30 features by importance + diversity

**Tools:** `manage_features(action="add|add_batch|test_transformations|discover")`

**Key principle:** Features are cheap; test aggressively. Good features beat tuned hyperparameters.

---

### 🟢 Phase 3: Model Selection

Find model architectures that generalize best via honest CV evaluation.

- Configure CV strategy (leave-one-season-out, expanding window, etc.)
- Add baseline model (simple preset)
- Add comparison models (different architectures: XGB, LGB, MLP)
- Run backtest across CV folds
- Inspect diagnostics (calibration, agreement, correlation)
- Keep winners, disable underperformers
- Configure ensemble (stacked or average)

**Tools:** `manage_models(action="add|update|remove|list")`, `configure(action="backtest|ensemble")`, `pipeline(action="run_backtest|diagnostics")`

**Why before hyperparams?** Good architectures beat bad ones with good hyperparameters. Diversity improves ensembles.

---

### 🔵 Phase 4: Hyperparameter Tuning

Fine-tune best architectures within computational budget.

**Only if:**
- ✅ Data quality validated (Phase 1)
- ✅ Features engineered (Phase 2)
- ✅ Architectures selected (Phase 3)

**Not if:**
- ❌ Features are weak (correlation < 0.3)
- ❌ Models show obvious overfitting
- ❌ Haven't tried different architectures

**Approaches:**
1. **Manual** — Edit overlay YAML, run experiments, compare to baseline
2. **Bayesian** (recommended) — Define search space, let Optuna explore intelligently

**Tools:** `experiment(action="create|run|promote")`, `pipeline(action="explore")`

**Expected ROI:**
- Phase 2: 5-20% gain (high ROI)
- Phase 3: 2-10% gain (medium ROI)
- Phase 4: 0.5-2% gain (low ROI)

---

## Python API

```python
from easyml.runner import validate_project, PipelineRunner, Project, FeatureStore

# Validate YAML config
result = validate_project("config/")
assert result.valid

# Fluent API: Build config programmatically
project = Project("my_project")
project.set_data(features_dir="data/features", task="classification", target_column="result")
project.add_model("xgb_baseline", "xgboost", features=["diff_seed", "diff_rating"])
project.configure_backtest(cv_strategy="leave_one_season_out", seasons=[2015, 2016, 2017])
project.to_yaml("config/")

# Run backtest
runner = PipelineRunner(".", config_dir="config/")
result = runner.backtest(seasons=[2015, 2016, 2017])
print(result.metrics)  # {model_name: {metric: value}}

# Get diagnostics
diag = runner.get_diagnostics()
print(diag)  # Per-model: brier, accuracy, ece, log_loss, agreement, calibration

# Run single-season prediction
predictions = runner.predict(season=2024)

# Bayesian exploration
from easyml.runner import run_exploration, ExplorationSpace, AxisDef
space = ExplorationSpace(
    axes=[
        AxisDef(key="models.xgb.params.max_depth", type="integer", low=3, high=10),
        AxisDef(key="ensemble.method", type="categorical", values=["stacked", "average"]),
    ],
    budget=50,
    primary_metric="brier"
)
result = run_exploration(".", space)
print(result["report"])  # Markdown report with best trial, all trials, parameter importance
```

## Key APIs

| API | Description |
|-----|-------------|
| `validate_project(config_dir)` | Validate YAML config against schema |
| `scaffold_project(path)` | Generate new project structure with starter config |
| `Project` | Fluent API for building config programmatically |
| `PipelineRunner` | Execute backtest, predict, diagnostics from config |
| `FeatureStore` | Declarative feature management (team/pairwise/matchup/regime) |
| `run_exploration(project_dir, space)` | Bayesian hyperparameter search via Optuna TPE |
| `ExperimentManager` | Experiment lifecycle (create, run, promote, DNR) |
| `generate_server(config, dir)` | Generate MCP server from config |
| `ProjectConfig` | Top-level Pydantic schema for all config |

## YAML Configuration

```yaml
# config/pipeline.yaml
data:
  features_path: data/features/matchup_features.parquet
  task: classification
  target_column: result        # 0 = away win, 1 = home win
  key_columns: [home_id, away_id, season]
  time_column: season

backtest:
  cv_strategy: leave_one_season_out
  seasons: [2015, 2016, 2017, 2018, 2019, 2020]
  metrics: [brier, accuracy, log_loss, ece]

ensemble:
  method: stacked              # or "average"
  meta_learner_type: logistic
  post_calibration: null       # or "platt", "isotonic", "spline"
  temperature: 1.0

# config/models.yaml
models:
  xgb_baseline:
    type: xgboost
    features: [diff_seed, diff_rating, diff_wins]
    params:
      max_depth: 5
      learning_rate: 0.05
      subsample: 1.0

  lgb_v1:
    type: lightgbm
    features: [diff_seed, diff_rating, diff_wins, home_court_advantage]
    params:
      num_leaves: 31
      learning_rate: 0.05

# config/features.yaml (Declarative)
features:
  seed:
    type: team
    source: schedule
    column: seed

  diff_seed:
    type: pairwise
    formula: "home_seed - away_seed"

  tournament_time:
    type: regime
    condition: "season_stage == 'tournament'"
```

## Guardrails

**Hard guardrails (non-overridable):**
- Feature leakage detection (temporal safety)
- Temporal ordering (train < test in CV)
- Critical path enforcement

**Advisory guardrails (overridable):**
- Naming conventions
- Do-Not-Repeat (DNR)
- Config protection
- Feature staleness
- Single-variable enforcement
- Audit logging (all tool invocations recorded)

## Further Reading

- [GETTING_STARTED.md](../../GETTING_STARTED.md) — Complete workflow guide with examples
- [../../README.md](../../README.md) — System overview and architecture
- [../../CLAUDE.md](../../CLAUDE.md) — Development conventions
