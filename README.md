# EasyML

**AI-driven machine learning without the context overhead.**

EasyML solves the problem of AI agents spending precious context tokens manipulating code, data pipelines, and YAML files instead of doing data science. It's an **automated ML orchestration framework** designed specifically for iterative AI-driven experimentation—with hard guardrails to enforce hygiene, automatic logging to prevent lost work, and declarative interfaces that let agents focus entirely on modeling decisions.

## The Problem

When an AI agent runs ML experiments iteratively, it spends token budget on:
- Writing and debugging feature engineering code
- Configuring data pipeline orchestration
- Creating boilerplate experiment tracking
- Managing file I/O and caching
- Manually logging results that might get lost
- Verifying data leakage and temporal integrity

This leaves less context for the actual data science: exploring feature spaces, testing hypotheses, comparing model architectures, and navigating trade-offs.

## The Solution

EasyML removes this overhead through:

1. **Declarative Feature System** — Define features as Python functions with type annotations (`entity`, `pairwise`, `instance`, `regime`). The system handles caching, deduplication, and entity/period alignment automatically. No more DataFrame wrangling in feature code.

2. **Hard Guardrails** — Non-overridable safety checks: data leakage detection, temporal integrity verification, critical path enforcement. Guardrails are advisory until locked; once locked, they cannot be bypassed. Violations are logged for audit.

3. **Automatic Logging & Measurement** — Every experiment run is fingerprinted, timestamped, and logged. Metrics are computed across all models and folds. No experiment result is lost; nothing requires manual tracking.

4. **YAML-Driven Orchestration** — The entire ML pipeline—sources, features, models, ensemble, backtest strategy—is defined in YAML. Agents manipulate structured config, not code. Config overlays enable isolated hypothesis testing without touching production.

5. **Bayesian Exploration** — Define a search space (feature mixes, model subsets, hyperparams, ensemble settings) and the system intelligently explores it across budget constraints using Optuna TPE, caching predictions across trials so unchanged models never retrain.

6. **MCP Interface** — All operations (backtest, experiment, explore, etc.) are single MCP tool calls. Agents invoke workflows, not pipelines.

## Architecture

```
+-------------------+
|    easyml-core    |   All engine code
|  schemas, config, |
|  guardrails,      |
|  models, runner,  |
|  feature_eng      |
+---------+---------+
          |
    +-----+-----+
    |           |
+---v---+  +---v------+
|plugin |  | sports   |
| MCP   |  | optional |
| server|  | domain   |
+-------+  +----------+
```

Three packages in a uv workspace monorepo:
- **easyml-core** — All core engine code (schemas, config, guardrails, models, runner, feature engineering, metrics, data sources)
- **easyml-plugin** — MCP server (thin async dispatcher with hot-reloadable handlers)
- **easyml-sports** — Optional domain plugin for matchup prediction (registers via hook system)

## Quick Start

```bash
git clone <repo-url> && cd easyml
uv sync          # install all packages + dev deps

uv run pytest    # run full test suite (~1800+ tests)
```

### Basic Usage

```python
from easyml.core.config import resolve_config
from easyml.core.models import ModelRegistry, TrainOrchestrator
from easyml.core.schemas.metrics import MetricRegistry

# 1. Load config
config = resolve_config("config/", file_map={"models": "models.yaml"})

# 2. Train models
model_registry = ModelRegistry.with_defaults()
orchestrator = TrainOrchestrator(model_registry, config["models"], output_dir="models/")
trained = orchestrator.train_all(X, y, feature_columns=cols)

# 3. Evaluate with any registered metric
metrics = MetricRegistry()
print(f"Brier: {metrics.get('binary', 'brier')(y_true, y_prob):.4f}")
```

### YAML-Driven Pipeline

```bash
# Initialize and validate project
easyml validate --config-dir config/

# Run backtest
easyml run backtest

# Create and run experiments
easyml experiment create exp-001-test-hyperparams
easyml experiment run exp-001
easyml experiment promote exp-001
```

### MCP Tool Interface

When running the MCP server, the framework provides tools for:
- `experiments` — Create, run, promote experiments; define overlays
- `data` — Ingest sources, validate, fill nulls, rename columns, manage views
- `features` — Register features, test transformations, discover correlations, analyze diversity
- `models` — Add models, adjust ensembles, control active models
- `configure` — Initialize projects, update backtest/ensemble config, run guardrails checks
- `pipeline` — Run backtests, make predictions, get diagnostics, list/show/compare runs
- `competitions` — Simulations, brackets, scoring for tournament-style events

## Key Capabilities

### Model Wrappers (8 types)
- **XGBoost, LightGBM, CatBoost** — Full eval_set/early stopping support
- **Random Forest, Logistic Regression, ElasticNet** — scikit-learn wrappers with param filtering
- **MLP** — PyTorch with optional normalize, batch_norm, early stopping, weight decay
- **TabNet** — PyTorch-TabNet with optional normalize, val_fraction, LR scheduler

All model params are configurable via YAML. The `ModelRegistry` uses inspect-based kwargs forwarding for generic model construction.

### Feature System (4 types)
- **entity** — Entity-level features (auto-generates pairwise diffs)
- **pairwise** — Instance-level formula features
- **instance** — Context columns passed through
- **regime** — Boolean flags that gate feature sets

### Metrics (45 across 6 task types)
- **binary** — brier, accuracy, log_loss, ece, auroc, f1, precision, recall, etc.
- **multiclass** — macro/micro/weighted variants
- **regression** — rmse, mae, r2, mape, etc.
- **ranking** — ndcg, mrr, map
- **survival** — concordance_index, brier_survival
- **probabilistic** — crps, calibration, sharpness

### Calibration (4 methods)
- **Spline (PCHIP)** — Monotonic interpolation with isotonic pre-processing
- **Isotonic** — Non-parametric monotonic regression
- **Platt** — Logistic regression scaling
- **Beta** — Beta distribution calibration

### CV Strategies
- **leave_one_out** — Symmetric LOSO (all other folds for training)
- **expanding_window** — Temporal expanding window
- **sliding_window** — Fixed-size sliding window
- **k_fold** — Standard k-fold
- **stratified** — Stratified k-fold

### View Engine (22 transform steps)
filter, select, derive, group_by, join, union, unpivot, sort, head, rolling, cast, distinct, rank, isin, cond_agg, lag, ewm, diff, trend, encode, bin, datetime, null_indicator

### Guardrails (12 total)
- **3 non-overridable** — Data leakage, temporal integrity, critical path enforcement
- **9 overridable** — Feature naming, model config, feature diversity, etc.

### Exploration
- **Auto-search** — Discover feature interactions, lags, rolling aggregations
- **Feature diversity** — Overlap matrix, diversity score, redundant pair detection, removal suggestions
- **Bayesian search** — Optuna TPE over features, models, hyperparams, ensemble settings

## Development

```bash
uv sync                                          # install workspace
uv run pytest packages/easyml-core/tests/ -q     # core tests (~1800+)
uv run pytest packages/easyml-sports/tests/ -q   # sports plugin tests
uv run pytest -v                                 # verbose all tests
```

Requires Python 3.11+. Managed by [uv](https://github.com/astral-sh/uv) workspaces.

## Using EasyML as an Agent

When an AI agent is connected to EasyML via MCP, it never needs to:

1. **Write data pipeline code** — Use `data` to ingest sources, define views, validate outputs
2. **Engineer features manually** — Use `features` to register declarative features; EasyML handles caching and routing
3. **Track experiments** — All runs are fingerprinted and logged; history is searchable
4. **Re-run identical experiments** — DNR (Do Not Repeat) prevents accidental duplication
5. **Mutate production config** — Use experiment overlays to test hypotheses in isolation
6. **Manage retraining** — Prediction cache ensures unchanged models skip retraining across trials
7. **Check data hygiene** — Guardrails automatically verify leakage, temporal integrity, critical paths

The agent focuses entirely on:
- Defining feature mixes and model architectures
- Stating hypotheses about what will improve metrics
- Analyzing results from bulk explorations
- Making data science decisions (not pipeline decisions)

## Design Philosophy

**EasyML is built for agents, not humans.**

- **Declarative over imperative** — YAML config and registry-based registration reduce code boilerplate
- **Defaults over decision fatigue** — Sensible presets for models, CV, metrics, guardrails
- **Structured contracts over strings** — Pydantic schemas everywhere; no magic field names
- **Automatic over manual** — Caching, logging, fingerprinting, and guardrails happen without agent intervention
- **Single source of truth** — YAML config is the contract; overlays enable isolated testing without mutations
- **Predictable, deterministic** — Fingerprinting ensures identical configs always produce identical results; no hidden state
- **Everything configurable** — No hardcoded thresholds, metric lists, or domain assumptions

## License

MIT
