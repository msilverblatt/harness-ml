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

1. **Declarative Feature System** — Define features as Python functions with type annotations (`team`, `pairwise`, `matchup`, `regime`). The system handles caching, deduplication, and entity/period alignment automatically. No more DataFrame wrangling in feature code.

2. **Hard Guardrails** — Non-overridable safety checks: data leakage detection, temporal integrity verification, critical path enforcement. Guardrails are advisory until locked; once locked, they cannot be bypassed. Violations are logged for audit.

3. **Automatic Logging & Measurement** — Every experiment run is fingerprinted, timestamped, and logged. Metrics are computed across all models and seasons. No experiment result is lost; nothing requires manual tracking.

4. **YAML-Driven Orchestration** — The entire ML pipeline—sources, features, models, ensemble, backtest strategy—is defined in YAML. Agents manipulate structured config, not code. Config overlays enable isolated hypothesis testing without touching production.

5. **Bayesian Exploration** — Define a search space (feature mixes, model subsets, hyperparams, ensemble settings) and the system intelligently explores it across budget constraints using Optuna TPE, caching predictions across trials so unchanged models never retrain.

6. **MCP Interface** — All operations (backtest, experiment, explore, etc.) are single MCP tool calls. Agents invoke workflows, not pipelines.

## Architecture

```
                        +------------------+
                        |  easyml-schemas  |   Pydantic contracts + metrics
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                  |                  |
     +--------v-------+ +-------v--------+ +-------v--------+
     |  easyml-config | | easyml-features| |  easyml-models |
     |  YAML + merge  | | registry +     | | 8 wrappers, CV |
     |  + variants    | | caching +      | | ensemble, cal  |
     +--------+-------+ | pairwise       | +-------+--------+
              |          +-------+--------+         |
              |                  |                  |
     +--------v-------+ +-------v--------+ +-------v--------+
     |  easyml-data   | | easyml-exper.  | | easyml-guard.  |
     |  sources, DVC  | | overlay mgmt,  | | 11 guardrails  |
     |  guards, refresh| | DNR, promote   | | MCP server,    |
     +----------------+ +----------------+ | audit logger   |
                                           +----------------+
```

All packages share contracts through `easyml-schemas` (the core). Config, features, and models form the middle tier. Data, experiments, and guardrails are the outer tool ring.

## Quick Start

```bash
git clone <repo-url> && cd easyml
uv sync          # install all packages + dev deps

uv run pytest    # run full test suite
```

### Basic Usage

```python
from easyml.config import resolve_config
from easyml.features import FeatureRegistry, FeatureBuilder
from easyml.models import ModelRegistry, TrainOrchestrator, StackedEnsemble
from easyml.schemas import brier_score, accuracy

# 1. Load config
config = resolve_config("config/", file_map={"models": "models.yaml"})

# 2. Register and build features
registry = FeatureRegistry()

@registry.register(name="win_rate", category="resume", level="team", output_columns=["win_rate"])
def compute_win_rate(df, cfg):
    result = df[["entity_id", "period_id"]].copy()
    result["win_rate"] = df["wins"] / (df["wins"] + df["losses"])
    return result

# 3. Train models
model_registry = ModelRegistry.with_defaults()
orchestrator = TrainOrchestrator(model_registry, config["models"], output_dir="models/")
trained = orchestrator.train_all(X, y, feature_columns=cols)

# 4. Evaluate
from easyml.schemas import brier_score
print(f"Brier: {brier_score(y_true, y_prob):.4f}")
```

## YAML-Driven Interface

The `easyml-runner` package provides a CLI and Python API that drives the entire framework from YAML configuration:

```bash
# Initialize and validate project
easyml init my-project && cd my-project
easyml validate --config pipeline.yaml

# Run backtest on production config
easyml run backtest --seasons 1 2 3

# Create and run experiments
easyml experiment create exp-001-test-hyperparams
easyml experiment run exp-001
easyml experiment promote exp-001

# Bayesian exploration: define search space, run bulk experiments
easyml explore --search-space '{
  "axes": [
    {"key": "models.xgb.params.max_depth", "type": "integer", "low": 3, "high": 10},
    {"key": "ensemble.method", "type": "categorical", "values": ["mean", "stack"]}
  ],
  "budget": 20
}'

# Server with dynamic MCP tool generation
easyml serve
```

### MCP Tool Interface

When running `easyml serve`, the framework generates an MCP server with tools for:
- `manage_experiments` — Create, run, promote experiments; define overlays
- `manage_data` — Ingest sources, validate, fill nulls, rename columns, manage views
- `manage_features` — Register features, test transformations, discover correlations
- `manage_models` — Add models, adjust ensembles, control active models
- `configure` — Initialize projects, update backtest/ensemble config, run guardrails checks
- `pipeline` — Run backtests, make predictions, get diagnostics, list/show runs

## Key Features

### For Iterative AI-Driven Development

- **MCP Tools** — Single-call operations: `backtest(experiment_overlay)`, `explore(search_space)`, `promote(experiment_id)`, `run_feature_discovery()`, all return structured results
- **Fingerprinting** — Every feature, model, and config combination has a deterministic hash. Unchanged work is never recomputed; predictions are cached globally
- **Experiment Overlays** — Hypothesis testing via config patches. No mutation of production YAML; all changes are isolated and reversible
- **Parameter Importance** — Optuna-based analysis shows which hyperparameters and feature choices matter most

### For Safety & Hygiene

- **Hard Guardrails** — 3 non-overridable checks (data leakage, temporal integrity, critical paths) that block problematic experiments
- **Advisory Guardrails** — 8 additional safety checks with explicit overrides logged for audit
- **Audit Logger** — All guardrail violations, overrides, and critical decisions are recorded with timestamps
- **Experiment DNR** — Do Not Repeat: prevents accidental retraining on identical configs

### For Automation

- **Feature Store** — Declarative features with automatic caching via fingerprinting, type routing, and deduplication; 4 feature types: team (entity-level), pairwise (all matchups), matchup (context-level), regime (boolean flags)
- **Pairwise Features** — Define matchup-level features (e.g., home team vs away team) and they're automatically expanded for all unique pairs without duplication
- **Regime Features** — Boolean feature flags (e.g., "is_playoff") that gate entire feature sets or routed through as discrete columns
- **View Management** — Lazy DAG resolution for complex data transformations with full source tracing, fingerprint caching to skip unchanged computations
- **Bayesian Exploration** — Optuna TPE-based search over features (include/exclude lists), models (activate/deactivate), hyperparameters (continuous/integer/categorical), and ensemble settings (method, temperature, weights); budget-constrained with prediction caching across trials
- **Discovery Tools** — Automatic feature correlation analysis, importance ranking (XGBoost or mutual information), redundancy detection, and grouping by type/category
- **Transformation Testing** — Test math operations (log, sqrt, standardize, interact, polynomial) on features and validate correctness before ingestion

## Packages

| Package | Description |
|---------|-------------|
| `easyml-schemas` | Pydantic v2 contracts, probability metrics (Brier, log loss, ECE), regression and ensemble metrics |
| `easyml-config` | Split YAML loading, variant resolution, deep merge (OmegaConf), nested key access |
| `easyml-features` | Declarative registry with 4 types (team, pairwise, matchup, regime), caching via fingerprinting, source hashing |
| `easyml-models` | 8 model wrappers (XGBoost, LightGBM, CatBoost, PyTorch, SKLearn, etc.), 5 CV strategies, calibration (Platt, Isotonic, Spline), stacked ensembles, SHAP feature importance |
| `easyml-data` | Source registry, DVC pipeline generation, stage guards, refresh orchestration, view definitions, lazy DAG resolution |
| `easyml-experiments` | Experiment lifecycle: auto-naming, change detection, DNR (Do Not Repeat), metrics logging, config promotion |
| `easyml-guardrails` | 11 guardrails (3 hard non-overridable + 8 advisory): data leakage, temporal integrity, critical paths, guardrail enforcement, audit logging |
| `easyml-runner` | YAML-driven CLI, PipelineRunner (backtest + predict), project scaffold, server generation, Bayesian exploration (Optuna TPE), MCP interface |
| `easyml-plugin` | Claude MCP server generator, MCP tool registration, agent-facing interfaces for all runner operations |

## Development

```bash
uv sync                        # install workspace
uv run pytest -v               # run all tests
uv run pytest packages/easyml-models/tests/  # run one package's tests
```

Requires Python 3.11+. Managed by [uv](https://github.com/astral-sh/uv) workspaces.

## Using EasyML as an Agent

When an AI agent is connected to EasyML via MCP (Model Context Protocol), it never needs to:

1. **Write data pipeline code** — Use `manage_data` to ingest sources, define views, validate outputs
2. **Engineer features manually** — Use `manage_features` to register declarative features; EasyML handles caching and routing
3. **Track experiments** — All runs are fingerprinted and logged; history is searchable
4. **Re-run identical experiments** — DNR (Do Not Repeat) prevents accidental duplication
5. **Mutate production config** — Use experiment overlays to test hypotheses in isolation
6. **Manage retraining** — Prediction cache ensures unchanged models skip retraining across trials
7. **Check data hygiene** — Guardrails automatically verify leakage, temporal integrity, critical paths
8. **Hunt for lost results** — Audit log records every experiment run and guardrail decision

The agent focuses entirely on:
- Defining feature mixes and model architectures
- Stating hypotheses about what will improve metrics
- Analyzing results from bulk explorations
- Making data science decisions (not pipeline decisions)

## Design Philosophy

**EasyML is built for agents, not humans.**

- **Declarative over imperative** — YAML config and decorator-based registration reduce code boilerplate
- **Defaults over decision fatigue** — Sensible presets for models, CV, metrics, guardrails
- **Structured contracts over strings** — Pydantic schemas everywhere; no magic field names
- **Automatic over manual** — Caching, logging, fingerprinting, and guardrails happen without agent intervention
- **Single source of truth** — YAML config is the contract; overlays enable isolated testing without mutations
- **Predictable, deterministic** — Fingerprinting ensures identical configs always produce identical results; no hidden state

## License

MIT
