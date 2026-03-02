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
easyml init my-project && cd my-project
easyml validate
easyml run pipeline
easyml experiment create exp-001-test
easyml explore --search-space '{"axes": [...]}'
easyml serve
```

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

- **Feature Store** — Declarative features with automatic caching, type routing, and deduplication
- **Pairwise Features** — Define matchup-level features (e.g., home team vs away team) automatically expanded for all pairs
- **Regime Features** — Boolean feature flags (e.g., "is_playoff") routed through the pipeline
- **View Management** — Lazy DAG resolution for complex data transformations with full source tracing

## Packages

| Package | Description |
|---------|-------------|
| `easyml-schemas` | Pydantic contracts, probability/regression/ensemble metrics |
| `easyml-config` | Split YAML loading, variant resolution, deep merge (OmegaConf) |
| `easyml-features` | Declarative feature registry, caching, pairwise builder, type routing |
| `easyml-models` | 8 model wrappers, 5 CV strategies, calibration, stacked ensembles |
| `easyml-data` | Source registry, DVC generation, stage guards, refresh orchestrator, views |
| `easyml-experiments` | Experiment lifecycle: naming, change detection, DNR, logging, promote |
| `easyml-guardrails` | 11 guardrails (3 hard + 8 advisory), MCP server, audit logging |
| `easyml-runner` | YAML-driven CLI, PipelineRunner, project scaffold, MCP server generation, Bayesian exploration |

## Development

```bash
uv sync                        # install workspace
uv run pytest -v               # run all tests
uv run pytest packages/easyml-models/tests/  # run one package's tests
```

Requires Python 3.11+. Managed by [uv](https://github.com/astral-sh/uv) workspaces.

## Design Philosophy

**EasyML is built for agents, not humans.**

- **Declarative over imperative** — YAML config and decorator-based registration reduce code boilerplate
- **Defaults over decision fatigue** — Sensible presets for models, CV, metrics, guardrails
- **Structured contracts over strings** — Pydantic schemas everywhere; no magic field names
- **Automatic over manual** — Caching, logging, fingerprinting, and guardrails happen without agent intervention
- **Single source of truth** — YAML config is the contract; overlays enable isolated testing without mutations

## License

MIT
