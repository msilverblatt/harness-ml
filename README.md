
```
 ██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗    ███╗   ███╗██╗
 ██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝    ████╗ ████║██║
 ███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗    ██╔████╔██║██║
 ██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██║
 ██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║    ██║ ╚═╝ ██║███████╗
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝
```

**An ML framework built for AI agents.** Guardrails, experiment tracking, and full pipeline orchestration — so your agent does data science, not plumbing.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-2242%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

### From raw CSV to stacked ensemble in under a minute


https://github.com/user-attachments/assets/c180d2b2-7ed1-4805-a08a-01b6fb3738ac


<sub>Full example: [examples/titanic](examples/titanic/)</sub>

---

## Why HarnessML?

AI agents burn context tokens on DataFrame wrangling, YAML editing, file I/O, and experiment bookkeeping. That's not data science — that's plumbing.

HarnessML gives agents a **single MCP tool call** for every ML operation: backtest, experiment, feature engineering, model tuning. The framework handles caching, logging, guardrails, and orchestration automatically. The agent focuses on hypotheses and results.

```
Agent: "Add an XGBoost model with these features and run a backtest"
                          ↓
              models(action="add", ...)
              pipeline(action="run_backtest")
                          ↓
       Automatic: CV splits, training, calibration,
       ensemble weighting, metrics, logging, fingerprinting
                          ↓
Agent: "Brier improved 0.003. Let's try adding interaction features."
```

## Quick Start

```bash
git clone https://github.com/msilverblatt/harness-ml.git && cd harness-ml
uv sync
uv run pytest  # 2242 tests
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     harness-core                        │
│  schemas ∙ config ∙ guardrails ∙ models ∙ runner        │
│  feature_eng ∙ calibration ∙ views ∙ sources            │
├────────────────────────┬────────────────────────────────┤
│    harness-plugin      │       harness-sports           │
│    MCP server          │       domain plugin            │
│    hot-reload handlers │       matchup prediction       │
└────────────────────────┴────────────────────────────────┘
```

Three packages, one `uv` workspace:

| Package | What it does |
|---------|-------------|
| **harness-core** | Engine: schemas, config, guardrails, 8 model wrappers, runner, feature store, views, calibration, metrics, data sources |
| **harness-plugin** | MCP server with hot-reloadable handlers — change handler code, no restart needed |
| **harness-sports** | Optional domain plugin for matchup prediction (hooks into core via registry) |

## What's Inside

### Models (8 wrappers)

XGBoost, LightGBM, CatBoost, Random Forest, Logistic Regression, ElasticNet, MLP (PyTorch), TabNet — all configurable via YAML with eval_set/early stopping, normalization, and inspect-based kwargs forwarding.

### Metrics (45 across 6 task types)

| Task | Examples |
|------|----------|
| Binary | brier, accuracy, log_loss, ece, auroc, f1, precision, recall |
| Multiclass | macro/micro/weighted variants of all classification metrics |
| Regression | rmse, mae, r2, mape |
| Ranking | ndcg, mrr, map |
| Survival | concordance_index, brier_survival |
| Probabilistic | crps, calibration, sharpness |

### Features (4 types)

- **entity** — per-entity stats, auto-generates pairwise diffs
- **pairwise** — formula features across entity pairs
- **instance** — context columns passed through
- **regime** — boolean flags that gate feature sets

### Calibration (4 methods)

Spline (PCHIP), Isotonic, Platt, Beta — all with save/load and fitted state tracking.

### CV Strategies

`leave_one_out` (symmetric LOSO), `expanding_window`, `sliding_window`, `k_fold`, `stratified`

### View Engine (22 transform steps)

`filter` `select` `derive` `group_by` `join` `union` `unpivot` `sort` `head` `rolling` `cast` `distinct` `rank` `isin` `cond_agg` `lag` `ewm` `diff` `trend` `encode` `bin` `datetime` `null_indicator`

### Guardrails (12 total)

3 **non-overridable** (data leakage, temporal integrity, critical path) + 9 overridable. Violations are logged for audit. Once locked, they cannot be bypassed.

### Exploration

- **Auto-search** — discover feature interactions, lags, rolling aggregations
- **Feature diversity** — overlap matrix, diversity score, redundant detection
- **Bayesian search** — Optuna TPE over features, models, hyperparams, ensemble settings

## MCP Tools

When connected via MCP, agents get these tools:

| Tool | Actions |
|------|---------|
| `data` | ingest sources, validate, fill nulls, rename, derive columns, manage views, upload to Drive/Kaggle |
| `features` | register features, test transforms, discover correlations, analyze diversity |
| `models` | add/update/clone models, batch operations, view presets |
| `configure` | init projects, set backtest/ensemble config, run guardrail checks |
| `pipeline` | run backtests, predict, diagnostics, compare runs, explain models, export notebooks |
| `experiments` | create/run/promote experiments with config overlays |
| `competitions` | simulations, brackets, scoring for tournament events |

## Usage

### Python API

```python
from harnessml.core.config import resolve_config
from harnessml.core.models import ModelRegistry, TrainOrchestrator
from harnessml.core.schemas.metrics import MetricRegistry

config = resolve_config("config/", file_map={"models": "models.yaml"})

model_registry = ModelRegistry.with_defaults()
orchestrator = TrainOrchestrator(model_registry, config["models"], output_dir="models/")
trained = orchestrator.train_all(X, y, feature_columns=cols)

metrics = MetricRegistry()
print(f"Brier: {metrics.get('binary', 'brier')(y_true, y_prob):.4f}")
```

### YAML-Driven Pipeline

```yaml
# config/pipeline.yaml
data:
  target_column: result
  fold_column: season
  entity_columns: [home, away]

models:
  xgb_main:
    type: xgboost
    preset: binary_default
    features: [elo_diff, win_pct_diff, scoring_margin_diff]
    params:
      max_depth: 4
      learning_rate: 0.05

ensemble:
  method: stacked
  calibration: spline
```

### Agent Workflow

```
# The agent never writes pipeline code. It declares intent:

models(action="add", name="lgb_tempo", preset="binary_default",
       features=["tempo_diff", "adj_efficiency_diff"])

pipeline(action="run_backtest")
# → Automatic: CV, training, calibration, ensemble, metrics, logging

pipeline(action="diagnostics")
# → Per-model breakdown, ensemble weights, calibration curves

experiments(action="create", name="exp-003-tempo-features")
# → Isolated overlay — production config untouched
```

## Design Philosophy

- **Declarative over imperative** — YAML config and registries, not boilerplate code
- **Defaults over decision fatigue** — sensible presets for models, CV, metrics
- **Automatic over manual** — caching, logging, fingerprinting, guardrails happen without intervention
- **Single source of truth** — config is the contract; overlays enable isolated testing
- **Deterministic** — fingerprinting ensures identical configs produce identical results
- **Everything configurable** — no hardcoded thresholds, metric lists, or domain assumptions

## Development

```bash
uv sync                                          # install workspace
uv run pytest packages/harness-core/tests/ -q    # core tests
uv run pytest packages/harness-sports/tests/ -q  # sports plugin tests
uv run pytest -v                                 # verbose, all tests
```

Requires Python 3.11+. Managed by [uv](https://github.com/astral-sh/uv).

## License

MIT
