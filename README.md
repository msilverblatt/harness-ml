```
 ██╗  ██╗ █████╗ ██████╗ ███╗   ██╗███████╗███████╗███████╗    ███╗   ███╗██╗
 ██║  ██║██╔══██╗██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝    ████╗ ████║██║
 ███████║███████║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗    ██╔████╔██║██║
 ██╔══██║██╔══██║██╔══██╗██║╚██╗██║██╔══╝  ╚════██║╚════██║    ██║╚██╔╝██║██║
 ██║  ██║██║  ██║██║  ██║██║ ╚████║███████╗███████║███████║    ██║ ╚═╝ ██║███████╗
 ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝
```
<div align="center">

**An Agent-Computer Interface (ACI) for machine learning.**<br>
Built natively on the [Model Context Protocol](https://modelcontextprotocol.io/).

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![MCP](https://img.shields.io/badge/MCP-native-6366f1.svg?style=flat-square)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/tests-1983%20passing-22c55e.svg?style=flat-square)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-f59e0b.svg?style=flat-square)](LICENSE)

<br>

</div>

<br>

## Why HarnessML?

Agents writing Python for ML hit a death spiral: boilerplate → error → stack trace → context exhaustion → hallucinated results. Existing frameworks weren't built for this — they're libraries for humans in notebooks.

HarnessML is an **Agent-Computer Interface**. One tool call per operation. Deterministic execution. Structured results back — no tracebacks, no codegen, no state management.

| | |
|---|---|
| **Zero boilerplate** | The agent declares a hypothesis, not a training loop |
| **Guardrailed** | 12 constraints block data leakage and temporal contamination before training starts |
| **Structured I/O** | Deterministic results back — not stack traces that burn context |
| **Persistent** | Every run fingerprinted and logged. Experiments survive session boundaries |

```
models(action="add", name="xgb_main", features=[...])
pipeline(action="run_backtest")
  → CV splits, training, calibration, ensemble, metrics, logging
pipeline(action="compare_latest")
  → "Brier: 0.182 → 0.179 (↑ +0.003)"
```
<br>
**Raw CSV to stacked ensemble, under a minute**

https://github.com/user-attachments/assets/c180d2b2-7ed1-4805-a08a-01b6fb3738ac

<sub>[examples/titanic](examples/titanic/)</sub>

<br>

## Quick Start

```bash
git clone https://github.com/msilverblatt/harness-ml.git && cd harness-ml
uv sync
uv run pytest  # 1983 tests
```

<br>

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     harness-core                        │
│  schemas · config · guardrails · models · runner        │
│  feature_eng · calibration · views · sources            │
├────────────────────────┬────────────────────────────────┤
│    harness-plugin      │       harness-sports           │
│    MCP server          │       domain plugin            │
│    hot-reload handlers │       matchup prediction       │
└────────────────────────┴────────────────────────────────┘
```

| Package | What it does |
|---------|-------------|
| **harness-core** | Engine: schemas, config, guardrails, 8 model wrappers, runner, feature store, views, calibration, metrics, data sources |
| **harness-plugin** | MCP server with hot-reloadable handlers — change handler code, no restart needed |
| **harness-sports** | Optional domain plugin for matchup prediction (hooks into core via registry) |

<br>

## What's Inside

<details>
<summary><b>Models</b> — 8 wrappers</summary>
<br>

XGBoost · LightGBM · CatBoost · Random Forest · Logistic Regression · ElasticNet · MLP (PyTorch) · TabNet

All configurable via YAML with eval_set/early stopping, normalization, class weighting, and inspect-based kwargs forwarding.
</details>

<details>
<summary><b>Metrics</b> — 45 across 6 task types</summary>
<br>

| Task | Examples |
|------|----------|
| Binary | brier, accuracy, log_loss, ece, auroc, f1, precision, recall |
| Multiclass | macro/micro/weighted variants of all classification metrics |
| Regression | rmse, mae, r2, mape |
| Ranking | ndcg, mrr, map |
| Survival | concordance_index, brier_survival |
| Probabilistic | crps, calibration, sharpness |
</details>

<details>
<summary><b>Features</b> — 4 types</summary>
<br>

- **entity** — per-entity stats, auto-generates pairwise diffs
- **pairwise** — formula features across entity pairs
- **instance** — context columns passed through
- **regime** — boolean flags that gate feature sets
</details>

<details>
<summary><b>Calibration</b> — 4 methods</summary>
<br>

Spline (PCHIP) · Isotonic · Platt · Beta — all with save/load and fitted state tracking.
</details>

<details>
<summary><b>CV Strategies</b></summary>
<br>

`leave_one_out` (symmetric LOSO) · `expanding_window` · `sliding_window` · `k_fold` · `stratified`
</details>

<details>
<summary><b>View Engine</b> — 22 transform steps</summary>
<br>

`filter` `select` `derive` `group_by` `join` `union` `unpivot` `sort` `head` `rolling` `cast` `distinct` `rank` `isin` `cond_agg` `lag` `ewm` `diff` `trend` `encode` `bin` `datetime` `null_indicator`
</details>

<details>
<summary><b>Guardrails</b> — 12 total</summary>
<br>

3 **non-overridable** (data leakage, temporal integrity, critical path) + 9 overridable. Violations are logged for audit. Once locked, they cannot be bypassed.
</details>

<details>
<summary><b>Exploration</b></summary>
<br>

- **Auto-search** — discover feature interactions, lags, rolling aggregations
- **Feature diversity** — overlap matrix, diversity score, redundant detection
- **Bayesian search** — Optuna TPE over features, models, hyperparams, ensemble settings
</details>

<br>

## MCP Tools

| Tool | Actions |
|------|---------|
| `data` | ingest sources, validate, fill nulls, rename, derive columns, manage views, upload to Drive/Kaggle |
| `features` | register features, test transforms, discover correlations, analyze diversity |
| `models` | add/update/clone models, batch operations, class weighting, append/remove features |
| `configure` | init projects, set backtest/ensemble config, run guardrail checks |
| `pipeline` | run backtests, predict, diagnostics, compare runs, compare latest, explain models, export notebooks |
| `experiments` | create/run/promote experiments with config overlays |
| `competitions` | simulations, brackets, scoring for tournament events |

<br>

## Usage

<details>
<summary><b>Python API</b></summary>
<br>

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
</details>

<details>
<summary><b>YAML-Driven Pipeline</b></summary>
<br>

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
</details>

<details open>
<summary><b>Agent Workflow</b></summary>
<br>

```python
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
</details>

<br>

## Design Philosophy

- **Agent UX over human UX** — designed for the cognitive profile and context limitations of an LLM, not for a human in a Jupyter notebook. Tools return structured summaries, not massive tracebacks
- **Declarative over imperative** — YAML config and registries, not boilerplate code
- **Defaults over decision fatigue** — sensible presets for models, CV, metrics
- **Automatic over manual** — caching, logging, fingerprinting, guardrails happen without intervention
- **Single source of truth** — config is the contract; overlays enable isolated testing
- **Deterministic** — fingerprinting ensures identical configs produce identical results
- **Everything configurable** — no hardcoded thresholds, metric lists, or domain assumptions

<br>

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
