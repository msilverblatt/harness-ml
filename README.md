<div align="center">
<img src="https://raw.githubusercontent.com/msilverblatt/harness-ml/main/docs/assets/logo.svg" alt="HarnessML" width="600">
</div>

<div align="center">

**An Agent-Computer Interface (ACI) for machine learning.**<br>
Built on [protomcp](https://github.com/msilverblatt/protomcp) and the [Model Context Protocol](https://modelcontextprotocol.io/).

[![CI](https://github.com/msilverblatt/harness-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/msilverblatt/harness-ml/actions)
[![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/msilverblatt/2f3d3f20d7fedaca8332a75394eb7864/raw/tests.json)](https://github.com/msilverblatt/harness-ml/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lines of Code](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/msilverblatt/176db496aed8a6532131c9aa7447f59c/raw/loc.json)](https://github.com/msilverblatt/harness-ml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-de5fe9.svg?style=flat-square&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)

</div>

[![harness-core](https://img.shields.io/pypi/v/harness-core?style=flat-square&label=core)](https://pypi.org/project/harness-core/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-189fdd.svg?style=flat-square)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-2980b9.svg?style=flat-square)](https://lightgbm.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-ffcc00.svg?style=flat-square)](https://catboost.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c.svg?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-0078d4.svg?style=flat-square)](https://optuna.org/)
[![SHAP](https://img.shields.io/badge/SHAP-7c3aed.svg?style=flat-square)](https://shap.readthedocs.io/)

[![harness-plugin](https://img.shields.io/pypi/v/harness-plugin?style=flat-square&label=plugin)](https://pypi.org/project/harness-plugin/)
[![protomcp](https://img.shields.io/pypi/v/protomcp?style=flat-square&label=protomcp)](https://github.com/msilverblatt/protomcp)
[![MCP](https://img.shields.io/badge/MCP-native-6366f1.svg?style=flat-square)](https://modelcontextprotocol.io/)

[![harness-studio](https://img.shields.io/pypi/v/harness-studio?style=flat-square&label=studio)](https://pypi.org/project/harness-studio/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57.svg?style=flat-square&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg?style=flat-square&logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6.svg?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-646cff.svg?style=flat-square&logo=vite&logoColor=white)](https://vite.dev/)

[![harness-sports](https://img.shields.io/pypi/v/harness-sports?style=flat-square&label=sports)](https://pypi.org/project/harness-sports/)

## Claude Code for Machine Learning

> **Note:** HarnessML is an active research project exploring what agent-driven ML workflows can look like. It is functional and demonstrates the core ideas, but it has not been stress-tested for production use. Expect rough edges, missing error messages, and workflows that assume a cooperative agent. Feedback and contributions welcome.

Training machine learning models with coding agents is a frustrating experience. They generate endless boilerplate, waste tokens debugging it, and forget why they were running an experiment in the first place. Coding agents want to write code, so every experiment becomes an engineering project instead of a scientific one.

HarnessML fixes this. Built on [protomcp](https://github.com/msilverblatt/protomcp), the agent calls structured MCP tools instead of writing training loops. Data ingestion, feature engineering, cross-validation, calibration, ensembling, and diagnostics all run through deterministic tool calls. The experiment lifecycle is a server-defined workflow with dynamic tool visibility — the agent literally cannot skip logging results or start a new experiment without completing the current one, because the tools for those steps aren't visible until the right moment.

Every experiment requires a hypothesis. Every run is fingerprinted and logged. Experiments survive session boundaries.

_No code. Just data science._

https://github.com/user-attachments/assets/b5205517-c9c2-403b-8bbb-d0b15a79e807

## Quick Start

### Install the Plugin

In Claude Code:

```
/plugin marketplace add msilverblatt/harness-ml
/plugin install harnessml@msilverblatt-harness-ml
```

This installs the MCP server, experiment discipline skills, and everything Claude needs to start building models.

### Full Setup (with Studio + Demo)

```bash
git clone https://github.com/msilverblatt/harness-ml.git && cd harness-ml
uv sync
uv run harness-setup
```

For full setup options, see [For Humans](https://msilverblatt.github.io/harness-ml/for-humans).

## How It Works

The agent never writes training loops. It declares intent through MCP tool calls:

```python
# Ingest data
data(action="ingest", path="data/raw/housing.csv")

# Add models
models(action="add", name="xgb_main", type="xgboost", features=[...])
models(action="add", name="lgb_main", type="lightgbm", features=[...])

# Train and evaluate
pipeline(action="run_backtest")
# → CV splits, training, calibration, ensemble, metrics, logging — all automatic

# Compare to previous
pipeline(action="compare_latest")
# → "RMSE: 24,312 → 22,847 (improved)"

# Iterate with discipline
experiments(action="create", hypothesis="Adding neighborhood features captures location premium")
pipeline(action="run_backtest")
# → Isolated overlay — production config untouched until promoted
```

Every experiment requires a hypothesis. Every run is fingerprinted and logged. Experiments survive session boundaries.

## See It Work

### Raw CSV to stacked ensemble, under a minute

https://github.com/user-attachments/assets/c180d2b2-7ed1-4805-a08a-01b6fb3738ac

### Tuned model 5 minutes later

<img width="815" height="766" alt="Tuned model diagnostics" src="https://github.com/user-attachments/assets/684fb4e1-ae8a-41a5-85fe-a8e8a2882897" />

<sub>[examples/titanic](examples/titanic/) · [examples/ames-housing](examples/ames-housing/) · [examples/wine-quality](examples/wine-quality/)</sub>

## Harness Studio

A companion web dashboard that runs alongside the agent, giving you full observability into what it's doing and how the model is performing. See the [harness-studio repo](https://github.com/msilverblatt/harness-studio) for details.

### Dashboard

<img width="1292" height="948" alt="Studio dashboard" src="https://github.com/user-attachments/assets/4184f4d4-c5aa-4de6-8b2e-6c77b299c8df" />

Project vitals, experiment verdict breakdown, primary metric trend with error bars, live MCP activity feed, and a mini pipeline DAG -- all updating in real time as the agent works.

### Pipeline DAG

<img width="1555" height="944" alt="Pipeline DAG" src="https://github.com/user-attachments/assets/78580e3a-a1de-481d-b6af-886457533df9" />

Interactive pipeline topology. Click any node for full config details. Models added by experiments show with dashed borders and EXP badges. Running nodes pulse during training.

### Activity Monitor

<img width="1429" height="944" alt="Activity monitor" src="https://github.com/user-attachments/assets/d502e121-3850-4681-872a-603ac4113c06" />

Track experiments as they run with MCP progress messages for all long-running tool calls.

### Experiments

<img width="1561" height="936" alt="Experiments tab" src="https://github.com/user-attachments/assets/223a5f69-70b8-46ef-a807-9243507432e7" />

Every experiment with its hypothesis, verdict, and metric deltas. Trend chart tracks the primary metric across iterations with error bars from cross-validation folds. Side-by-side comparison for any two runs.

### Diagnostics

<img width="1559" height="944" alt="Diagnostics tab" src="https://github.com/user-attachments/assets/e80a321e-96fd-4b22-b374-408a8da11102" />

Per-run deep dive: headline metrics, meta-learner coefficients, model correlation heatmap, calibration curves, residual plots, per-fold breakdown, and the full markdown report.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     harness-core                        │
│  schemas · config · guardrails · models · runner        │
│  feature_eng · calibration · views · sources            │
├────────────────────────┬────────────────────────────────┤
│    harness-plugin      │       harness-sports           │
│    MCP server (pmcp)   │       domain plugin            │
│    workflows + groups  │       matchup prediction       │
├────────────────────────┴────────────────────────────────┤
│                    harness-studio                       │
│    companion dashboard · real-time observability        │
│    FastAPI + React · SQLite events · WebSocket live     │
└─────────────────────────────────────────────────────────┘
```

| Package | What it does |
|---------|-------------|
| **harness-core** | Engine: schemas, config, guardrails, 13 model wrappers, runner, feature store, views, calibration, 45 metrics across 6 task types, data sources, preprocessing, feature selection, drift detection, conformal prediction, explainability |
| **harness-plugin** | MCP server built on [protomcp](https://github.com/msilverblatt/protomcp) -- `@tool_group` actions + `@workflow` for experiment discipline via dynamic tool visibility. 10 skills |
| **harness-studio** | Companion dashboard -- live activity, pipeline DAG, experiments, diagnostics, HTML report export. FastAPI + React + SQLite |
| **harness-sports** | Optional domain plugin for matchup prediction (hooks into core via registry) |

## MCP Tools

| Tool | What the agent can do |
|------|----------------------|
| `data` | Ingest sources, validate, fill nulls, rename, derive columns, manage views, upload to Drive/Kaggle |
| `features` | Register features, test transforms, discover correlations, analyze diversity, auto-search interactions |
| `models` | Add/update/clone models, batch operations, class weighting, append/remove features |
| `configure` | Init projects, set backtest/ensemble config, run guardrail checks |
| `pipeline` | Run backtests, predict, diagnostics, compare runs, explain models, workflow progress, export notebooks |
| `experiments` | Create/run/promote experiments with config overlays, required hypothesis, Bayesian exploration |
| `competitions` | Simulations, brackets, scoring for tournament events |

## What's Inside

<details>
<summary><b>Models</b> -- 13 wrappers</summary>
<br>

XGBoost · LightGBM · CatBoost · Random Forest · Logistic Regression · ElasticNet · MLP (PyTorch) · TabNet · TabPFN · SVM · HistGradientBoosting · GAM (PyGAM) · NGBoost

All configurable via YAML with eval_set/early stopping, normalization, class weighting, and inspect-based kwargs forwarding.
</details>

<details>
<summary><b>Metrics</b> -- 45 across 6 task types</summary>
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
<summary><b>Calibration</b> -- 4 methods</summary>
<br>

Spline (PCHIP) · Isotonic · Platt · Beta -- all with save/load and fitted state tracking.
</details>

<details>
<summary><b>CV Strategies</b> -- 8 strategies</summary>
<br>

`leave_one_out` (symmetric LOSO) · `expanding_window` · `sliding_window` · `k_fold` · `purged_kfold` · `stratified_kfold` · `group_kfold` · `bootstrap`
</details>

<details>
<summary><b>View Engine</b> -- 22 transform steps</summary>
<br>

`filter` `select` `derive` `group_by` `join` `union` `unpivot` `sort` `head` `rolling` `cast` `distinct` `rank` `isin` `cond_agg` `lag` `ewm` `diff` `trend` `encode` `bin` `datetime` `null_indicator`
</details>

<details>
<summary><b>Guardrails</b> -- 12 total</summary>
<br>

3 **non-overridable** (data leakage, temporal integrity, critical path) + 9 overridable. Violations are logged for audit. Once locked, they cannot be bypassed.
</details>

<details>
<summary><b>Preprocessing</b></summary>
<br>

- **Scaling** -- zscore, robust, quantile (leakage-safe: fit on train, transform both)
- **Imputation** -- median, mean, zero, KNN, iterative
- **Encoding** -- frequency encoding for categoricals
- **Feature selection** -- k_best, RFE, correlation clustering
</details>

<details>
<summary><b>Exploration</b></summary>
<br>

- **Auto-search** -- discover feature interactions, lags, rolling aggregations
- **Feature diversity** -- overlap matrix, diversity score, redundant detection
- **Bayesian search** -- Optuna TPE over features, models, hyperparams, ensemble settings
- **Workflow tracking** -- phased gates ensure feature discovery and model diversity before tuning
</details>

<details>
<summary><b>Advanced</b></summary>
<br>

- **Ensemble diversity** -- disagreement, Q-statistic, kappa, correlation metrics
- **Meta-learner types** -- logistic (default), ridge, GBM
- **Drift detection** -- KS test, PSI, multi-feature drift monitoring
- **Conformal prediction** -- split-conformal with finite-sample correction
- **Explainability** -- SHAP values, partial dependence plots, feature interactions
- **Text features** -- TF-IDF, count vectorizer extraction
- **Cyclical encoding** -- sin/cos pairs for periodic features
</details>

<details>
<summary><b>Experiment Discipline</b></summary>
<br>

- **Required hypothesis** -- every experiment must state what it expects and why
- **Required conclusion** -- document what was learned, not just pass/fail
- **Phased workflow** -- EDA → model diversity → feature engineering → tuning → ensemble
- **Workflow gates** -- soft warnings by default, hard blocks with `workflow.enforce_phases: true`
</details>

## Development

```bash
uv sync                                          # install workspace
uv run pytest packages/harness-core/tests/ -q    # core tests
uv run pytest packages/harness-sports/tests/ -q  # sports plugin tests
uv run pytest packages/harness-studio/tests/ -q  # studio tests
uv run pytest -v                                 # verbose, all tests
```

### Studio Frontend

```bash
cd packages/harness-studio/frontend
bun install
bun run dev                                      # Vite dev server on :5173
bash ../scripts/build_frontend.sh                # build + copy to Python package
```

Requires Python 3.11+. Managed by [uv](https://github.com/astral-sh/uv). Studio frontend uses [bun](https://bun.sh/).

## License

MIT
