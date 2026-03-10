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

[![CI](https://github.com/msilverblatt/harness-ml/actions/workflows/tests.yml/badge.svg)](https://github.com/msilverblatt/harness-ml/actions)
[![harness-core](https://img.shields.io/pypi/v/harness-core?style=flat-square&label=core)](https://pypi.org/project/harness-core/)
[![harness-plugin](https://img.shields.io/pypi/v/harness-plugin?style=flat-square&label=plugin)](https://pypi.org/project/harness-plugin/)
[![harness-studio](https://img.shields.io/pypi/v/harness-studio?style=flat-square&label=studio)](https://pypi.org/project/harness-studio/)
[![harness-sports](https://img.shields.io/pypi/v/harness-sports?style=flat-square&label=sports)](https://pypi.org/project/harness-sports/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Lines of Code](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/msilverblatt/176db496aed8a6532131c9aa7447f59c/raw/loc.json)](https://github.com/msilverblatt/harness-ml)
[![Tests](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/msilverblatt/2f3d3f20d7fedaca8332a75394eb7864/raw/tests.json)](https://github.com/msilverblatt/harness-ml/actions)

[![MCP](https://img.shields.io/badge/MCP-native-6366f1.svg?style=flat-square)](https://modelcontextprotocol.io/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-e92063.svg?style=flat-square&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg?style=flat-square&logo=react)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6.svg?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-646cff.svg?style=flat-square&logo=vite&logoColor=white)](https://vite.dev/)

</div>

## Claude Code for Machine Learning

Training machine learning models with coding agents is a frustrating experience. 
- They are fantastic at generating endless boilerplate, and even better at wasting tokens debugging it.
- They are great at setting up your experiment, but terrible at writing down the results (or even remembering why they were doing it in the first place).

Coding agents will always want to do what they do best: write code. So building a model becomes an engineering project for every experiment you want to run. And keeping them focused on the task? They are simply not built for the rigor of the scientific method.

That changes now. Claude is no longer a software engineer working on machine learning. _Claude is now a data scientist._

**HarnessML is a complete ML framework, built for agents-first.** Claude Code calls `models(action="add")` or `pipeline(action="run_backtest")` instead of writing training loops — data ingestion, feature engineering, cross-validation, calibration, ensembling, and diagnostics all run through structured tool calls with deterministic results back.

_No code. Just data science.
_


https://github.com/user-attachments/assets/b5205517-c9c2-403b-8bbb-d0b15a79e807


## Harness Studio

A companion web dashboard that runs alongside the agent, giving you full observability into what it's doing and how the model is performing.

### Dashboard
<!-- TODO: Add screenshot — Dashboard overview -->
<img width="1292" height="948" alt="Screenshot 2026-03-08 at 12 06 19 AM" src="https://github.com/user-attachments/assets/4184f4d4-c5aa-4de6-8b2e-6c77b299c8df" />

Project vitals, experiment verdict breakdown, primary metric trend with error bars, live MCP activity feed, and a mini pipeline DAG — all updating in real time as the agent works.

### Pipeline DAG
<!-- TODO: Add screenshot — Full DAG view -->
<img width="1555" height="944" alt="Screenshot 2026-03-07 at 11 48 34 PM" src="https://github.com/user-attachments/assets/78580e3a-a1de-481d-b6af-886457533df9" />

Interactive pipeline topology. Click any node for full config details. Models added by experiments show with dashed borders and EXP badges. Running nodes pulse during training.

## Activity Monitor
<img width="1429" height="944" alt="Screenshot 2026-03-08 at 12 32 15 AM" src="https://github.com/user-attachments/assets/d502e121-3850-4681-872a-603ac4113c06" />

Track experiments as they run with MCP progress messages for all long-running tool calls.

### Experiments
<!-- TODO: Add screenshot — Experiments table with trend chart -->
<img width="1561" height="936" alt="Screenshot 2026-03-07 at 11 45 24 PM" src="https://github.com/user-attachments/assets/223a5f69-70b8-46ef-a807-9243507432e7" />

Every experiment with its hypothesis, verdict, and metric deltas. Trend chart tracks the primary metric across iterations with error bars from cross-validation folds. Side-by-side comparison for any two runs.

### Diagnostics
<!-- TODO: Add screenshot — Diagnostics page -->
<img width="1559" height="944" alt="Screenshot 2026-03-07 at 11 47 32 PM" src="https://github.com/user-attachments/assets/e80a321e-96fd-4b22-b374-408a8da11102" />

Per-run deep dive: headline metrics, meta-learner coefficients, model correlation heatmap, calibration curves, residual plots, per-fold breakdown, and the full markdown report.

<br>


## Quick Start

### Install the Plugin

In Claude Code:

```
/plugin marketplace add msilverblatt/harness-ml
/plugin install harnessml@msilverblatt-harness-ml
```

This installs the MCP server, experiment discipline skills, and everything Claude needs to start building models.

### Full Setup (with Studio + Demo)

To also get the companion dashboard and a demo project:

```bash
git clone https://github.com/msilverblatt/harness-ml.git && cd harness-ml
uv sync
uv run harness-setup
```

For full setup options, see [For Humans](https://msilverblatt.github.io/harness-ml/for-humans).

<br>

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

<br>

## See It Work In Claude Code

### Raw CSV to stacked ensemble, under a minute

https://github.com/user-attachments/assets/c180d2b2-7ed1-4805-a08a-01b6fb3738ac

### Tuned model 5 minutes later

<img width="815" height="766" alt="Tuned model diagnostics" src="https://github.com/user-attachments/assets/684fb4e1-ae8a-41a5-85fe-a8e8a2882897" />

<sub>[examples/titanic](examples/titanic/) · [examples/ames-housing](examples/ames-housing/) · [examples/wine-quality](examples/wine-quality/)</sub>

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
├────────────────────────┴────────────────────────────────┤
│                    harness-studio                       │
│    companion dashboard · real-time observability        │
│    FastAPI + React · SQLite events · WebSocket live     │
└─────────────────────────────────────────────────────────┘
```

| Package | What it does |
|---------|-------------|
| **harness-core** | Engine: schemas, config, guardrails, 12 model wrappers, runner, feature store, views, calibration, 45 metrics across 6 task types, data sources, preprocessing, feature selection, drift detection, conformal prediction, explainability |
| **harness-plugin** | MCP server with hot-reloadable handlers — change handler code, no restart needed |
| **harness-studio** | Companion dashboard — live activity, pipeline DAG, experiments, diagnostics. FastAPI + React + SQLite |
| **harness-sports** | Optional domain plugin for matchup prediction (hooks into core via registry) |

<br>

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

<br>

## What's Inside

<details>
<summary><b>Models</b> — 12 wrappers</summary>
<br>

XGBoost · LightGBM · CatBoost · Random Forest · Logistic Regression · ElasticNet · MLP (PyTorch) · TabNet · SVM · HistGradientBoosting · GAM (PyGAM) · NGBoost

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
<summary><b>Calibration</b> — 4 methods</summary>
<br>

Spline (PCHIP) · Isotonic · Platt · Beta — all with save/load and fitted state tracking.
</details>

<details>
<summary><b>CV Strategies</b> — 7 strategies</summary>
<br>

`leave_one_out` (symmetric LOSO) · `expanding_window` · `sliding_window` · `k_fold` · `purged_kfold` · `stratified_kfold` · `group_kfold` · `bootstrap`
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
<summary><b>Preprocessing</b></summary>
<br>

- **Scaling** — zscore, robust, quantile (leakage-safe: fit on train, transform both)
- **Imputation** — median, mean, zero, KNN, iterative
- **Encoding** — frequency encoding for categoricals
- **Feature selection** — k_best, RFE, correlation clustering
</details>

<details>
<summary><b>Exploration</b></summary>
<br>

- **Auto-search** — discover feature interactions, lags, rolling aggregations
- **Feature diversity** — overlap matrix, diversity score, redundant detection
- **Bayesian search** — Optuna TPE over features, models, hyperparams, ensemble settings
- **Workflow tracking** — phased gates ensure feature discovery and model diversity before tuning
</details>

<details>
<summary><b>Advanced</b></summary>
<br>

- **Ensemble diversity** — disagreement, Q-statistic, kappa, correlation metrics
- **Meta-learner types** — logistic (default), ridge, GBM
- **Drift detection** — KS test, PSI, multi-feature drift monitoring
- **Conformal prediction** — split-conformal with finite-sample correction
- **Explainability** — SHAP values, partial dependence plots, feature interactions
- **Text features** — TF-IDF, count vectorizer extraction
- **Cyclical encoding** — sin/cos pairs for periodic features
</details>

<details>
<summary><b>Experiment Discipline</b></summary>
<br>

- **Required hypothesis** — every experiment must state what it expects and why
- **Required conclusion** — document what was learned, not just pass/fail
- **Phased workflow** — EDA → model diversity → feature engineering → tuning → ensemble
- **Workflow gates** — soft warnings by default, hard blocks with `workflow.enforce_phases: true`
</details>

<br>

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
