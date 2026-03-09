# HarnessML -- AI Agent Instructions

## What This Is

HarnessML is a general-purpose agentic ML framework with 4 packages in a uv
workspace monorepo. It supports any ML task type (binary classification,
multiclass, regression, ranking, survival, probabilistic forecasting).

Architecture: **harness-core** (schemas, config, guardrails, models, runner,
feature engineering, metrics, data sources) + **harness-plugin** (MCP server,
thin async dispatcher with hot-reloadable handlers) + **harness-studio**
(companion web dashboard for real-time observability) + **harness-sports**
(optional domain plugin for matchup prediction).

## Tech Stack

- Python 3.11+, managed by **uv** (always use `uv run`, never bare `python`)
- Pydantic v2 for all schemas and contracts
- OmegaConf for config deep merge
- FastMCP for async MCP server with hot-reload
- scikit-learn, XGBoost, CatBoost, LightGBM, PyTorch (MLP, TabNet)
- Optional: shap, matplotlib, pandera, optuna, nbformat, google-api-python-client, kaggle
- Studio: FastAPI, uvicorn, sqlite3 (stdlib), React 19, Vite, bun
- Namespace packages: no `__init__.py` at `src/harnessml/` level

## Package Map

| Package | Purpose | Key Modules |
|---------|---------|-------------|
| `harness-core` | All core engine code | `schemas/`, `config/`, `guardrails/`, `models/`, `runner/`, `feature_eng/` |
| `harness-plugin` | MCP server (thin dispatcher) | `mcp_server.py`, `handlers/` (models, data, features, experiments, config, pipeline) |
| `harness-studio` | Companion web dashboard | `server.py`, `event_store.py`, `routes/`, `frontend/` (React/Vite) |
| `harness-sports` | Optional sports domain plugin | `matchups.py`, `hooks.py` (registers into core extension points) |

### harness-core submodules

| Submodule | Purpose |
|-----------|---------|
| `core.schemas` | Pydantic contracts (`contracts.py`) + MetricRegistry with 45 metrics across 6 task types (`metrics.py`) |
| `core.config` | YAML loading + OmegaConf deep merge |
| `core.guardrails` | Safety guardrails (leakage, temporal, naming) |
| `core.models` | Model wrappers (XGBoost, LightGBM, CatBoost, RF, Logistic, ElasticNet, MLP, TabNet) + registry |
| `core.runner` | Pipeline orchestration, project, hooks, CLI, DAG, matchups |
| `core.runner.data` | Data ingestion, pipeline, profiling, utils, loaders |
| `core.runner.features` | Feature store, engine, cache, discovery, diversity, selection, auto-search, utils |
| `core.runner.training` | Trainer, CV strategies, preprocessing, meta-learner, calibration (Spline/Isotonic/Platt/Beta), postprocessing, prediction cache, fingerprint |
| `core.runner.experiments` | Experiment schema, journal, manager, logger |
| `core.runner.views` | View executor (pandas + polars), resolver, polars compat |
| `core.runner.analysis` | Diagnostics, reporting, explainability, drift, conformal, ensemble diversity, viz |
| `core.runner.optimization` | HPO, exploration, sweep, pipeline planner |
| `core.runner.validation` | Guards, stage guards, validation, validator |
| `core.runner.scaffold` | Project scaffold, presets, server gen, notebook generation |
| `core.runner.workflow` | Workflow tracker, run manager |
| `core.runner.config_writer` | Config writing helpers |
| `core.runner.sources` | Source registry, freshness tracking, schema validation, adapters (file/url/api/computed) |
| `core.runner.drives` | Cloud adapters: Google Drive (OAuth upload/folders), Kaggle (dataset/notebook upload) |
| `core.feature_eng` | Feature engineering registry + transforms |

## Key Conventions

- All packages under `packages/`, source in `packages/<name>/src/harnessml/<subpkg>/`
- Namespace packages: `harnessml.core.schemas`, `harnessml.core.runner`, etc. (no root `__init__.py`)
- Import paths use `harnessml.core.*` for all core code, `harnessml.plugin.*` for MCP, `harnessml.sports.*` for sports
- Registry pattern used in features, models, sources, metrics, and guardrails
- Hook system for domain plugins (`core.runner.hooks.HookRegistry`)
- TDD: write tests alongside implementation, run with `uv run pytest`
- MCP handlers are hot-reloadable in dev mode (`HARNESS_DEV=1`) — no server restart needed for handler changes

## How to Add a New Model

1. Create `packages/harness-core/src/harnessml/core/models/wrappers/my_model.py`
2. Subclass `BaseModel`, implement `fit`, `predict_proba`, `save`, `load`
3. Register in `ModelRegistry.with_defaults()` (wrap in try/except for optional deps)
4. Add tests in `packages/harness-core/tests/models/`

## How to Add a New Feature

Use the MCP tool or config_writer:
1. `features(action="add", name="...", formula="...", type="grouped|instance|formula|regime")`
2. For batch: `features(action="add_batch", features=[{...}, ...])`
3. For auto-discovery: `features(action="auto_search", features=[...], search_types=["interactions","lags","rolling"])`

## How to Add a New View Step

1. Add Pydantic model to `packages/harness-core/src/harnessml/core/runner/schema.py` (in TransformStep union)
2. Add executor function to `packages/harness-core/src/harnessml/core/runner/view_executor.py`
3. Register in `_dispatch` dict
4. Available steps: filter, select, derive, group_by, join, union, unpivot, sort, head, rolling, cast, distinct, rank, isin, cond_agg, lag, ewm, diff, trend, encode, bin, datetime, null_indicator

## How to Add a New Metric

1. Write metric function in `packages/harness-core/src/harnessml/core/schemas/metrics.py`
2. Register: `MetricRegistry.register("task_type", "name", fn)`
3. Task types: binary, multiclass, regression, ranking, survival, probabilistic

## MCP Server Architecture

- `mcp_server.py` — thin async dispatcher, tool signatures + docstrings only
- `handlers/*.py` — all business logic, hot-reloadable in dev mode
- `handlers/_validation.py` — enum validation with fuzzy match, cross-parameter hints
- `handlers/_common.py` — shared helpers (resolve_project_dir, parse_json_param)
- Handler dispatch pattern: `ACTIONS` dict → `dispatch(action, **kwargs)`
- Changes to handler code: no restart needed (hot-reload)
- Changes to tool signatures/docstrings: restart required

## Harness Studio

Companion web dashboard providing real-time observability. Reads existing project
artifacts (config YAMLs, journal JSONL, run outputs) + SQLite event log. Zero
changes to harness-core.

- **Event layer**: MCP server emits events to SQLite via fail-safe emitter
- **Backend**: FastAPI serves REST + WebSocket, reads project artifacts directly
- **Frontend**: React/Vite with 4 tabs (Activity, DAG, Experiments, Diagnostics)
- **Build**: `bash packages/harness-studio/scripts/build_frontend.sh` copies Vite output to Python package
- **Run**: `uv run harness-studio --project-dir <path>` serves on port 8421
- **Frontend dev**: `cd packages/harness-studio/frontend && bun run dev`

### Studio Architecture

- `event_store.py` — SQLite WAL-mode event store (record, query, session_stats)
- `broadcaster.py` — asyncio.Queue fan-out for WebSocket live streaming
- `routes/` — events, project (config + DAG), experiments (journal), runs (metrics, calibration, correlations)
- `frontend/` — React 19 + TypeScript + CSS Modules + design tokens
- Event emission in `harness-plugin/mcp_server.py` — fail-safe, swallows exceptions

## Testing

```bash
uv run pytest                                    # all tests
uv run pytest packages/harness-core/tests/        # core tests
uv run pytest packages/harness-sports/tests/      # sports plugin tests
uv run pytest packages/harness-studio/tests/      # studio tests
uv run pytest packages/harness-core/tests/runner/  # runner subsystem
uv run pytest -v                                 # verbose
```

## Experiment Discipline

- **Hypothesis is required** when creating experiments (via MCP or config_writer)
- **Conclusion** should be logged after every experiment explaining what was learned
- **Exhaust strategies** — try 3-5 meaningfully different configs before abandoning a strategy
- **Workflow phases**: EDA → Feature Discovery → Model Diversity → Feature Engineering → Tuning → Ensemble
- Use `pipeline(action="progress")` to check workflow phase completion
- Optional hard gates: set `workflow.enforce_phases: true` in pipeline.yaml to block premature tuning

## Skills (docs/skills/)

| Skill | Purpose |
|-------|---------|
| `harness-run-experiment` | Disciplined experiment execution with required hypothesis/conclusion |
| `harness-explore-space` | Phased workflow preventing premature tuning |
| `harness-domain-research` | Hypothesis-driven domain research for feature engineering |

## What NOT to Do

- Do not bypass guardrails — non-overridable ones exist for safety (leakage, temporal, critical path)
- Do not hardcode paths, thresholds, or magic numbers
- Do not create parallel versions of files — extend existing modules
- Do not skip `uv run` — bare `python` will not find workspace packages
- Do not put `__init__.py` at the `src/harnessml/` level (breaks namespace packages)
- Do not modify production config directly — use experiment overlays
- Do not import from old paths (`harnessml.schemas`, `harnessml.runner`, etc.) — use `harnessml.core.*`
