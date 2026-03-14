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
- protomcp for MCP server (`@tool_group`/`@action` pattern, `pmcp run server.py`)
- scikit-learn, XGBoost, CatBoost, LightGBM, PyTorch (MLP, TabNet)
- Optional: shap, matplotlib, pandera, optuna, nbformat, google-api-python-client, kaggle, pygam, ngboost
- Studio: FastAPI, uvicorn, sqlite3 (stdlib), React 19, Vite, bun
- Namespace packages: no `__init__.py` at `src/harnessml/` level

## Package Map

| Package | Purpose | Key Modules |
|---------|---------|-------------|
| `harness-core` | All core engine code | `schemas/`, `config/`, `guardrails/`, `models/`, `runner/`, `feature_eng/` |
| `harness-plugin` | MCP server (protomcp) | `server.py`, `handlers/` (8 tool_group classes: models, data, features, experiments, config, pipeline, notebook, competitions) |
| `harness-studio` | Companion web dashboard | `server.py`, `event_store.py`, `routes/`, `frontend/` (React/Vite) |
| `harness-sports` | Optional sports domain plugin | `matchups.py`, `hooks.py` (registers into core extension points) |

### harness-core submodules

| Submodule | Purpose |
|-----------|---------|
| `core.schemas` | Pydantic contracts (`contracts.py`) + MetricRegistry with 45 metrics across 6 task types (`metrics.py`) |
| `core.config` | YAML loading + OmegaConf deep merge |
| `core.guardrails` | Safety guardrails (leakage, temporal, naming) |
| `core.models` | Model wrappers (XGBoost, LightGBM, CatBoost, RF, Logistic, ElasticNet, MLP, TabNet, TabPFN, SVM, HistGBM, GAM, NGBoost) + registry |
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
- MCP handlers are hot-reloadable via `pmcp dev server.py` — all changes take effect automatically

## How to Add a New Model

1. Create `packages/harness-core/src/harnessml/core/models/wrappers/my_model.py`
2. Subclass `BaseModel`, implement `fit`, `predict_proba`, `save`, `load`
3. Register in `ModelRegistry.with_defaults()` (wrap in try/except for optional deps)
4. Add tests in `packages/harness-core/tests/models/`

## How to Add a New Feature

**For the feature registry** (FeatureStore — declarative features used in training):
1. `features(action="add", name="...", formula="...", type="entity|pairwise|regime")` — type is auto-inferred if omitted (formula→pairwise, condition→regime, source→entity)
2. For batch: `features(action="add_batch", features=[{...}, ...])`
3. For auto-discovery: `features(action="auto_search", features=[...], search_types=["interactions","lags","rolling"])`

**For simple derived columns** (pandas expressions applied to the data directly):
- `data(action="derive_column", name="col_name", expression="df['a'] + df['b']")`
- Use this for straightforward column transforms before feature engineering

## How to Add a New View Step

1. Add Pydantic model to `packages/harness-core/src/harnessml/core/runner/schema.py` (in TransformStep union)
2. Add executor function to `packages/harness-core/src/harnessml/core/runner/views/executor.py`
3. Register in `_dispatch` dict
4. Available steps: filter, select, derive, group_by, join, union, unpivot, sort, head, rolling, cast, distinct, rank, isin, cond_agg, lag, ewm, diff, trend, encode, bin, datetime, null_indicator

## How to Add a New Metric

1. Write metric function in `packages/harness-core/src/harnessml/core/schemas/metrics.py`
2. Register: `MetricRegistry.register("task_type", "name", fn)`
3. Task types: binary, multiclass, regression, ranking, survival, probabilistic

## MCP Server Architecture

Built on [protomcp](https://github.com/msilverblatt/protomcp) — a language-agnostic MCP runtime.

- `server.py` — 25-line entry point, imports infrastructure + handler modules, calls `protomcp.run()`
- `handlers/*.py` — each file is a `@tool_group` class with `@action` methods delegating to `_handle_*` business logic
- `pmcp_middleware.py` — local middleware for error formatting and auto-install of missing packages
- `pmcp_telemetry.py` — telemetry sink forwarding tool call events to Studio's SQLite event store
- `pmcp_sidecar.py` — auto-starts Studio as a companion process on first tool call
- `handlers/_validation.py` — runtime validation helpers (validate_required, validate_enum with fuzzy match)
- `handlers/_common.py` — shared helpers (resolve_project_dir, parse_json_param)
- Server runs via `pmcp run server.py` — pmcp handles MCP protocol, transport, and session management
- Hot-reload: `pmcp dev server.py` watches for file changes and reloads automatically

### Experiment Workflow (Dynamic Tool Visibility)

The experiment lifecycle is a protomcp `@workflow` that controls which tools the agent can see at each step:

```
experiment.create → [experiment.write_overlay] → experiment.run → experiment.log_result → [experiment.promote | experiment.done]
```

- **Structural discipline enforcement**: The agent cannot start a new experiment without completing `log_result` — the `experiment.create` tool is literally not visible while a workflow is active
- **Dynamic tool lists**: At each step, only the valid next steps are visible. After `experiment.run`, the agent sees only `experiment.log_result` (plus all non-experiment tools via `allow_during`)
- **`allow_during` globs**: During an active experiment workflow, all non-experiment tools remain visible (`notebook.*`, `configure.*`, `data.*`, `features.*`, `models.*`, `pipeline.*`). The agent can still explore data, check features, and write notebook entries
- **`no_cancel` on run**: Once a backtest is running, the agent cannot cancel (avoids wasted compute)
- **Discipline gates in `create`**: Plan-exists check, previous-experiment-logged check, and plan-freshness check run inside the `create` step handler

Standalone experiment actions (not part of the workflow): `experiments(action="quick_run")`, `experiments(action="explore")`, `experiments(action="compare")`, `experiments(action="journal")`

### Notable MCP Actions

- `data(action="snapshot", name="...")` / `data(action="restore_snapshot", name="...")` — save and restore config YAMLs + features.parquet for rollback
- `data(action="derive_column", name="...", expression="...")` — derive a new column from a pandas expression
- `configure(action="suggest_cv")` — analyze features data and recommend a CV strategy
- `experiments(action="quick_run", description="...", overlay={...})` — create+configure+run experiment in one call
- `models(action="clone", name="source", new_name="target")` — clone a model config with optional overrides (features, params, active)
- `pipeline(action="explain", method="builtin")` — use tree `feature_importances_` instead of SHAP (faster, no extra dependency)
- `notebook(action="write", type="phase_transition", content="...")` — record explicit workflow phase transitions for the phase tracker

## Plugin Auto-Discovery

Domain plugins are discovered via Python entry points. harness-plugin declares the
`harnessml.plugins` entry point group, and harness-sports registers its hooks there:

```toml
# In harness-plugin/pyproject.toml
[project.entry-points."harnessml.plugins"]
sports = "harnessml.sports.hooks:register"
```

This means `import harnessml.sports` automatically registers sports hooks (column
candidates, renames, competition narrative) into `HookRegistry`. New domain plugins
follow the same pattern: add an entry point pointing to a `register()` function.

## Harness Studio

Companion web dashboard providing real-time observability. Reads existing project
artifacts (config YAMLs, journal JSONL, run outputs) + SQLite event log. Zero
changes to harness-core.

- **Event layer**: MCP server emits events to SQLite via fail-safe emitter
- **Backend**: FastAPI serves REST + WebSocket, reads project artifacts directly
- **Frontend**: React/Vite with tabs (Dashboard, Activity, DAG, Experiments, Diagnostics, Data, Features, Models, Ensemble, Config, Predictions, Notebook, Preferences)
- **Export**: Static HTML report generation via Jinja2 templates (`export.py`, `routes/export.py`)
- **Build**: `bash packages/harness-studio/scripts/build_frontend.sh` copies Vite output to Python package
- **Run**: `uv run harness-studio --project-dir <path>` serves on port 8421
- **Frontend dev**: `cd packages/harness-studio/frontend && bun run dev`

### Studio Architecture

- `event_store.py` — SQLite WAL-mode event store (record, query, session_stats)
- `broadcaster.py` — asyncio.Queue fan-out for WebSocket live streaming
- `routes/` — events, project (config + DAG), experiments (journal), runs (metrics, calibration, correlations)
- `frontend/` — React 19 + TypeScript + CSS Modules + design tokens
- Event emission in `harness-plugin/pmcp_telemetry.py` — telemetry sink, fail-safe, swallows exceptions

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

## Experiment Discipline (Programmatic Gates)

The following are enforced in code (`config_writer/experiments.py`). Experiment creation, running, and quick_run will return errors if gates fail:

1. **No experiments without a plan** — A `notebook(action="write", type="plan", ...)` entry must exist before any experiment can be created or run
2. **No next experiment without logging the previous one** — If the most recent experiment is completed but has no conclusion, you must call `experiments(action="log_result", ...)` before creating/running the next
3. **No more than 3 experiments without updating the plan** — After 3 experiments since the last plan entry, you must write a new plan before continuing

Additional conventions (not code-enforced):
- Write a theory (`notebook(action="write", type="theory", ...)`) before writing a plan
- Record findings after each experiment: `notebook(action="write", type="finding", experiment_id="...", ...)`
- On phase transitions, write both a finding summary and a new theory + plan
- Check `notebook(action="summary")` at session start

## Skills

10 experiment discipline skills installed to `skills/` and `packages/harness-plugin/skills/`:

| Skill | Purpose |
|-------|---------|
| `run-experiment` | Disciplined experiment execution with required hypothesis/conclusion |
| `experiment-design` | Phased workflow preventing premature tuning |
| `domain-research` | Hypothesis-driven domain research for feature engineering |
| `eda` | Exploratory data analysis workflow |
| `feature-engineering` | Systematic feature engineering process |
| `model-diversity` | Ensuring model diversity in ensembles |
| `diagnosis` | Diagnosing model performance issues |
| `synthesis` | Synthesizing experiment results into conclusions |
| `mindset` | Data science mindset and best practices |
| `project-setup` | Project initialization workflow |

## What NOT to Do

- Do not bypass guardrails — non-overridable ones exist for safety (leakage, temporal, critical path)
- Do not hardcode paths, thresholds, or magic numbers
- Do not create parallel versions of files — extend existing modules
- Do not skip `uv run` — bare `python` will not find workspace packages
- Do not put `__init__.py` at the `src/harnessml/` level (breaks namespace packages)
- Do not modify production config directly — use experiment overlays
- Do not import from old paths (`harnessml.schemas`, `harnessml.runner`, etc.) — use `harnessml.core.*`
