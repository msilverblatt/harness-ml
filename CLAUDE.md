# EasyML -- AI Agent Instructions

## What This Is

EasyML is a general-purpose agentic ML framework with 3 packages in a uv
workspace monorepo. It supports any ML task type (binary classification,
multiclass, regression, ranking, survival, probabilistic forecasting).

Architecture: **easyml-core** (schemas, config, guardrails, models, runner,
feature engineering, metrics, data sources) + **easyml-plugin** (MCP server,
thin async dispatcher with hot-reloadable handlers) + **easyml-sports**
(optional domain plugin for matchup prediction).

## Tech Stack

- Python 3.11+, managed by **uv** (always use `uv run`, never bare `python`)
- Pydantic v2 for all schemas and contracts
- OmegaConf for config deep merge
- FastMCP for async MCP server with hot-reload
- scikit-learn, XGBoost, CatBoost, LightGBM, PyTorch (MLP, TabNet)
- Optional: shap, matplotlib, pandera, optuna
- Namespace packages: no `__init__.py` at `src/easyml/` level

## Package Map

| Package | Purpose | Key Modules |
|---------|---------|-------------|
| `easyml-core` | All core engine code | `schemas/`, `config/`, `guardrails/`, `models/`, `runner/`, `feature_eng/` |
| `easyml-plugin` | MCP server (thin dispatcher) | `mcp_server.py`, `handlers/` (models, data, features, experiments, config, pipeline) |
| `easyml-sports` | Optional sports domain plugin | `matchups.py`, `hooks.py` (registers into core extension points) |

### easyml-core submodules

| Submodule | Purpose |
|-----------|---------|
| `core.schemas` | Pydantic contracts (`contracts.py`) + MetricRegistry with 45 metrics across 6 task types (`metrics.py`) |
| `core.config` | YAML loading + OmegaConf deep merge |
| `core.guardrails` | Safety guardrails (leakage, temporal, naming) |
| `core.models` | Model wrappers (XGBoost, LightGBM, CatBoost, RF, Logistic, ElasticNet, MLP, TabNet) + registry |
| `core.runner` | Pipeline, training, meta-learner, calibration (Spline/Isotonic/Platt/Beta), feature store, views, config writer, diagnostics, exploration, sources |
| `core.runner.sources` | Source registry, freshness tracking, schema validation, adapters (file/url/api/computed) |
| `core.feature_eng` | Feature engineering registry + transforms |

## Key Conventions

- All packages under `packages/`, source in `packages/<name>/src/easyml/<subpkg>/`
- Namespace packages: `easyml.core.schemas`, `easyml.core.runner`, etc. (no root `__init__.py`)
- Import paths use `easyml.core.*` for all core code, `easyml.plugin.*` for MCP, `easyml.sports.*` for sports
- Registry pattern used in features, models, sources, metrics, and guardrails
- Hook system for domain plugins (`core.runner.hooks.HookRegistry`)
- TDD: write tests alongside implementation, run with `uv run pytest`
- MCP handlers are hot-reloadable in dev mode (`EASYML_DEV=1`) — no server restart needed for handler changes

## How to Add a New Model

1. Create `packages/easyml-core/src/easyml/core/models/wrappers/my_model.py`
2. Subclass `BaseModel`, implement `fit`, `predict_proba`, `save`, `load`
3. Register in `ModelRegistry.with_defaults()` (wrap in try/except for optional deps)
4. Add tests in `packages/easyml-core/tests/models/`

## How to Add a New Feature

Use the MCP tool or config_writer:
1. `features(action="add", name="...", formula="...", type="grouped|instance|formula|regime")`
2. For batch: `features(action="add_batch", features=[{...}, ...])`
3. For auto-discovery: `features(action="auto_search", features=[...], search_types=["interactions","lags","rolling"])`

## How to Add a New View Step

1. Add Pydantic model to `packages/easyml-core/src/easyml/core/runner/schema.py` (in TransformStep union)
2. Add executor function to `packages/easyml-core/src/easyml/core/runner/view_executor.py`
3. Register in `_dispatch` dict
4. Available steps: filter, select, derive, group_by, join, union, unpivot, sort, head, rolling, cast, distinct, rank, isin, cond_agg, lag, ewm, diff, trend, encode, bin, datetime, null_indicator

## How to Add a New Metric

1. Write metric function in `packages/easyml-core/src/easyml/core/schemas/metrics.py`
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

## Testing

```bash
uv run pytest                                    # all tests
uv run pytest packages/easyml-core/tests/        # core tests
uv run pytest packages/easyml-sports/tests/      # sports plugin tests
uv run pytest packages/easyml-core/tests/runner/  # runner subsystem
uv run pytest -v                                 # verbose
```

## What NOT to Do

- Do not bypass guardrails — non-overridable ones exist for safety (leakage, temporal, critical path)
- Do not hardcode paths, thresholds, or magic numbers
- Do not create parallel versions of files — extend existing modules
- Do not skip `uv run` — bare `python` will not find workspace packages
- Do not put `__init__.py` at the `src/easyml/` level (breaks namespace packages)
- Do not modify production config directly — use experiment overlays
- Do not import from old paths (`easyml.schemas`, `easyml.runner`, etc.) — use `easyml.core.*`
