# Exploration Engine Design

**Goal:** Let the agent define a search space (features, models, hyperparams, ensemble settings) and run Bayesian-optimized experiments in bulk, returning a full report.

**Problem:** The agent currently burns time and API spend defining 20+ sequential experiments tweaking the same parameters. It should define the space once and get results back.

## Architecture

```
Agent MCP call (action="explore", search_space={...})
  → ExplorationSpace (Pydantic-validated)
  → Optuna Study (TPE sampler)
      → Trial N: suggest params → build overlay → PipelineRunner.backtest()
      → All metrics → Optuna reports back
  → ExplorationReport (best config, all trials, parameter importance)
```

Shared `PredictionCache` across all trials — unchanged models are never retrained.

## Search Space Schema

### Axis Types

| Type | Optuna mapping | Example |
|------|---------------|---------|
| `continuous` | `suggest_float(key, low, high, log=)` | learning_rate 0.01–0.3 |
| `integer` | `suggest_int(key, low, high)` | max_depth 3–10 |
| `categorical` | `suggest_categorical(key, values)` | ensemble method |
| `subset` | N independent booleans with min_size constraint | features/models to toggle |

### Subset Key Routing

| Subset key | Overlay translation |
|-----------|-------------------|
| `models.active` | `models.{candidate}.active = True/False` |
| `features.include` | `data.feature_defs.{candidate}.enabled = True/False` |
| `features.categories` | All features in category → enabled True/False |

### Pydantic Models

```python
class AxisDef(BaseModel):
    key: str
    type: Literal["continuous", "integer", "categorical", "subset"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    values: list[Any] | None = None
    candidates: list[str] | None = None
    min_size: int = 1

class ExplorationSpace(BaseModel):
    axes: list[AxisDef]
    budget: int = 20
    primary_metric: str = "brier"
    baseline: bool = True
    description: str = ""
```

## Trial Execution

1. Optuna TPE suggests values for each axis
2. Build config overlay: continuous/integer/categorical → `set_nested_key`, subset → semantic routing
3. `PipelineRunner(overlay=overlay, prediction_cache=shared_cache).backtest()`
4. Return primary_metric to Optuna, store all metrics

Failed trials are pruned, not fatal.

## Report

- **Summary**: best trial, delta vs baseline, cache hit rate
- **All trials**: ranked table with ALL metrics from backtest + axis values
- **Parameter importance**: Optuna's fANOVA-based importance analysis

## File Layout

```
experiments/
  expl-001/
    space.yaml
    .cache/
    baseline/results.json
    trials/trial-001/results.json ...
    study.json
    best_overlay.yaml
    report.md
```

## MCP Integration

New `"explore"` action on `manage_experiments`. Delegates to `config_writer.run_exploration()`.
Promotion via existing `manage_experiments(action="promote", experiment_id="expl-001")`.

## Dependencies

Optuna as optional dependency: `[project.optional-dependencies] explore = ["optuna>=3.0"]`

## Module Map

| File | Contents |
|------|----------|
| `exploration.py` | `ExplorationSpace`, `AxisDef`, `run_exploration()`, overlay builder, report formatter |
| `config_writer.py` | `run_exploration()` wrapper |
| `mcp_server.py` | `"explore"` action |
