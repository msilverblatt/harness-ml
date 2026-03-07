# HarnessML v2 Refactor Design

## Problem Statement

HarnessML was intended as a general-purpose agentic MLOps framework, but the first
implementation leaked domain-specific assumptions (NCAA sports/matchup prediction)
throughout the core. ~1,738 lines across 10 files are sports-specific, and 3 of
8 packages are dead weight. The MCP server lacks batch operations, progress
reporting, and response efficiency controls.

This refactor: (1) consolidates 8 packages into 3, (2) extracts sports-specific
code into a plugin, (3) adds general-purpose feature engineering primitives,
(4) comprehensive metrics for all ML task types, (5) data source management with
freshness tracking, and (6) optimizes the MCP server architecture.

## Migration Strategy

Break-and-fix: do the full extraction, then update mm-women to use the sports
plugin. No backward compatibility maintained during refactor.

---

## 1. Package Structure

### From 8 packages → 3

```
packages/
  harness-core/          ← schemas + config + guardrails + models + runner
    src/harnessml/core/
      schemas/
        contracts.py    ← ModelConfig, GuardrailViolation, ExperimentResult
        metrics.py      ← MetricRegistry + all task-type metrics
      config/
        loader.py       ← YAML loading + env var substitution
        merge.py        ← OmegaConf deep merge
      guardrails/
        base.py         ← Guardrail ABC (overridable vs non-overridable)
        inventory.py    ← 7 concrete guardrails
        audit.py        ← violation logging
      models/
        base.py         ← BaseModel ABC
        registry.py     ← ModelRegistry factory
        wrappers/       ← xgboost, lightgbm, catboost, rf, logistic,
                           elastic_net, mlp, tabnet
      runner/
        pipeline.py
        training.py
        meta_learner.py
        calibration.py  ← Spline, Isotonic, Platt, Beta
        feature_store.py
        feature_cache.py
        view_resolver.py
        view_executor.py
        dag.py
        cv_strategies.py
        config_writer.py
        exploration.py
        postprocessing.py
        reporting.py
        diagnostics.py
        project.py
        schema.py
        sources/
          registry.py   ← SourceDef + SourceRegistry
          freshness.py  ← FreshnessTracker
          validation.py ← Pandera schema validation
          adapters.py   ← file, url, api, computed adapters
      feature_eng/
        transforms.py   ← formula function registry
        encoders.py     ← target_loo, target_temporal, frequency, ordinal

  harness-plugin/        ← MCP server (thin dispatcher)
    src/harnessml/plugin/
      mcp_server.py     ← tool signatures + validation only
      handlers/         ← all business logic, hot-reloadable
        models.py
        data.py
        features.py
        experiments.py
        config.py
        pipeline.py
        _validation.py  ← enum validation, fuzzy match, cross-param hints
      progress.py       ← Context-based progress reporting helpers

  harness-sports/        ← domain plugin (optional install)
    src/harnessml/sports/
      matchups.py       ← generate_pairwise_matchups, predict_all_matchups
      pairwise.py       ← auto-pairwise generation (diff/ratio)
      augmentation.py   ← matchup symmetry augmentation
      postprocessing.py ← seed compression, availability adjustment
      hooks.py          ← registers into core extension points
```

### Deleted packages
- `harnessml-features` (462 LOC) — dead code, superseded by feature_store.py
- `harnessml-config` (126 LOC) — absorbed into harness-core/config
- `harnessml-data` (434 LOC) — superseded by feature_store + new sources module

### Design decisions
- `harness-core` is the only required install
- `harness-sports` registers via hooks — core defines extension points,
  sports plugin registers implementations
- `harness-plugin` depends on core, optionally discovers installed plugins
  (harness-sports, future domain plugins)

---

## 2. MCP Server Architecture

### Hot-reload dispatcher

```python
# mcp_server.py — changes only when tool signatures change
_DEV_MODE = os.environ.get("HARNESS_DEV", "0") == "1"

def _load_handler(module_name: str):
    mod = importlib.import_module(f"harnessml.plugin.handlers.{module_name}")
    if _DEV_MODE:
        importlib.reload(mod)
    return mod

@mcp.tool()
@_safe_tool
async def manage_models(action: str, ctx: Context, ...) -> str:
    return await _load_handler("models").dispatch(action, ctx=ctx, **kwargs)
```

### Handler dispatch pattern

```python
# handlers/models.py — hot-reloadable, all logic here
ACTIONS = {
    "add": _handle_add,
    "add_batch": _handle_add_batch,
    "update": _handle_update,
    "update_batch": _handle_update_batch,
    "remove": _handle_remove,
    "remove_batch": _handle_remove_batch,
    "list": _handle_list,
    "presets": _handle_presets,
    "clone": _handle_clone,
}

def dispatch(action, **kwargs):
    if action not in ACTIONS:
        valid = ", ".join(sorted(ACTIONS))
        closest = _fuzzy_match(action, ACTIONS)
        msg = f"**Error**: Unknown action '{action}'. Valid: {valid}"
        if closest:
            msg += f"\n\nDid you mean **{closest}**?"
        return msg
    return ACTIONS[action](**kwargs)
```

### What requires MCP restart vs not

| Change | Restart? |
|--------|----------|
| New tool (`@mcp.tool()`) | Yes |
| New typed parameter on signature | Yes |
| Tool docstring change | Yes |
| New action within existing tool | No |
| Action implementation change | No (dev mode hot-reload) |
| Response format change | No |

### Batch actions

| Handler | New Batch Actions |
|---------|-------------------|
| models.py | `add_batch`, `update_batch`, `remove_batch`, `clone` |
| data.py | `add_sources_batch`, `fill_nulls_batch`, `add_views_batch` |
| features.py | (already has `add_batch`) |
| experiments.py | `compare` (diff two experiments) |
| pipeline.py | `compare_runs`, `progress` |

### Progress reporting via MCP protocol

Tools are async. Progress uses `ctx.report_progress()` from FastMCP Context:

```python
async def _handle_run_backtest(ctx: Context, **kwargs):
    for i, (train, test) in enumerate(cv_folds):
        await ctx.report_progress(
            progress=i, total=len(cv_folds),
            message=f"Fold {i+1}/{len(cv_folds)}: season {test}"
        )
        runner.run_fold(train, test)
```

Applied to: run_backtest, explore, discover, add_features_batch, refresh_all.

Context also provides logging: `ctx.info()`, `ctx.warning()`, `ctx.error()`.

### Response modes

`detail` parameter on verbose tools:

| Tool.Action | summary | full (default) |
|-------------|---------|----------------|
| pipeline.show_run | 5-line metrics | Full per-season + picks + coefficients |
| pipeline.diagnostics | Per-model table | + calibration + confusion + SHAP |
| configure.show | Key settings | Entire YAML |
| experiments.explore | Best trial + delta | All trials + param importance |
| data.profile | Row/col + nulls | Full per-column stats |

`section` parameter on `configure(action="show", section="models")`.

### Input validation

- Enum validation with fuzzy match ("did you mean?") on all fixed-value params
- Cross-parameter hints (regressor without cdf_scale, ensemble model without features)
- Actionable error context: every known failure mode maps to suggested next steps

---

## 3. Core Generalization

### Schema changes

`FeatureType` enum becomes general-purpose:

```python
class FeatureType(str, Enum):
    INSTANCE = "instance"   # row-level (was MATCHUP)
    GROUPED = "grouped"     # group-level, expanded to instance (was ENTITY)
    REGIME = "regime"       # boolean flag from condition
    FORMULA = "formula"     # computed from other features (was PAIRWISE)
    # Sports plugin registers PAIRWISE via hook
```

Removed from core:
- `entity_a_column` / `entity_b_column` → sports plugin config
- `PairwiseMode` (diff/ratio/both/none) → sports plugin
- Hardcoded "TeamA"/"TeamB" column detection → sports plugin

Kept/renamed:
- `entity_column` → `group_column` (generic grouping)
- `period_column` stays (general for temporal data)
- `seed_diffs` → `prior_feature` (optional, None by default)
- All `df["season"]` → `df[config.data.time_column]`
- `group_columns: list[str]` added for multi-column grouping

### Extension points

| Hook | Default | Sports Plugin |
|------|---------|---------------|
| `feature_expansion_hook` | Left join on group_columns | Pairwise diff/ratio |
| `provider_injection_hook` | Column merge | Entity→matchup with diff |
| `pre_training_hook` | No-op | Matchup symmetry augmentation |
| `post_prediction_hook` | No-op | Seed compression |
| `feature_type_hook` | Core types only | Registers PAIRWISE type |

### Feature store

- `_generate_pairwise()` → moves to harness-sports/pairwise.py
- Core `_expand_grouped_feature()` calls registered hooks, falls back to
  default left join on group_columns
- All feature computation stays in the view/feature system, never in training

### Pipeline

- Provider injection uses hook instead of hardcoded column detection
- Training loop receives fully materialized feature DataFrame — no feature
  engineering at training time

---

## 4. Feature Engineering Extensions

### 8 new view steps

| Step | Purpose | Schema |
|------|---------|--------|
| `lag` | Lagged values per group | `{keys, order_by, columns: {new: "src:lag_n"}}` |
| `ewm` | Exponentially weighted stats | `{keys, order_by, span, aggs: {new: "src:mean"}}` |
| `diff` | First/second differences | `{keys, order_by, columns: {new: "src:n"}, pct: bool}` |
| `trend` | OLS slope over window | `{keys, order_by, window, columns: {new: "src"}}` |
| `encode` | Categorical encoding | `{column, method, output}` (target_loo, target_temporal, frequency, ordinal) |
| `bin` | Discretization | `{column, method, n_bins, output}` (quantile, uniform, custom, kmeans) |
| `datetime` | Calendar features | `{column, extract: [...], cyclical: [...]}` |
| `null_indicator` | Missing flags | `{columns: [...], prefix: "missing_"}` |

Target encoding uses leave-one-out (LOO) or temporal approach — both are
stateless and leakage-safe by construction. No train/test split awareness
needed. All feature computation happens in the view/feature system.

### 15 new formula functions

Power/distribution: `reciprocal`, `exp`, `expm1`, `power`
Cyclical: `sin_cycle(col, period)`, `cos_cycle(col, period)`
Statistical: `zscore`, `minmax`, `rank_pct`, `winsorize(col, lower, upper)`
Comparison: `maximum`, `minimum`, `where(cond, t, f)`, `isnull`
Composition: `safe_div(x, y)`, `pct_of_total(x, y)`

### 10 new rolling aggregations

`median`, `skew`, `kurt`, `slope`, `ema`, `range`, `cv`, `pct_change`,
`first`, `last`

### Enhanced auto-feature search

```
manage_features(action="auto_search",
    features=["col_a", "col_b"],
    search_types=["interactions", "lags", "rolling"],
    top_n=20)
```

- interactions: all numeric pairs × {+, -, *, /, abs_diff} → rank by lift
- lags: each feature × lag(1,3,5,10) → rank by correlation
- rolling: each feature × rolling_mean(3,5,10,20) → rank by correlation

---

## 5. Metrics & Diagnostics

### MetricRegistry — task-type dispatch

```python
class MetricRegistry:
    _metrics: dict[str, dict[str, Callable]] = {}

    @classmethod
    def register(cls, task, name, fn, requires=None): ...

    @classmethod
    def compute_all(cls, task, **data) -> dict[str, float]: ...
```

### Full metric coverage

| Task | Metrics |
|------|---------|
| binary | brier, accuracy, log_loss, ece, auc_roc, auc_pr, f1, precision, recall, mcc, specificity, confusion_matrix, cohen_kappa |
| multiclass | accuracy, macro/micro/weighted f1/precision/recall, confusion_matrix (NxN), per_class_report, log_loss, mcc, cohen_kappa |
| regression | rmse, mae, r2, mape, median_ae, explained_variance, mean_bias, quantile_loss |
| ranking | ndcg_at_k, map, mrr, precision_at_k, recall_at_k, spearman_rank_correlation |
| survival | concordance_index, time_dependent_brier, cumulative_incidence_auc |
| probabilistic | crps, pit_histogram_data, sharpness, coverage_at_level |

### Calibration (enhanced)

Calibrators: Spline (PCHIP), Isotonic, Platt, **Beta** (new).

Diagnostics: reliability_diagram_data, hosmer_lemeshow_test,
calibration_slope_intercept, calibration_decomposition,
bootstrap_ci, per_class_calibration.

### Per-model diagnostics

- All task-appropriate metrics from MetricRegistry
- Feature importance (gain/split/cover for trees, coefficients for linear)
- SHAP values (optional dep: `shap`)
- Permutation importance (sklearn.inspection)
- Confusion matrix at optimal threshold
- ROC curve data (fpr, tpr, thresholds)
- PR curve data (precision, recall, thresholds)
- Per-model calibration curve data

### Visualization

Optional `matplotlib` dependency. `pipeline(action="diagnostics", render="png",
output_dir="./plots/")` generates: roc_curve.png, pr_curve.png,
calibration.png, confusion_matrix.png, shap_summary.png,
feature_importance.png.

Without matplotlib: returns structured data only.

---

## 6. Data Source Management

### SourceDef schema

```yaml
# config/sources.yaml
sources:
  kenpom_ratings:
    source_type: api           # file | api | url | computed
    path_pattern: "data/external/kenpom/ratings_{year}.json"
    refresh_frequency: yearly  # hourly | daily | weekly | monthly | yearly | manual
    auth:
      type: api_key
      env_var: KENPOM_API_KEY
    rate_limit: 1.0
    incremental: false
    depends_on: []
    schema:
      required_columns: [Season, TeamName, AdjO, AdjD, AdjEM]
      types: {Season: int64, AdjO: float64}
    leakage_notes: "Use Selection Sunday snapshot only"
```

### Freshness tracking

`sources_state.json` (gitignored) tracks last_fetched + row_count per source.
`FreshnessTracker.check_all()` compares timestamps against refresh_frequency.

### Schema validation

Optional Pandera integration. Validates at ingest and view materialization
boundaries. Returns structured violation list on failure.

### Built-in adapters

| Adapter | What |
|---------|------|
| file | Read CSV/parquet/Excel from local path |
| url | Download from URL with optional auth headers |
| api | REST API with pagination + rate limiting |
| computed | Python function deriving data from other sources |

Not full scrapers. Complex scraping stays in domain code. Future: browser
adapter wrapping Playwright for agent-configurable scraping.

### Dependency-ordered refresh

`refresh_all` runs stale sources in topological order with progress reporting.
Validates schema before saving — won't ingest data that fails validation.

### MCP actions

```
manage_data(action="add_source", ...)
manage_data(action="add_sources_batch", sources=[...])
manage_data(action="list_sources")
manage_data(action="check_freshness")
manage_data(action="check_freshness", source="...")
manage_data(action="refresh", source="...")
manage_data(action="refresh_all")
manage_data(action="validate_source", source="...")
```

---

## 7. Implementation Phases

### Phase 1: Package Consolidation (no behavior changes)
- Merge 5 packages into harness-core
- Delete 3 dead packages
- Fix imports, update pyproject.toml
- All existing tests pass

### Phase 2: MCP Architecture (depends on Phase 1)
- Thin dispatcher + handlers/ with hot-reload
- Async tools with ctx.report_progress()
- Batch actions, convenience actions
- Enum validation, fuzzy match, cross-param hints
- Response modes (detail, section)

### Phase 3: Domain Extraction (depends on Phase 1)
- Create harness-sports with hooks
- Define 5 extension points in core
- Move ~1,738 lines of sports code
- Generalize schemas
- Fix mm-women to use sports plugin

### Phase 4: Feature Engineering (depends on Phase 1)
- 8 new view steps
- 15 formula functions
- 10 rolling aggregations
- Enhanced auto_search

### Phase 5: Metrics & Diagnostics (depends on Phase 1)
- MetricRegistry with task-type dispatch
- All metrics (binary, multiclass, regression, ranking, survival, probabilistic)
- Enhanced calibration (Beta + full diagnostics)
- SHAP, permutation importance, ROC/PR curves
- Optional matplotlib rendering
- Progress reporting in backtest and exploration

### Phase 6: Data Source Management (depends on Phase 2)
- Source registry + freshness tracker
- Pandera validation
- Built-in adapters
- Dependency-ordered refresh
- MCP actions

### Parallelism

Phases 2, 3, 4, 5 can run in parallel after Phase 1 completes.
Phase 6 depends on Phase 2.
