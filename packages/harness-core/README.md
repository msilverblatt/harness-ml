# Harness Core

Core engine for [HarnessML](https://github.com/msilverblatt/harness-ml) -- a general-purpose agentic ML framework supporting binary classification, multiclass, regression, ranking, survival analysis, and probabilistic forecasting.

## Architecture

```
harnessml.core
├── schemas/        # Pydantic v2 contracts + MetricRegistry (45 metrics, 6 task types)
├── config/         # YAML loading + OmegaConf deep merge
├── guardrails/     # Safety guardrails (17 total, 4 non-overridable)
├── models/         # Model wrappers (13 algorithms) + ModelRegistry
├── runner/         # Pipeline, training, CV, meta-learner, calibration, views, sources
│   ├── sources/    # Source registry + adapters (file, url, api, computed)
│   ├── drives/     # Cloud adapters (Google Drive, Kaggle)
│   ├── notebook/   # Jupyter notebook generation (Colab, Kaggle, local)
│   └── config_writer/  # Programmatic config generation
└── feature_eng/    # Feature engineering registry + transforms
```

## Key Imports

```python
from harnessml.core.runner.pipeline import PipelineRunner
from harnessml.core.runner.project import ProjectManager
from harnessml.core.runner.schema import BacktestConfig
from harnessml.core.models.registry import ModelRegistry
from harnessml.core.schemas.metrics import MetricRegistry
from harnessml.core.guardrails.server import GuardrailServer
from harnessml.core.feature_eng.registry import FeatureRegistry
from harnessml.core.runner.hooks import HookRegistry
```

## Model Wrappers

13 model wrappers, all sharing a common `BaseModel` interface (`fit`, `predict_proba`, `save`, `load`):

| Model | Backend | Notes |
|-------|---------|-------|
| XGBoost | xgboost | Early stopping, eval_set |
| LightGBM | lightgbm | Early stopping via callback |
| CatBoost | catboost | Silent mode, eval_set early stopping |
| Random Forest | scikit-learn | Invalid param filtering with warnings |
| Logistic Regression | scikit-learn | Standard binary/multiclass |
| Elastic Net | scikit-learn | StandardScaler normalization, NaN handling |
| MLP | PyTorch | Batch norm, early stopping, weight decay, seed stride |
| TabNet | pytorch-tabnet | Scheduler, eval set, val fraction, seed stride |
| SVM | scikit-learn | Support vector classification |
| HistGBM | scikit-learn | Histogram-based gradient boosting |
| GAM | pygam | Generalized additive model |
| NGBoost | ngboost | Natural gradient boosting |

Models are registered through `ModelRegistry.with_defaults()`, with try/except guards for optional dependencies.

## Registry Pattern

Five registries share a common pattern for extensibility:

- **ModelRegistry** -- register model wrappers by name, create via `registry.create(name, **kwargs)`
- **MetricRegistry** -- 45 metrics across 6 task types (binary, multiclass, regression, ranking, survival, probabilistic)
- **FeatureRegistry** -- feature engineering transforms (grouped, instance, formula, regime)
- **GuardrailServer** -- safety checks that run before/after tool execution
- **SourceRegistry** -- data source adapters with freshness tracking and schema validation

## Metrics

45 metrics organized by task type:

| Task Type | Examples |
|-----------|----------|
| Binary | AUC, Brier score, log loss, accuracy, precision, recall, F1, ECE |
| Multiclass | Multiclass AUC, macro/micro F1, Cohen's kappa, confusion matrix |
| Regression | RMSE, MAE, MAPE, R-squared, explained variance |
| Ranking | NDCG, MAP, MRR, precision@k |
| Survival | Concordance index, Brier survival score, integrated Brier |
| Probabilistic | CRPS, calibration error, prediction interval coverage |

## Cross-Validation Strategies

7 CV strategies with alias support:

| Strategy | Aliases | Notes |
|----------|---------|-------|
| `leave_one_out` | `loso`, `loo` | Leave-one-season-out (default) |
| `expanding_window` | `expanding` | Growing training window |
| `sliding_window` | `sliding` | Fixed-size window (requires `window_size`) |
| `purged_kfold` | `purged` | K-fold with temporal purging (requires `n_folds`) |
| `stratified_kfold` | `skf` | Stratified K-fold (requires `n_folds`) |
| `group_kfold` | `gkf` | Group-aware K-fold (requires `group_column`) |
| `bootstrap` | -- | Bootstrap resampling |

## Calibration

4 post-hoc calibration methods applied after ensemble aggregation:

- **Spline (PCHIP)** -- monotonic piecewise cubic Hermite interpolation
- **Isotonic** -- isotonic regression (non-parametric)
- **Platt** -- logistic sigmoid scaling (parametric)
- **Beta** -- beta distribution CDF calibration

## Declarative View Transforms

22 transform steps for data preparation, all defined as Pydantic models and executed by the view executor:

```
filter, select, derive, group_by, join, union, unpivot, cast, sort,
distinct, rolling, head, rank, cond_agg, isin, lag, ewm, diff, trend,
encode, bin, datetime, null_indicator
```

Views are composable -- chain multiple steps to build complex data pipelines declaratively in YAML.

## Guardrails

17 safety guardrails, 4 of which are non-overridable:

| Guardrail | Overridable | Purpose |
|-----------|-------------|---------|
| FeatureLeakage | No | Prevents target leakage in features |
| CriticalPath | No | Protects critical pipeline paths |
| TemporalOrdering | No | Enforces temporal consistency |
| PredictionSanity | No | Validates prediction output ranges |
| SanityCheck | Yes | General data sanity checks |
| NamingConvention | Yes | Enforces column naming standards |
| DoNotRetry | Yes | Prevents retry loops on permanent failures |
| SingleVariable | Yes | Limits single-variable experiments |
| ConfigProtection | Yes | Warns on config overwrites |
| RateLimit | Yes | Throttles rapid tool calls |
| ExperimentLogged | Yes | Ensures experiments are logged |
| FeatureStaleness | Yes | Flags stale feature data |
| FeatureDiversity | Yes | Encourages diverse feature types |
| DataDistribution | Yes | Monitors data distribution shifts |
| ClassImbalance | Yes | Warns on severe class imbalance |
| ModelComplexity | Yes | Flags overly complex configurations |
| FeatureCount | Yes | Limits feature count relative to samples |

## Hook System

Domain plugins (like `harness-sports`) extend core behavior without modifying it:

```python
from harnessml.core.runner.hooks import HookRegistry, COLUMN_CANDIDATES

HookRegistry.register(COLUMN_CANDIDATES, my_column_candidates_fn)
```

Hook points include column candidates, column renames, and competition narratives.

## Data Sources

Source registry with 4 adapter types:

- **File** -- local CSV, Parquet, Excel files
- **URL** -- remote file downloads with caching
- **API** -- REST API data fetching
- **Computed** -- derived sources from view transforms

All sources support freshness tracking and schema validation.

## Quick Start

```bash
# From the monorepo root
uv sync

# Run tests
uv run pytest packages/harness-core/tests/ -v
```

## Adding a New Model

1. Create `packages/harness-core/src/harnessml/core/models/wrappers/my_model.py`
2. Subclass `BaseModel`, implement `fit`, `predict_proba`, `save`, `load`
3. Register in `ModelRegistry.with_defaults()` (wrap in try/except for optional deps)
4. Add tests in `packages/harness-core/tests/models/`

## Adding a New Metric

1. Write metric function in `packages/harness-core/src/harnessml/core/schemas/metrics.py`
2. Register: `MetricRegistry.register("task_type", "name", fn)`
3. Task types: binary, multiclass, regression, ranking, survival, probabilistic
