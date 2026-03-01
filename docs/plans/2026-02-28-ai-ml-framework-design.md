# AI-Driven ML Framework — Design Document

**Date:** 2026-02-28
**Status:** Draft (reviewed)
**Origin:** Extracted from March Madness bracket optimizer pipeline

## Problem

Building ML pipelines that an AI agent (Claude) can safely and autonomously operate requires two things that don't exist together today:

1. **ML infrastructure** — features, models, ensembles, experiments, versioning — that's config-driven so the agent doesn't need to write boilerplate code for every change.
2. **AI guardrails** — experiment protocol enforcement, do-not-retry registries, sanity checks, single-variable discipline — that prevent the agent from repeating known failures or making undisciplined changes.

Existing ML frameworks (scikit-learn, MLflow, Weights & Biases) handle #1 partially but aren't designed for AI-agent operation. Existing AI tool frameworks (MCP servers, LangChain tools) handle the interface but know nothing about ML discipline.

This framework combines both layers into a unix-style suite of composable tools.

## Target Users

- **Primary:** The author, for reuse across sports prediction projects (March Madness, then NFL/NBA/etc).
- **Secondary:** Open-source users building AI-assisted ML pipelines. Architecture designed for this from day one; public release later.

## Core Value Proposition

The AI guardrails layer is the differentiator. But guardrails without structure to enforce them on are just documentation. The ML infrastructure gives the guardrails teeth — structured config, typed artifacts, feature registries, and experiment protocols create the surface area that guardrails can actually validate.

## Architecture: Schema Core + Tool Ring

Seven independent packages, one shared contracts library, composable via unix-style conventions.

```
                    ┌─────────────────┐
                    │  mcp-guardrails  │   ← The differentiator
                    │  (MCP server gen │
                    │   + AI safety)   │
                    └────────┬────────┘
                             │
     ┌──────────┬────────────┼────────────┬────────────┐
     │          │            │            │            │
┌────┴────┐ ┌──┴──────┐ ┌───┴────┐ ┌─────┴─────┐ ┌───┴────┐
│  exp-   │ │feature- │ │ model- │ │   data-   │ │config- │
│  runner │ │registry │ │ensemble│ │  pipeline │ │overlay │
└────┬────┘ └──┬──────┘ └───┬────┘ └─────┬─────┘ └───┬────┘
     │         │             │            │           │
     └─────────┴─────────────┴────────────┴───────────┘
                             │
                    ┌────────┴────────┐
                    │   ml-schemas    │   ← Shared contracts
                    │  (tiny, stable) │
                    └─────────────────┘
```

### Interface Between Tools

A shared schema library (`ml-schemas`) defines the contracts: config schema, feature metadata schema, run manifest format, experiment result schema. Tools import this to validate inputs/outputs but don't depend on each other at runtime.

### Build vs Buy Decisions

| Component | Decision | Library | Rationale |
|-----------|----------|---------|-----------|
| Pydantic schemas | **Build** | — | Bespoke contracts, no existing ML config schema library fits |
| Built-in metrics | **Wrap** sklearn | sklearn.metrics | ECE, calibration table, ensemble diagnostics are custom (~200 LOC) |
| Config deep merge | **Wrap** OmegaConf | OmegaConf | Gains variable interpolation (`${model.lr}`); ~150 LOC of file-loading glue around it |
| Config file loading + variants | **Build** | — | Variant suffix pattern not in any library; ~150 LOC |
| Feature registry | **Build** | — | Hamilton is close but wrong abstraction (function-name-as-feature, bundled execution, no source hashing); 400 LOC |
| Model wrappers | **Build** | — | Thin (~50 LOC each); AutoGluon/PyCaret/FLAML are all-or-nothing AutoML frameworks |
| Temporal CV | **Build** | — | No library handles grouped folds + nested calibration splits; ~300 LOC |
| Fingerprint caching | **Build** | — | 35 LOC total; joblib/DVC/MLflow solve different problems |
| Stacking ensemble | **Build** | — | Takes pre-computed predictions; sklearn's StackingClassifier wants to own the training loop |
| Platt/Isotonic calibration | **Wrap** sklearn | sklearn | Already thin wrappers |
| Spline calibrator | **Build** | — | PCHIP + quantile bins not in any library; ~85 LOC |
| RunManager | **Build** | — | Filesystem-only, no server needed; ~100 LOC. MLflow/W&B are tracking servers — overkill |
| Pipeline DAG | **Keep DVC** | DVC | Already using DVC; custom DAG is NIH. Add thin typed-artifact layer on top |
| Source registry | **Build** | — | ~100 LOC; Great Expectations is overkill for freshness checks |
| Stage guards | **Build** | — | ~50 LOC declarative checks |
| Experiment protocol | **Build** | — | Overlay + do-not-retry is genuinely novel; MLflow/W&B/Neptune are tracking tools, not protocol enforcement |
| MCP guardrails | **Build** | FastMCP foundation | Nothing exists in MCP ecosystem for ML-domain guardrails |

---

## Package 1: `ml-schemas` — Shared Contracts

Pure data shapes + standard metric implementations. No business logic, no file I/O beyond Pydantic and sklearn wrappers.

### Schemas

| Schema | Purpose | Key Fields |
|--------|---------|------------|
| `FeatureMeta` | Feature registration metadata | `name`, `category`, `level`, `depends_on`, `output_columns`, `valid_range`, `nan_strategy`, **`temporal_filter`**, **`tainted_columns`** |
| `SourceMeta` | Data source registration metadata | `name`, `category`, `outputs`, `schema`, **`leakage_notes`**, **`temporal_safety`** |
| `ModelConfig` | Single model definition | `name`, `type`, `mode`, `features`, `params`, `train_seasons`, `pre_calibrate` |
| `EnsembleConfig` | Ensemble/stacking definition | `method`, `meta_learner_params`, `calibration`, `exclude_models`, `pre_calibrate`, **`cv_strategy`** |
| `StageConfig` | Pipeline stage (for DVC generation) | `script`, `consumes`, `produces` (list of `ArtifactDecl`) |
| `ArtifactDecl` | Typed pipeline artifact | `name`, `type` (features/predictions/embeddings/model), `path` |
| `PipelineConfig` | Full resolved config | `stages`, `models`, `ensemble`, `data_paths`, `backtest_folds` |
| `RunManifest` | Versioned run metadata | `run_id`, `created_at`, `labels`, `stage`, `metrics` |
| `ExperimentResult` | Experiment outcome | `experiment_id`, `baseline_metrics`, `result_metrics`, `delta`, `verdict`, `models_trained` |
| `Metrics` | Universal metrics container | `dict[str, float]` — not hardcoded to any specific metrics |
| `GuardrailViolation` | Structured block response | `blocked`, `rule`, `message`, `source`, `override_hint` |
| `Fold` | CV fold definition | `fold_id`, `train_idx`, `test_idx`, `calibration_idx` |

### Leakage Prevention Schemas

Two schemas specifically address data leakage — the most dangerous failure mode for an AI-operated ML pipeline.

**`FeatureMeta.temporal_filter`** — declares what data a feature computation must exclude:

```python
@dataclass
class TemporalFilter:
    exclude_event_types: list[str] = []   # e.g., ["tournament", "postseason"]
    cutoff_date_field: str | None = None  # e.g., "tournament_start_date"
    max_date_field: str | None = None     # e.g., "game_date"
```

The feature registry validates that compute function outputs don't violate these constraints. This prevents an LLM from creating features that include tournament/postseason data.

**`FeatureMeta.tainted_columns`** — columns known to contain end-of-season or post-event data:

```python
# Example: KenPom end-of-season ratings are leaky
@registry.register(
    name="kenpom_ratings",
    tainted_columns=["kp_adj_o", "kp_adj_d", "kp_sos"],  # end-of-season values
    ...
)
```

Sanity checks block training if any model references a tainted column.

**`SourceMeta.temporal_safety`** — tags data sources by leakage risk:

```python
@sources.register(
    name="kenpom_archive",
    temporal_safety="pre_tournament",     # safe — snapshot before event
    leakage_notes="Archive endpoint returns ratings as of Selection Sunday",
    ...
)

@sources.register(
    name="kenpom_ratings",
    temporal_safety="end_of_season",      # LEAKY — includes tournament results
    leakage_notes="Current ratings endpoint includes tournament game data",
    ...
)
```

### Design Principles

- `Metrics` is `dict[str, float]`. MM uses Brier/ECE/Accuracy; NFL might use RMSE/MAE/AUC. The schema doesn't care.
- `ModelConfig.type` is a string, not an enum. The model registry resolves it.
- `ArtifactDecl.type` determines how an artifact can be consumed — "features" artifacts can feed into model training; "predictions" artifacts can feed into meta-ensembles.
- Versioning: schemas get semver. Breaking changes = major bump. Other packages pin `ml-schemas >= 1.0, < 2.0`.

### Built-In Metrics

The `Metrics` container is `dict[str, float]` (flexible), but `ml-schemas` ships standard metric implementations so users don't reimplement Brier score for every project. These are pure functions — numpy/sklearn wrappers with no heavy dependencies.

**Probability metrics:**

| Metric | Implementation | Notes |
|--------|---------------|-------|
| `brier_score` | sklearn wrapper | Primary for calibrated probability models |
| `log_loss` | sklearn wrapper | |
| `ece` | Custom | Expected Calibration Error — not in sklearn, not worth pulling `netcal` for 30 LOC |
| `calibration_table` | Custom | Binned accuracy vs confidence, returns structured data |

**Classification metrics:**

| Metric | Implementation | Notes |
|--------|---------------|-------|
| `accuracy` | sklearn wrapper | Configurable threshold (default 0.5) |
| `auc_roc` | sklearn wrapper | |
| `f1` / `precision` / `recall` | sklearn wrapper | |

**Regression metrics:**

| Metric | Implementation | Notes |
|--------|---------------|-------|
| `rmse` / `mae` / `mse` | sklearn wrapper | For margin/spread models |
| `r_squared` | sklearn wrapper | |

**Ensemble diagnostics:**

| Metric | Implementation | Notes |
|--------|---------------|-------|
| `model_correlations` | Custom | Pairwise prediction correlation matrix |
| `model_audit` | Custom | Per-model metrics (Brier, accuracy, etc.) |

**API:**

```python
from ml_schemas.metrics import brier_score, ece, calibration_table, model_correlations

metrics = {
    "brier": brier_score(y_true, y_prob),
    "accuracy": accuracy(y_true, y_prob, threshold=0.5),
    "ece": ece(y_true, y_prob, n_bins=10),
    "log_loss": log_loss(y_true, y_prob),
}

# Calibration diagnostics
cal_table = calibration_table(y_true, y_prob, bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# Ensemble diagnostics
corr_matrix = model_correlations({"xgb_core": preds_xgb, "mlp_margin": preds_mlp})
audit = model_audit(model_predictions, y_true, metrics=["brier", "accuracy"])
```

### What It Does NOT Include

- No file I/O
- No validation beyond type checking
- No defaults (each tool decides its own)

---

## Package 2: `config-overlay` — Split Config Resolution

YAML manipulation layer. Uses **OmegaConf** for deep merge (gains variable interpolation as a bonus), custom code for file loading and variant resolution.

### Capabilities

- Load split YAML files from a directory and merge into one resolved config
- Deep merge overlays (experiment configs) on top of base config via OmegaConf
- Variable interpolation (`${ensemble.meta_learner.C}` in YAML)
- Variant support (gender, environment) via suffix conventions
- Validate merged result against `PipelineConfig` schema

### API

```python
from config_overlay import resolve_config, deep_merge

config = resolve_config(
    config_dir="config/",
    file_map={
        "pipeline": "pipeline.yaml",
        "models": ["models/production.yaml"],
        "ensemble": "ensemble.yaml",
    },
    variant="w",          # loads pipeline_w.yaml, etc.
    overlay={"models": {"xgb_core": {"params": {"max_depth": 4}}}},
)

merged = deep_merge(base, experiment)  # OmegaConf.merge() under the hood
```

### Key Behaviors

- File map is explicit — user declares which files map to which config sections
- Variant suffix resolution: if `variant="w"` and file is `pipeline.yaml`, looks for `pipeline_w.yaml` first, falls back to `pipeline.yaml`
- `deep_merge` semantics via OmegaConf: dicts merge recursively, lists and scalars replace
- Outputs validated `PipelineConfig` (or raw dict if you opt out)

### Why OmegaConf, Not Hydra

OmegaConf provides the merge primitive and interpolation. Hydra adds decorator-based entry points (`@hydra.main()`), working directory management, and multirun orchestration — all of which conflict with an MCP-driven pipeline where an AI agent calls tools, not a CLI. Hydra is too opinionated for this use case.

---

## Package 3: `feature-registry` — Registration, Discovery, Resolution, Building

Domain-agnostic feature management. Extracted from MM's `src/mm/features/registry.py` plus the builder and resolution layers that connect features to models.

### Capabilities

- `@register_feature` decorator capturing metadata and compute function
- Auto-discovery via `pkgutil` — import a package, all features register themselves
- Source code hashing for change detection (feeds into fingerprint caching)
- Validation: declared output columns, valid ranges, NaN strategies, **temporal filters**
- Query API: list/filter/inspect features without computing them
- **Feature resolver**: maps model config feature references to actual DataFrame columns
- **Feature builder**: incremental per-feature computation with manifest-based staleness detection
- **Pairwise feature builder**: constructs matchup-level features from entity-level features

### 3a. Registration + Discovery

```python
from feature_registry import FeatureRegistry

registry = FeatureRegistry()

@registry.register(
    name="scoring_margin",
    category="offense",
    level="team",
    output_columns=["scoring_margin", "scoring_margin_std"],
    depends_on=["game_results"],
    valid_range=(-50, 50),
    nan_strategy="median",
    temporal_filter=TemporalFilter(exclude_event_types=["tournament"]),
)
def compute_scoring_margin(df, config):
    return df[["entity_id", "period_id", "scoring_margin", "scoring_margin_std"]]

registry.discover("my_project.features")
registry.list_features(category="offense")
registry.source_hash("scoring_margin")      # → "a3f2b1c..."
```

### 3b. Feature Resolver

Bridges model config to actual DataFrame columns. Supports explicit column names and category-based resolution.

```python
from feature_registry import FeatureResolver

resolver = FeatureResolver(registry=registry)

# Explicit: model config lists exact column names
columns = resolver.resolve(["diff_seed_num", "diff_adj_em"], available_columns=df.columns)

# Category-based: model config lists feature categories
columns = resolver.resolve_categories(["offense", "defense"], available_columns=df.columns)
```

Category-to-column mapping is derived from registry metadata (not a hardcoded prefix map). The resolver validates that all requested columns exist in the DataFrame.

### 3c. Feature Builder

Orchestrates incremental feature computation with manifest-based cache invalidation.

```python
from feature_registry import FeatureBuilder

builder = FeatureBuilder(
    registry=registry,
    cache_dir="data/features/.cache/",
    manifest_path="data/features/manifest.json",
)

# Build all features — skips unchanged (by source hash), computes stale
features_df = builder.build_all(raw_data=df, config=config)
# - Per-feature parquet shards cached in cache_dir
# - Manifest tracks source hash per feature
# - Output columns validated against declarations
# - NaN strategies applied per feature metadata
```

### 3d. Pairwise Feature Builder

Constructs matchup/comparison features from entity-level features. Common pattern in sports (team vs team), elections (candidate vs candidate), competitive gaming.

```python
from feature_registry import PairwiseFeatureBuilder

pairwise = PairwiseFeatureBuilder(
    methods=["diff"],          # diff, ratio, concat, or custom
    id_columns=["entity_a_id", "entity_b_id", "period_id"],
)

matchup_df = pairwise.build(entity_features=team_df, matchups=matchup_pairs)
# → DataFrame with diff_scoring_margin, diff_adj_em, etc.
```

### Design Decisions

- `level` and `category` are free-form strings, not enums. MM uses "team"/"matchup"; NFL might use "player"/"team"/"game".
- `depends_on` is declarative — the registry tracks it, builder enforces order.
- `entity_id` and `period_id` are generic names for what MM calls "TeamID" and "Season".
- Compute functions receive raw DataFrame + config dict. Registry doesn't own the DataFrame schema.
- `temporal_filter` on `FeatureMeta` enables automated leakage validation — the builder checks that compute outputs don't include filtered event types.

---

## Package 4: `model-ensemble` — Training, Stacking, Backtesting

The heaviest package. Ships with pre-built wrappers for common model types, temporal cross-validation, training orchestration, ensemble stacking, backtesting, and versioned output management.

### 4a. Built-In Model Wrappers

| Type String | Wraps | Classifier | Regressor | Notes |
|------------|-------|------------|-----------|-------|
| `xgboost` | XGBClassifier/Regressor | yes | yes | Early stopping, GPU support |
| `catboost` | CatBoostClassifier/Regressor | yes | yes | Categorical feature support |
| `lightgbm` | LGBMClassifier/Regressor | yes | yes | |
| `random_forest` | RandomForestClassifier/Regressor | yes | yes | |
| `logistic_regression` | LogisticRegression | yes | no | |
| `elastic_net` | SGDClassifier (log loss) | yes | no | L1/L2 via `l1_ratio` |
| `mlp` | Custom PyTorch MLP | yes | yes | Multi-seed, dropout, early stopping |
| `tabnet` | pytorch-tabnet | yes | yes | Multi-seed averaging |

Each wrapper implements `BaseModel`:
- `fit(X, y, X_val, y_val)` — including early stopping where supported
- `predict_proba(X)` — classifiers return probabilities; regressors use CDF conversion
- `predict_margin(X)` — regressors return raw margins
- `save(path)` / `load(path)` — serialization
- Multi-seed averaging — set `n_seeds: 5` in config
- Margin → probability CDF conversion for regression models (configurable scale)

**Installation via optional extras:**

```
pip install model-ensemble[xgboost]
pip install model-ensemble[torch]      # MLP + TabNet
pip install model-ensemble[all]
```

**Custom model types remain pluggable:**

```python
registry = ModelRegistry.with_defaults()
registry.register("gnn", MyGNNModel)           # project-specific
registry.register("survival", MySurvivalModel)
```

**Config-driven model creation (what Claude actually does):**

```yaml
# config/models/production.yaml
models:
  xgb_core:
    type: xgboost
    mode: classifier
    features: [diff_seed_num, diff_adj_em, diff_sr_srs]
    params:
      max_depth: 3
      learning_rate: 0.05
      n_estimators: 500
```

```python
model = registry.create_from_config(config["models"]["xgb_core"])
model.fit(X_train, y_train, X_val, y_val)
```

### 4b. Temporal Cross-Validation

Sports and financial ML share a fundamental constraint: you can never use future data to predict the past. We build our own CV strategies (~300 LOC) because existing libraries don't fit:

- **sklearn `TimeSeriesSplit`**: no group awareness, no embargo, no nested calibration splits
- **`tscv` / `timeseriescv`**: designed for continuous indices (daily financial data), not discrete season groups. Forking would require reworking core splitting logic — at that point you've rewritten most of it. The continuous-index assumption is in every function signature.
- **None** support `NestedCV` with inner calibration carve-out — the critical feature for leakage prevention

The purged k-fold algorithm from `tscv` is worth referencing (~20 lines of embargo-gap logic), but not a reason to fork the whole library.

**Built-in strategies:**

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `LeaveOneSeasonOut` | Train on all folds < target, predict target | MM: train on 2015-2024, predict 2025 |
| `ExpandingWindow` | Train on folds 1..N-1, predict fold N | NFL: train on weeks 1-16, predict week 17 |
| `SlidingWindow` | Train on folds N-K..N-1, predict fold N | Handle distribution shift: only recent data |
| `PurgedKFold` | K-fold with embargo gap between train/test | Prevent near-boundary leakage |
| `NestedCV` | Wraps any outer strategy + inner calibration split | Per-fold pre-calibration without leakage |

**Protocol — all strategies implement the same interface:**

```python
from model_ensemble.cv import FoldGenerator, Fold

class FoldGenerator(Protocol):
    def split(self, data, fold_ids, **kwargs) -> list[Fold]: ...

@dataclass
class Fold:
    fold_id: str | int
    train_idx: np.ndarray
    test_idx: np.ndarray
    calibration_idx: np.ndarray | None  # for nested CV / per-fold pre-cal
```

**API:**

```python
from model_ensemble.cv import (
    LeaveOneSeasonOut, ExpandingWindow, SlidingWindow, PurgedKFold, NestedCV,
)

# LOSO — what MM uses today
loso = LeaveOneSeasonOut(min_train_folds=3)
folds = loso.split(data, fold_ids=data["season"])

# Nested CV — any strategy + inner calibration split
nested = NestedCV(
    outer=LeaveOneSeasonOut(),
    inner_calibration_fraction=0.2,
)
folds = nested.split(data, fold_ids=data["season"])
# → Fold(..., calibration_idx=[20% of train rows])
```

**Config-driven:**

```yaml
# config/ensemble.yaml
cross_validation:
  strategy: leave_one_out
  min_train_folds: 3
  nested:
    enabled: true
    calibration_fraction: 0.2
```

**Critical invariants (enforced at runtime, not just by convention):**

1. **Temporal ordering**: After `cv.split()`, the framework asserts `max(train_fold_ids) < test_fold_id` for every fold. This catches bugs in custom `FoldGenerator` implementations. Non-bypassable — no `human_override`.
2. **Per-fold pre-calibration**: Pre-calibration MUST happen inside the fold loop, not before it. `NestedCV` enforces this by only exposing calibration indices per fold. Fitting pre-calibration once on the full training set is the leakage bug discovered in EXP-039 — the framework makes it structurally impossible.

### 4c. Training Orchestrator

The glue between model wrappers, feature resolver, fingerprint caching, and the CV system. Every project needs this — without it, users write ~150 LOC of boilerplate to iterate models.

```python
from model_ensemble import TrainOrchestrator

orchestrator = TrainOrchestrator(
    model_registry=registry,
    feature_resolver=resolver,
    config=config,
)

# Train all models for one fold (or production = all data)
results = orchestrator.train_all(
    data=features_df,
    target_fold=2025,     # None = production (all data)
    output_dir=Path("models/runs/20260228/"),
)
```

**What it handles:**
- Iterates all model configs, resolves features per model via `FeatureResolver`
- Checks fingerprint cache — unchanged models skip retraining
- Carves validation split (most recent fold as val for early stopping)
- Selects target column per model (margin for regressors, binary for classifiers)
- Computes feature medians at train time, persists as `feature_medians.json` for prediction-time NaN imputation
- Saves sidecar artifacts: `model_features.json` (column mapping), fingerprint files
- Supports dry-run mode (validate config + features without training)
- **Failure policy**: if one model fails, logs warning and continues (configurable: `skip_and_continue`, `abort`)

### 4d. Per-Fold Model Checkpoints

For backtesting, you need N complete model copies — one per fold — trained only on data from before that fold.

```python
# Train production models (all data)
orchestrator.train_all(data, target_fold=None, output_dir=production_dir)

# Train per-fold backtest models
for fold in cv.split(data, fold_ids):
    orchestrator.train_all(
        data=data.loc[fold.train_idx],
        target_fold=fold.fold_id,
        output_dir=backtest_dir / str(fold.fold_id),
    )
```

The orchestrator manages the per-fold artifact directories. Fingerprint caching works per-fold — if fold 2018's models haven't changed, they skip retraining even when fold 2019's models need updating.

### 4e. Fingerprint Caching

Deterministic hashing of model config + feature source hashes + data mtime. Same inputs = skip retraining.

```python
from model_ensemble import Fingerprint

fp = Fingerprint.compute(model_config, feature_hash, data_mtime)
if fp.matches(Path("models/xgb_core/.fingerprint")):
    model = BaseModel.load("models/xgb_core/")
else:
    model = train(...)
    fp.save("models/xgb_core/.fingerprint")
```

**Known failure mode:** Fingerprint caching can preserve stale sidecar artifacts (e.g., `model_features.json` with bad column names from pandas merge artifacts). The `mcp-guardrails` package exposes an `invalidate_cache` tool for manual cache busting when fixing bugs.

### 4f. Meta-Learner + Calibration

Stacked ensemble with cross-validated training.

```python
from model_ensemble import StackedEnsemble, SplineCalibrator

ensemble = StackedEnsemble(
    method="stacked",
    meta_learner_type="logistic",
    meta_learner_params={"C": 2.5},
    pre_calibrate={"mlp_margin": SplineCalibrator(n_bins=20, prob_max=0.985)},
    post_calibrate=SplineCalibrator(n_bins=20),
)

ensemble.fit(
    base_predictions={"xgb_core": preds_xgb, "mlp_margin": preds_mlp},
    y_true=labels,
    cv=LeaveOneSeasonOut(min_train_folds=3),
    fold_ids=data["season"],
)

final_probs = ensemble.predict(new_base_predictions)
ensemble.coefficients()  # → {"xgb_core": +1.93, "mlp_margin": +3.04}
```

**Calibrator implementations:**
- `SplineCalibrator` — PCHIP interpolation through quantile-binned calibration points. Custom, ~85 LOC. This is the method that works best (EXP-009, EXP-063).
- `PlattCalibrator` — sklearn `CalibratedClassifierCV(method="sigmoid")` wrapper
- `IsotonicCalibrator` — sklearn `CalibratedClassifierCV(method="isotonic")` wrapper

### 4g. Ensemble Post-Processing Chain

Extensible chain of post-ensemble probability adjustments. The ensemble output goes through:

```
pre-calibration → meta-learner → post-calibration → temperature scaling → clipping → post-hoc adjustments
```

```python
from model_ensemble import EnsemblePostprocessor, LogitAdjustment

postprocessor = EnsemblePostprocessor(
    steps=[
        ("temperature", TemperatureScaling(T=1.0)),
        ("clip", ProbabilityClipping(floor=0.02, ceiling=0.98)),
        ("availability", LogitAdjustment(feature="availability_score", strength=0.1)),
    ]
)

final_probs = postprocessor.apply(ensemble_probs, context_features=matchup_df)
```

Users register arbitrary post-hoc adjustments (injury adjustments, weather adjustments, etc.) via the `LogitAdjustment` base class. The chain is config-driven.

### 4h. Backtest Runner

The primary evaluation system. Loads per-fold predictions, trains fold-specific meta-learners, evaluates against actuals, computes baselines and diagnostics.

```python
from model_ensemble import BacktestRunner

runner = BacktestRunner(
    ensemble=ensemble,
    cv=LeaveOneSeasonOut(),
    metrics=["brier", "accuracy", "ece", "log_loss"],
    baseline_fn=seed_probability_baseline,  # domain-specific baseline
)

results = runner.run(
    per_fold_predictions=predictions_by_fold,
    actuals=actuals_by_fold,
)
# → BacktestResult(
#     pooled_metrics={"brier": 0.1752, "accuracy": 0.7452, ...},
#     per_fold_metrics={2018: {...}, 2019: {...}, ...},
#     baseline_metrics={"brier": 0.220, ...},
#     diagnostics=DiagnosticsReport(...),
# )
```

**Two-pass architecture:**
1. Load pre-computed per-fold base model predictions
2. For each held-out fold, train a fold-specific meta-learner on all OTHER folds' predictions, apply to held-out fold (LOSO prevents leakage)
3. Apply post-processing chain per fold
4. Evaluate against actuals, compute pooled metrics, compare to baselines

### 4i. Tracking Callbacks

Hook for external experiment tracking tools (MLflow, W&B, Neptune).

```python
from model_ensemble import TrackingCallback

class MLflowTracker(TrackingCallback):
    def on_model_trained(self, model_name, params, metrics): ...
    def on_backtest_complete(self, pooled_metrics, per_fold_metrics): ...
    def on_experiment_logged(self, experiment_id, result): ...

orchestrator = TrainOrchestrator(..., callbacks=[MLflowTracker()])
```

The callback protocol is optional. The framework's own experiment logging (`exp-runner`) works without it. This just provides hooks for teams that want MLflow/W&B integration alongside the built-in system.

### 4j. Versioned Output Management (RunManager)

```python
from model_ensemble import RunManager

rm = RunManager(base_dir="models/")
run_dir = rm.new_run()                  # models/runs/20260228_143021/
rm.promote(run_dir.name)                # updates manifest + "latest" symlink
rm.get_latest()                         # resolves current run
rm.list_runs()                          # all runs with timestamps and labels
```

Timestamped directories, JSON manifest, symlink promotion. Works for models, predictions, backtest outputs — any artifact category.

---

## Package 5: `data-pipeline` — Source Registry, Stage Guards, DVC Integration

Manages everything between raw data sources and the feature/model layers. **Uses DVC for DAG orchestration** (not a custom DAG engine) with typed artifact metadata layered on top.

### 5a. Pipeline DAG — DVC with Typed Artifacts

DVC already handles dependency resolution, change detection, and parallel execution. We don't rebuild that. Instead, we layer typed artifact declarations on top of DVC stages:

```yaml
# config/pipeline.yaml — typed artifact metadata (not a replacement for dvc.yaml)
artifacts:
  raw_data:
    type: data
    path: data/processed/
    produced_by: ingest

  base_features:
    type: features
    path: data/features/
    produced_by: featurize

  survival_curves:
    type: features              # ← output is features, not a model
    path: data/survival/
    produced_by: survival

  gnn_embeddings:
    type: features              # ← embeddings used as features
    path: data/gnn/embeddings/
    produced_by: gnn

  tree_ensemble_preds:
    type: predictions
    path: data/predictions/
    produced_by: train
    consumes: [base_features, survival_curves, gnn_embeddings]
```

**DVC config generator** — if Claude needs to add stages via experiment overlay, a thin generator (~50 LOC) translates artifact declarations into `dvc.yaml` stages:

```python
from data_pipeline import generate_dvc_yaml

dvc_config = generate_dvc_yaml(config["artifacts"])
# Writes dvc.yaml with correct deps/outs based on artifact type and consumes declarations
```

This preserves DVC's battle-tested dependency resolution while giving the framework typed artifact awareness (for validation, inspection tools, and DAG visualization).

### 5b. Data Source Registry

```python
from data_pipeline import SourceRegistry

sources = SourceRegistry()

@sources.register(
    name="barttorvik",
    category="external",
    outputs=["data/external/barttorvik/"],
    freshness_check=lambda path: file_age_hours(path / "team_stats.csv") < 24,
    schema={"required_columns": ["team", "adj_o", "adj_d", "adj_t"]},
    temporal_safety="pre_tournament",
    leakage_notes="Barttorvik data is updated daily but excludes tournament games from team stats",
)
def scrape_barttorvik(output_dir, config):
    ...  # domain-specific scraping

sources.check_freshness_all()  # → {"barttorvik": False, "kenpom": True}
```

**Leakage metadata** (`temporal_safety`, `leakage_notes`) gives the AI agent structured information about what data is safe to use when creating features. The `mcp-guardrails` inspection tools surface this.

### 5c. Stage Guards

Declarative prerequisite checks with **staleness detection** — compares artifact mtime to source code and config file mtimes.

```python
from data_pipeline import StageGuard

guard = StageGuard(
    name="train_ready",
    requires=["data/features/team_season_features.parquet"],
    min_rows=1000,
    max_age_hours=168,
    stale_if_older_than=[            # staleness vs source code/config
        "config/models/production.yaml",
        "src/my_project/features/",
    ],
)

guard.check()  # raises GuardrailViolation if stale or missing
```

Guards are called imperatively at the start of each pipeline script (not via decorators — the imperative pattern is simpler and matches the actual codebase's approach).

### 5d. Refresh Orchestration

Failure-tolerant scrape → ingest → featurize chain. Scraper failures are warnings, not blockers.

```python
from data_pipeline import RefreshOrchestrator

refresh = RefreshOrchestrator(
    source_registry=sources,
    dvc_stages=["ingest", "featurize", "survival", "train_gnn"],
    continue_on_scraper_failure=True,
    stop_on_stage_failure=True,
)

result = refresh.run()
# → RefreshResult(
#     scrapers={"barttorvik": "ok", "kenpom": "failed: 429"},
#     stages={"ingest": "ok", "featurize": "ok"},
#     warnings=["kenpom: using stale data (72h old)"],
# )
```

---

## Package 6: `exp-runner` — Experiment Protocol Enforcement

Manages the experiment lifecycle. Doesn't run models — manages the protocol around running them.

### Capabilities

- Create experiment directories with overlay templates
- Detect changes between production config and overlay
- Compare results to baseline with structured deltas
- Log every experiment to a markdown ledger
- Maintain do-not-retry registry of known failed patterns
- **Enforce single-variable discipline** (programmatic, not just documentation)
- **Enforce mandatory logging** (block next experiment if previous wasn't logged)
- **Atomic promotes with rollback** (backup → write → verify)

### API

```python
from exp_runner import ExperimentManager

mgr = ExperimentManager(
    experiments_dir="experiments/",
    baseline_metrics={"brier": 0.1752, "accuracy": 0.7452},
    naming_pattern=r"exp-\d{3}-[a-z0-9-]+$",
    log_path="EXPERIMENT_LOG.md",
)

# Create
exp = mgr.create("exp-055-new-feature")

# Detect changes — powers single-variable enforcement
changes = mgr.detect_changes(production_config, exp.overlay)
# → ChangeSet(changed_models=["xgb_core"], new_models=[], ensemble_changed=False)

# Log (mandatory after every experiment)
mgr.log(
    experiment_id="exp-055-new-feature",
    hypothesis="Adding pace features improves late-round accuracy",
    changes="Added pace_adj, tempo_rank to xgb_core",
    metrics={"brier": 0.1748, "accuracy": 0.7460},
    verdict="keep",
    notes="Brier improved 0.0004, directionally positive",
)

# Do-not-retry
mgr.add_do_not_retry(
    pattern="temperature scaling",
    reference="EXP-002",
    reason="T>1.0 hurts Brier monotonically",
)
mgr.check_do_not_retry(overlay_text)  # raises GuardrailViolation if match
```

### Single-Variable Enforcement

The `detect_changes()` method returns a `ChangeSet`. The guardrail blocks if too many things changed:

```python
changes = mgr.detect_changes(production, overlay)
total_changes = len(changes.changed_models) + len(changes.new_models) + int(changes.ensemble_changed)
if total_changes > 1:
    raise GuardrailViolation(rule="single_variable", message=f"Overlay changes {total_changes} things...")
```

### Mandatory Logging Enforcement

The experiment manager tracks state: was the last experiment logged?

```python
if mgr.has_unlogged_experiment():
    raise GuardrailViolation(
        rule="experiment_not_logged",
        message=f"Experiment {mgr.last_experiment_id} has not been logged. Call log_experiment first.",
    )
```

This blocks `create_experiment` and `run_experiment` if the previous experiment's results haven't been logged. Soft reminders aren't enough — the LLM will forget.

### Atomic Promote with Rollback

```python
mgr.promote(
    experiment_id="exp-055-new-feature",
    model_name="xgb_core",
    production_config_path="config/models/production.yaml",
)
# 1. Validates experiment has logged verdict of "keep"
# 2. Creates backup of production.yaml
# 3. Writes to temp file, then atomic rename
# 4. If write fails, restores from backup
```

### Design Decisions

- `naming_pattern` is a regex you provide — not hardcoded.
- `baseline_metrics` is `dict[str, float]` — not tied to Brier.
- The manager doesn't run models. It manages the protocol: create → detect → (you train) → log → guard.
- Log format is markdown (human-readable, git-friendly) + structured JSON for programmatic access.
- Do-not-retry patterns are regex-based, stored in JSON alongside the experiment log.

---

## Package 7: `mcp-guardrails` — The Crown Jewel

AI-agent interface layer. Wraps any pipeline using the other tools and makes it safe for Claude to operate. Nothing exists in the MCP ecosystem for ML-domain-specific guardrails — this is the differentiator.

### Capabilities

- Generate MCP server from pipeline definition
- Pre-execution guardrails (sanity checks, do-not-retry, naming, single-variable)
- **Leakage guardrails** (tainted column denylist, feature temporal filter validation)
- **Production config protection** (diff against git HEAD before training)
- **Destructive action prevention** (critical path protection)
- **Rate limiting** (cooldown between expensive operations)
- **Mandatory experiment logging enforcement**
- Auto-generated read-only inspection tools from registries
- Structured output formatting for agent consumption
- **Structured audit log** (JSON, every tool invocation)
- **Cache invalidation tool** (manual fingerprint busting)
- `human_override` escape hatch on every guardrail

### API

```python
from mcp_guardrails import PipelineServer, Guardrail, tool

class MyMLServer(PipelineServer):
    stages = ["ingest", "featurize", "train", "predict", "evaluate"]

    guardrails = [
        Guardrail.sanity_check(script="scripts/sanity_check.py"),
        Guardrail.naming_convention(pattern=r"exp-\d{3}-[a-z0-9-]+$"),
        Guardrail.do_not_retry(registry_path="experiments/do_not_retry.json"),
        Guardrail.single_variable(max_changes=1),
        Guardrail.feature_leakage(denylist=["kp_adj_o", "kp_sos"]),
        Guardrail.config_protection(git_diff=True),
        Guardrail.critical_paths(protected=["models/", "data/features/", "config/"]),
        Guardrail.rate_limit(min_interval_minutes=5, category="training"),
        Guardrail.experiment_logged(experiment_manager=mgr),
        Guardrail.feature_staleness(manifest_path="data/features/manifest.json"),
    ]

    @tool(stage="train", guardrails=["sanity_check", "do_not_retry", "config_protection", "rate_limit"])
    def train_models(self, run_id=None, experiment_id=None, human_override=False):
        return self.run_stage("train", run_id=run_id, experiment_id=experiment_id)

    @tool(category="analysis")
    def show_baseline(self):
        return self.format_metrics(self.experiment_manager.baseline_metrics)

    @tool(category="maintenance")
    def invalidate_cache(self, model_name=None):
        """Delete fingerprint files to force retraining."""
        return self.cache_manager.invalidate(model_name)
```

### Guardrail Inventory

| Guardrail | Purpose | Overridable |
|-----------|---------|-------------|
| `sanity_check` | Run validation script before training | yes |
| `naming_convention` | Enforce experiment naming pattern | yes |
| `do_not_retry` | Block experiments matching known failure patterns | yes |
| `single_variable` | Block overlays that change multiple things | yes |
| `feature_leakage` | Block models referencing tainted columns | **no** |
| `config_protection` | Diff production config against git HEAD, warn on unauthorized changes | yes |
| `critical_paths` | Refuse to delete/overwrite protected directories via tools | **no** |
| `rate_limit` | Cooldown between expensive operations (training, full pipeline) | yes |
| `experiment_logged` | Block next experiment if previous wasn't logged | yes |
| `feature_staleness` | Warn if feature source code changed but featurize hasn't re-run | yes |
| `temporal_ordering` | (In CV) Assert train folds < test fold | **no** |

**Non-overridable guardrails** protect against data leakage and data destruction. These are the two failure modes where "human_override" is not an appropriate escape hatch because the consequences are silent and hard to detect.

### Auto-Generated Tools

By using the other packages, these tools come free:

| From Package | Tools Generated |
|---|---|
| `config-overlay` | `show_config`, `show_config --section X` |
| `feature-registry` | `list_features`, `show_feature_details` |
| `model-ensemble` | `list_models`, `show_model_details`, `show_ensemble_config`, `show_calibration_table`, `show_model_correlations`, `model_audit` |
| `data-pipeline` | `check_freshness`, `show_pipeline_dag`, `show_source_leakage_notes` |
| `exp-runner` | `list_experiments`, `show_experiment`, `compare_experiments`, `show_do_not_retry` |
| `RunManager` | `list_runs`, `show_run_details`, `compare_runs` |

### Structured Audit Log

Every tool invocation is logged to a structured JSON audit trail:

```json
{
  "timestamp": "2026-02-28T14:30:21Z",
  "tool": "train_models",
  "args": {"run_id": null, "experiment_id": "exp-055-new-feature"},
  "guardrails_checked": ["sanity_check", "do_not_retry", "config_protection"],
  "guardrails_passed": true,
  "result_status": "success",
  "duration_s": 342.1,
  "log_path": "/tmp/mm_train_20260228_143021.log"
}
```

This enables post-hoc review of what the agent did and why.

### Guardrail Execution Flow

```
Agent calls tool → guardrails check (ordered) → if blocked: return GuardrailViolation
                                               → if passed: execute pipeline command
                                               → format output → return to agent
                                               → log to audit trail
```

Every guardrail (except non-overridable ones) accepts `human_override=True`. Agent can't set this autonomously — MCP protocol surfaces confirmation to the human.

### Logging

Dual-stream by default:
- `stderr` → INFO (visible in Claude Code CLI)
- File → DEBUG at configurable path (persistent across sessions)

---

## Package Summary

| Package | Size (est.) | Dependencies | Key Responsibility |
|---------|------------|-------------|-------------------|
| `ml-schemas` | ~500 LOC | pydantic, numpy, sklearn | Contracts, built-in metrics, leakage prevention schemas |
| `config-overlay` | ~300 LOC | ml-schemas, omegaconf, pyyaml | OmegaConf merge, file loading, variant resolution |
| `feature-registry` | ~700 LOC | ml-schemas | Registration, discovery, resolution, building, pairwise features |
| `model-ensemble` | ~2500 LOC | ml-schemas, numpy, scipy, + optional ML libs | Wrappers, temporal CV, orchestrator, backtest, stacking, calibration, post-processing, versioning |
| `data-pipeline` | ~400 LOC | ml-schemas | Source registry, DVC integration, stage guards, refresh |
| `exp-runner` | ~600 LOC | ml-schemas, config-overlay | Experiment protocol, mandatory logging, do-not-retry, atomic promotes |
| `mcp-guardrails` | ~800 LOC | ml-schemas, fastmcp, exp-runner | MCP server gen, 11 guardrails, auto-inspection, audit log |

**Total: ~5800 LOC** across 7 packages.

## What Claude Can Do Without Writing Code

With this framework, the AI agent operates entirely through config:

1. **Add a new model** — edit `config/models/production.yaml`: specify type, features, params
2. **Wire model output as features** — edit `config/pipeline.yaml`: add artifact with `type: features`
3. **Create nested ensembles** — overlay: add meta_ensemble stage consuming predictions
4. **Run experiments** — create overlay YAML with single-variable change, run via MCP
5. **Compare to baseline** — MCP inspection tools, structured deltas
6. **Log results** — mandatory `log_experiment` call (enforced, not optional)
7. **Avoid past mistakes** — do-not-retry guardrails block known failures automatically
8. **Inspect leakage safety** — `show_source_leakage_notes`, tainted column denylist
9. **Switch CV strategies** — overlay `cross_validation.strategy` in ensemble config

The only time code is needed: domain-specific data loaders (scrapers), custom model types (GNN, survival), and project-specific feature computations.

## Next Steps

1. Validate abstractions against second project (NFL/NBA prediction)
2. Determine package naming and namespace
3. Write implementation plan (skill: writing-plans)
4. Extract `ml-schemas` first — everything depends on it
5. Extract packages in dependency order: schemas → config → features → model-ensemble → data-pipeline → exp-runner → mcp-guardrails
