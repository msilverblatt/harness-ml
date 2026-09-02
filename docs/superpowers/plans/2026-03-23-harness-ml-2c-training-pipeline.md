# harness-ml Plan 2c: Training Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete training pipeline — the heart of harness-ml. DAG-driven model orchestration, cross-validation, fingerprint caching, meta-learner stacking, calibration, post-processing, and the top-level backtest runner that composes everything.

**Architecture:** Stateless runners composed by a top-level backtest orchestrator. The pipeline follows a precisely verified 4-phase flow: (1) base model training per fold with DAG wave ordering, (2) meta-learner + post-processing via nested LOSO, (3) metrics + diagnostics, (4) production artifact fitting. See spec Section 7 for the complete flow.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, pydantic, concurrent.futures (ThreadPoolExecutor)

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Section 7 (Training Pipeline)

**Dependencies:** harness-ml Plans 2a (task types, models) + 2b (features, evals) must be complete.

**E2E testing mandate:** After every 2-3 tasks, run real e2e tests that train actual models on actual data through the actual pipeline. Verify metric values, not just "did it run."

---

## File Structure

```
packages/harness-ml/src/harness/ml/
├── config/
│   ├── __init__.py
│   ├── project.py             # ProjectConfig schema (task type, target, CV, metrics)
│   ├── models.py              # ModelConfig schema (per-model: type, params, features, active)
│   ├── ensemble.py            # EnsembleConfig schema (method, meta-learner, calibration, post-processing)
│   └── loader.py              # Load configs from YAML workspace files
├── runners/
│   ├── __init__.py
│   ├── backtest.py            # Top-level orchestrator (THE entry point)
│   ├── cross_validation.py    # 8 CV fold generation strategies
│   ├── training.py            # Per-fold model training (single model)
│   ├── preprocessing.py       # Leakage-safe fit/transform
│   ├── meta_learner.py        # OOF collection + nested LOSO stacking
│   ├── calibration.py         # Pre/post calibration
│   ├── postprocessing.py      # 9-step ordered pipeline
│   ├── prediction_cache.py    # SHA256 fingerprint-based skip-unchanged
│   ├── provider_context.py    # Per-fold provider outputs (instance + entity)
│   ├── dag.py                 # Dependency graph, topological waves, cycle detection
│   └── progress.py            # Progress callback protocol

tests/
├── test_config/
│   ├── test_project.py
│   ├── test_models.py
│   └── test_ensemble.py
├── test_runners/
│   ├── test_cross_validation.py
│   ├── test_preprocessing.py
│   ├── test_prediction_cache.py
│   ├── test_provider_context.py
│   ├── test_dag.py
│   ├── test_training.py
│   ├── test_meta_learner.py
│   ├── test_postprocessing.py
│   └── test_backtest.py
└── test_e2e.py                # Extended with full pipeline e2e tests
```

---

### Task 1: Config Schemas

**Files:**
- Create: `src/harness/ml/config/__init__.py`
- Create: `src/harness/ml/config/project.py`
- Create: `src/harness/ml/config/models.py`
- Create: `src/harness/ml/config/ensemble.py`
- Create: `src/harness/ml/config/loader.py`
- Create: `tests/test_config/` with tests

These are Pydantic schemas representing the workspace config files (`project.yaml`, `models.yaml`, `ensemble.yaml`).

**ProjectConfig:**
```python
class CVConfig(BaseModel):
    strategy: str = "kfold"           # One of 8 strategies
    n_folds: int = 5
    fold_column: str | None = None    # Column containing fold assignments
    fold_values: list | None = None   # Explicit fold values (for LOSO)
    min_train_folds: int = 2

class ProjectConfig(BaseModel):
    task_type: str = "binary"
    target_column: str = "target"
    cv: CVConfig = CVConfig()
    metrics: list[str] = Field(default_factory=lambda: ["brier", "accuracy"])
    eval_filter: str | None = None    # Pandas query for metrics filtering
```

**ModelConfig:**
```python
class SingleModelConfig(BaseModel):
    name: str
    model_type: str                   # "xgboost", "logistic", etc.
    params: dict = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)   # Feature names this model uses
    active: bool = True
    include_in_ensemble: bool = True
    n_seeds: int = 1
    depends_on: list[str] = Field(default_factory=list)  # Provider models
    provides: str | None = None       # What this model provides (for DAG)
    provides_level: str = "instance"  # "instance" or "entity"
    training_filter: str | None = None
    zero_fill_features: list[str] = Field(default_factory=list)
    class_weight: str | dict | None = None
    augment_symmetry: bool = False

class ModelsConfig(BaseModel):
    models: dict[str, SingleModelConfig] = Field(default_factory=dict)
```

**EnsembleConfig:**
```python
class EnsembleConfig(BaseModel):
    method: str = "stacked"           # "stacked" or "average"
    meta_learner_type: str = "logistic"  # "logistic", "ridge", "gbm"
    meta_learner_params: dict = Field(default_factory=dict)
    exclude_models: list[str] = Field(default_factory=list)
    calibration: str = "none"         # "spline", "isotonic", "platt", "none"
    pre_calibration: dict[str, str] = Field(default_factory=dict)  # {model: method}
    temperature: float = 1.0
    clip_floor: float | None = None
    meta_features: list[str] = Field(default_factory=list)
    prior_feature: str | None = None
```

**ConfigLoader:**
```python
class ConfigLoader:
    @staticmethod
    def load_project(path: Path) -> ProjectConfig: ...
    @staticmethod
    def load_models(path: Path) -> ModelsConfig: ...
    @staticmethod
    def load_ensemble(path: Path) -> EnsembleConfig: ...
```

Tests: creation with defaults, from YAML dict, validation, missing fields use defaults.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement config schemas**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): config schemas (project, models, ensemble)"
```

---

### Task 2: Cross-Validation Strategies

**Files:**
- Create: `src/harness/ml/runners/__init__.py`
- Create: `src/harness/ml/runners/cross_validation.py`
- Create: `tests/test_runners/__init__.py`
- Create: `tests/test_runners/test_cross_validation.py`

Implement all 8 CV strategies. Each takes a DataFrame + CVConfig and returns a list of `(train_indices, test_indices)` tuples.

```python
def generate_folds(df: pd.DataFrame, config: CVConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate train/test index splits based on the CV strategy."""
    strategy = config.strategy
    if strategy == "kfold":
        return _kfold(df, config)
    elif strategy == "stratified_kfold":
        return _stratified_kfold(df, config)
    # ... etc
```

**Strategies:**
1. `kfold` — sklearn KFold
2. `stratified_kfold` — sklearn StratifiedKFold
3. `group_kfold` — sklearn GroupKFold (using fold_column)
4. `leave_one_out` — each unique value in fold_column is a test fold
5. `expanding_window` — train on all prior folds, test on current
6. `sliding_window` — fixed-size train window
7. `purged_kfold` — KFold with temporal gap between train/test
8. `bootstrap` — random resampling

Tests: verify fold count, no train/test overlap, correct index types, each strategy produces expected behavior.

- [ ] **Step 1: Write tests for all 8 strategies**
- [ ] **Step 2: Implement cross_validation.py**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): 8 cross-validation strategies"
```

---

### Task 3: DAG + Provider Context

**Files:**
- Create: `src/harness/ml/runners/dag.py`
- Create: `src/harness/ml/runners/provider_context.py`
- Create: `tests/test_runners/test_dag.py`
- Create: `tests/test_runners/test_provider_context.py`

**DAG:**
```python
class ModelDAG:
    def __init__(self, models: dict[str, SingleModelConfig]):
        self._models = models
        self._graph = self._build_graph()

    def topological_waves(self) -> list[list[str]]:
        """Groups of models that can train in parallel. Wave N depends only on waves 0..N-1."""

    def validate(self) -> list[str]:
        """Returns list of error messages (empty = valid). Checks cycles, missing deps."""

    def dependencies(self, model_name: str) -> set[str]:
        """Get upstream provider dependencies for a model."""
```

**ProviderContext:**
```python
class ProviderContext:
    """Per-fold storage for provider model outputs. Transient — not persisted."""

    def store_instance(self, model_name: str, train_preds: np.ndarray, test_preds: np.ndarray): ...
    def store_entity(self, model_name: str, entity_df: pd.DataFrame): ...
    def get_instance(self, model_name: str) -> tuple[np.ndarray, np.ndarray] | None: ...
    def get_entity(self, model_name: str) -> pd.DataFrame | None: ...
    def inject_features(self, df: pd.DataFrame, split: str, model_deps: list[str]) -> pd.DataFrame:
        """Inject provider outputs as feature columns for a downstream model."""
```

Tests:
- DAG: no deps → single wave, linear chain → sequential waves, parallel models → same wave, cycle detection, missing dep detection.
- ProviderContext: store/get instance, store/get entity, inject features adds columns.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement DAG + ProviderContext**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): model DAG + provider context"
```

---

### Task 4: Prediction Cache + Fingerprinting

**Files:**
- Create: `src/harness/ml/runners/prediction_cache.py`
- Create: `tests/test_runners/test_prediction_cache.py`

```python
class PredictionCache:
    def __init__(self, cache_dir: Path):
        self._dir = cache_dir

    def compute_fingerprint(
        self, model_config: dict, feature_schema: str,
        upstream_fingerprints: dict[str, str] | None = None,
    ) -> str:
        """SHA256 hash of model config + feature schema + upstream fingerprints."""

    def get(self, model_name: str, fold_id: str, fingerprint: str) -> np.ndarray | None:
        """Retrieve cached predictions, or None if cache miss."""

    def put(self, model_name: str, fold_id: str, fingerprint: str, predictions: np.ndarray) -> None:
        """Store predictions in cache."""

    def has(self, model_name: str, fold_id: str, fingerprint: str) -> bool:
        """Check if cache entry exists."""
```

Tests: compute fingerprint deterministic, upstream changes cascade, get/put roundtrip, cache miss returns None, provider models not cached.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement prediction cache**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): prediction cache with fingerprint cascading"
```

---

### Task 5: Preprocessing + Training Runner

**Files:**
- Create: `src/harness/ml/runners/preprocessing.py`
- Create: `src/harness/ml/runners/training.py`
- Create: `src/harness/ml/runners/progress.py`
- Create: `tests/test_runners/test_preprocessing.py`
- Create: `tests/test_runners/test_training.py`

**Preprocessing (leakage-safe):**
```python
class Preprocessor:
    def fit(self, X_train: pd.DataFrame) -> Preprocessor:
        """Fit on training data only. Computes medians for imputation."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted parameters. Never sees test data during fit."""

    @property
    def feature_medians(self) -> dict[str, float]: ...
```

**Training runner (single model per fold):**
```python
def train_single_model(
    model_wrapper, task_type, X_train, y_train, X_test,
    model_config: SingleModelConfig, preprocessor: Preprocessor,
) -> TrainingResult:
    """Train one model on one fold. Handles the full inner loop:
    - Apply training filter
    - Zero-fill specified features
    - Drop NaN rows
    - Compute class weights
    - Apply symmetric augmentation if configured
    - Carve early-stopping split if configured
    - Multi-seed training + averaging
    - Predict on both train + test
    Returns TrainingResult with train_preds, test_preds, model, metadata.
    """
```

**Progress callback:**
```python
class BacktestProgress(Protocol):
    def on_fold_start(self, fold_id: str, fold_num: int, total_folds: int) -> None: ...
    def on_model_trained(self, model_name: str, fold_id: str, duration_s: float) -> None: ...
    def on_wave_complete(self, wave_num: int, total_waves: int) -> None: ...
    def on_backtest_complete(self, metrics: dict[str, float]) -> None: ...
```

Tests:
- Preprocessor: fit on train, transform test, medians correct, no data leakage (test values don't affect fit).
- Training: train_single_model produces predictions of correct length, handles n_seeds > 1, applies training filter, applies augmentation.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement preprocessing + training + progress**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): preprocessing + training runner + progress protocol"
```

---

### Task 6: E2E — Training Components

- [ ] **Step 1: Write e2e tests**

Test the full Phase 1 flow: load data → generate folds → build DAG → for each fold, train models in wave order with provider context and prediction caching. Verify:
- Models actually learn (accuracy > random)
- Provider predictions flow to downstream models
- Fingerprint cache hits on re-run with same config
- Different configs produce cache misses

- [ ] **Step 2: Run e2e tests, fix any integration issues**
- [ ] **Step 3: Commit**

```bash
git commit -m "test(harness-ml): e2e tests for Phase 1 training components"
```

---

### Task 7: Meta-Learner + Calibration

**Files:**
- Create: `src/harness/ml/runners/meta_learner.py`
- Create: `src/harness/ml/runners/calibration.py`
- Create: `tests/test_runners/test_meta_learner.py`

**Meta-learner (nested LOSO stacking):**
```python
class MetaLearner:
    def train_nested_loso(
        self, fold_predictions: dict[str, pd.DataFrame],
        ensemble_config: EnsembleConfig, task_type: str,
    ) -> MetaLearnerResult:
        """Phase 2: For each holdout fold, train meta on others, predict holdout.
        Returns processed predictions per fold + production artifacts."""
```

**Calibration:**
```python
class Calibrator:
    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, method: str) -> Any:
        """Fit a calibrator on training data."""

    def transform(self, y_pred: np.ndarray, calibrator: Any) -> np.ndarray:
        """Apply fitted calibrator to predictions."""
```

Tests: meta-learner produces predictions for each fold, simple average fallback works, calibration fit/transform roundtrip.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement meta-learner + calibration**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): meta-learner (nested LOSO) + calibration"
```

---

### Task 8: Post-Processing Pipeline

**Files:**
- Create: `src/harness/ml/runners/postprocessing.py`
- Create: `tests/test_runners/test_postprocessing.py`

The 9-step ordered pipeline:
```python
def apply_postprocessing(
    predictions: pd.DataFrame,
    meta_learner_result,
    ensemble_config: EnsembleConfig,
    task_type: str,
) -> pd.DataFrame:
    """Apply the 9-step post-processing pipeline in strict order:
    1. Extract base model prediction columns
    2. Filter ensemble.exclude_models
    3. Apply pre-calibration (per-model)
    4. Meta-learner prediction (stacking)
    5. Post-calibration
    6. Temperature scaling
    7. Probability clipping
    8. Logit adjustments
    9. Prior-proximity compression
    """
```

Tests: each step applied in correct order, temperature scaling works, clipping enforces bounds, logit adjustments modify predictions.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement postprocessing**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): 9-step post-processing pipeline"
```

---

### Task 9: Backtest Runner (Top-Level Orchestrator)

**Files:**
- Create: `src/harness/ml/runners/backtest.py`
- Create: `tests/test_runners/test_backtest.py`

This is THE entry point. It composes all previous components following the verified 4-phase flow.

```python
@dataclass
class BacktestResult:
    metrics: dict[str, float]
    per_fold_metrics: list[dict[str, float]]
    predictions: pd.DataFrame
    models_trained: int
    models_cached: int
    models_failed: list[dict]
    duration_s: float
    eval_report: EvalReport | None = None

def run_backtest(
    data: pd.DataFrame,
    feature_set: FeatureSet,
    models_config: ModelsConfig,
    project_config: ProjectConfig,
    ensemble_config: EnsembleConfig,
    cache_dir: Path | None = None,
    progress: BacktestProgress | None = None,
    eval_config: dict | None = None,
) -> BacktestResult:
    """Execute the complete backtest pipeline (4 phases)."""
```

Tests: basic backtest with 2 models produces metrics, multi-fold produces per-fold metrics, models_failed tracks errors, eval_report populated when eval_config provided.

- [ ] **Step 1: Write tests**
- [ ] **Step 2: Implement backtest runner**
- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(harness-ml): backtest runner (4-phase orchestrator)"
```

---

### Task 10: Full Pipeline E2E Tests

**Files:**
- Update: `tests/test_e2e.py`

The most important tests in the entire project. Exercise the complete chain:

```python
class TestE2EBacktest:
    def test_simple_backtest_two_models(self):
        """Train logistic + random_forest, verify ensemble beats either alone."""

    def test_backtest_with_features_and_evals(self):
        """Features → resolve → train → metrics → eval report — full chain."""

    def test_prediction_cache_speeds_up_rerun(self):
        """Run backtest twice with same config — second should be faster (cache hits)."""

    def test_backtest_metrics_are_honest(self):
        """OOF predictions produce honest (not overfit) metrics."""

    def test_different_cv_strategies_produce_different_folds(self):
        """kfold vs stratified_kfold produce different splits."""
```

- [ ] **Step 1: Write comprehensive e2e tests**
- [ ] **Step 2: Run all tests, fix integration issues**
- [ ] **Step 3: Update __init__.py with runner exports**
- [ ] **Step 4: Commit**

```bash
git commit -m "feat(harness-ml): full pipeline e2e tests + exports (Plan 2c complete)"
```
