# Gap Closure: harnessml-runner → Full Pipeline Recreation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** After this plan is implemented, a user should be able to recreate the mm pipeline's model training, ensemble stacking, backtesting, prediction, and diagnostics by pointing harnessml-runner at data files and YAML config. Every capability in `~/mm` that goes through `config/`, `pipelines/`, `src/mm/models/`, `src/mm/ensemble/`, and `src/mm/backtest/` must have a corresponding harnessml-runner path. Domain-specific components (bracket optimization, tournament simulation) are out of scope — those are project-level extensions, not general ML framework concerns.

**Scope:** Only `harnessml-runner` is modified. The 7 library packages remain untouched. Where library packages lack needed APIs, harnessml-runner implements the logic itself (it's the orchestration layer).

**Working directory:** `/Users/msilverblatt/harnessml/packages/harnessml-runner/`

---

## What the mm pipeline does that harnessml-runner cannot

The mm pipeline has a 9-step ensemble post-processing chain, a two-pass backtester with LOSO meta-learner training, per-fold nested pre-calibration, regressor-to-probability CDF conversion, multi-seed neural net averaging, training season filtering, fingerprint-based model caching, a production predict stage, run management, and rich diagnostics. harnessml-runner currently does none of this — it trains models and averages their predictions.

---

## Phase 15: Schema Extensions

### Task 15.1: Extend ModelDef, EnsembleDef, and ProjectConfig schemas

**File:** `src/harnessml/runner/schema.py`

**Why:** The mm config has fields that harnessml-runner's Pydantic models don't know about, so they'd be rejected or silently dropped.

**Changes to ModelDef:**

```python
class ModelDef(BaseModel):
    type: str
    features: list[str] = []
    feature_sets: list[str] = []        # NEW — alternative to explicit features
    params: dict[str, Any] = {}
    active: bool = True
    mode: Literal["classifier", "regressor"] = "classifier"
    prediction_type: str | None = None  # NEW — "margin" for regression→CDF models
    train_seasons: str = "all"          # NEW — "all" or "last_N"
    n_seeds: int = 1
    pre_calibration: str | None = None  # CHANGED — was dict, now string (e.g. "spline")
    training_filter: dict[str, Any] | None = None
    cdf_scale: float | None = None      # NEW — override CDF scale for regression models
```

Also add to `_KNOWN_MODEL_TYPES`:
- `"xgboost_regression"` — XGBRegressor (margin prediction, CDF conversion)
- `"gnn"` — Graph Neural Network
- `"survival"` — Survival hazard model

**Changes to EnsembleDef:**

```python
class EnsembleDef(BaseModel):
    method: Literal["stacked", "average"]
    meta_learner: dict[str, Any] = {}   # e.g. {"C": 2.5, "type": "stacked"}
    pre_calibration: dict[str, str] = {} # CHANGED — was dict[str, Any]|None, now model_name→method
    calibration: str = "spline"         # CHANGED — was dict|None, now string
    spline_prob_max: float = 0.985      # NEW
    spline_n_bins: int = 20             # NEW
    meta_features: list[str] = []       # NEW — extra features for meta-learner input
    temperature: float = 1.0
    clip_floor: float = 0.0
    availability_adjustment: float = 0.1
    seed_compression: float = 0.0       # NEW — seed-proximity compression strength
    seed_compression_threshold: int = 4  # NEW
    exclude_models: list[str] = []
```

**New FeaturesConfig class:**

```python
class FeaturesConfig(BaseModel):
    first_season: int = 2003
    momentum_window: int = 10
```

**Changes to ProjectConfig:**

```python
class ProjectConfig(BaseModel):
    data: DataConfig
    models: dict[str, ModelDef]
    ensemble: EnsembleDef
    backtest: BacktestConfig
    feature_config: FeaturesConfig | None = None # NEW
    features: dict[str, FeatureDecl] | None = None
    sources: dict[str, SourceDecl] | None = None
    experiments: ExperimentDef | None = None
    guardrails: GuardrailDef | None = None
    server: ServerDef | None = None
```

**Changes to DataConfig:**

```python
class DataConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    features_dir: str
    gender: str = "M"
    predictions_dir: str | None = None    # NEW
    survival_dir: str | None = None       # NEW
    outputs_dir: str | None = None        # NEW
```

**Tests:** Update `tests/test_schema.py`:
- Test ModelDef with `prediction_type="margin"`, `train_seasons="last_5"`, `cdf_scale=5.5`
- Test EnsembleDef with `pre_calibration={"v2_mlp_margin": "spline"}`, `calibration="spline"`, `spline_prob_max=0.985`, `meta_features=["diff_seed_num"]`
- Test FeaturesConfig defaults
- Test `xgboost_regression`, `gnn`, `survival` pass ModelDef type validation
- Test that old config format (without new fields) still validates (all new fields have defaults)

**Acceptance criteria:**
- mm's `config/pipeline.yaml`, `config/models/production.yaml`, and `config/ensemble.yaml` validate through ProjectConfig without error when loaded as dicts
- All new fields have backward-compatible defaults
- Existing tests still pass

### Task 15.2: Update validator.py for new config sections

**File:** `src/harnessml/runner/validator.py`

**Changes:**
1. Load `features` section from pipeline.yaml (nested under `features:` key) → maps to `feature_config` in ProjectConfig
2. Handle the fact that mm's pipeline.yaml has `data`, `features`, and `backtest` all in the same file — the validator should extract each as a top-level key

Currently `validate_project()` loads pipeline.yaml and treats it as the `data` section. Fix this: pipeline.yaml is a multi-section file. Parse it, extract `data`, `backtest`, `features` keys, and merge them into the right ProjectConfig fields. Unknown top-level keys (like `bracket`) are silently ignored — they're domain-specific and validated by project-level code, not the framework.

**Tests:** Add test that loads a pipeline.yaml structured like mm's (with data, backtest, features sections) and validates it.

---

## Phase 16: Stacked Ensemble & Calibration Pipeline

This is the critical phase. The mm pipeline's core value is in its ensemble post-processing chain.

### Task 16.1: Implement calibration utilities

**File:** Create `src/harnessml/runner/calibration.py`

Implement the calibration infrastructure that mm has in `src/mm/models/calibration.py`:

```python
class SplineCalibrator:
    """Spline-based probability calibration."""
    def __init__(self, prob_max: float = 0.985, n_bins: int = 20): ...
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None: ...
    def transform(self, y_prob: np.ndarray) -> np.ndarray: ...
    @property
    def is_fitted(self) -> bool: ...

class IsotonicCalibrator:
    """Isotonic regression calibration."""
    ...

class PlattCalibrator:
    """Platt scaling (logistic regression on logits)."""
    ...

def build_calibrator(method: str, ensemble_config: dict) -> SplineCalibrator | IsotonicCalibrator | PlattCalibrator | None:
    """Factory: create calibrator from config string."""

def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to probabilities."""
```

Implementation details (from mm):
- SplineCalibrator uses `scipy.interpolate.UnivariateSpline` fitted on binned probabilities
- IsotonicCalibrator wraps `sklearn.isotonic.IsotonicRegression`
- PlattCalibrator uses logistic regression on log-odds
- `temperature_scale` converts to logits, divides by T, converts back

**Tests:** In `tests/test_calibration.py`:
- Test SplineCalibrator fit/transform round-trip preserves monotonicity
- Test IsotonicCalibrator basic fit/transform
- Test PlattCalibrator basic fit/transform
- Test `build_calibrator()` factory for each method
- Test `temperature_scale()` with T=1.0 (identity), T=2.0 (softens), T=0.5 (sharpens)
- Test `is_fitted` property before and after fit

### Task 16.2: Implement stacked meta-learner

**File:** Create `src/harnessml/runner/meta_learner.py`

This is the core of the mm ensemble. Implements:

```python
class StackedEnsemble:
    """Logistic regression meta-learner over base model predictions."""

    def __init__(self, model_names: list[str]):
        self.model_names = model_names
        self._model = None  # sklearn LogisticRegression

    def fit(
        self,
        model_preds: dict[str, np.ndarray],
        seed_diffs: np.ndarray,
        y_true: np.ndarray,
        C: float = 1.0,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Fit meta-learner on base model predictions.

        Feature matrix: [model_pred_1, ..., model_pred_N, seed_diff, extra_1, ...]
        """
        ...

    def predict(
        self,
        model_preds: dict[str, np.ndarray],
        seed_diffs: np.ndarray,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Predict ensemble probabilities."""
        ...

    def get_coefficients(self) -> dict[str, float]:
        """Return model_name → coefficient mapping for inspection."""
        ...

    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...


def train_meta_learner_loso(
    y_true: np.ndarray,
    model_preds: dict[str, np.ndarray],
    seed_diffs: np.ndarray,
    season_labels: np.ndarray,
    model_names: list[str],
    ensemble_config: dict,
    extra_features: dict[str, np.ndarray] | None = None,
) -> tuple[StackedEnsemble, Any, dict]:
    """Train stacked meta-learner with LOSO + nested calibrator CV.

    This is the mm pipeline's secret sauce. The algorithm:

    1. For each training season (nested CV):
       a. Hold out that season
       b. Fit per-model pre-calibration on the remaining seasons
       c. Train meta-learner on pre-calibrated predictions
       d. Predict on held-out season → OOF meta-learner predictions
    2. Fit post-calibrator on the nested OOF meta-learner predictions
    3. Fit final pre-calibrators on ALL data
    4. Fit final meta-learner on ALL pre-calibrated data

    Returns: (meta_learner, post_calibrator, pre_calibrators_dict)
    """
```

Key implementation details from mm's `src/mm/models/meta_learner.py`:
- Uses `sklearn.linear_model.LogisticRegression(C=C, max_iter=1000, solver='lbfgs')`
- Feature matrix is `[prob_model1, prob_model2, ..., seed_diff, extra1, extra2, ...]`
- Pre-calibration fitted PER nested fold to avoid leakage (this was a critical fix in mm)
- Post-calibrator fitted on OOF predictions from the nested loop
- Final meta-learner fitted on all data after pre-calibrating all data with global pre-calibrators
- Minimum 10 samples per fold, minimum 20 samples for calibrator fitting

**Tests:** In `tests/test_meta_learner.py`:
- Test StackedEnsemble fit/predict with synthetic data (3 models, 100 samples)
- Test coefficients extraction matches expected shape
- Test `train_meta_learner_loso` with 3 seasons of synthetic data
- Test pre-calibration is applied per-fold (not global)
- Test save/load round-trip
- Test with `extra_features` parameter

### Task 16.3: Implement ensemble post-processing chain

**File:** Create `src/harnessml/runner/postprocessing.py`

This is the 9-step pipeline from mm's `src/mm/ensemble/postprocessing.py`:

```python
def apply_ensemble_postprocessing(
    preds: pd.DataFrame,
    meta_learner: StackedEnsemble,
    calibrator,
    ensemble_config: dict,
    team_features: pd.DataFrame | None = None,
    pre_calibrators: dict | None = None,
) -> pd.DataFrame:
    """Apply full ensemble post-processing pipeline.

    Steps:
    1. Extract base model probability columns (prob_*)
    2. Filter out excluded models
    3. Apply pre-calibration to specified models
    4. Run meta-learner prediction
    5. Apply post-calibration
    6. Temperature scaling
    7. Probability clipping
    8. Availability adjustment (if team_features provided)
    9. Seed-proximity compression (if configured)

    Adds 'prob_ensemble' column to preds DataFrame.
    """
```

Also implement the adjustment helpers:

```python
def apply_availability_adjustment(
    probs: np.ndarray,
    preds: pd.DataFrame,
    team_features: pd.DataFrame,
    strength: float,
) -> np.ndarray:
    """Adjust probabilities based on team availability metrics."""

def apply_seed_compression(
    probs: np.ndarray,
    preds: pd.DataFrame,
    compression: float,
    threshold: int,
) -> np.ndarray:
    """Compress probabilities toward 0.5 for close seed matchups."""
```

**Tests:** In `tests/test_postprocessing.py`:
- Test full pipeline with mock meta-learner and calibrator
- Test each step independently (pre-cal, meta, post-cal, temp, clip, avail, compression)
- Test exclude_models filtering
- Test with ensemble_config having all default values (no-op pipeline)
- Test prob_ensemble column is added to DataFrame

---

## Phase 17: Pipeline Runner Overhaul

### Task 17.1: Rewrite PipelineRunner for real ensemble

**File:** `src/harnessml/runner/pipeline.py`

The current PipelineRunner does simple model training and per-fold averaging. Rewrite it to match mm's actual pipeline architecture.

**Key changes:**

1. **Training season filtering**: Use `train_seasons` from ModelDef to filter data per model.

```python
def _filter_train_seasons(
    df: pd.DataFrame,
    train_seasons: str,
    target_season: int | None = None,
) -> pd.DataFrame:
    """Filter training data. 'all' keeps everything, 'last_N' keeps N most recent."""
    if target_season is not None:
        df = df[df["season"] < target_season]
    if train_seasons == "all":
        return df
    elif train_seasons.startswith("last_"):
        n = int(train_seasons.split("_")[1])
        max_season = df["season"].max()
        return df[df["season"] > max_season - n]
    return df
```

2. **Regressor model handling**: Models with `mode="regressor"` or `prediction_type="margin"` predict margins, then convert to probabilities via CDF. The CDF scale is either explicit in config (`cdf_scale`) or auto-fitted from training data.

```python
def _margin_to_prob(margins: np.ndarray, cdf_scale: float) -> np.ndarray:
    """Convert predicted margins to win probabilities via normal CDF."""
    from scipy.stats import norm
    return norm.cdf(margins / cdf_scale)
```

3. **Multi-seed averaging**: Models with `n_seeds > 1` are trained N times with different random seeds, predictions averaged.

4. **Two-pass backtesting**: Match mm's architecture exactly:
   - Pass 1: Per season, train all active models on everything except that season, predict that season's matchups, save per-season prediction DataFrames with `prob_{model_name}` columns
   - Pass 2: Train LOSO meta-learner (one per held-out season), apply ensemble post-processing

5. **Exclude models**: Filter out models listed in `ensemble.exclude_models` from meta-learner training.

6. **Feature-per-model training**: Each model trains on its own feature subset (not a shared X matrix). Build per-model X from the model's `features` list.

7. **Column name conventions**: Support both `result`/`season` (harnessml convention) and `TeamAWon`/`TeamAMargin`/`Season` (mm convention) via auto-detection.

**Updated PipelineRunner API:**

```python
class PipelineRunner:
    def load(self) -> None: ...
    def train(self, run_id: str | None = None) -> dict: ...
    def predict(self, season: int, run_id: str | None = None) -> pd.DataFrame: ...  # NEW
    def backtest(self) -> dict: ...  # REWRITTEN
    def run_full(self) -> dict: ...
```

**Rewritten backtest() method pseudo-code:**

```python
def backtest(self) -> dict:
    seasons = self.config.backtest.seasons
    ensemble_config = self.config.ensemble.model_dump()

    # Pass 1: Generate per-season OOF predictions
    season_data = {}
    for holdout_season in seasons:
        fold_preds = {}
        for model_name, model_def in self._active_models():
            train_df = self._filter_train_seasons(
                self._df, model_def.train_seasons, target_season=holdout_season
            )
            test_df = self._df[self._df["season"] == holdout_season]

            model = self._train_single_model(model_name, model_def, train_df)
            probs = self._predict_model(model, model_def, test_df)
            fold_preds[f"prob_{model_name}"] = probs

        season_data[holdout_season] = self._build_prediction_df(
            test_df, fold_preds
        )

    # Pass 2: Train LOSO meta-learners and apply post-processing
    if ensemble_config["method"] == "stacked":
        for holdout_season in seasons:
            meta, cal, pre_cals = self._train_meta_for_season(
                season_data, ensemble_config, holdout_season
            )
            season_data[holdout_season] = apply_ensemble_postprocessing(
                season_data[holdout_season], meta, cal,
                ensemble_config, pre_calibrators=pre_cals,
            )

    # Compute metrics
    return self._compute_backtest_metrics(season_data)
```

**Tests:** Update `tests/test_pipeline.py`:
- Test backtest with stacked ensemble (mock 3 models, 3 seasons)
- Test train_seasons filtering ("all" vs "last_3")
- Test regressor model with CDF conversion
- Test exclude_models
- Test multi-seed averaging
- Test that prob_ensemble column exists in backtest output

### Task 17.2: Implement model training internals

**File:** `src/harnessml/runner/training.py` (new)

Extract the per-model training logic into a dedicated module. This handles what mm's `src/mm/models/train.py` does:

```python
def train_single_model(
    model_name: str,
    model_def: ModelDef,
    train_df: pd.DataFrame,
    registry: ModelRegistry,
    target_season: int | None = None,
    augment_symmetry: bool = False,
) -> tuple[Any, list[str], dict]:
    """Train a single model from its ModelDef config.

    Handles:
    - Feature column extraction from model_def.features
    - Training season filtering
    - Validation set split (most recent season for early stopping)
    - Data augmentation via matchup symmetry
    - Regressor target (margin) vs classifier target (win/loss)
    - Multi-seed training and prediction averaging
    - CDF scale fitting for regressors

    Returns: (fitted_model, feature_columns, metrics_dict)
    """

def predict_single_model(
    model: Any,
    model_def: ModelDef,
    test_df: pd.DataFrame,
    feature_medians: dict | None = None,
) -> np.ndarray:
    """Generate predictions from a fitted model.

    Handles:
    - Regressor → CDF probability conversion
    - NaN imputation with feature medians
    - Multi-seed prediction averaging

    Returns: probability array
    """
```

Also implement early stopping validation set logic:
- When `early_stopping_rounds` is in model params and `target_season` is set, split off the most recent training season as validation
- Pass `eval_set=(X_val, y_val)` to model.fit()

**Tests:** In `tests/test_training.py`:
- Test `train_single_model` with a classifier
- Test `train_single_model` with a regressor (verify CDF conversion)
- Test multi-seed training (n_seeds=3)
- Test validation set splitting
- Test matchup symmetry augmentation (doubled data, flipped labels)

### Task 17.3: Implement fingerprint caching

**File:** Create `src/harnessml/runner/fingerprint.py`

```python
def compute_fingerprint(model_config: dict, data_mtime: float | None = None) -> str:
    """SHA-256 hash of model config + data modification time."""

def is_cached(models_dir: Path, model_name: str, fingerprint: str) -> bool:
    """Check if a model's fingerprint file matches."""

def save_fingerprint(models_dir: Path, model_name: str, fingerprint: str) -> None:
    """Write fingerprint file for a trained model."""

def compute_meta_fingerprint(
    predictions_dir: Path,
    ensemble_config: dict,
) -> str:
    """Hash backtest predictions + ensemble config for meta-learner cache."""

def save_meta_cache(cache_dir: Path, meta, calibrator, pre_calibrators, fingerprint: str) -> None:
    """Save meta-learner artifacts + fingerprint."""

def load_meta_cache(cache_dir: Path, fingerprint: str) -> tuple | None:
    """Load cached meta-learner if fingerprint matches. Returns None on miss."""
```

Wire into PipelineRunner:
- Before training each model, check `is_cached()` — skip if config unchanged
- After training, `save_fingerprint()`
- Before training meta-learner, check `load_meta_cache()` — skip if predictions unchanged

**Tests:** In `tests/test_fingerprint.py`:
- Test compute_fingerprint determinism (same config → same hash)
- Test compute_fingerprint changes when config changes
- Test is_cached / save_fingerprint round-trip
- Test compute_meta_fingerprint
- Test save_meta_cache / load_meta_cache round-trip

---

## Phase 18: Prediction Mode

### Task 18.1: Implement prediction mode and matchup generation

**File:** Extend `src/harnessml/runner/pipeline.py` with `predict()` method

The mm pipeline has a dedicated `predict.py` that:
1. Loads production-trained models
2. Generates all pairwise matchup probabilities for a target season
3. Trains a production meta-learner on all backtest prediction data
4. Applies ensemble post-processing
5. Saves prediction parquets

harnessml-runner needs:

```python
def predict(self, season: int, run_id: str | None = None) -> pd.DataFrame:
    """Generate predictions for a target season.

    1. Load trained models from models_dir
    2. Build all pairwise matchups from team features + seeds
    3. Predict each matchup with each model
    4. Train production meta-learner on backtest predictions
    5. Apply ensemble post-processing
    6. Return predictions DataFrame with prob_* and prob_ensemble columns
    """
```

This requires a matchup generator:

**File:** Create `src/harnessml/runner/matchups.py`

```python
def generate_pairwise_matchups(
    team_features: pd.DataFrame,
    seeds: pd.DataFrame,
    season: int,
    feature_medians: dict | None = None,
) -> pd.DataFrame:
    """Generate all pairwise matchup features for seeded teams.

    For each pair of seeded teams (A < B) in a season:
    - Look up team-season features for both teams
    - Compute diff_* features (TeamA value - TeamB value)
    - Impute NaN with feature medians
    - Include TeamA, TeamB, Season, diff_seed_num columns

    Returns DataFrame with N*(N-1)/2 rows.
    """

def predict_all_matchups(
    matchups: pd.DataFrame,
    models: dict,
    model_defs: dict[str, ModelDef],
    feature_medians: dict | None = None,
) -> pd.DataFrame:
    """Run all models on all matchups, return DataFrame with prob_* columns.

    Handles regressor models (CDF conversion) and multi-seed models (averaging).
    """
```

**Tests:** In `tests/test_matchups.py`:
- Test matchup generation produces N*(N-1)/2 rows for N teams
- Test diff features are computed correctly (teamA_feat - teamB_feat)
- Test NaN imputation with medians
- Test predict_all_matchups with a regressor model (CDF conversion applied)

---

## Phase 19: CLI & Operational Features

### Task 19.1: Add predict and experiment run CLI commands

**File:** `src/harnessml/runner/cli.py`

**New commands:**

```python
@run.command()
@click.option("--season", required=True, type=int)
@click.option("--run-id", default=None)
@click.pass_context
def predict(ctx, season, run_id):
    """Generate predictions for a target season."""
    runner = PipelineRunner(...)
    runner.load()
    preds = runner.predict(season, run_id=run_id)
    click.echo(f"Predicted {len(preds)} matchups for season {season}")

@experiment.command("run")
@click.argument("experiment_id")
@click.pass_context
def experiment_run(ctx, experiment_id):
    """Run an experiment: load overlay, train, backtest, compare to baseline."""
    # 1. Load experiment overlay from experiments/{id}/overlay.yaml
    # 2. Create PipelineRunner with overlay
    # 3. Run backtest
    # 4. Compare to baseline
    # 5. Output results
```

**Tests:** In `tests/test_cli.py`:
- Test predict command invocation (with mock runner)
- Test experiment run command creates overlay runner

### Task 19.2: Implement run management

**File:** Create `src/harnessml/runner/run_manager.py`

```python
class RunManager:
    """Manages versioned pipeline output directories.

    Directory structure:
        outputs_base/
            YYYYMMDD_HHMMSS/      # run directory
                predictions/
                diagnostics/
            current -> YYYYMMDD_HHMMSS  # symlink to promoted run
    """

    def __init__(self, outputs_base: Path): ...
    def new_run(self, run_id: str | None = None) -> Path: ...
    def promote(self, run_id: str) -> None: ...
    def list_runs(self) -> list[dict]: ...
    def get_current(self) -> Path | None: ...
```

**Tests:** In `tests/test_run_manager.py`:
- Test new_run creates directory with timestamp
- Test promote creates symlink
- Test list_runs returns sorted entries
- Test get_current returns promoted path

### Task 19.3: Implement diagnostics

**File:** Create `src/harnessml/runner/diagnostics.py`

```python
def compute_pooled_metrics(
    season_predictions: list[pd.DataFrame],
) -> dict[str, dict]:
    """Compute pooled metrics across all seasons.

    Returns per-model metrics: accuracy, brier_score, ece, log_loss.
    Pooled = computed on concatenated predictions (more stable than averaging).
    """

def evaluate_season_predictions(
    preds: pd.DataFrame,
    actuals: dict[str, int],
    season: int,
) -> list[dict]:
    """Compute per-model metrics for a single season.

    Returns list of dicts with model, season, accuracy, brier_score, ece, log_loss.
    """

def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float: ...
def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float: ...
def compute_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple: ...
```

**Tests:** In `tests/test_diagnostics.py`:
- Test compute_brier_score against sklearn
- Test compute_ece with known calibration
- Test compute_pooled_metrics with synthetic multi-season data
- Test evaluate_season_predictions returns expected structure

### Task 19.4: Wire serve command with guardrails

**File:** `src/harnessml/runner/cli.py` (update serve command) and `src/harnessml/runner/server_gen.py`

Turn the serve stub into a working command:

```python
@main.command()
@click.pass_context
def serve(ctx):
    """Start the MCP server."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    if result.config.server is None:
        click.echo("No server configuration found in server.yaml")
        raise SystemExit(1)

    server = generate_server(result.config.server, config_dir)
    mcp = server.to_fastmcp()
    mcp.run()
```

Add guardrail enforcement to generated server tools:

In `server_gen.py`, update `_make_execution_tool()`:
- Before executing a command, check guardrails defined on the tool
- Load guardrail config from project config
- Apply relevant guardrails (sanity check, naming validation, etc.)
- Include guardrail results in tool output

**Tests:**
- Test serve command starts server (mock FastMCP)
- Test execution tools with guardrails reject invalid inputs

---

## Phase 20: Integration Test & Config Validation

### Task 20.1: Write mm-config integration test

**File:** `tests/test_mm_integration.py`

This is the acid test. Write a test that:
1. Takes the actual mm config files (`config/pipeline.yaml`, `config/models/production.yaml`, `config/ensemble.yaml`)
2. Validates them through harnessml-runner's `validate_project()`
3. Asserts every model in production.yaml passes ModelDef validation
4. Asserts ensemble config passes EnsembleDef validation

Use the actual mm config as fixture data (copy relevant portions into test fixtures, don't depend on mm repo path).

Create test fixture files:
- `tests/fixtures/mm_pipeline.yaml` — copy of mm's `config/pipeline.yaml`
- `tests/fixtures/mm_models.yaml` — copy of mm's `config/models/production.yaml` (just a few representative models: logreg_seed, xgb_core, xgb_spread_broad, v2_mlp_margin, v2_tabnet_clf)
- `tests/fixtures/mm_ensemble.yaml` — copy of mm's `config/ensemble.yaml`

```python
def test_mm_config_validates():
    """The mm pipeline's actual config must validate through harnessml-runner."""
    config_dir = Path(__file__).parent / "fixtures"
    result = validate_project(config_dir)
    assert result.valid, f"Validation failed:\n{result.format()}"

    config = result.config
    assert len(config.models) >= 5
    assert config.ensemble.method == "stacked"
    assert config.ensemble.meta_learner.get("C") == 2.5

    # Check regression models
    spread = config.models["xgb_spread_broad"]
    assert spread.type == "xgboost_regression"
    assert spread.mode == "regressor" or spread.prediction_type == "margin"

    # Check MLP with pre-calibration
    mlp = config.models["v2_mlp_margin"]
    assert mlp.prediction_type == "margin"
    assert mlp.n_seeds == 5

    # Check ensemble pre-calibration
    assert "v2_mlp_margin" in config.ensemble.pre_calibration
    assert config.ensemble.calibration == "spline"

def test_mm_backtest_pipeline_smoke():
    """Smoke test: PipelineRunner can load mm-style config and run backtest
    with synthetic data matching mm's column conventions."""
    # Create synthetic matchup_features.parquet with TeamAWon, Season, diff_seed_num, etc.
    # Run PipelineRunner.backtest()
    # Assert prob_ensemble column exists
    # Assert metrics include brier, accuracy
```

**Acceptance criteria:**
- mm's actual config validates without error
- Smoke test with synthetic data produces ensemble predictions with full post-processing chain
- All 14 active mm models' configs pass ModelDef validation
- All 3 regressor models recognized (xgboost_regression type)
- Pre-calibration, post-calibration, meta-learner all exercised

### Task 20.2: Update package exports and documentation

**Files:**
- `src/harnessml/runner/__init__.py` — add new exports
- Update any docstrings

New exports to add:
- `SplineCalibrator`, `IsotonicCalibrator`, `PlattCalibrator`, `build_calibrator`, `temperature_scale` from calibration
- `StackedEnsemble`, `train_meta_learner_loso` from meta_learner
- `apply_ensemble_postprocessing` from postprocessing
- `RunManager` from run_manager
- `compute_pooled_metrics`, `compute_brier_score`, `compute_ece` from diagnostics
- `compute_fingerprint`, `is_cached`, `save_fingerprint` from fingerprint
- `FeaturesConfig` from schema

**Tests:** Verify all new symbols are importable from `harnessml.runner`.

---

## Summary: What This Enables

After implementation, a user can recreate the mm pipeline's training and evaluation with:

```yaml
# config/pipeline.yaml
data:
  gender: M
  raw_dir: data/raw
  processed_dir: data/processed
  features_dir: data/features
features:
  first_season: 2003
  momentum_window: 10
backtest:
  cv_strategy: leave_one_season_out
  seasons: [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
```

```yaml
# config/models/production.yaml
xgb_core:
  type: xgboost
  features: [diff_seed_num, diff_bt_barthag, ...]
  train_seasons: all
  params: {max_depth: 4, learning_rate: 0.01, ...}

xgb_spread_broad:
  type: xgboost_regression
  features: [diff_seed_num, diff_scoring_margin, ...]
  train_seasons: all
  params: {max_depth: 1, ...}

v2_mlp_margin:
  type: mlp
  prediction_type: margin
  features: [...]
  train_seasons: all
  n_seeds: 5
  params: {hidden_layers: [128, 64], dropout: 0.4, ...}
```

```yaml
# config/ensemble.yaml
method: stacked
meta_learner:
  C: 2.5
pre_calibration:
  v2_mlp_margin: spline
calibration: spline
spline_prob_max: 0.985
spline_n_bins: 20
temperature: 1.0
clip_floor: 0.0
availability_adjustment: 0.1
exclude_models: [cat_broad, lgbm_broad, ...]
```

And run:
```bash
harnessml run backtest
harnessml run predict --season 2025
harnessml experiment run exp-066-test
```

Each of these commands exercises the full pipeline: training season filtering, per-model feature subsets, regressor CDF conversion, multi-seed averaging, fingerprint caching, stacked meta-learner with nested LOSO pre-calibration, 9-step post-processing, and diagnostics.
