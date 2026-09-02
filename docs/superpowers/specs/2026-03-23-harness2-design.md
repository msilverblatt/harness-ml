# Harness 2 — Design Specification

**Date:** 2026-03-23
**Status:** Draft
**Author:** msilverblatt + Claude

---

## 1. Vision & Identity

Harness is an agent-first ML platform that lets AI agents do real data science work — building, evaluating, and iterating on tabular ML pipelines through structured tool calls rather than writing code. The human's role is collaborative: guiding strategy, discussing hypotheses, and steering the agent toward productive research directions.

Harness 2 is a ground-up redesign of [harness-ml](https://github.com/msilverblatt/harness-ml) (v1), preserving its core strengths — MCP-driven workflow, experiment discipline, fingerprint caching, nested calibration, model diversity — while dramatically tightening the architecture, reducing tool surface area, and eliminating the operational burden placed on the agent.

### Core Principles

1. **The agent thinks in hypotheses, not operations.** The primary interface is `experiment.propose`. One tool call: state a hypothesis, provide typed parameters, get back metrics. The system handles all orchestration — data pipeline, feature resolution, training, evaluation, artifact production.

2. **Nothing is deleted, only versioned.** Every experiment creates a version in a tree. Rejected experiments persist as research artifacts. The agent can branch from any version, backtrack without pain, and compare across branches.

3. **Separation of concerns through composition.** Three independent libraries (research-loop, harness-data, harness-ml) composed by a thin application layer. Each is independently useful and testable.

4. **Task types are encapsulated.** Adding a new task type means adding one directory. No changes to the runner, no changes to existing task types.

5. **Model diversity is a feature, not a luxury.** 13+ model architectures ship by default. The wrapper contract is so clean that adding a new model is a single file.

---

## 2. Architecture Overview

### Three Libraries + One Application

```
┌─────────────────────────────────────────────┐
│  HARNESS (the application)                  │
│                                             │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │   CLI   │  │ MCP      │  │ Studio    │  │
│  │         │  │ Server   │  │ Dashboard │  │
│  └────┬────┘  └────┬─────┘  └─────┬─────┘  │
│       │            │              │         │
│       └────────┬───┘──────────────┘         │
│                │                            │
│         ┌──────▼──────┐                     │
│         │  Workspace  │                     │
│         │  Manager    │                     │
│         └──────┬──────┘                     │
│                │                            │
│   ┌────────────┼────────────┐               │
│   │            │            │               │
│   ▼            ▼            ▼               │
│ research-  harness-     harness-            │
│ loop       data         ml                  │
│ (npm)      (Python)     (Python)            │
└─────────────────────────────────────────────┘
```

| Component | Identity | Language | Purpose |
|-----------|----------|----------|---------|
| **Harness** | The application | TS + Python | CLI, MCP server, Studio dashboard, workspace management |
| **research-loop** | Experiment discipline library | TypeScript | Generic scientific method enforcement via MCP workflow |
| **harness-data** | Data engineering library | Python | Source ingestion, declarative transforms, profiling |
| **harness-ml** | Tabular ML library | Python | Features, models, training, calibration, ensemble, diagnostics |

**Key boundaries:**
- **research-loop** knows the scientific method. It knows nothing about ML.
- **harness-ml** knows tabular ML. It knows nothing about experiment discipline.
- **harness-data** knows data transforms. It knows nothing about models or experiments.
- **Harness (the app)** composes them via a thin `server.ts` that wires research-loop hooks to harness-ml, registers data tools from harness-data, and serves the Studio dashboard.

### How They Compose

The MCP server (TypeScript) uses research-loop's `workflow()` primitive. Harness implements the domain hooks:

| research-loop hook | Harness implementation |
|-------------------|----------------------|
| `onRun(args, ctx)` | Validate typed experiment → generate config diff from parent version → resolve features via harness-data transforms → run full training pipeline via harness-ml → create new version → return metrics |
| `onEvaluate(result, ctx)` | Compare run metrics against parent version baseline |
| `onCompare(baseline, result)` | Format metric deltas for agent readability |
| `onLog(exp, ctx)` | Write experiment metadata to `versions/vNNN/meta.yaml` |
| `getBaseline()` | Return parent version's metrics |
| `detectChange(ctx)` | Verify typed diff is non-empty |

### Distribution & Packaging (TS + Python Hybrid)

The MCP server and research-loop integration are TypeScript. The ML engine, data library, and Studio backend are Python. These are bridged at the process level:

- The `harness serve` CLI command starts the TypeScript MCP server (via `npx`/bundled Node binary)
- The MCP server spawns harness-ml and harness-data as Python subprocesses via a JSON-RPC bridge
- The Python packages are installed via `pip`/`uv` as normal Python packages
- The TypeScript MCP server is bundled as a pre-built Node package, vendored into the Python distribution

**Installation:**

```bash
# Primary install (includes bundled TS server + Python packages)
pip install harness
# or
uv tool install harness

# Zero-install trial
uvx harness init my-project

# With all model backends pre-installed
pip install harness[all]

# GPU support
pip install harness[gpu]
```

**Runtime dependencies:** Node.js 20+ (for protomcp/research-loop MCP server). The `harness doctor` command checks for this and provides install guidance.

**Internal package boundaries (invisible to user):**

```
harness/
├── packages/
│   ├── harness-ml/           # Pure Python ML library
│   ├── harness-data/         # Pure Python data engineering library
│   └── harness-studio/       # FastAPI + React dashboard
├── server/
│   ├── server.ts             # MCP server entry point (protomcp + research-loop)
│   ├── hooks.ts              # Domain hooks wiring research-loop to Python bridge
│   ├── tools/                # MCP tool definitions
│   ├── bridge.ts             # JSON-RPC bridge to Python subprocesses
│   └── package.json          # TS dependencies (bundled at build time)
├── cli.py                    # CLI entry point
└── pyproject.toml            # Single metapackage install
```

---

## 3. Workspace Design

The workspace is the on-disk contract between all layers. Human-inspectable, agent-readable, complete source of truth.

### Directory Structure

```
my-project/
├── harness.yaml                   # Workspace marker file (workspace name, created timestamp)
├── config/                        # Current version's config (convenience — mirrors current pointer)
│   ├── project.yaml               # Task type, target, CV strategy, metrics
│   ├── models.yaml                # Model definitions + hyperparams
│   ├── features.yaml              # Feature definitions + formulas
│   ├── ensemble.yaml              # Ensemble method, calibration, post-processing
│   └── evals.yaml                 # Eval criteria definitions (user-defined, from preset)
├── data/
│   ├── sources.yaml               # Data source declarations (harness-data lifecycle)
│   ├── transforms.yaml            # Declarative transform pipeline definition
│   ├── raw/                       # Untouched originals
│   └── clean/
│       ├── dataset.parquet        # Analysis-ready output from harness-data
│       └── schema.json            # Column names, types, stats
├── versions/
│   ├── v001/
│   │   ├── meta.yaml              # Parent, hypothesis, conclusion, verdict, timestamp, data hash
│   │   ├── diff.yaml              # Typed diff from parent (null for root — full config as diff)
│   │   ├── config/                # Full resolved config snapshot
│   │   └── run/                   # Run results for this version (1:1 mapping)
│   │       ├── metrics.json       # Per-fold + aggregate metrics
│   │       ├── calibration.json   # Calibration curve data
│   │       ├── predictions.parquet # Full prediction table
│   │       └── diagnostics.json   # ECE, Brier decomposition, etc.
│   ├── v002/
│   │   ├── meta.yaml              # Parent: v001
│   │   ├── diff.yaml
│   │   ├── config/
│   │   └── run/
│   └── ...
├── artifacts/                     # Shared artifact cache (fingerprint-based)
│   ├── models/                    # Serialized trained models
│   ├── predictions/               # Cached predictions per (model, fold, fingerprint)
│   └── fingerprints.json          # SHA256 cache index
├── current                        # Pointer to active version (e.g., "v007")
└── .harness/
    ├── studio.db                  # Derived SQLite cache for dashboard queries
    ├── events.db                  # MCP event log (agent activity — observability only)
    └── state.json                 # Workspace metadata
```

**Workspace marker:** `harness.yaml` is the file discovered by walking up the directory tree. It contains the workspace name and creation timestamp — nothing else. All config lives in `config/`.

**Version-to-run mapping:** Each version has exactly one run, stored inside `versions/vNNN/run/`. There is no separate `runs/` directory. This eliminates the ambiguity of mapping versions to runs — they are structurally co-located. Re-running a version overwrites its `run/` directory.

**Data files co-located:** `sources.yaml` and `transforms.yaml` both live inside `data/` since they are both harness-data concerns.

### Split Config Files

Config is split by concern to prevent monolithic blast radius:

| File | Contents | Change frequency |
|------|----------|-----------------|
| `project.yaml` | Task type, target column, CV strategy, metrics | Rare (set once) |
| `models.yaml` | Model definitions, hyperparameters, feature lists | Frequent |
| `features.yaml` | Feature definitions, formulas, types | Frequent |
| `ensemble.yaml` | Ensemble method, meta-learner, calibration, post-processing pipeline | Moderate |
| `evals.yaml` | Eval criteria definitions (checks, comparisons, judgment prompts) | Rare (set up, then tuned) |

Each file has its own schema and validation. A schema change to `models.yaml` does not affect `features.yaml`.

### Version Tree Model

Every experiment creates a version. Versions are never deleted. The tree structure emerges from parent pointers.

**Key behaviors:**
- `config/` is the working directory — always reflects the `current` version
- `versions.switch(v)` overwrites `config/` with that version's snapshot (like `git checkout`)
- Artifacts are shared across versions via fingerprint cache (no duplication)
- Branching from any version: `experiment.propose(parent="v003")` — defaults to `current` if omitted

**Branching from non-current versions:** When `experiment.propose(parent="v003")` is called but `current` points to `v007`, the system resolves the parent config **in memory** from `versions/v003/config/`. It never modifies `config/` on disk during the experiment — the resolved config is passed directly to the training pipeline. The new version's `config/` snapshot is written from the in-memory resolved config. After the experiment completes, `config/` still reflects `v007` (unchanged). The agent can later `versions.switch` to the new version if desired.

**Baseline experiment (v001):** The `baseline` experiment type has no parent. Its `diff.yaml` contains the full config (diff from null). `getBaseline()` returns empty/null metrics for baseline experiments — the first run establishes the baseline, it doesn't compare against one.

**Data hash and staleness:** `data_hash` records the SHA256 of `data/clean/dataset.parquet` at run time. This is an auditability field — it lets you know whether a version's metrics were computed against the current data or stale data. The `analyze.diagnostics` tool flags versions whose `data_hash` doesn't match the current dataset. The agent can re-run any version against current data by proposing a new experiment with the same config (the version tree naturally tracks the re-evaluation as a new version).

**Version metadata (`meta.yaml`):**
```yaml
id: v007
parent: v003
experiment_type: feature
hypothesis: "Adding momentum feature improves calibration"
conclusion: "Improved ECE by 0.02 but Brier unchanged"
verdict: mixed            # improved | degraded | inconclusive | mixed
timestamp: 2026-03-23T14:30:00Z
data_hash: sha256:abc123  # Hash of clean dataset at run time
metrics:
  brier: 0.198
  ece: 0.031
  log_loss: 0.412
```

### Workspace-Layer Boundaries

| Layer | Reads | Writes |
|-------|-------|--------|
| harness-data | `data/sources.yaml`, `data/raw/` | `data/clean/`, `data/transforms.yaml` |
| harness-ml | `data/clean/`, `config/` | `config/`, `artifacts/` |
| research-loop | (in-memory state) | — |
| Harness app | everything | `versions/` (including `run/` subdirectories), `current`, `.harness/` |
| Studio | everything | `.harness/studio.db` (derived cache) |

No overlapping writes between harness-data and harness-ml.

---

## 4. The Agent Interface — MCP Tool Surface

### Design Philosophy

The agent thinks in hypotheses, not operations. `experiment.propose` is THE tool — one call handles validation, config diffing, data pipeline re-runs, feature resolution, training, evaluation, versioning, and result reporting. The agent never separately "adds a feature" then "configures the ensemble" then "runs a backtest."

### Tool Inventory

**Project Setup (one-time):**

| Tool | Purpose |
|------|---------|
| `project.init` | Task type, target column, CV strategy, data source. One call bootstraps everything. |

**Data Tools (separate phase, used early then occasionally):**

| Tool | Purpose |
|------|---------|
| `data.add_source` | Declare a data source (one or many) |
| `data.transform` | Add/modify transform steps in the pipeline |
| `data.run` | Execute transform pipeline → produce clean dataset |
| `data.profile` | Column stats, distributions, quality checks |
| `data.inspect` | Preview rows, examine schema |

**Experiment Tools (the main loop — via research-loop):**

| Tool | Purpose |
|------|---------|
| `experiment.propose` | Hypothesis + experiment type + typed params + optional parent version. System handles everything: validates → diffs → re-runs data pipeline if needed → resolves features → trains → evaluates → creates version → returns metrics + diagnostics + parent comparison. |
| `experiment.conclude` | Record conclusion (string) + verdict (improved/degraded/inconclusive/mixed). Ends the experiment. Version already exists. |

**Analysis Tools (read-only, anytime):**

| Tool | Purpose |
|------|---------|
| `analyze.diagnostics` | Metrics, calibration curves, per-fold stats for any version |
| `analyze.explain` | Feature importance, SHAP values for any version |
| `analyze.compare` | Side-by-side comparison of any N versions |
| `analyze.discover` | Auto-suggest features, interactions, transformations worth testing |

**Version Tools (tree navigation):**

| Tool | Purpose |
|------|---------|
| `versions.list` | Tree view with metrics summary per version |
| `versions.show` | Full detail for a specific version |
| `versions.switch` | Change working config to a different version |
| `versions.ancestry` | Experiment narrative from root to a version |

**Workspace Tool:**

| Tool | Purpose |
|------|---------|
| `workspace.open` | Point the server at a different workspace (for multi-project workflows) |

**Total: 17 tools + MCP resources.**

### Typed Experiment System

Each experiment type constrains what can change and what parameters are accepted:

| Type | Params | What the system does |
|------|--------|---------------------|
| `baseline` | Initial features + initial models | Sets up first pipeline, runs, creates v001 |
| `feature` | Feature definition (name, formula, type) | Adds feature to config, adds to relevant models, runs |
| `model` | Model type, hyperparams, feature list | Adds model to config, runs |
| `hyperparameter` | Model name, param changes | Updates model params, runs |
| `ensemble` | Ensemble config changes | Updates ensemble config, runs |
| `calibration` | Calibration method changes | Updates calibration config, runs |
| `cv_strategy` | CV strategy changes | Updates CV config, runs |
| `feature_selection` | Model name, feature list changes | Updates model's feature list, runs |

Each type maps to specific config files:

| Experiment Type | Diffs Against |
|----------------|---------------|
| `feature` | `features.yaml` |
| `model` | `models.yaml` |
| `hyperparameter` | `models.yaml` |
| `ensemble` | `ensemble.yaml` |
| `calibration` | `ensemble.yaml` |
| `cv_strategy` | `project.yaml` |
| `feature_selection` | `models.yaml` |

### MCP Resources (always-available context)

| Resource | Content |
|----------|---------|
| `harness://data/schema` | Current clean dataset schema (columns, types) |
| `harness://versions/tree` | Full version tree structure |
| `harness://versions/current` | Current version config + metrics |
| `harness://models/available` | Model types, default params, task compatibility |
| `harness://tasks/supported` | Task types, available metrics |

### What Happens Inside `experiment.propose`

```
1. Validate typed params against experiment type schema
   → Immediate error if formula references nonexistent column,
     model type is incompatible with task, etc.
2. Generate config diff from parent version
3. If data sources changed or transforms modified:
   → Re-run harness-data pipeline automatically
4. Resolve all features (calls harness-data transform engine for computed features)
5. Run full training pipeline (see Section 6 for complete flow)
6. Create new version in tree (snapshot config, record metrics)
7. Return: metrics, diagnostics, comparison vs parent baseline
```

### Research-Loop Workflow

```
propose → run → conclude (terminal)
```

- `propose`: Agent states hypothesis, selects experiment type and parent version
- `run`: System executes everything (handled by `onRun` hook)
- `conclude`: Agent records what they learned (verdict + conclusion)

No promote/discard ceremony. The version exists regardless. `versions.switch` handles navigation independently — the agent can switch to any version at any time.

Research-loop extension needed: `propose` must accept a `parent` parameter. Baseline (`getBaseline`) returns parent version's metrics rather than a global baseline.

### Tool Response Schemas

**`experiment.propose` success response:**
```json
{
  "version": "v007",
  "parent": "v003",
  "metrics": {"brier": 0.198, "ece": 0.031, "log_loss": 0.412},
  "parent_metrics": {"brier": 0.201, "ece": 0.051, "log_loss": 0.418},
  "deltas": {"brier": -0.003, "ece": -0.020, "log_loss": -0.006},
  "per_fold": [{"fold": 1, "brier": 0.195}, {"fold": 2, "brier": 0.201}],
  "models_trained": 5,
  "models_cached": 8,
  "models_failed": [],
  "duration_s": 45.2
}
```

**`experiment.propose` validation failure:**
```json
{
  "error": "validation",
  "message": "Feature formula references nonexistent column 'momentum_score'",
  "available_columns": ["points", "rebounds", "assists", "seed"]
}
```

**`experiment.propose` partial failure (some models failed):**
```json
{
  "version": "v007",
  "metrics": {"brier": 0.205},
  "models_trained": 4,
  "models_failed": [{"name": "tabnet_main", "error": "CUDA out of memory"}],
  "warnings": ["1 model failed — metrics computed from 4/5 models"]
}
```

---

## 5. harness-ml — Task Type Architecture

Each task type is a self-contained module implementing a protocol. The runner never branches on task type.

### Protocol

```python
class TaskType(Protocol):
    name: str                              # "binary", "multiclass", "regression"

    def metrics(self) -> list[Metric]:
        """Available metrics for this task type."""

    def default_metrics(self) -> list[str]:
        """Default metric set for backtesting."""

    def validate_target(self, series: pd.Series) -> ValidationResult:
        """Validate that the target column is appropriate."""

    def validate_predictions(self, predictions: np.ndarray) -> ValidationResult:
        """Sanity-check model outputs."""

    def calibration_methods(self) -> list[CalibrationType]:
        """Available calibration methods for this task type."""

    def adapt(self, model: Model, params: dict) -> AdaptedModel:
        """Wrap a model with task-specific objective, prediction extraction, defaults."""

    def postprocess(self, predictions: np.ndarray, config: dict) -> np.ndarray:
        """Task-specific post-processing."""

    def format_results(self, metrics: dict, fold_results: list) -> ResultSummary:
        """Format results for agent consumption."""
```

### Directory Structure

```
harness-ml/src/harness/ml/tasks/
├── protocol.py           # TaskType protocol
├── registry.py           # TaskRegistry — discovers and loads task types
├── binary/
│   ├── __init__.py
│   ├── task.py           # Implements TaskType protocol
│   ├── metrics.py        # Brier, log_loss, AUC, ECE, precision, recall, f1, etc.
│   ├── calibration.py    # Spline, Isotonic, Platt, Beta
│   ├── adaptation.py     # Objective mappings, predict_proba extraction, defaults per family
│   ├── validation.py     # Target is 0/1, predictions in [0,1]
│   └── tests/
├── multiclass/
│   ├── ...
│   ├── adaptation.py     # Per-class probabilities, softmax, etc.
│   └── tests/
└── regression/
    ├── ...
    ├── adaptation.py     # Raw value output, conformal prediction intervals
    └── tests/
```

**Adding a new task type = adding one directory.** Implement the protocol, register it. No changes to the runner, no changes to existing task types. Later task types (ranking, survival, probabilistic) drop in without touching existing code.

### Task Adaptation Layer

Each task type's `adaptation.py` contains a mostly-declarative mapping of how each model family behaves for that task:

```python
# binary/adaptation.py
OBJECTIVES = {
    "xgboost":   {"objective": "binary:logistic", "eval_metric": "logloss"},
    "lightgbm":  {"objective": "binary", "metric": "binary_logloss"},
    "catboost":   {"loss_function": "Logloss"},
    "mlp":        {"loss": "bce", "output_dim": 1, "activation": "sigmoid"},
    # ...
}
```

Models never mention tasks. Tasks never mention specific models. The mapping is explicit and auditable in one place.

---

## 6. harness-ml — Model Architecture

### Model Protocol

```python
class Model(Protocol):
    name: str                              # "xgboost", "lightgbm", etc.
    supports_tasks: list[str]              # ["binary", "multiclass", "regression"]
    requires_packages: list[str]           # ["xgboost"] — for auto-install

    def fit(self, X_train, y_train, X_val, y_val, params: dict) -> FitResult:
        """Train the model."""

    def predict(self, model, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

    def default_params(self, task_type: str) -> dict:
        """Sensible defaults per task type."""

    def param_schema(self) -> dict:
        """JSON schema for valid hyperparameters (used for tool-time validation)."""

    def save(self, model, path: Path) -> None:
        """Serialize to disk."""

    def load(self, path: Path):
        """Deserialize from disk."""

    def supports_multi_seed(self) -> bool:
        """Whether this model type benefits from seed averaging."""
```

### Model Families

Models are organized by their underlying framework, with shared base classes that absorb duplicated logic:

```
harness-ml/src/harness/ml/models/
├── protocol.py
├── registry.py                 # Auto-discover, auto-install
├── families/
│   ├── boosting/
│   │   ├── base.py             # Shared: early stopping, feature importance,
│   │   │                       #   iterative training, categorical support
│   │   ├── xgboost.py          # Constructor, serialization, param mapping (~50 lines)
│   │   ├── lightgbm.py
│   │   ├── catboost.py
│   │   └── hist_gbm.py
│   ├── linear/
│   │   ├── base.py             # Shared: sklearn fit/predict, coefficient extraction
│   │   ├── logistic.py
│   │   └── elastic_net.py
│   ├── neural/
│   │   ├── base.py             # Shared: device mgmt, batch training, LR scheduling,
│   │   │                       #   early stopping, checkpointing
│   │   ├── mlp.py
│   │   ├── tabnet.py
│   │   ├── tabpfn.py
│   │   └── realmlp.py          # New in v2
│   ├── tree/
│   │   ├── base.py             # Shared: sklearn tree interface
│   │   └── random_forest.py
│   └── kernel/
│       ├── base.py             # Shared: sklearn kernel model interface
│       └── svm.py
└── tests/
    └── test_model_contract.py  # Parametrized: runs every model through full protocol
```

**Adding a new model = one file in the appropriate family.** No changes anywhere else. The registry auto-discovers models. `test_model_contract.py` automatically includes new models in protocol verification.

---

## 7. harness-ml — Training Pipeline

### Runner Architecture

```
harness-ml/src/harness/ml/runners/
├── backtest.py            # Top-level orchestrator — composes everything below
├── cross_validation.py    # Fold generation (8 strategies)
├── training.py            # Per-fold model training (parallel within wave)
├── preprocessing.py       # Leakage-safe fit/transform
├── meta_learner.py        # OOF collection, nested stacking
├── calibration.py         # Pre/post calibration (delegates to task type)
├── postprocessing.py      # Ordered post-processing pipeline
├── prediction_cache.py    # Fingerprint-based skip-unchanged
├── provider_context.py    # Per-fold provider outputs (instance + entity level)
├── dag.py                 # Dependency graph, topological waves, cycle detection
└── progress.py            # Progress callback protocol
```

### DAG-Driven Orchestration

The training pipeline supports complex model dependency graphs, not just flat model → ensemble flows:

```
                    ┌──────────────┐
                    │ team_strength │  (wave 0 — no dependencies)
                    │ (xgboost)     │
                    └──────┬───────┘
                           │ predictions become features
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ matchup  │ │ xgb_alt  │ │ lgbm_alt │  (wave 1)
        │ enhanced │ │          │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │            │            │
             │      ┌─────▼────────────▼──┐
             │      │ base_stack          │  (sub-ensemble)
             │      │ (stacked, logistic) │
             │      └─────────┬───────────┘
             │                │
             └────────┬───────┘
                      ▼
              ┌───────────────┐
              │ final         │  (top-level ensemble)
              │ (stacked,     │
              │  ridge)       │
              └───────────────┘
```

**Provider system:**
- **Instance-level providers**: Model outputs pairwise predictions directly. Stored in ProviderContext as arrays, injected directly as features for downstream models.
- **Entity-level providers**: Model outputs per-entity predictions. Transformed into pairwise derivatives (diff, ratio) via entity ID lookup before injection into downstream models.

**Fingerprint cascade:** Each model's fingerprint includes upstream provider fingerprints as hash inputs. If a provider's params change → its fingerprint changes → all downstream fingerprints change → automatic cache invalidation. Zero bookkeeping.

### Complete Backtest Flow

```
 1. Load & validate
    a. Load config, data, features, models from workspace
    b. Run pre-training guards (features exist, min rows, config valid)
    c. Compute feature schema hash (for fingerprinting)
    d. Resolve feature sets → concrete feature lists per model
    e. Load entity features if any model is an entity-level provider
    f. Filter models:
       - active: false → skip entirely (don't train, don't predict)
       - include_in_ensemble: false → train + predict, but excluded
         from meta-learner feature matrix at construction time
       - ensemble.exclude_models → included in meta-learner training,
         filtered during post-processing

 2. Build & validate DAG
    a. Build provider map (column → provider model)
    b. Infer dependencies (model → {upstream providers})
    c. Compute topological waves
    d. Validate: cycle detection, missing providers, task compatibility

 3. Generate CV folds (8 strategies available)

 ──── PHASE 1: BASE MODEL TRAINING ────

 4. For each fold (optionally parallel via ThreadPoolExecutor):
    a. Split train/test by fold
    b. Create fresh ProviderContext (transient, per-fold — NOT persisted)

    c. For each wave (topological order):
       - Resolve features (including upstream provider outputs from ProviderContext)

       - For each model in wave (parallel within wave):
         i.   Compute fingerprint (includes model config + feature schema
              + upstream provider fingerprints for cascade invalidation)
         ii.  Check prediction cache
              → IF HIT: use cached predictions, skip ALL steps below
              → IF MISS: proceed to training
         iii. [Inside train_single_model — all below only on cache miss]:
              - Apply training filter (exclude rows per model config)
              - Zero-fill specified features
              - Drop NaN rows in feature columns
              - Extract feature columns
              - Compute feature medians from training data (for NaN imputation)
              - Compute class weights (balanced or custom)
              - Apply data augmentation (symmetric rows for pairwise models)
                — re-compute sample weights after augmentation
              - Carve early-stopping validation split if configured
              - For each seed (if n_seeds > 1):
                  Train model with random_state=seed
              - Average predictions across seeds
              - CDF scale fitting for regressors (ONLY if task_type != "regression")
                — multi-seed: fit scale on first seed's margins, apply to all
              - Predict on BOTH train and test splits
         iv.  If provider: store predictions in ProviderContext
              (both train and test, for downstream models in this fold)
         v.   Cache predictions + fingerprint (providers are NOT cached)

       - Report wave progress

    d. Collect base model predictions as OOF (out-of-fold) for this fold
    e. Report fold progress

 ──── PHASE 2: META-LEARNER + POST-PROCESSING (NESTED LOSO) ────

 5. For each holdout fold (second loop over folds):
    a. Construct feature matrix from OOF predictions:
       [model_preds, prior_diffs, meta_features]
       - Binary: one column per model
       - Multiclass: per-class columns (prob_model_c0, prob_model_c1, ...)
    b. Train set = all folds EXCEPT holdout
    c. Fit per-model pre-calibrators on train portion ONLY
    d. Pre-calibrate train + holdout predictions
    e. Train meta-learner on pre-calibrated train predictions
       (logistic / ridge / GBM)
    f. Predict on holdout fold
    g. Apply full post-processing to holdout fold predictions:
       1. Extract base model prediction columns
       2. Filter ensemble.exclude_models
       3. Apply pre-calibration (per-model, fitted in 5c)
       4. Meta-learner prediction (stacking)
       5. Post-calibration
       6. Temperature scaling
       7. Probability clipping
       8. Logit adjustments (paired/diff mode)
       9. Prior-proximity compression
    h. Store processed predictions for holdout fold

 6. Fit final production artifacts on ALL data:
    a. Fit final pre-calibrators on all data
    b. Fit final meta-learner on all pre-calibrated data
    c. Fit final post-calibrator on nested OOF predictions
    d. Serialize production artifacts → version's artifact directory
    e. On failure at any point → fallback to simple average
       - Binary: average prob_ columns
       - Multiclass: per-class averaging

 ──── PHASE 3: METRICS + DIAGNOSTICS ────

 7. Compute metrics
    a. Apply eval filter (row filtering for metrics ONLY —
       predictions still exist for all rows)
    b. Compute per-fold metrics (task-type specific)
    c. Compute pooled metrics across all folds

 8. Generate diagnostics (post-hoc)
    a. Per-fold stats
    b. Calibration curves
    c. ECE, Brier decomposition
    d. Model agreement / diversity metrics
    e. Diagnostics report

 ──── PHASE 4: PRODUCTION ARTIFACT FITTING ────

 9. Retrain all base models on full dataset (all folds combined)
    - Same preprocessing, same hyperparams as CV
    - Fingerprint cache helps — only retrain changed models
    - Serialize to version's artifact directory

10. Write results to workspace
    a. Predictions, metrics, diagnostics → versions/vNNN/run/
    b. Production artifacts → artifacts/ (shared cache, fingerprint-keyed)
    c. Fingerprints → artifacts/fingerprints.json

11. Report completion

### Error Handling

- Model failure within a wave → log error, skip model, continue wave
- Provider failure → downstream dependent models also fail (cascade), independent models continue
- Meta-learner failure → fallback to simple average (graceful degradation)
- Zero successful models → raise error
```

### Cross-Validation Strategies (8)

1. **leave_one_out** — Leave-one-season-out (LOSO)
2. **expanding_window** — All prior rows as train
3. **sliding_window** — Fixed train/test windows
4. **kfold** — Standard K-fold
5. **purged_kfold** — K-fold with temporal purging
6. **stratified_kfold** — Stratified K-fold (class balance)
7. **group_kfold** — Group-aware K-fold
8. **bootstrap** — Bootstrap with .632 estimator

### Progress Tracking

```python
class BacktestProgress(Protocol):
    def on_fold_start(self, fold_id, fold_num, total_folds): ...
    def on_model_trained(self, model_name, fold_id, duration_s, metrics): ...
    def on_wave_complete(self, wave_num, total_waves): ...
    def on_meta_learner_trained(self, metrics): ...
    def on_backtest_complete(self, final_metrics): ...
```

Studio subscribes via WebSocket and streams live updates.

---

## 8. harness-data

Data engineering library. Separate lifecycle from ML. Consumed by harness-ml for both clean dataset production AND feature computation.

### Structure

```
harness-data/src/harness/data/
├── sources/
│   ├── protocol.py          # Source protocol (load, validate, refresh)
│   ├── registry.py          # Source type registry
│   ├── file.py              # CSV, Parquet, Excel
│   ├── url.py               # HTTP/HTTPS fetch
│   └── api.py               # API adapters (paginated, authenticated)
├── transforms/
│   ├── protocol.py          # Transform step protocol
│   ├── engine.py            # Transform pipeline executor
│   ├── steps/               # One file per step type — full set, not trimmed
│   │   ├── filter.py
│   │   ├── select.py
│   │   ├── join.py
│   │   ├── derive.py        # Formula-based column derivation
│   │   ├── aggregate.py     # group_by + agg
│   │   ├── rolling.py       # Rolling windows
│   │   ├── lag.py           # Lag features
│   │   ├── ewm.py           # Exponentially weighted
│   │   ├── cast.py          # Type casting
│   │   ├── fill.py          # Null handling
│   │   ├── unpivot.py       # Wide → long
│   │   ├── sort.py
│   │   ├── head.py
│   │   ├── distinct.py
│   │   ├── rank.py
│   │   ├── encode.py        # Categorical encoding
│   │   ├── bin.py           # Binning/discretization
│   │   ├── diff.py          # Differencing
│   │   ├── trend.py         # Trend computation
│   │   └── datetime.py      # Datetime extraction
│   └── tests/
├── profiling/
│   ├── profiler.py          # Column stats, distributions, cardinality
│   └── validation.py        # Schema validation, quality checks
├── workspace.py             # Reads/writes sources.yaml, transforms.yaml, data/
└── runner.py                # Stateless: sources + transforms → clean dataset
```

### Dual Use

harness-data serves two roles:

1. **Data pipeline execution** (used directly by data tools): Ingest sources, apply declarative transforms, produce clean datasets.
2. **Feature computation engine** (used by harness-ml): When harness-ml resolves a feature formula like `rolling_mean_3(points)`, it calls harness-data's rolling transform. One implementation, no duplication.

### Contract with harness-ml

**File-level contract (data pipeline output):**
```
harness-data outputs:
  data/clean/dataset.parquet    # Analysis-ready dataframe
  data/clean/schema.json        # Column names, types, stats, row count

harness-ml reads:
  data/clean/dataset.parquet    # Input to feature resolution + training
  data/clean/schema.json        # Used for feature formula validation
```

**Programmatic API (feature computation):** harness-ml imports harness-data as a Python library for computing features at training time. This is a direct function-level dependency, not file I/O or IPC:

```python
# harness-ml calls harness-data's transform engine directly
from harness.data.transforms import engine

# Feature formula: "rolling_mean_3(points)"
# harness-ml parses the formula, identifies the transform type ("rolling"),
# and calls harness-data's engine:
result = engine.apply_step(
    data=fold_train_df,
    step_type="rolling",
    params={"column": "points", "window": 3, "agg": "mean"}
)
```

This means harness-data is both a standalone data pipeline tool AND a library dependency of harness-ml. The transform engine is the shared code — used by `data.run` for full pipeline execution and by harness-ml for individual feature computation during training.

### Transform Steps

Each step implements a protocol. Adding a new step = one file. The engine discovers and applies them by config in `transforms.yaml`.

### Expression Engine

Formula evaluation is a first-class subsystem, not a utility function. It is the language the agent uses to express data transformations and feature formulas without writing Python code. Every derive step, feature formula, and computed column flows through it.

```
harness-data/src/harness/data/expressions/
├── __init__.py
├── engine.py          # Expression parser + evaluator
├── registry.py        # Function registry (register, list, describe, type info)
├── validator.py       # Validate expression against schema without executing
├── functions/
│   ├── math.py        # abs, log, sqrt, exp, clip, sign, floor, ceil, round
│   ├── stats.py       # zscore, rank_pct, rolling_mean, rolling_std
│   ├── comparison.py  # where, safe_div, minimum, maximum
│   ├── null.py        # isna, fillna, coalesce
│   └── string.py      # lower, upper, contains, extract
└── tests/
```

**Key capabilities:**

- **Validation without execution** — `validator.validate("abs(rating_a - rating_b)", schema)` checks that columns exist and functions are known, returning clear error messages, without running the computation. This prevents "run for 10 minutes then fail on a typo."
- **Function registry with discoverability** — functions are registered with name, description, input types, and output type. The MCP resource `harness://expressions/functions` lets the agent discover what's available. Extensions can register custom functions.
- **Composable expressions** — `zscore(rolling_mean(points, 3))` works. Nested function calls, not just flat pd.eval.
- **Agent-optimized error messages** — not `KeyError: 'momentum'` but `Column 'momentum' not found in dataset. Available columns: seed, rating, tempo, ...`
- **Safe evaluation** — no raw `eval`/`exec`. Whitelisted functions only. Attribute access blocked.

The expression engine is used by:
- The `derive` transform step (harness-data)
- Feature formula resolution (harness-ml)
- Filter expressions (harness-data)
- Computed columns in the pipeline

---

## 9. Eval System

### Philosophy

Evaluation is a first-class, configurable, multi-dimensional assessment system — not just "compute a metric and show a number." Inspired by the constitutional AI approach: define principles (eval criteria), evaluate against them, provide structured feedback.

### Three Layers

```
┌─────────────────────────────────────────┐
│  Layer 3: LLM Judgments (agent)         │
│  Natural language criteria, qualitative │
│  assessment, domain expertise           │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  Layer 2: Comparative Evals (system)    │
│  Context-aware: vs parent, vs baseline, │
│  vs ensemble. Computes deltas.          │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  Layer 1: Threshold Checks (system)     │
│  Deterministic, configurable rules.     │
│  Pass/fail with explanation.            │
└─────────────────────────────────────────┘
```

**Layers 1 and 2** are computed by harness-ml during the run. Deterministic, structured, returned as data. **Layer 3** is the agent's job — harness-ml provides structured data and prompts; the agent provides qualitative judgment recorded in `experiment.conclude`.

### Generic Framework, User-Defined Dimensions

The eval system does NOT hardcode what matters (calibration, diversity, etc.). Users define their own eval dimensions in `evals.yaml` using metrics from the task type's metric registry. The system provides operators (threshold checks, comparisons). The dimensions are pure data, not code.

**Eval definition:**

```yaml
# config/evals.yaml — user defines what matters for their project
evals:
  probability_accuracy:
    description: "Are predicted probabilities trustworthy?"
    checks:
      - metric: ece
        op: "<"
        value: 0.05
        severity: error
      - metric: brier
        op: "<"
        value: 0.25
        severity: warning
    comparisons:
      - vs: parent
        metric: ece
        expect: lower
    judgment: |
      Look at the calibration curve. Any systematic biases?
      Are there probability ranges where the model is over/under-confident?

  model_value:
    description: "Does this model contribute something new?"
    checks:
      - metric: max_ensemble_correlation
        op: "<"
        value: 0.95
        severity: error
    comparisons:
      - vs: ensemble
        metric: prediction_correlation
    judgment: |
      Does this model disagree with the ensemble on interesting cases
      or just on noise?

  stability:
    description: "Consistent across evaluation folds?"
    checks:
      - metric: fold_std_brier
        op: "<"
        value: 0.03
        severity: warning
    comparisons:
      - vs: parent
        metric: fold_std_brier
        expect: lower
```

### Architecture

```
harness-ml/src/harness/ml/evals/
├── __init__.py
├── schema.py            # EvalDefinition, EvalCheck, EvalComparison, EvalReport (Pydantic)
├── runner.py            # Generic: load defs → compute metrics → check → compare → report
├── checks.py            # Threshold operators (<, >, between, !=)
├── comparisons.py       # vs_parent, vs_baseline, vs_ensemble
└── presets/
    ├── binary.yaml      # Suggested eval template for binary classification
    ├── regression.yaml  # Suggested eval template for regression
    └── multiclass.yaml  # Suggested eval template for multiclass
```

**No domain-specific Python code.** The runner is generic. Eval definitions are pure YAML. Presets are suggestions — `harness init` copies a preset into `evals.yaml` and the user customizes.

**The metric registry is the bridge.** Eval definitions reference metrics by name (`ece`, `brier`, `fold_std_brier`). These resolve through the task type's metric registry. The eval runner doesn't know what "ECE" is — it asks the task type to compute it, then checks the result against the threshold.

### Eval Report (returned by `experiment.propose`)

```json
{
  "version": "v007",
  "metrics": {"brier": 0.198, "ece": 0.031, "auroc": 0.72},
  "eval_report": {
    "probability_accuracy": {
      "checks": [
        {"metric": "ece", "value": 0.031, "op": "<", "threshold": 0.05, "pass": true},
        {"metric": "brier", "value": 0.198, "op": "<", "threshold": 0.25, "pass": true}
      ],
      "comparisons": [
        {"vs": "parent", "metric": "ece", "parent_value": 0.051, "delta": -0.020, "improved": true}
      ],
      "judgment_prompt": "Look at the calibration curve..."
    },
    "model_value": {
      "checks": [{"metric": "max_ensemble_correlation", "value": 0.87, "pass": true}],
      "comparisons": [...],
      "judgment_prompt": "Does this model disagree on interesting cases..."
    },
    "summary": {
      "checks_passed": 4,
      "checks_total": 4,
      "checks_failed_error": 0,
      "checks_failed_warning": 0,
      "regressions": 0,
      "improvements": 2,
      "dimensions_needing_judgment": ["probability_accuracy", "model_value"]
    }
  }
}
```

### What This Enables

1. **Automated regression detection** — an experiment might improve Brier but degrade calibration. Structured evals flag this automatically.
2. **Eval-driven experimentation** — the agent sees "stability is the weakest dimension" and focuses experiments there.
3. **Version comparison by dimensions** — "v005 has best calibration, v007 has best discrimination" — richer than just "lowest Brier."
4. **Domain-agnostic** — sports prediction defines calibration + diversity checks. Medical diagnosis defines recall + fairness checks. The framework is the same.
5. **Progressive rigor** — start with simple threshold checks. Add comparisons as the project matures. Add judgment criteria when you want qualitative assessment. No dimension is mandatory.

### Workspace Integration

```
config/
├── project.yaml
├── models.yaml
├── features.yaml
├── ensemble.yaml
└── evals.yaml          # Eval criteria (user-defined, initialized from preset)
```

`evals.yaml` is created by `harness init` from a task-type-appropriate preset. Users customize freely. Adding a new eval dimension = adding YAML. No code changes.

---

## 10. Pairwise Features (Native, Not a Plugin)


V1's sports plugin is eliminated. Pairwise features are a native concept in harness-ml, applicable to any domain with entity comparisons (sports matchups, A/B testing, head-to-head comparisons).

### Feature Types

| Type | Description | Example |
|------|-------------|---------|
| `entity` | Per-entity, per-period metric from a source. Auto-generates pairwise derivatives (diff, ratio). | `seed` → `diff_seed`, `ratio_seed` |
| `pairwise` | Per-instance feature computed from formula over entity values. | `rating_a - rating_b` |
| `instance` | Per-instance context property (column or formula). | `tournament_stage` |
| `model_output` | Predictions from an upstream provider model. Auto-generates pairwise derivatives for entity-level providers. | `pred_team_strength` → `pred_team_strength_a`, `pred_team_strength_b` |

### Symmetric Data Augmentation

For pairwise models, training data is doubled with reversed rows:
- `diff_*` features are negated
- Labels are flipped (binary: `1-y`, pairwise regression with signed targets like score spreads: `-y`)

This teaches the model symmetry (A vs B is the inverse of B vs A). **Only applicable to pairwise tasks with symmetric semantics** — not for general regression targets where negation is meaningless (e.g., counts, magnitudes). The task type's `adaptation.py` controls whether augmentation is valid. Handled by the training layer, not the runner.

---

## 11. Studio Dashboard

### Views (8)

| # | View | Purpose | Data Source |
|---|------|---------|-------------|
| 1 | **Version Tree** | Home page. Visual tree of all versions with metrics, colored by verdict. Metric trend lines. Experiment narrative. | `versions/*/meta.yaml` |
| 2 | **Version Detail** | Clicked from tree. Hypothesis, diff, metrics with parent comparison, per-model performance, ensemble weights, calibration curve, per-fold breakdown. | `versions/vNNN/` |
| 3 | **Pipeline Explorer** | Interactive DAG. Click nodes to expand: feature list + importance, model hyperparams + metrics, ensemble composition + correlations. Raw config panel. | `config/`, computed DAG |
| 4 | **Diagnostics** | Deep dive. Cross-version comparison, calibration curve overlay, ECE, Brier decomposition, model diversity matrix, per-fold stats, residual analysis. | `versions/*/run/` |
| 5 | **Predictions** | Browse individual outputs. Sortable paginated table, distribution histogram, confidence analysis. Export CSV/Parquet. | `versions/*/run/predictions.parquet` |
| 6 | **Data Profile** | Source inventory, column-level stats, distributions, data preview. | `data/clean/schema.json`, profiling |
| 7 | **MCP Monitor** | Live agent activity stream. Tool calls, params, results, timing, errors. Experiment flow visualization. | `.harness/events.db` |
| 8 | **Preferences** | Theme selection. | localStorage |

### Architecture

```
harness-studio/
├── server.py              # FastAPI — reads workspace directly
├── routes/
│   ├── versions.py        # Version tree + detail endpoints
│   ├── pipeline.py        # DAG + config endpoints
│   ├── diagnostics.py     # Cross-version metrics comparison
│   ├── predictions.py     # Prediction browsing + export
│   ├── data.py            # Data profile endpoints
│   └── monitor.py         # MCP event log stream
├── event_log.py           # Append-only MCP event log (SQLite WAL)
│                          # Records tool calls — NOT workspace state
├── cache.py               # Optional derived SQLite cache for fast queries
├── websocket.py           # Streams: backtest progress + agent activity
└── frontend/
    └── src/views/          # React components for each view
```

**Key design choices:**
- **No event store for workspace state.** Routes read workspace files directly. The event log exists solely for agent activity observability (MCP Monitor).
- **SQLite cache is derived and rebuildable.** Exists for fast queries ("all versions sorted by Brier"). Can be deleted and rebuilt from workspace files.
- **WebSocket streams both** backtest progress (from BacktestProgress callback) and agent activity (from MCP event log).

---

## 12. CLI & Installation

### Commands

```bash
harness init [project-name]    # Create workspace, auto-configure MCP client
harness serve                  # Start MCP server (auto-detect workspace from cwd)
harness studio                 # Open dashboard (auto-detect workspace)
harness status                 # Show current version, metrics, data freshness
harness doctor                 # Check Python, dependencies, MCP config
```

### Workspace Discovery

Like git, harness walks up the directory tree looking for `harness.yaml`:

```bash
~/projects/march-madness/        # ← workspace root (has config/)
~/projects/march-madness/data/   # ← harness finds workspace by walking up
```

No hardcoded project paths. Each Claude Code session auto-discovers the nearest workspace.

### MCP Client Auto-Configuration

`harness init` detects the MCP client and writes the config entry:

```json
{
  "harness": {
    "command": "harness",
    "args": ["serve"]
  }
}
```

No workspace path needed — the server auto-discovers from cwd at runtime.

### No-Workspace Mode

If no workspace is found, the MCP server starts but all tools return a helpful message guiding the agent to initialize a project or navigate to one.

### First-Run Experience

```
$ harness init march-madness
  Project type? [binary/multiclass/regression]: binary
  Data source path: ./data/games.csv
  Target column: home_win

  ✓ Created workspace at ./march-madness/
  ✓ Detected Claude Code — added MCP server config
  ✓ Run `harness studio` to open the dashboard

  Start a conversation with Claude and say:
  "Set up a baseline model for predicting home_win"
```

---

## 13. research-loop Extensions

Harness 2 requires the following extensions to research-loop:

### 1. Parent Version Selection

`propose` must accept a `parent` parameter (defaults to current version):

```typescript
proposeSchema: z.object({
  experiment_type: z.enum(['baseline', 'feature', 'model', 'hyperparameter',
                           'ensemble', 'calibration', 'cv_strategy', 'feature_selection']),
  parent: z.string().optional(),  // Version ID, defaults to current
  // ... type-specific params via runSchema
})
```

### 2. Dynamic Baseline

`getBaseline()` returns the parent version's metrics (not a global baseline). This means the baseline changes per-experiment based on which version the agent branches from.

### 3. Terminal State Change

The workflow becomes:

```
propose → setup_and_run → conclude (terminal)
```

`promote` and `discard` are removed as workflow steps. `conclude` is the only terminal state. Version navigation (`versions.switch`) is handled outside the experiment workflow.

### 4. Verdict Vocabulary

Replace `keep | discard | inconclusive` with:

```typescript
type Verdict = 'improved' | 'degraded' | 'inconclusive' | 'mixed'
```

Descriptive assessment for the research narrative, not prescriptive action.

---

## 14. What's Deferred to Later Phases

### Phase 2: Extensibility
- **Additional task types:** Ranking (NDCG, MRR, MAP), Survival (concordance index), Probabilistic (CRPS, calibration, sharpness). Each drops in as a new directory under `tasks/` with no changes to existing code.
- **Additional model architectures:** SOTA tabular models as they emerge. Each is a single file in the appropriate model family.

### Phase 3: Advanced Version Tree Operations
- **Cross-branch experiments:** Cherry-pick config elements from multiple versions. Merge branches (combine features from v005 + models from v007). Currently approximated manually by switching versions and using tools.
- **Meta-ensemble across versions:** Ensemble the best model from each branch into a super-ensemble. Use version tree to identify diverse, high-performing branches.
- **Concurrent experiments:** Run experiments on multiple branches simultaneously. Would require multiple research-loop instances or significant extension.

### Phase 4: Downstream Applications
- **Competition / scoring tools:** Bracket simulation, tournament scoring, leaderboard generation. Consumes harness-ml predictions — separate tools, not part of training.

---

## 15. Migration from v1

Harness 2 is a ground-up redesign, not a refactor. V1 projects are not directly compatible. However:

- V1's trained models and predictions can be imported as a baseline version (v001) in a v2 workspace
- V1's experiment journal can be converted to v2 version tree entries
- V1's config files (pipeline.yaml, models.yaml, etc.) can be auto-converted to v2 format
- A `harness migrate <v1-project-path>` CLI command handles this conversion

---

## 16. Key Differences from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Tool count | ~80 actions | 17 tools |
| Agent cognitive model | Manage state through many tool calls | Propose hypothesis, get results |
| Experiment model | Linear (promote/discard, overwrite production) | Version tree (branch from anywhere, nothing deleted) |
| Package count (user-facing) | 4 packages to understand | 1 install |
| Task types | Logic scattered across codebase | Self-contained modules |
| Models | 13 wrappers in flat directory | Families with shared base classes |
| Pipeline runner | 72K-line god file | Stateless, composable runners |
| Data engineering | Mixed with ML tools | Separate library (harness-data) |
| Experiment discipline | Custom (built into harness) | Generic (research-loop) |
| Dashboard | 13 tabs, event-store-driven | 8 views, reads workspace directly |
| Pairwise features | Plugin (harness-sports) | Native feature type |
| Config | Split across many files, unclear boundaries | Split by concern with clear ownership |
| Production model | Linear: promote overwrites baseline | No production concept — version tree with current pointer |
| Backtest vs predict vs train | Three separate concepts | One concept: run |
| Formula evaluation | Inline pd.eval with minimal safety | First-class expression engine with registry, validation, discoverability |
| Model evaluation | Raw metric computation | Three-layer eval system (threshold checks + comparisons + LLM judgment) |
