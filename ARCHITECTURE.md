# HarnessML Architecture

## System Overview

HarnessML is a general-purpose agentic ML framework organized as a 4-package
uv workspace monorepo. It supports six ML task types: binary classification,
multiclass classification, regression, ranking, survival analysis, and
probabilistic forecasting.

### Package Roles

| Package | Role | Key Surface |
|---------|------|-------------|
| **harness-core** | Engine: schemas, config, guardrails, models, runner, feature engineering, metrics, data sources | `harnessml.core.*` |
| **harness-plugin** | MCP server: thin async dispatcher with hot-reloadable handlers | `harnessml.plugin.*` |
| **harness-studio** | Companion web dashboard: real-time observability via FastAPI + React | `harnessml.studio.*` |
| **harness-sports** | Optional domain plugin: matchup prediction, pairwise features, symmetric LOSO | `harnessml.sports.*` |

### System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENT / USER INTERFACE                            │
│                                                                             │
│   Claude Code ──► MCP Server (harness-plugin)                               │
│                     │  7 tools, ~80 actions                                 │
│                     │  hot-reload handlers (HARNESS_DEV=1)                  │
│                     │                                                       │
│                     ├──► Event Emitter ──► SQLite ──► Studio (harness-studio)│
│                     │                                  WebSocket live stream │
│                     │                                  React 19 dashboard   │
│                     ▼                                                       │
│               Config Writer                                                 │
│          (reads/writes YAML configs)                                        │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION & ETL                                │
│                                                                             │
│   Data Sources (CSV, Parquet, URL, API, Computed)                           │
│       │  Source Registry: freshness tracking, schema validation              │
│       ▼                                                                     │
│   View Executor (22-step declarative ETL, Polars or Pandas backend)         │
│       │  Steps: filter, select, derive, group_by, join, union, unpivot,     │
│       │         sort, head, rolling, cast, distinct, rank, isin, cond_agg,  │
│       │         lag, ewm, diff, trend, encode, bin, datetime, null_indicator│
│       ▼                                                                     │
│   Feature Store (parquet files, SHA256 fingerprint caching)                 │
│       │  Feature Registry: formula, grouped, instance, regime types          │
│       │  Feature Selection: SelectKBest, RFE, correlation clustering         │
│       │  Text Features: TF-IDF, count vectorizer                            │
│       │  Cyclical Encoding: sin/cos pairs for periodic features              │
│       ▼                                                                     │
│   Preprocessing (leakage-safe: fit on train, transform test)                │
│       Scaling: zscore, robust, quantile                                     │
│       Imputation: median, mean, zero, KNN, iterative (MICE)                 │
│       Categorical: frequency encoding                                       │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GUARDRAILS & VALIDATION                              │
│                                                                             │
│   Pre-Training Validation (validate_project)                                │
│       │  Leakage detection (non-overridable)                                │
│       │  Temporal ordering (non-overridable)                                │
│       │  Critical path protection (non-overridable)                         │
│       │  Data distribution, class imbalance, model complexity (overridable)  │
│       │  Feature count, prediction sanity (overridable)                      │
│       │  Formula syntax validation, feature existence checks                 │
│       ▼                                                                     │
│   Data Profiler: cardinality, type inference, distribution hints             │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      CROSS-VALIDATION & TRAINING                            │
│                                                                             │
│   CV Strategies (8 strategies)                                              │
│       LOSO, Expanding Window, Sliding Window, KFold, Purged KFold,          │
│       Stratified KFold, Group KFold, Bootstrap (.632)                       │
│       │                                                                     │
│       ▼                                                                     │
│   Model Training (per fold, parallel via ThreadPoolExecutor)                │
│       │  12 model wrappers:                                                 │
│       │    XGBoost, LightGBM, CatBoost, RandomForest, Logistic, ElasticNet, │
│       │    MLP, TabNet, SVM, HistGradientBoosting, GAM, NGBoost             │
│       │  GPU routing: CUDA / MPS / CPU (detect_device)                      │
│       │  Multi-seed averaging (seed_stride)                                 │
│       │  Error recovery: partial results on model failure                    │
│       │  Prediction caching: skip unchanged model+feature+data combos       │
│       ▼                                                                     │
│   Meta-Learner Ensemble (LOSO training)                                     │
│       │  Types: logistic (default), ridge, GBM                              │
│       │  Per-fold coefficients, multiclass per-class weighting              │
│       │  Ensemble diversity: disagreement, Q-statistic, Kappa, correlation   │
│       ▼                                                                     │
│   Calibration (per-fold, 4 methods)                                         │
│       Spline (PCHIP), Isotonic, Platt, Beta                                 │
│       │                                                                     │
│       ▼                                                                     │
│   Post-Processing                                                           │
│       Temperature scaling, clip floor, prior compression, logit adjustments  │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OUTPUTS & ANALYSIS                                   │
│                                                                             │
│   Predictions (CSV/Parquet export)                                          │
│   Diagnostics (45 metrics, markdown + JSON, per-fold breakdowns)            │
│   Explainability (SHAP values, PDP, feature interactions)                   │
│   Drift Detection (KS test, PSI)                                            │
│   Conformal Prediction (calibrated confidence intervals)                    │
│   Experiment Journal (JSONL, hypothesis/conclusion, compare, rollback)       │
│   HPO (Optuna: pruning, multi-objective, hyperparameter importance)          │
│   Notebook Generation (Colab, Kaggle, local)                                │
│   Cloud Upload (Google Drive, Kaggle)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

Each stage is driven by YAML configuration and orchestrated through the
MCP server, making the entire pipeline agent-readable and agent-writable.

### Core Submodules

| Submodule | Purpose |
|-----------|---------|
| `core.schemas` | Pydantic v2 contracts + MetricRegistry (45 metrics across 6 task types) |
| `core.config` | YAML loading + OmegaConf deep merge for experiment overlays |
| `core.guardrails` | Safety guardrails: leakage detection, temporal validation, naming conventions |
| `core.models` | Model wrappers (XGBoost, LightGBM, CatBoost, RF, Logistic, ElasticNet, MLP, TabNet, SVM, HistGBM, GAM, NGBoost) + ModelRegistry |
| `core.runner` | Pipeline orchestration, project, hooks, CLI, DAG, matchups |
| `core.runner.data` | Data ingestion, pipeline, profiling, utils, loaders |
| `core.runner.features` | Feature store, engine, cache, discovery, diversity, selection, auto-search, utils |
| `core.runner.training` | Trainer, CV strategies, preprocessing, meta-learner, calibration, postprocessing, prediction cache, fingerprint |
| `core.runner.experiments` | Experiment schema, journal, manager, logger |
| `core.runner.views` | View executor (pandas + polars), resolver, polars compat |
| `core.runner.analysis` | Diagnostics, reporting, explainability, drift, conformal, ensemble diversity, viz |
| `core.runner.optimization` | HPO, exploration, sweep, pipeline planner |
| `core.runner.validation` | Guards, stage guards, validation, validator |
| `core.runner.scaffold` | Project scaffold, presets, server gen, notebook generation |
| `core.runner.workflow` | Workflow tracker, run manager |
| `core.runner.config_writer` | Config writing helpers |
| `core.runner.sources` | Source registry, freshness tracking, schema validation, adapters (file/url/api/computed) |
| `core.runner.drives` | Cloud adapters: Google Drive (OAuth), Kaggle (dataset/notebook upload) |
| `core.feature_eng` | Feature engineering registry + transforms |

---

## Build-vs-Buy Analysis

Each subsystem was evaluated against existing open-source alternatives.
The table below documents what was built custom versus adopted, and why.

| Subsystem | Choice | Alternatives Considered | Why This Choice |
|-----------|--------|------------------------|-----------------|
| View Engine | Custom 22-step declarative ETL | dbt, Pandas pipes, Hamilton | YAML-driven for agent consumption; domain-specific steps (symmetric unpivot, matchup joins); tightly integrated with feature store and source registry |
| Feature Store | Custom parquet-based with registry | Feast, Tecton, Hopsworks, Featuretools | Zero infrastructure requirement; file-based portability; MCP-integrated add/remove/search; SHA256 fingerprint-based caching eliminates redundant recomputation |
| Experiment Tracking | Custom JSONL journal + discipline enforcement | MLflow, W&B, Neptune, ClearML | Hypothesis-driven methodology is a core value proposition; YAML-native config diffs via OmegaConf; phased workflow enforcement (EDA through Ensemble) not available in any alternative |
| Model Wrappers | Custom BaseModel + ModelRegistry | sklearn Pipeline, MLflow Models, ONNX Runtime | Unified interface across 12 model types with per-fold calibration; multi-seed averaging (seed_stride); inspect-based kwargs forwarding; normalize/batch_norm/early_stopping per wrapper |
| Dashboard | Custom FastAPI + React 19 | Streamlit, Gradio, TensorBoard, Grafana | Real-time WebSocket streaming of MCP events; pipeline DAG visualization; experiment discipline integration (hypothesis/conclusion); 45-metric diagnostics panel |
| Config Management | OmegaConf + YAML | Hydra, Pydantic Settings, dynaconf | Deep merge enables experiment overlays on immutable production config; YAML-native format is agent-readable and agent-writable; no decorator or annotation overhead |
| MCP Server | FastMCP | Custom WebSocket, gRPC, REST API | Native Claude Code integration; hot-reload handlers in dev mode (HARNESS_DEV=1); tool signatures serve as self-documenting API |
| Guardrails | Custom overridable / non-overridable system | Great Expectations, Pandera, whylogs | ML-specific checks (leakage detection, temporal validation, critical path protection); agent-facing override hints with explanations; two-tier severity model |
| Calibration | Custom Spline/Isotonic/Platt/Beta | sklearn CalibratedClassifierCV, netcal | PCHIP spline interpolation; per-fold calibration within cross-validation; integrated into ensemble meta-learner; sigmoid CDF conversion for probability output |
| HPO | Optuna wrapper | Hyperopt, Ray Tune, sklearn GridSearchCV | Pruning support for early termination; multi-objective optimization; lightweight single-process execution; integrates with experiment journal |
| Cross-Validation | Custom 8-strategy system | sklearn CV splitters, TimeSeriesSplit | Temporal integrity enforcement; nested calibration within folds; symmetric LOSO for matchup prediction; custom fold column support |
| Source Adapters | Custom registry with 4 adapter types | Intake, fsspec, Kedro DataCatalog | Freshness tracking; schema validation at load time; computed sources for derived datasets; MCP-integrated source management |
| Metric Registry | Custom registry with 45 metrics | torchmetrics, sklearn.metrics | Covers 6 task types in one registry; agent-discoverable via MCP; custom metrics (ECE, Brier decomposition) not available elsewhere as a unified set |

### Adopted Libraries (Buy)

| Library | Role | Why Adopted |
|---------|------|-------------|
| Pydantic v2 | Schema validation, contracts | Industry standard for Python data validation; native JSON Schema generation |
| OmegaConf | Config deep merge | Purpose-built for hierarchical YAML config with merge semantics |
| FastMCP | MCP protocol implementation | Official MCP SDK; handles protocol negotiation and transport |
| scikit-learn | Base ML algorithms, preprocessing | De facto standard; RandomForest, LogisticRegression, ElasticNet, StandardScaler |
| XGBoost / LightGBM / CatBoost | Gradient boosting | Best-in-class gradient boosting implementations; each has unique strengths |
| PyTorch | Neural network backend | MLP and TabNet implementations; GPU support when available |
| FastAPI + uvicorn | Studio HTTP server | Async-native; WebSocket support; OpenAPI docs |
| React 19 + Vite | Studio frontend | Component model for dashboard tabs; fast HMR in development |
| Optuna | Hyperparameter optimization | Pruning, multi-objective, study persistence |
| structlog | Structured logging | JSON-native logging for agent-parseable output |

---

## Design Decisions

### Prediction Caching

Predictions are cached using SHA256 fingerprints derived from: model
parameters, feature list, data hash, and fold configuration. This allows
the runner to skip recomputation when a model+feature+data combination has
already been evaluated, which is critical for iterative experiment workflows
where only one variable changes at a time.

### Experiment Overlays

Production configuration is treated as immutable. Experiments create YAML
overlay files that are deep-merged via OmegaConf at runtime. This preserves
a clean diff between baseline and experiment, enables rollback by simply
deleting the overlay, and produces human-readable experiment definitions
that the agent can inspect and modify.

### Symmetric LOSO Cross-Validation

For matchup prediction (e.g., sports), each game appears twice in the
dataset (once per team perspective). Symmetric Leave-One-Season-Out ensures
that both perspectives of the same game land in the same fold, preventing
data leakage. The system supports 8 CV strategies total, with temporal
integrity checks built into each.

### Registry Pattern

Five subsystems use the registry pattern: models, metrics, features,
guardrails, and sources. Each registry provides `register()`, `get()`, and
`list()` operations with a `with_defaults()` factory that wraps optional
dependencies in try/except blocks. This keeps the core import lightweight
while allowing optional backends (CatBoost, TabNet, etc.) to register
themselves when available.

### Hot-Reload Handlers

When `HARNESS_DEV=1` is set, the MCP server calls `importlib.reload()` on
handler modules before each dispatch. This means handler business logic
can be edited without restarting the server, which tightens the
development loop. Changes to tool signatures or docstrings still require
a restart since those are registered at server startup.

### Namespace Packages

There is no `__init__.py` at the `src/harnessml/` level in any package.
This enables Python namespace packages, allowing multiple packages
(harness-core, harness-plugin, harness-studio, harness-sports) to
contribute to the `harnessml` namespace without conflicting. Each package
owns its own sub-namespace (e.g., `harnessml.core`, `harnessml.plugin`).

### Hook System for Domain Plugins

Domain plugins (like harness-sports) register into core extension points
via `HookRegistry`. Hooks are declared in harness-core and invoked at
specific pipeline stages. Plugins register hook implementations at import
time via entry points (`[project.entry-points."harnessml.plugins"]`). This
keeps domain-specific logic out of the core engine.

### Meta-Learner Ensemble

The ensemble uses a meta-learner that computes per-fold coefficients for
each base model. This avoids the common pitfall of fitting ensemble weights
on the same data used to train base models. For multiclass tasks, the
meta-learner produces per-class coefficients, enabling different model
weighting across classes.

### Two-Tier Guardrail Severity

Guardrails are split into overridable and non-overridable categories.
Non-overridable guardrails (leakage detection, temporal validation,
critical path protection) cannot be bypassed even by explicit agent
request. Overridable guardrails (naming conventions, soft limits) can be
suppressed with documented justification. This prevents the most dangerous
ML pitfalls while keeping the system flexible.

---

## Package Dependencies

```
harness-core
  Dependencies: pydantic, numpy, pandas, pyarrow, scikit-learn, scipy,
                omegaconf, pyyaml, click, structlog, polars
  Optional: xgboost, catboost, lightgbm, torch, pytorch-tabnet, optuna,
            shap, matplotlib, pandera, nbformat, google-api-python-client,
            google-auth-oauthlib, kaggle, pygam, ngboost

harness-plugin
  Dependencies: harness-core, mcp (FastMCP), click
      |
      v
harness-core

harness-sports
  Dependencies: harness-core
      |
      v
harness-core

harness-studio
  Dependencies: harness-core, fastapi, uvicorn, aiosqlite, websockets
      |
      v
harness-core
```

### Dependency Graph

```
harness-plugin ──> harness-core <── harness-sports
                       ^
                       |
                  harness-studio
```

All three downstream packages depend on harness-core. There are no
cross-dependencies between harness-plugin, harness-studio, and
harness-sports. The studio reads project artifacts (config YAMLs, journal
JSONL, run outputs) from disk and uses harness-core for schema definitions.

### Build System

All packages use Hatchling as the build backend. The workspace is managed
by uv, with workspace source references in each package's `pyproject.toml`
(e.g., `harness-core = { workspace = true }` under `[tool.uv.sources]`).

---

## Security Assumptions

This section documents the current security posture and known limitations.
HarnessML is designed for local development and trusted environments.

### Token Storage

OAuth tokens for Google Drive integration are stored as plaintext JSON at
the path configured in the drive adapter (default:
`credentials_dir/drive_token.json`). These tokens grant `drive.file` scope
(access only to files created by the application). The token file has no
encryption at rest and relies on filesystem permissions for protection.

### Kaggle Credentials

Kaggle API key handling is inherited from the `kaggle` package, which reads
`~/.kaggle/kaggle.json`. HarnessML does not manage, rotate, or encrypt
these credentials. Users are responsible for securing this file per
Kaggle's documentation.

### Configuration Security

No secrets should be stored in project config YAML files. This is enforced
by convention and documentation, not by code-level scanning. Config files
are expected to live in version-controlled project directories and should
never contain API keys, tokens, or passwords.

### SQLite Event Store

The Studio event store uses SQLite in WAL mode. The database file is
created in the project directory with default filesystem permissions
(typically owner read/write). There is no row-level access control or
encryption. The event store contains operational telemetry (tool calls,
timing, parameters) but not model predictions or training data.

### Network Exposure

Studio serves on `localhost` by default and is not intended for network
exposure. The `--host` flag can override this, but there is no
authentication layer, TLS termination, or CORS restriction beyond
same-origin. Running Studio on a non-localhost interface in an untrusted
network is not recommended without a reverse proxy.

### MCP Server Permissions

The MCP server runs as a subprocess of Claude Code and inherits the
parent process's filesystem and network permissions. It can read and write
any file the user can access. There is no sandboxing, capability
restriction, or audit log beyond what the Studio event store captures.

### Process Isolation

There is no process isolation between the MCP server, the training
pipeline, and the Studio dashboard. All run under the same user account
and share filesystem access. Model training can consume arbitrary CPU,
memory, and disk without resource limits.

### Supply Chain

Optional dependencies (XGBoost, CatBoost, LightGBM, PyTorch, TabNet) are
pinned to minimum versions in `pyproject.toml` but not hash-locked. The
`uv.lock` file provides reproducible installs within the workspace, but
does not verify package integrity beyond what PyPI provides.

### Recommendations for Hardened Deployment

If HarnessML is deployed in a shared or networked environment, consider:

- Encrypting OAuth tokens at rest (e.g., via OS keychain integration)
- Running Studio behind a reverse proxy with TLS and authentication
- Restricting MCP server filesystem access via containerization
- Adding secret scanning to CI for config YAML files
- Pinning and hash-verifying all dependencies in `uv.lock`
- Setting filesystem permissions on the SQLite event store to owner-only

---

## File Layout

```
harness-ml/
├── ARCHITECTURE.md                    # This file
├── CLAUDE.md                          # AI agent instructions
├── CONTRIBUTING.md                    # Development guide
├── README.md                          # Project overview
├── pyproject.toml                     # Workspace root (uv)
├── uv.lock                           # Lockfile
│
├── .claude-plugin/
│   └── plugin.json                    # Claude Code Plugin manifest
├── .mcp.json                          # MCP server configuration
├── skills/
│   ├── harness-run-experiment/SKILL.md
│   ├── harness-explore-space/SKILL.md
│   └── harness-domain-research/SKILL.md
│
├── .github/workflows/
│   ├── tests.yml                      # CI test suite
│   ├── publish.yml                    # PyPI publishing
│   └── deploy-docs.yml               # GitHub Pages
│
├── docs/
│   ├── for-agents.md                  # MCP tool reference
│   ├── for-humans.md                  # Human-readable guide
│   ├── troubleshooting.md             # Common issues + fixes
│   ├── examples/
│   │   ├── binary-classification.md
│   │   ├── temporal-cv.md
│   │   └── feature-engineering.md
│   ├── skills/                        # Skill source markdown
│   └── plans/                         # Implementation plans
│
├── packages/
│   ├── harness-core/                  # ── ML ENGINE ──
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── src/harnessml/core/
│   │   │   ├── __init__.py
│   │   │   ├── errors.py                  # ErrorCode enum, format_error/warning
│   │   │   ├── logging.py                 # structlog-based get_logger, configure_logging
│   │   │   │
│   │   │   ├── schemas/
│   │   │   │   ├── contracts.py           # Pydantic v2 contracts, GuardrailRule enum
│   │   │   │   └── metrics.py             # MetricRegistry (45 metrics, 6 task types)
│   │   │   │
│   │   │   ├── config/
│   │   │   │   ├── loader.py              # YAML loading
│   │   │   │   ├── merge.py               # OmegaConf deep merge
│   │   │   │   └── resolver.py            # Path and variable resolution
│   │   │   │
│   │   │   ├── guardrails/
│   │   │   │   ├── base.py                # Guardrail base class, severity levels
│   │   │   │   ├── inventory.py           # 12 guards (DataDistribution, ClassImbalance, etc.)
│   │   │   │   ├── execution.py           # Guardrail execution engine
│   │   │   │   ├── audit.py               # Audit trail
│   │   │   │   └── server.py              # Guardrail server integration
│   │   │   │
│   │   │   ├── models/
│   │   │   │   ├── base.py                # BaseModel abstract class
│   │   │   │   ├── registry.py            # ModelRegistry with with_defaults()
│   │   │   │   ├── params.py              # Per-model Pydantic param schemas
│   │   │   │   ├── calibration.py         # Spline, Isotonic, Platt, Beta calibrators
│   │   │   │   ├── device.py              # GPU detection (CUDA/MPS/CPU)
│   │   │   │   ├── ensemble.py            # Ensemble utilities
│   │   │   │   ├── backtest.py            # BacktestRunner
│   │   │   │   ├── cv.py                  # Cross-validation utilities
│   │   │   │   ├── fingerprint.py         # Model fingerprinting
│   │   │   │   ├── orchestrator.py        # Training orchestration
│   │   │   │   ├── postprocessing.py      # Prediction post-processing
│   │   │   │   ├── run_manager.py         # Run lifecycle management
│   │   │   │   ├── tracking.py            # Model tracking
│   │   │   │   └── wrappers/
│   │   │   │       ├── xgboost.py         # XGBoost
│   │   │   │       ├── lightgbm.py        # LightGBM
│   │   │   │       ├── catboost.py        # CatBoost
│   │   │   │       ├── random_forest.py   # RandomForest
│   │   │   │       ├── logistic.py        # LogisticRegression (auto max_iter)
│   │   │   │       ├── elastic_net.py     # ElasticNet (normalize, NaN handling)
│   │   │   │       ├── mlp.py             # MLP (GPU, batch_norm, early_stopping)
│   │   │   │       ├── tabnet.py          # TabNet (GPU, scheduler, seed_stride)
│   │   │   │       ├── svm.py             # SVM (SVC/SVR, probability=True)
│   │   │   │       ├── hist_gbm.py        # HistGradientBoosting (native NaN)
│   │   │   │       ├── gam.py             # GAM (PyGAM, optional dep)
│   │   │   │       └── ngboost.py         # NGBoost (optional dep)
│   │   │   │
│   │   │   ├── runner/
│   │   │   │   ├── schema.py              # ProjectConfig, DataConfig, BacktestConfig, etc.
│   │   │   │   ├── pipeline.py            # PipelineRunner (fold parallelization)
│   │   │   │   ├── project.py             # Project abstraction
│   │   │   │   ├── hooks.py               # HookRegistry for domain plugins
│   │   │   │   ├── dag.py                 # Provider dependency DAG
│   │   │   │   ├── matchups.py            # Matchup generation
│   │   │   │   ├── cli.py                 # CLI entry point
│   │   │   │   ├── transformation_tester.py # Transform testing
│   │   │   │   │
│   │   │   │   ├── data/                  # Data ingestion & pipeline
│   │   │   │   │   ├── ingest.py          # Data ingestion + raw preservation
│   │   │   │   │   ├── pipeline.py        # Data pipeline orchestration
│   │   │   │   │   ├── profiler.py        # Cardinality, type inference
│   │   │   │   │   ├── loaders.py         # Config loaders
│   │   │   │   │   └── utils.py           # Data utilities
│   │   │   │   │
│   │   │   │   ├── features/              # Feature store & engineering
│   │   │   │   │   ├── store.py           # Parquet-based feature store
│   │   │   │   │   ├── engine.py          # Feature computation engine
│   │   │   │   │   ├── cache.py           # Feature caching
│   │   │   │   │   ├── discovery.py       # Auto feature discovery
│   │   │   │   │   ├── diversity.py       # Feature diversity analysis
│   │   │   │   │   ├── selection.py       # SelectKBest, RFE, correlation clustering
│   │   │   │   │   ├── auto_search.py     # Auto feature search
│   │   │   │   │   └── utils.py           # Feature utilities
│   │   │   │   │
│   │   │   │   ├── training/              # Training & ensemble
│   │   │   │   │   ├── trainer.py         # Model training loop, NaN handling
│   │   │   │   │   ├── cv_strategies.py   # 8 CV strategies
│   │   │   │   │   ├── preprocessing.py   # Leakage-safe Preprocessor
│   │   │   │   │   ├── meta_learner.py    # StackedEnsemble (logistic/ridge/gbm)
│   │   │   │   │   ├── calibration.py     # build_calibrator factory
│   │   │   │   │   ├── postprocessing.py  # Ensemble post-processing
│   │   │   │   │   ├── prediction_cache.py # SHA256 fingerprint-based caching
│   │   │   │   │   └── fingerprint.py     # Config fingerprinting
│   │   │   │   │
│   │   │   │   ├── experiments/           # Experiment tracking
│   │   │   │   │   ├── schema.py          # ExperimentRecord, TrialRecord Pydantic models
│   │   │   │   │   ├── journal.py         # JSONL journal (read/write/update)
│   │   │   │   │   ├── manager.py         # Experiment lifecycle, compare, rollback
│   │   │   │   │   ├── logger.py          # Experiment logging
│   │   │   │   │   └── experiment.py      # Experiment utilities
│   │   │   │   │
│   │   │   │   ├── views/                 # Declarative ETL
│   │   │   │   │   ├── executor.py        # View engine (Pandas backend)
│   │   │   │   │   ├── executor_polars.py # View engine (Polars backend, 3-13x faster)
│   │   │   │   │   ├── resolver.py        # View dependency resolution
│   │   │   │   │   └── polars_compat.py   # Pandas ↔ Polars converters
│   │   │   │   │
│   │   │   │   ├── analysis/              # Diagnostics & reporting
│   │   │   │   │   ├── diagnostics.py     # Markdown + JSON diagnostics
│   │   │   │   │   ├── reporting.py       # Report generation
│   │   │   │   │   ├── explainability.py  # SHAP values, PDP, interactions
│   │   │   │   │   ├── drift.py           # KS test, PSI drift detection
│   │   │   │   │   ├── conformal.py       # Conformal prediction intervals
│   │   │   │   │   ├── ensemble_diversity.py # Disagreement, Q-stat, Kappa, correlation
│   │   │   │   │   └── viz.py             # Visualization utilities
│   │   │   │   │
│   │   │   │   ├── optimization/          # HPO & exploration
│   │   │   │   │   ├── hpo.py             # Optuna: pruning, multi-objective
│   │   │   │   │   ├── exploration.py     # Hyperparameter exploration
│   │   │   │   │   ├── sweep.py           # Parameter sweep
│   │   │   │   │   └── pipeline_planner.py # Pipeline planning
│   │   │   │   │
│   │   │   │   ├── validation/            # Guards & validation
│   │   │   │   │   ├── guards.py          # Pipeline stage guards
│   │   │   │   │   ├── stage_guards.py    # PipelineGuards class
│   │   │   │   │   ├── validation.py      # validate_project pre-training checks
│   │   │   │   │   └── validator.py       # Config validator
│   │   │   │   │
│   │   │   │   ├── scaffold/              # Project scaffolding
│   │   │   │   │   ├── scaffold.py        # Project scaffolding
│   │   │   │   │   ├── presets.py         # Configuration presets
│   │   │   │   │   ├── notebook.py        # Jupyter notebook generation
│   │   │   │   │   └── server_gen.py      # Server generation
│   │   │   │   │
│   │   │   │   ├── workflow/              # Workflow & run management
│   │   │   │   │   ├── tracker.py         # Phased workflow enforcement
│   │   │   │   │   └── run_manager.py     # Run lifecycle
│   │   │   │   │
│   │   │   │   ├── config_writer/         # Config YAML management (10 submodules)
│   │   │   │   │   ├── __init__.py        # Re-exports 65 public functions
│   │   │   │   │   ├── _helpers.py        # Shared helpers
│   │   │   │   │   ├── _init.py           # Project initialization
│   │   │   │   │   ├── features.py        # Feature CRUD + formula validation
│   │   │   │   │   ├── models.py          # Model CRUD + feature existence checks
│   │   │   │   │   ├── data.py            # Data config management
│   │   │   │   │   ├── pipeline.py        # Pipeline + backtest config
│   │   │   │   │   ├── experiments.py     # Experiment CRUD + JSONL journal
│   │   │   │   │   ├── views.py           # View definitions
│   │   │   │   │   └── sources.py         # Source definitions
│   │   │   │   │
│   │   │   │   ├── sources/               # Data source adapters
│   │   │   │   │   ├── registry.py        # Source registry
│   │   │   │   │   ├── adapters.py        # File, URL, API, computed adapters
│   │   │   │   │   ├── freshness.py       # Freshness tracking
│   │   │   │   │   └── validation.py      # Schema validation at load time
│   │   │   │   │
│   │   │   │   └── drives/                # Cloud integrations
│   │   │   │       ├── drive.py           # Google Drive (OAuth upload/folders)
│   │   │   │       └── kaggle.py          # Kaggle (dataset/notebook upload)
│   │   │   │
│   │   │   └── feature_eng/
│   │   │       ├── registry.py            # Feature engineering registry
│   │   │       ├── text.py                # TF-IDF, count vectorizer
│   │   │       └── transforms.py          # Cyclical encoding, transform functions
│   │   │
│   │   └── tests/                         # ~1780 tests
│   │       ├── schemas/                   # Contract + metric tests
│   │       ├── config/                    # Config loading tests
│   │       ├── guardrails/                # Guardrail tests
│   │       ├── models/                    # Model wrapper + calibration tests
│   │       ├── runner/                    # Pipeline, training, CV, experiment tests
│   │       ├── feature_eng/               # Feature engineering tests
│   │       └── benchmarks/                # View executor benchmarks
│   │
│   ├── harness-plugin/                # ── MCP SERVER ──
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── src/harnessml/plugin/
│   │   │   ├── mcp_server.py              # Tool signatures, async dispatcher, threading locks
│   │   │   ├── event_emitter.py           # Fail-safe SQLite event emission
│   │   │   ├── setup.py                   # harness-setup CLI, skill installation
│   │   │   └── handlers/
│   │   │       ├── data.py                # Data source management actions
│   │   │       ├── features.py            # Feature add/remove/search/auto_search
│   │   │       ├── models.py              # Model add/update/remove/list
│   │   │       ├── config.py              # Config read/write/validate
│   │   │       ├── pipeline.py            # Backtest, predict, train, progress
│   │   │       ├── experiments.py         # Create/log/compare/rollback experiments
│   │   │       ├── competitions.py        # Kaggle/competition integrations
│   │   │       ├── _validation.py         # Enum fuzzy match, required params, cross-param
│   │   │       └── _common.py             # Shared helpers (resolve_project_dir, etc.)
│   │   └── tests/                         # ~152 tests
│   │
│   ├── harness-studio/               # ── WEB DASHBOARD ──
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── scripts/build_frontend.sh
│   │   ├── src/harnessml/studio/
│   │   │   ├── server.py                  # FastAPI app, static file serving
│   │   │   ├── cli.py                     # CLI: harness-studio command
│   │   │   ├── event_store.py             # SQLite WAL-mode, parameterized queries
│   │   │   ├── broadcaster.py             # asyncio.Queue fan-out, bounded queues
│   │   │   ├── errors.py                  # ErrorCategory enum, classify_error
│   │   │   └── routes/
│   │   │       ├── events.py              # Event log REST + WebSocket
│   │   │       ├── project.py             # Config + DAG endpoints
│   │   │       ├── experiments.py         # Experiment journal endpoints
│   │   │       ├── runs.py                # Metrics, calibration, correlations (cached)
│   │   │       ├── predictions.py         # Prediction export (CSV/Parquet), pagination
│   │   │       ├── config.py              # Config viewer
│   │   │       ├── data.py                # Data viewer
│   │   │       ├── features.py            # Feature viewer
│   │   │       ├── models.py              # Model viewer
│   │   │       ├── ensemble.py            # Ensemble viewer
│   │   │       └── ws.py                  # WebSocket handler
│   │   │
│   │   ├── frontend/                      # React 19 + TypeScript + Vite + bun
│   │   │   └── src/
│   │   │       ├── App.tsx                # Root component, tab routing
│   │   │       ├── main.tsx               # Entry point
│   │   │       ├── types/api.ts           # TypeScript API response types
│   │   │       ├── utils/time.ts          # Timestamp humanization
│   │   │       ├── hooks/
│   │   │       │   ├── useApi.ts          # API fetch hook
│   │   │       │   ├── useWebSocket.ts    # WebSocket connection
│   │   │       │   ├── useKeyboardShortcuts.ts  # Keyboard shortcuts (1-4, r, /, ?)
│   │   │       │   ├── useProject.ts      # Project context
│   │   │       │   ├── useTheme.ts        # Theme switching
│   │   │       │   └── useRefreshKey.ts   # Auto-refresh
│   │   │       ├── components/
│   │   │       │   ├── Layout/            # App shell, sidebar, navigation
│   │   │       │   ├── Toast/             # Toast notifications (auto-dismiss)
│   │   │       │   ├── EmptyState/        # Contextual empty states
│   │   │       │   ├── ErrorBoundary/     # React error boundary per tab
│   │   │       │   ├── StatBox/           # Metric stat boxes
│   │   │       │   ├── ExpandableRow/     # Expandable table rows
│   │   │       │   ├── MarkdownRenderer/  # Markdown rendering
│   │   │       │   └── Tooltip/           # Tooltips with glossary
│   │   │       ├── views/
│   │   │       │   ├── Dashboard/         # Overview dashboard
│   │   │       │   ├── Activity/          # Live event log + stat bar
│   │   │       │   ├── DAG/               # Pipeline DAG (React Flow nodes)
│   │   │       │   ├── Experiments/       # Sortable table + metric charts
│   │   │       │   ├── Diagnostics/       # Metrics, calibration, correlations, residuals
│   │   │       │   ├── Data/              # Data viewer
│   │   │       │   ├── Features/          # Feature viewer
│   │   │       │   ├── Models/            # Model viewer
│   │   │       │   ├── Ensemble/          # Ensemble viewer
│   │   │       │   ├── Config/            # Config viewer
│   │   │       │   ├── Predictions/       # Prediction viewer + export
│   │   │       │   └── Preferences/       # Theme + settings
│   │   │       └── styles/
│   │   │           ├── tokens.css         # Design tokens
│   │   │           ├── reset.css          # CSS reset
│   │   │           ├── colors.ts          # Color utilities
│   │   │           └── themes/            # 11 themes (claude, nord, matrix, etc.)
│   │   │
│   │   └── tests/                         # ~33 tests
│   │
│   └── harness-sports/               # ── DOMAIN PLUGIN ──
│       ├── pyproject.toml
│       ├── README.md
│       ├── src/harnessml/sports/
│       │   ├── hooks.py                   # Hook registration into core
│       │   ├── matchups.py                # Matchup prediction logic
│       │   ├── pairwise.py                # Pairwise feature generation
│       │   └── competitions/
│       │       ├── schemas.py             # Tournament/bracket schemas
│       │       ├── structure.py           # Bracket structure generation
│       │       ├── simulator.py           # Monte Carlo simulation
│       │       ├── scorer.py              # Scoring systems
│       │       ├── optimizer.py           # Bracket optimization
│       │       ├── confidence.py          # Confidence intervals
│       │       ├── adjustments.py         # Probability adjustments
│       │       ├── explainer.py           # Pick explanations
│       │       └── export.py              # Export formats
│       └── tests/                         # ~20 tests
```
