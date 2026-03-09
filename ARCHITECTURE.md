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

### Data Flow

```
CSV / Parquet / URL / API
        |
    Source Registry (file, url, api, computed adapters)
        |
    Views: 22-step declarative ETL (YAML-driven)
        |
    Feature Store (parquet, fingerprint-cached)
        |
    Model Registry (8 model types, multi-seed averaging)
        |
    Meta-Learner Ensemble (per-fold coefficients)
        |
    Calibration (Spline / Isotonic / Platt / Beta)
        |
    Predictions + Diagnostics
```

Each stage is driven by YAML configuration and orchestrated through the
MCP server, making the entire pipeline agent-readable and agent-writable.

### Core Submodules

| Submodule | Purpose |
|-----------|---------|
| `core.schemas` | Pydantic v2 contracts + MetricRegistry (45 metrics across 6 task types) |
| `core.config` | YAML loading + OmegaConf deep merge for experiment overlays |
| `core.guardrails` | Safety guardrails: leakage detection, temporal validation, naming conventions |
| `core.models` | Model wrappers (XGBoost, LightGBM, CatBoost, RF, Logistic, ElasticNet, MLP, TabNet) + ModelRegistry |
| `core.runner` | Pipeline orchestration, training, meta-learner, calibration, feature store, views, config writer, diagnostics, exploration |
| `core.runner.sources` | Source registry, freshness tracking, schema validation, adapters (file/url/api/computed) |
| `core.runner.drives` | Cloud adapters: Google Drive (OAuth), Kaggle (dataset/notebook upload) |
| `core.runner.notebook` | Jupyter notebook generation (Colab, Kaggle, local destinations) |
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
| Model Wrappers | Custom BaseModel + ModelRegistry | sklearn Pipeline, MLflow Models, ONNX Runtime | Unified interface across 8+ model types with per-fold calibration; multi-seed averaging (seed_stride); inspect-based kwargs forwarding; normalize/batch_norm/early_stopping per wrapper |
| Dashboard | Custom FastAPI + React 19 | Streamlit, Gradio, TensorBoard, Grafana | Real-time WebSocket streaming of MCP events; pipeline DAG visualization; experiment discipline integration (hypothesis/conclusion); 45-metric diagnostics panel |
| Config Management | OmegaConf + YAML | Hydra, Pydantic Settings, dynaconf | Deep merge enables experiment overlays on immutable production config; YAML-native format is agent-readable and agent-writable; no decorator or annotation overhead |
| MCP Server | FastMCP | Custom WebSocket, gRPC, REST API | Native Claude Code integration; hot-reload handlers in dev mode (HARNESS_DEV=1); tool signatures serve as self-documenting API |
| Guardrails | Custom overridable / non-overridable system | Great Expectations, Pandera, whylogs | ML-specific checks (leakage detection, temporal validation, critical path protection); agent-facing override hints with explanations; two-tier severity model |
| Calibration | Custom Spline/Isotonic/Platt/Beta | sklearn CalibratedClassifierCV, netcal | PCHIP spline interpolation; per-fold calibration within cross-validation; integrated into ensemble meta-learner; sigmoid CDF conversion for probability output |
| HPO | Optuna wrapper | Hyperopt, Ray Tune, sklearn GridSearchCV | Pruning support for early termination; multi-objective optimization; lightweight single-process execution; integrates with experiment journal |
| Cross-Validation | Custom 7-strategy system | sklearn CV splitters, TimeSeriesSplit | Temporal integrity enforcement; nested calibration within folds; symmetric LOSO for matchup prediction; custom fold column support |
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
data leakage. The system supports 7 CV strategies total, with temporal
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
            google-auth-oauthlib, kaggle

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
packages/
  harness-core/
    src/harnessml/core/
      schemas/         # Pydantic contracts, MetricRegistry
      config/          # YAML loading, OmegaConf merge
      guardrails/      # Leakage, temporal, naming checks
      models/          # BaseModel, wrappers/, ModelRegistry
      runner/          # Pipeline, training, meta-learner, calibration
        sources/       # Source registry, adapters
        drives/        # Google Drive, Kaggle cloud adapters
        notebook/      # Jupyter notebook generation
      feature_eng/     # Feature registry, transforms
    tests/

  harness-plugin/
    src/harnessml/plugin/
      mcp_server.py    # Tool signatures, async dispatcher
      handlers/        # Business logic (hot-reloadable)
        models.py      # Model CRUD actions
        data.py        # Data source management
        features.py    # Feature add/remove/search
        experiments.py # Experiment lifecycle
        config.py      # Config read/write
        pipeline.py    # Pipeline orchestration + progress
        _validation.py # Enum validation, fuzzy match
        _common.py     # Shared helpers

  harness-studio/
    src/harnessml/studio/
      server.py        # FastAPI application
      event_store.py   # SQLite WAL-mode event store
      broadcaster.py   # WebSocket fan-out
      routes/          # REST + WebSocket endpoints
    frontend/          # React 19 + TypeScript + Vite

  harness-sports/
    src/harnessml/sports/
      matchups.py      # Matchup prediction logic
      hooks.py         # Hook registration into core
```
