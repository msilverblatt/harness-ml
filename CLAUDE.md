# EasyML -- AI Agent Instructions

## What This Is

EasyML is a modular ML framework with 8 packages in a uv workspace monorepo.
Architecture: schema-core (shared Pydantic contracts) + tool-ring (config,
features, models, data, experiments, guardrails) + runner (YAML-driven CLI).

## Tech Stack

- Python 3.11+, managed by **uv** (always use `uv run`, never bare `python`)
- Pydantic v2 for all schemas and contracts
- OmegaConf for config deep merge
- scikit-learn, XGBoost, CatBoost, LightGBM, PyTorch (MLP, TabNet)
- Namespace packages: no `__init__.py` at `src/easyml/` level

## Package Map

| Package | Purpose | Key Classes |
|---------|---------|-------------|
| `easyml-schemas` | Shared contracts + metrics | `ModelConfig`, `brier_score`, `GuardrailViolation` |
| `easyml-config` | YAML loading + merge | `resolve_config`, `deep_merge` |
| `easyml-features` | Feature engineering | `FeatureRegistry`, `FeatureBuilder` |
| `easyml-models` | Training + ensembles | `ModelRegistry`, `TrainOrchestrator`, `StackedEnsemble` |
| `easyml-data` | Sources + DVC | `SourceRegistry`, `generate_dvc_yaml`, `StageGuard` |
| `easyml-experiments` | Experiment lifecycle | `ExperimentManager` |
| `easyml-guardrails` | Safety guardrails | `Guardrail`, `PipelineServer`, `AuditLogger` |
| `easyml-runner` | YAML-driven CLI + orchestration | `PipelineRunner`, `validate_project`, `scaffold_project`, `generate_server` |

## Key Conventions

- All packages under `packages/`, source in `packages/<name>/src/easyml/<subpkg>/`
- Namespace packages: `easyml.schemas`, `easyml.config`, etc. (no root `__init__.py`)
- Registry pattern used in features, models, sources, and guardrails
- TDD: write tests alongside implementation, run with `uv run pytest`
- Integration tests at repo root: `tests/test_integration.py`

## How to Add a New Model

1. Create `packages/easyml-models/src/easyml/models/wrappers/my_model.py`
2. Subclass `BaseModel`, implement `fit`, `predict_proba`, `save`, `load`
3. Register in `ModelRegistry.with_defaults()` (wrap in try/except for optional deps)
4. Add tests in `packages/easyml-models/tests/`

## How to Add a New Feature

1. Get or create a `FeatureRegistry` instance
2. Use `@registry.register(name=..., category=..., level=..., output_columns=[...])`
3. Function signature: `def compute(df, config) -> DataFrame` with entity_id + period_id cols
4. The `FeatureBuilder` handles caching automatically via source hashing

## How to Run Experiments

1. Create: `ExperimentManager.create("exp-NNN-description")`
2. Edit the overlay YAML in the experiment directory
3. Detect changes: `mgr.detect_changes(production, overlay)`
4. Log results: `mgr.log(experiment_id, hypothesis, changes, verdict)`
5. Promote winners: `mgr.promote(experiment_id, production_config_path)`

## Testing

```bash
uv run pytest                    # all tests
uv run pytest tests/             # integration tests only
uv run pytest packages/easyml-models/tests/  # one package
uv run pytest -v                 # verbose
```

## What NOT to Do

- Do not bypass guardrails -- non-overridable ones exist for safety (leakage, temporal, critical path)
- Do not hardcode paths, thresholds, or magic numbers
- Do not create parallel versions of files -- extend existing modules
- Do not skip `uv run` -- bare `python` will not find workspace packages
- Do not put `__init__.py` at the `src/easyml/` level (breaks namespace packages)
- Do not modify production config directly -- use experiment overlays
