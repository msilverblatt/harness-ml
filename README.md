# EasyML

AI-driven ML framework with guardrails. Schema-core architecture with
composable packages for config, features, models, data, experiments, and
safety guardrails.

## Architecture

```
                        +------------------+
                        |  easyml-schemas  |   Pydantic contracts + metrics
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                  |                  |
     +--------v-------+ +-------v--------+ +-------v--------+
     |  easyml-config | | easyml-features| |  easyml-models |
     |  YAML + merge  | | registry +     | | 8 wrappers, CV |
     |  + variants    | | caching +      | | ensemble, cal  |
     +--------+-------+ | pairwise       | +-------+--------+
              |          +-------+--------+         |
              |                  |                  |
     +--------v-------+ +-------v--------+ +-------v--------+
     |  easyml-data   | | easyml-exper.  | | easyml-guard.  |
     |  sources, DVC  | | overlay mgmt,  | | 11 guardrails  |
     |  guards, refresh| | DNR, promote  | | MCP server,    |
     +----------------+ +----------------+ | audit logger   |
                                           +----------------+
```

All packages share contracts through `easyml-schemas` (the core). Config,
features, and models form the middle tier. Data, experiments, and guardrails
are the outer tool ring.

## Quick Start

```bash
git clone <repo-url> && cd easyml
uv sync          # install all packages + dev deps

uv run pytest    # run full test suite
```

### Basic Usage

```python
from easyml.config import resolve_config
from easyml.features import FeatureRegistry, FeatureBuilder
from easyml.models import ModelRegistry, TrainOrchestrator, StackedEnsemble
from easyml.schemas import brier_score, accuracy

# 1. Load config
config = resolve_config("config/", file_map={"models": "models.yaml"})

# 2. Register and build features
registry = FeatureRegistry()

@registry.register(name="win_rate", category="resume", level="team", output_columns=["win_rate"])
def compute_win_rate(df, cfg):
    result = df[["entity_id", "period_id"]].copy()
    result["win_rate"] = df["wins"] / (df["wins"] + df["losses"])
    return result

# 3. Train models
model_registry = ModelRegistry.with_defaults()
orchestrator = TrainOrchestrator(model_registry, config["models"], output_dir="models/")
trained = orchestrator.train_all(X, y, feature_columns=cols)

# 4. Evaluate
from easyml.schemas import brier_score
print(f"Brier: {brier_score(y_true, y_prob):.4f}")
```

## Packages

| Package | Description |
|---------|-------------|
| `easyml-schemas` | Pydantic contracts, probability/regression/ensemble metrics |
| `easyml-config` | Split YAML loading, variant resolution, deep merge (OmegaConf) |
| `easyml-features` | Decorator-based feature registry, incremental caching, pairwise builder |
| `easyml-models` | 8 model wrappers, 5 CV strategies, calibration, stacked ensembles |
| `easyml-data` | Source registry, DVC generation, stage guards, refresh orchestrator |
| `easyml-experiments` | Experiment lifecycle: naming, change detection, DNR, logging, promote |
| `easyml-guardrails` | 11 guardrails (3 non-overridable), MCP server, audit logging |

## Development

```bash
uv sync                        # install workspace
uv run pytest -v               # run all tests
uv run pytest packages/easyml-models/tests/  # run one package's tests
```

Requires Python 3.11+. Managed by [uv](https://github.com/astral-sh/uv) workspaces.
