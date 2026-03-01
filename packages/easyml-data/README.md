# easyml-data

Data ingestion and validation for EasyML -- source registry with leakage metadata,
typed artifacts, DVC config generation, stage guards, and refresh orchestration.

## Installation

```bash
pip install easyml-data
```

## Quick Start

```python
from easyml.data import SourceRegistry, generate_dvc_yaml, StageGuard, RefreshOrchestrator
from easyml.schemas import StageConfig, ArtifactDecl

# Register data sources with leakage metadata
sources = SourceRegistry()

@sources.register(
    name="kaggle", category="external",
    outputs=["data/raw/"], temporal_safety="pre_tournament",
)
def fetch_kaggle(output_dir, config):
    ...

# Generate DVC pipeline YAML from typed stages
stages = {
    "ingest": StageConfig(
        script="pipelines/ingest.py", consumes=[],
        produces=[ArtifactDecl(name="raw", type="data", path="data/processed/")],
    ),
    "featurize": StageConfig(
        script="pipelines/featurize.py", consumes=["raw"],
        produces=[ArtifactDecl(name="features", type="features", path="data/features/")],
    ),
}
dvc_yaml = generate_dvc_yaml(stages)

# Stage guards with staleness detection
guard = StageGuard("train", requires=["data/features/team.parquet"], min_rows=100)
guard.check()  # raises GuardrailViolationError if preconditions fail
```

## Key APIs

- `SourceRegistry` -- Decorator-based registration with leakage metadata and freshness checks
- `generate_dvc_yaml(stages)` -- Build DVC pipeline YAML from typed `StageConfig` declarations
- `StageGuard` -- Validate file existence, row counts, and staleness before pipeline stages
- `RefreshOrchestrator` -- Failure-tolerant batch execution of all registered sources
