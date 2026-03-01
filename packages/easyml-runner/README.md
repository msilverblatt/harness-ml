# easyml-runner

YAML-driven orchestration layer for easyml. Define models, ensemble, backtest,
and experiments in config files -- run everything from CLI or Python API.

## Install

```bash
pip install easyml-runner
```

## Quick Start

```bash
# Scaffold a new project
easyml init my-project && cd my-project

# Validate configuration
easyml validate

# Run the full pipeline (train + backtest)
easyml run pipeline

# Create and manage experiments
easyml experiment create exp-001-baseline
easyml experiment list

# Inspect config and models
easyml inspect config
easyml inspect models

# Start MCP server (if configured)
easyml serve
```

## Python API

```python
from easyml.runner import validate_project, PipelineRunner, scaffold_project

# Validate config
result = validate_project("config/")
assert result.valid

# Run pipeline
runner = PipelineRunner(project_dir=".", config_dir="config/")
result = runner.run_full()
print(result["metrics"])
```

## Key APIs

| API | Description |
|-----|-------------|
| `validate_project(config_dir)` | Validate YAML config, return `ValidationResult` |
| `PipelineRunner` | Train models, run backtesting from config |
| `scaffold_project(path)` | Generate new project with starter config |
| `generate_server(config, dir)` | Generate MCP server from `ServerDef` |
| `load_features(decls, registry)` | Auto-load features from YAML declarations |
| `load_sources(decls, registry)` | Auto-load data sources from YAML declarations |
| `ProjectConfig` | Top-level Pydantic schema for all config |
