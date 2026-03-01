# easyml-config

Configuration resolution for EasyML -- loads split YAML files, resolves variants,
and deep-merges overlays using OmegaConf.

## Installation

```bash
pip install easyml-config
```

## Quick Start

```python
from easyml.config import resolve_config, deep_merge, load_config_file

# Load and merge a split config directory
config = resolve_config(
    config_dir="config/",
    file_map={
        "pipeline": "pipeline.yaml",
        "models": "models/production.yaml",
        "ensemble": "ensemble.yaml",
    },
    variant="w",  # loads pipeline_w.yaml if it exists, falls back to pipeline.yaml
    overlay={"ensemble": {"temperature": 1.5}},
)

# Deep merge two dicts (OmegaConf-based, returns plain dict)
base = {"models": {"xgb": {"depth": 3}}}
override = {"models": {"xgb": {"depth": 5, "lr": 0.01}}}
merged = deep_merge(base, override)
# {"models": {"xgb": {"depth": 5, "lr": 0.01}}}
```

## Key APIs

- `resolve_config(config_dir, file_map, variant, overlay)` -- Main entry point
- `load_config_file(config_dir, filename, variant)` -- Load one YAML with variant resolution
- `deep_merge(base, override)` -- Recursive dict merge via OmegaConf
