# easyml-experiments

Experiment tracking and overlay management for EasyML -- naming validation,
change detection, do-not-retry enforcement, mandatory logging, and atomic promotion.

## Installation

```bash
pip install easyml-experiments
```

## Quick Start

```python
from easyml.experiments import ExperimentManager

mgr = ExperimentManager(
    experiments_dir="experiments/",
    naming_pattern=r"^exp-\d{3}-.+$",
    log_path="EXPERIMENT_LOG.md",
    do_not_retry_path="do_not_retry.json",
)

# Create experiment (validates naming, checks for unlogged experiments)
exp_dir = mgr.create("exp-042-new-feature")

# Detect what the overlay changes vs production
report = mgr.detect_changes(production_config, overlay)
print(report.changed_models, report.new_models)

# Check do-not-retry patterns
mgr.check_do_not_retry("temperature scaling T=1.5")  # raises if blocked

# Log results (mandatory before creating next experiment)
mgr.log("exp-042-new-feature", hypothesis="...", changes="...", verdict="keep")

# Promote to production (requires logged "keep" verdict, creates backup)
backup = mgr.promote("exp-042-new-feature", "config/models/production.yaml")
```

## Key APIs

- `ExperimentManager` -- Full experiment lifecycle: create, detect changes, log, promote
- `ChangeReport` -- Summary of changed/new/removed models and ensemble changes
- `ExperimentError` -- Exception wrapping `GuardrailViolation` for structured error handling
