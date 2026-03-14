# Contributing to HarnessML

Thank you for your interest in contributing to HarnessML. This guide covers
development setup, code conventions, and the PR workflow.

---

## Development Setup

### Prerequisites

- **Python 3.11+**
- **[uv](https://github.com/astral-sh/uv)** for workspace management
- **[bun](https://bun.sh/)** (only if working on Studio frontend)

### Install

```bash
git clone https://github.com/msilverblatt/harness-ml.git
cd harness-ml
uv sync
```

This installs all four workspace packages in development mode:

| Package | Path |
|---------|------|
| harness-core | `packages/harness-core/` |
| harness-plugin | `packages/harness-plugin/` |
| harness-studio | `packages/harness-studio/` |
| harness-sports | `packages/harness-sports/` |

Always use `uv run` to execute commands. Never use bare `python`.

---

## Running Tests

```bash
# All tests
uv run pytest

# Per-package
uv run pytest packages/harness-core/tests/
uv run pytest packages/harness-plugin/tests/
uv run pytest packages/harness-studio/tests/
uv run pytest packages/harness-sports/tests/

# Verbose output
uv run pytest -v

# Specific subsystem
uv run pytest packages/harness-core/tests/runner/
uv run pytest packages/harness-core/tests/models/
```

All tests must pass before submitting a PR.

---

## Code Style

We use **[ruff](https://docs.astral.sh/ruff/)** for linting and formatting.

```bash
uv run ruff check .          # lint
uv run ruff format .         # format
uv run ruff check --fix .    # auto-fix lint issues
```

Key style rules:
- 4-space indentation
- Double quotes for strings
- Type hints on public API functions
- Docstrings on all public classes and functions

---

## Namespace Packages

HarnessML uses Python namespace packages. The critical rule:

> **Never place an `__init__.py` at the `src/harnessml/` level.**

Each package owns its own subnamespace:

```
packages/harness-core/src/harnessml/core/
packages/harness-plugin/src/harnessml/plugin/
packages/harness-studio/src/harnessml/studio/
packages/harness-sports/src/harnessml/sports/
```

The `__init__.py` files live inside `core/`, `plugin/`, `studio/`, and `sports/`
-- never at the `harnessml/` directory itself.

---

## Import Paths

Always use the full namespace path:

```python
# Correct -- post-reorganization paths
from harnessml.core.schemas.contracts import ProjectConfig
from harnessml.core.models.registry import ModelRegistry
from harnessml.core.runner.config_writer import ConfigWriter
from harnessml.core.runner.training.trainer import Trainer
from harnessml.core.runner.data.ingest import ingest_data
from harnessml.core.runner.features.store import FeatureStore
from harnessml.core.runner.views.executor import ViewExecutor
from harnessml.core.runner.analysis.reporting import build_pick_log
from harnessml.core.runner.experiments.manager import ExperimentManager
from harnessml.plugin.handlers.models import ModelsTools  # @tool_group class

# Wrong -- old flat runner paths, do not use
from harnessml.core.runner.training import Trainer
from harnessml.core.runner.data_ingest import ingest_data
from harnessml.core.runner.feature_store import FeatureStore
from harnessml.core.runner.view_executor import ViewExecutor
```

---

## Registry Pattern

HarnessML uses a registry pattern for extensible components. Each registry
follows the same structure: register a name to a callable, then look it up at
runtime.

### Adding a New Model

1. Create `packages/harness-core/src/harnessml/core/models/wrappers/my_model.py`
2. Subclass `BaseModel`, implement `fit`, `predict_proba`, `save`, `load`
3. Register in `ModelRegistry.with_defaults()` (wrap in `try/except` for optional deps)
4. Add tests in `packages/harness-core/tests/models/`

### Adding a New Feature

Use the feature engineering registry:

1. Define a transform function
2. Register with `FeatureRegistry.register(name, fn, type=...)`
3. Feature types: `grouped`, `instance`, `formula`, `regime`

### Adding a New Metric

1. Write the metric function in `packages/harness-core/src/harnessml/core/schemas/metrics.py`
2. Register: `MetricRegistry.register("task_type", "name", fn)`
3. Supported task types: `binary`, `multiclass`, `regression`, `ranking`, `survival`, `probabilistic`

### Adding a New Guardrail

1. Add check function to `packages/harness-core/src/harnessml/core/guardrails/`
2. Register in the guardrail registry
3. Mark as `overridable=False` for safety-critical checks (leakage, temporal, critical path)

---

## MCP Handler Development

The MCP server (`harness-plugin`) is built on [protomcp](https://github.com/msilverblatt/protomcp).
Each handler file is a `@tool_group` class with `@action` methods:

- `server.py` -- 25-line entry point, imports handlers and calls `protomcp.run()`
- `handlers/*.py` -- `@tool_group` classes with `@action` methods delegating to `_handle_*` business logic

### Hot-Reload (Dev Mode)

Use `pmcp dev` for automatic hot-reload. All handler changes take effect
immediately without restarting the server.

```bash
pmcp dev packages/harness-plugin/src/harnessml/plugin/server.py
```

### Handler Structure

Each handler module defines a `@tool_group` class. Each method decorated
with `@action` becomes an MCP tool action with its own typed schema.
Business logic lives in `_handle_*` functions; the `@action` methods
are thin wrappers. Shared helpers live in `handlers/_common.py` and
`handlers/_validation.py`.

### Experiment Workflow

The experiment lifecycle (`experiment_workflow.py`) uses protomcp's
`@workflow`/`@step` feature with dynamic tool visibility. The agent can
only see valid next steps at each point in the sequence:

```
experiment.create → [experiment.write_overlay] → experiment.run
    → experiment.log_result → [experiment.promote | experiment.done]
```

During an active workflow, all non-experiment tools remain visible via
`allow_during` globs. The experiment discipline gates (plan-exists,
previous-logged, plan-freshness) run inside the `create` step handler.

---

## PR Workflow

### Branch Naming

Use feature branches off `main`:

```bash
git checkout -b feat/my-feature
git checkout -b fix/bug-description
git checkout -b docs/update-guide
```

### Commit Messages

We prefer [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add survival analysis metrics
fix: correct temporal split boundary condition
docs: update model wrapper guide
ci: add Python 3.12 to test matrix
refactor: extract calibration into separate module
test: add coverage for multiclass meta-learner
```

### Before Submitting

1. Run the full test suite: `uv run pytest`
2. Run linting: `uv run ruff check .`
3. Ensure no `__init__.py` at the `src/harnessml/` level
4. Verify imports use `harnessml.core.*` paths (not old `harnessml.*` paths)

### Review Process

- All PRs require passing CI
- Describe **what** changed and **why** in the PR description
- Link related issues if applicable
- Keep PRs focused -- one logical change per PR
