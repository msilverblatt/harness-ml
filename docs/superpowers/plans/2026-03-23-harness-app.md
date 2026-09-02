# Harness App Implementation Plan (Package 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Harness application layer — workspace management (version tree, config files, run orchestration), CLI entry points, and the composition layer that wires harness-data + harness-ml into a usable product.

**Architecture:** The workspace manager owns the version tree, config file management, and run orchestration. The CLI provides `harness init`, `harness serve`, `harness status`, `harness doctor`. The workspace is the on-disk contract — all state lives in files that humans can inspect and Studio can read.

**Tech Stack:** Python 3.11+, click (CLI), pydantic, pyyaml, harness-data, harness-ml

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Sections 3 (Workspace), 4 (Tool Surface), 12 (CLI)

**E2E testing mandate:** Every task includes e2e tests exercising the full workspace lifecycle.

---

## File Structure

```
packages/harness-app/
├── pyproject.toml
├── src/harness/app/
│   ├── __init__.py
│   ├── workspace/
│   │   ├── __init__.py
│   │   ├── manager.py           # WorkspaceManager — top-level orchestration
│   │   ├── versions.py          # VersionTree — create, switch, list, compare
│   │   ├── config.py            # ConfigManager — read/write/snapshot config files
│   │   └── discovery.py         # find_workspace — walk up directory tree
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── types.py             # ExperimentType enum + typed param schemas
│   │   ├── runner.py            # ExperimentRunner — propose → run → conclude
│   │   └── history.py           # Version tree navigation + comparison
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py              # Click CLI group
│   │   ├── init.py              # harness init
│   │   ├── status.py            # harness status
│   │   └── doctor.py            # harness doctor
│   └── analysis/
│       ├── __init__.py
│       └── diagnostics.py       # Diagnostics generation (metrics, calibration, per-fold)
└── tests/
    ├── conftest.py
    ├── test_workspace/
    │   ├── test_manager.py
    │   ├── test_versions.py
    │   ├── test_config.py
    │   └── test_discovery.py
    ├── test_experiments/
    │   ├── test_types.py
    │   └── test_runner.py
    ├── test_cli/
    │   ├── test_init.py
    │   └── test_status.py
    └── test_e2e.py
```

---

### Task 1: Workspace Discovery + Config Manager

**Files:**
- Create: `packages/harness-app/pyproject.toml`
- Create: `src/harness/app/workspace/discovery.py`
- Create: `src/harness/app/workspace/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_workspace/test_discovery.py`
- Create: `tests/test_workspace/test_config.py`

**discovery.py** — find workspace by walking up from cwd:
```python
from pathlib import Path

def find_workspace(start: Path | None = None) -> Path | None:
    """Walk up directory tree looking for harness.yaml. Returns workspace root or None."""
    current = Path(start or Path.cwd()).resolve()
    while current != current.parent:
        if (current / "harness.yaml").exists():
            return current
        current = current.parent
    if (current / "harness.yaml").exists():
        return current
    return None
```

**config.py** — read/write/snapshot workspace config files:
```python
class ConfigManager:
    def __init__(self, workspace_dir: Path):
        self._root = workspace_dir
        self._config_dir = workspace_dir / "config"

    def read_project(self) -> ProjectConfig: ...
    def read_models(self) -> ModelsConfig: ...
    def read_ensemble(self) -> EnsembleConfig: ...
    def read_features(self) -> FeatureSet: ...
    def read_evals(self) -> dict: ...  # Raw YAML for eval runner

    def write_project(self, config: ProjectConfig): ...
    def write_models(self, config: ModelsConfig): ...

    def snapshot_config(self, dest_dir: Path): ...  # Copy all config files to a directory
    def restore_config(self, source_dir: Path): ...  # Overwrite config/ from a snapshot
```

Tests: find_workspace locates harness.yaml, returns None when not found. ConfigManager reads/writes YAML, snapshot/restore roundtrip.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-app): workspace discovery + config manager"
```

---

### Task 2: Version Tree

**Files:**
- Create: `src/harness/app/workspace/versions.py`
- Create: `tests/test_workspace/test_versions.py`

The version tree is the core of harness 2's experiment model.

```python
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class VersionMeta:
    id: str
    parent: str | None
    experiment_type: str | None = None
    hypothesis: str = ""
    conclusion: str = ""
    verdict: str = ""  # improved, degraded, inconclusive, mixed
    timestamp: str = ""
    data_hash: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

class VersionTree:
    def __init__(self, workspace_dir: Path):
        self._root = workspace_dir
        self._versions_dir = workspace_dir / "versions"

    def create_version(self, meta: VersionMeta, config_snapshot: Path) -> str:
        """Create a new version directory with meta.yaml + config snapshot."""

    def get_version(self, version_id: str) -> VersionMeta | None:
        """Read version metadata."""

    def get_current(self) -> str | None:
        """Read the current version pointer."""

    def set_current(self, version_id: str) -> None:
        """Update the current pointer and restore config/ from this version's snapshot."""

    def list_versions(self) -> list[VersionMeta]:
        """List all versions with their metadata."""

    def next_version_id(self) -> str:
        """Generate next version ID (v001, v002, etc.)."""

    def get_tree(self) -> dict:
        """Return tree structure (parent pointers) for visualization."""

    def compare(self, v1: str, v2: str) -> dict:
        """Compare metrics between two versions."""

    def ancestry(self, version_id: str) -> list[VersionMeta]:
        """Path from root to this version."""
```

Tests: create version, get version, current pointer, list versions, tree structure, ancestry chain, compare metrics, next_version_id auto-increments.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-app): version tree (create, switch, compare, ancestry)"
```

---

### Task 3: Workspace Manager

**Files:**
- Create: `src/harness/app/workspace/manager.py`
- Create: `tests/test_workspace/test_manager.py`

Top-level orchestration — the main entry point for workspace operations.

```python
class WorkspaceManager:
    def __init__(self, workspace_dir: Path):
        self._root = workspace_dir
        self.config = ConfigManager(workspace_dir)
        self.versions = VersionTree(workspace_dir)
        self.data = DataWorkspace(workspace_dir)  # from harness-data

    @staticmethod
    def init(workspace_dir: Path, task_type: str, target_column: str) -> "WorkspaceManager":
        """Initialize a new workspace: create directories, harness.yaml, default configs."""

    def run_experiment(
        self, experiment_type: str, hypothesis: str,
        params: dict, parent: str | None = None,
    ) -> BacktestResult:
        """THE main operation: propose → generate diff → run backtest → create version."""

    def conclude_experiment(self, version_id: str, conclusion: str, verdict: str) -> None:
        """Record conclusion + verdict in version meta."""

    def switch_version(self, version_id: str) -> None:
        """Update current pointer, restore config/."""

    def status(self) -> dict:
        """Current version, metrics, model count, data freshness."""
```

Tests: init creates full workspace structure, run_experiment creates a version with metrics, conclude_experiment updates meta, switch_version changes config/, status returns useful info.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-app): workspace manager (init, run_experiment, conclude, switch)"
```

---

### Task 4: Experiment Types

**Files:**
- Create: `src/harness/app/experiments/types.py`
- Create: `src/harness/app/experiments/runner.py`
- Create: `tests/test_experiments/test_types.py`
- Create: `tests/test_experiments/test_runner.py`

Typed experiment system — each type knows what params it accepts and what config it modifies.

```python
from enum import Enum

class ExperimentType(str, Enum):
    BASELINE = "baseline"
    FEATURE = "feature"
    MODEL = "model"
    HYPERPARAMETER = "hyperparameter"
    ENSEMBLE = "ensemble"
    CALIBRATION = "calibration"
    CV_STRATEGY = "cv_strategy"
    FEATURE_SELECTION = "feature_selection"

class ExperimentRunner:
    """Applies typed experiment params to config, runs backtest, creates version."""

    def run(self, workspace: WorkspaceManager, experiment_type: ExperimentType,
            hypothesis: str, params: dict, parent: str | None = None) -> BacktestResult:
        """1. Resolve parent config
           2. Apply typed diff based on experiment_type
           3. Run backtest
           4. Create version with metrics
           5. Return result"""
```

Tests: each experiment type applies correct config changes, baseline creates initial config, feature adds to features.yaml, model adds to models.yaml, hyperparameter modifies existing model params.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-app): typed experiment system (8 experiment types)"
```

---

### Task 5: CLI

**Files:**
- Create: `src/harness/app/cli/main.py`
- Create: `src/harness/app/cli/init.py`
- Create: `src/harness/app/cli/status.py`
- Create: `src/harness/app/cli/doctor.py`
- Create: `tests/test_cli/test_init.py`
- Create: `tests/test_cli/test_status.py`

Using Click for CLI framework:

```python
# main.py
import click

@click.group()
def cli():
    """Harness — Agent-first ML platform."""
    pass

@cli.command()
@click.argument("project_name", required=False)
@click.option("--task-type", type=click.Choice(["binary", "multiclass", "regression"]), prompt=True)
@click.option("--target", prompt="Target column")
def init(project_name, task_type, target):
    """Initialize a new Harness workspace."""

@cli.command()
def status():
    """Show workspace status."""

@cli.command()
def doctor():
    """Check system dependencies and configuration."""

@cli.command()
def studio():
    """Open the Studio dashboard."""
```

Tests: init creates workspace, status shows current version, doctor checks dependencies.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-app): CLI (init, status, doctor)"
```

---

### Task 6: Full E2E — Workspace Lifecycle

**Files:**
- Create: `tests/test_e2e.py`

The most important test — exercises the complete workspace lifecycle:

```python
class TestE2EWorkspaceLifecycle:
    def test_full_lifecycle(self, tmp_path):
        """Init → add data → baseline experiment → feature experiment → compare versions."""

        # 1. Init workspace
        ws = WorkspaceManager.init(tmp_path, task_type="binary", target_column="target")

        # 2. Add data source + run data pipeline
        # ... create CSV, add source, run pipeline

        # 3. Run baseline experiment
        result1 = ws.run_experiment("baseline", "Establish baseline with logistic + RF", {
            "models": {"lr": {"model_type": "logistic"}, "rf": {"model_type": "random_forest"}},
            "features": {"f1": {"type": "instance", "source_column": "feature_a"}},
        })
        assert result1.metrics["accuracy"] > 0.5

        # 4. Check version tree
        versions = ws.versions.list_versions()
        assert len(versions) == 1

        # 5. Run feature experiment
        result2 = ws.run_experiment("feature", "Add feature_b", {
            "feature": {"name": "f2", "type": "instance", "source_column": "feature_b"},
        })

        # 6. Compare versions
        versions = ws.versions.list_versions()
        assert len(versions) == 2

        # 7. Conclude + switch
        ws.conclude_experiment(versions[1].id, "Feature improved accuracy", "improved")
        ws.switch_version(versions[1].id)
        assert ws.versions.get_current() == versions[1].id
```

- [ ] **Steps: Write test → implement any missing pieces → verify → commit**

```bash
git commit -m "feat(harness-app): full workspace lifecycle e2e (Plan 4 complete)"
```
