# MCP Server Implementation Plan (Package 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MCP server — THE interface the AI agent uses to interact with Harness. 17 tools + 5 resources. This is the composition layer that wires harness-data, harness-ml, harness-app, and research-loop into a coherent agent experience.

**Architecture:** Python protomcp server (`pmcp dev server.py`). Uses `@tool_group` / `@action` decorators from protomcp. Each tool group is a thin handler that delegates to workspace manager, data workspace, and backtest runner. No TS/Python bridge needed — everything is Python.

**Tech Stack:** Python 3.11+, protomcp (runtime), harness-app (workspace), harness-data, harness-ml

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Section 4 (Tool Surface)

---

## File Structure

```
packages/harness-server/
├── server.py                    # Entry point: pmcp dev server.py
├── tools/
│   ├── __init__.py
│   ├── project.py               # project.init
│   ├── data.py                  # data.add_source, data.transform, data.run, data.profile, data.inspect
│   ├── experiment.py            # experiment.propose, experiment.conclude (via research-loop)
│   ├── analyze.py               # analyze.diagnostics, analyze.explain, analyze.compare, analyze.discover
│   ├── versions.py              # versions.list, versions.show, versions.switch, versions.ancestry
│   └── workspace.py             # workspace.open
├── resources.py                 # MCP resources (harness://*)
├── context.py                   # Shared request context (workspace manager singleton)
└── tests/
    ├── test_tools_project.py
    ├── test_tools_data.py
    ├── test_tools_experiment.py
    ├── test_tools_analyze.py
    ├── test_tools_versions.py
    └── test_e2e.py
```

---

### Task 1: Server Entry Point + Context + Project Tool

**Files:**
- Create: `packages/harness-server/server.py`
- Create: `packages/harness-server/context.py`
- Create: `packages/harness-server/tools/__init__.py`
- Create: `packages/harness-server/tools/project.py`
- Create: `packages/harness-server/tests/test_tools_project.py`

**server.py** — protomcp entry point:
```python
"""Harness MCP Server — the agent's interface to the ML platform."""
from harness.app.workspace.discovery import find_workspace
from context import get_workspace_manager

# Import all tool groups to register them
import tools.project
import tools.data
import tools.experiment
import tools.analyze
import tools.versions
import tools.workspace

# Import resources
import resources
```

**context.py** — shared workspace manager:
```python
from pathlib import Path
from harness.app.workspace.manager import WorkspaceManager
from harness.app.workspace.discovery import find_workspace

_workspace_manager: WorkspaceManager | None = None

def get_workspace_manager() -> WorkspaceManager | None:
    global _workspace_manager
    if _workspace_manager is None:
        ws_dir = find_workspace()
        if ws_dir:
            _workspace_manager = WorkspaceManager(ws_dir)
    return _workspace_manager

def set_workspace(path: Path) -> WorkspaceManager:
    global _workspace_manager
    _workspace_manager = WorkspaceManager(path)
    return _workspace_manager

def require_workspace() -> WorkspaceManager:
    ws = get_workspace_manager()
    if ws is None:
        raise RuntimeError("No workspace found. Run 'harness init' or navigate to a workspace directory.")
    return ws
```

**tools/project.py** — `project.init`:
```python
from protomcp import tool_group, action
from pathlib import Path
from context import set_workspace

@tool_group("project")
class ProjectTools:

    @action("init")
    def init(self, task_type: str = "binary", target_column: str = "target",
             project_name: str | None = None) -> str:
        """Initialize a new Harness workspace.

        Args:
            task_type: ML task type (binary, multiclass, regression)
            target_column: Name of the target column in your data
            project_name: Optional project directory name (defaults to current directory)
        """
        from harness.app.workspace.manager import WorkspaceManager
        workspace_dir = Path.cwd() / project_name if project_name else Path.cwd()
        ws = WorkspaceManager.init(workspace_dir, task_type=task_type, target_column=target_column)
        set_workspace(workspace_dir)
        return f"Workspace initialized at {workspace_dir}\nTask type: {task_type}\nTarget: {target_column}"
```

Tests: call init, verify workspace created. Use protomcp's test utilities if available, otherwise test the underlying functions directly.

- [ ] **Steps: Write tests → implement → verify → commit**

---

### Task 2: Data Tools

**tools/data.py** — 5 data tools:
```python
@tool_group("data")
class DataTools:

    @action("add_source")
    def add_source(self, name: str, path: str, source_type: str = "file") -> str:
        """Declare a data source."""
        ws = require_workspace()
        ws.data.add_source(name, path, source_type=source_type)
        return f"Source '{name}' added: {path}"

    @action("transform")
    def transform(self, op: str, **params) -> str:
        """Add a transform step to the data pipeline."""
        ws = require_workspace()
        ws.data.add_transform({"op": op, "params": params})
        return f"Transform added: {op}"

    @action("run")
    def run(self) -> str:
        """Execute the data pipeline → produce clean dataset."""
        ws = require_workspace()
        result = ws.data.run_pipeline()
        return f"Pipeline complete: {result.row_count} rows, {result.column_count} columns"

    @action("profile")
    def profile(self) -> str:
        """Profile the clean dataset — column stats, types, quality."""
        ws = require_workspace()
        from harness.data.profiling.profiler import DataProfiler
        df = ws.data.load_clean_data()
        profiler = DataProfiler()
        profile = profiler.profile(df)
        lines = [f"Dataset: {profile.row_count} rows, {profile.column_count} columns\n"]
        for col in profile.columns:
            lines.append(f"  {col.name}: {col.inferred_type} (nulls: {col.null_pct:.1f}%, unique: {col.n_unique})")
        return "\n".join(lines)

    @action("inspect")
    def inspect(self, rows: int = 10) -> str:
        """Preview the clean dataset."""
        ws = require_workspace()
        df = ws.data.load_clean_data()
        schema = ws.data.load_schema()
        header = f"Schema: {schema['row_count']} rows, {schema['column_count']} columns\n"
        header += f"Columns: {', '.join(schema['columns'])}\n\n"
        return header + df.head(rows).to_string()
```

- [ ] **Steps: Write tests → implement → verify → commit**

---

### Task 3: Experiment Tools (THE core)

**tools/experiment.py** — `experiment.propose` and `experiment.conclude`:

```python
@tool_group("experiment")
class ExperimentTools:

    @action("propose")
    def propose(self, experiment_type: str, hypothesis: str,
                params: dict, parent: str | None = None) -> str:
        """Run a typed experiment. ONE CALL does everything.

        Args:
            experiment_type: baseline, feature, model, hyperparameter, ensemble, calibration, cv_strategy, feature_selection
            hypothesis: What you expect to happen and why
            params: Type-specific parameters (see docs)
            parent: Version to branch from (defaults to current)

        Returns:
            Structured results: version ID, metrics, comparison with parent, eval report
        """
        ws = require_workspace()
        result = ws.run_experiment(experiment_type, hypothesis, params, parent=parent)

        # Format response
        version = ws.versions.get_current()
        parent_id = parent or ws.versions.get_version(version).parent

        lines = [
            f"Version: {version}",
            f"Parent: {parent_id or 'none'}",
            f"\nMetrics:",
        ]
        for k, v in result.metrics.items():
            lines.append(f"  {k}: {v:.4f}")

        # Add parent comparison if available
        if parent_id:
            parent_meta = ws.versions.get_version(parent_id)
            if parent_meta and parent_meta.metrics:
                lines.append(f"\nComparison vs {parent_id}:")
                for k, v in result.metrics.items():
                    pv = parent_meta.metrics.get(k)
                    if pv is not None:
                        delta = v - pv
                        direction = "+" if delta > 0 else ""
                        lines.append(f"  {k}: {pv:.4f} → {v:.4f} ({direction}{delta:.4f})")

        lines.append(f"\nModels trained: {result.models_trained}")
        lines.append(f"Models cached: {result.models_cached}")
        if result.models_failed:
            lines.append(f"Models failed: {[m['name'] for m in result.models_failed]}")
        lines.append(f"Duration: {result.duration_s:.1f}s")

        return "\n".join(lines)

    @action("conclude")
    def conclude(self, conclusion: str, verdict: str, version: str | None = None) -> str:
        """Record your conclusion about the current experiment.

        Args:
            conclusion: What you learned from this experiment
            verdict: improved, degraded, inconclusive, or mixed
            version: Version to conclude (defaults to current)
        """
        ws = require_workspace()
        v = version or ws.versions.get_current()
        if not v:
            return "Error: No current version to conclude"
        ws.conclude_experiment(v, conclusion, verdict)
        return f"Version {v} concluded: {verdict}\n{conclusion}"
```

- [ ] **Steps: Write tests → implement → verify → commit**

---

### Task 4: Analysis + Version + Workspace Tools

**tools/analyze.py:**
```python
@tool_group("analyze")
class AnalyzeTools:
    @action("diagnostics")
    def diagnostics(self, version: str | None = None) -> str: ...

    @action("compare")
    def compare(self, versions: list[str]) -> str: ...

    @action("explain")
    def explain(self, version: str | None = None) -> str: ...

    @action("discover")
    def discover(self) -> str: ...
```

**tools/versions.py:**
```python
@tool_group("versions")
class VersionTools:
    @action("list")
    def list_versions(self) -> str: ...

    @action("show")
    def show(self, version: str) -> str: ...

    @action("switch")
    def switch(self, version: str) -> str: ...

    @action("ancestry")
    def ancestry(self, version: str | None = None) -> str: ...
```

**tools/workspace.py:**
```python
@tool_group("workspace")
class WorkspaceTools:
    @action("open")
    def open(self, path: str) -> str: ...
```

- [ ] **Steps: Write tests → implement → verify → commit**

---

### Task 5: MCP Resources

**resources.py:**
```python
from protomcp import resource

@resource("harness://data/schema")
def data_schema() -> str:
    """Current clean dataset schema."""

@resource("harness://versions/tree")
def versions_tree() -> str:
    """Full version tree structure."""

@resource("harness://versions/current")
def current_version() -> str:
    """Current version config + metrics."""

@resource("harness://models/available")
def available_models() -> str:
    """Model types, default params, task compatibility."""

@resource("harness://tasks/supported")
def supported_tasks() -> str:
    """Task types, available metrics."""
```

- [ ] **Steps: Implement → verify → commit**

---

### Task 6: Full E2E — Agent Workflow Simulation

The most important test. Simulates what an agent actually does:

```python
class TestE2EAgentWorkflow:
    def test_full_agent_session(self, tmp_path):
        """Simulate a complete agent session:
        1. project.init
        2. data.add_source + data.run
        3. experiment.propose (baseline)
        4. experiment.conclude
        5. experiment.propose (add model)
        6. experiment.conclude
        7. versions.list
        8. analyze.compare
        9. versions.switch
        """
```

- [ ] **Steps: Write e2e → verify → commit**

```bash
git commit -m "feat(harness-server): MCP server — 17 tools + 5 resources (THE agent interface)"
```
