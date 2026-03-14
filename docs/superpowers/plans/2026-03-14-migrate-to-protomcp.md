# Migrate harness-plugin from FastMCP to protomcp

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the FastMCP-based MCP server (`mcp_server.py`) with protomcp's tool_group/action pattern, eliminating the god-function signatures, `_safe_tool` wrapper, manual validation, and Studio lifecycle code.

**Architecture:** Each handler file becomes a `@tool_group` class with `@action` methods. The 1200-line `mcp_server.py` is replaced by a ~50-line `server.py` that registers middleware, telemetry, sidecars, and server context. Business logic in handler files is unchanged — only the registration/dispatch layer changes. The server runs via `pmcp dev server.py` instead of `uv run harness-ml`.

**Tech Stack:** protomcp Python SDK (from `/tmp/protomcp/sdk/python/`), pmcp binary (built at `/tmp/protomcp/bin/pmcp`)

**Key constraint:** The E2E test (`tests/test_e2e_mcp.py`) must pass after migration. It drives the server via stdio JSON-RPC and validates the full workflow.

---

## File Structure

### Files to create

| File | Responsibility |
|------|---------------|
| `packages/harness-plugin/src/harnessml/plugin/server.py` | New entry point — registers middleware, telemetry, sidecar, server_context, calls `protomcp.run()` |
| `packages/harness-plugin/src/harnessml/plugin/pmcp_middleware.py` | Local middleware: error formatting, auto-install, event emission |
| `packages/harness-plugin/src/harnessml/plugin/pmcp_telemetry.py` | Telemetry sink for Studio event store |
| `packages/harness-plugin/src/harnessml/plugin/pmcp_context.py` | Server context resolver for `project_dir` |
| `packages/harness-plugin/src/harnessml/plugin/pmcp_sidecar.py` | Studio sidecar definition |

### Files to modify

| File | Change |
|------|--------|
| `packages/harness-plugin/src/harnessml/plugin/handlers/models.py` | Add `@tool_group("models")` class wrapping existing `_handle_*` functions |
| `packages/harness-plugin/src/harnessml/plugin/handlers/data.py` | Same pattern |
| `packages/harness-plugin/src/harnessml/plugin/handlers/features.py` | Same pattern |
| `packages/harness-plugin/src/harnessml/plugin/handlers/experiments.py` | Same pattern (async handlers need sync conversion since protomcp is sync) |
| `packages/harness-plugin/src/harnessml/plugin/handlers/config.py` | Same pattern |
| `packages/harness-plugin/src/harnessml/plugin/handlers/pipeline.py` | Same pattern (async handlers need sync conversion) |
| `packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py` | Same pattern |
| `packages/harness-plugin/src/harnessml/plugin/handlers/competitions.py` | Same pattern |
| `packages/harness-plugin/pyproject.toml` | Replace `mcp` dependency with `protomcp`, update entry point |
| `tests/test_e2e_mcp.py` | Update server startup to use `pmcp dev server.py` |

### Files to delete

| File | Reason |
|------|--------|
| `packages/harness-plugin/src/harnessml/plugin/mcp_server.py` | Replaced by `server.py` + tool_groups + middleware |
| `packages/harness-plugin/src/harnessml/plugin/handlers/_validation.py` | Replaced by declarative validation on `@action` |

### Files to keep unchanged

| File | Reason |
|------|--------|
| `packages/harness-plugin/src/harnessml/plugin/handlers/_common.py` | Still needed for `parse_json_param`, `resolve_project_dir` (used inside handlers) |
| `packages/harness-plugin/src/harnessml/plugin/event_emitter.py` | Still the Studio event emission backend |
| `packages/harness-plugin/src/harnessml/plugin/setup.py` | harness-setup CLI — independent of MCP framework |
| All `harness-core` files | No changes — business logic is untouched |
| All `harness-studio` files | No changes |

---

## Async-to-Sync Migration Notes

protomcp handlers run in threads (sync). 6 harness-ml handlers are currently async:

- `experiments._handle_run` — uses `asyncio.run_coroutine_threadsafe` + `loop.run_in_executor`
- `experiments._handle_quick_run` — same pattern
- `experiments._handle_explore` — same pattern
- `features._handle_discover` — same pattern
- `pipeline._handle_run_backtest` — same pattern
- `pipeline._handle_compare_targets` — same pattern

These all follow the same structure: they use `await loop.run_in_executor(None, sync_function)` to run CPU-bound work off the event loop, and `asyncio.run_coroutine_threadsafe(ctx.report_progress(...), loop)` for progress.

In protomcp, handlers already run in threads, so the `run_in_executor` wrapper is unnecessary — just call the sync function directly. Progress reporting uses `ctx.report_progress(progress, total, message)` which is already sync in protomcp's `ToolContext`.

The conversion for each async handler is:
1. Remove `async` keyword
2. Remove `asyncio.get_running_loop()` and `loop.run_in_executor()` — call sync function directly
3. Replace `await ctx.report_progress(...)` with `ctx.report_progress(...)` (sync)
4. Replace `make_progress_callback(ctx, loop)` with a simpler callback that calls `ctx.report_progress()` directly

---

## Chunk 1: Infrastructure (server.py, middleware, telemetry, context, sidecar)

### Task 1: Install protomcp Python SDK into workspace

**Files:**
- Modify: `packages/harness-plugin/pyproject.toml`

- [ ] **Step 1: Add protomcp as a dependency**

In `packages/harness-plugin/pyproject.toml`, replace the `mcp` dependency with `protomcp`. Since protomcp isn't on PyPI yet, use a local path reference:

```toml
dependencies = [
    "harness-core>=0.1.0,<1.0",
    "harness-studio>=0.1.0,<1.0",
    "protomcp @ file:///tmp/protomcp/sdk/python",
    "click>=8.0",
]
```

Remove the `[project.scripts]` `harness-ml` entry (will be replaced later):
```toml
[project.scripts]
harness-setup = "harnessml.plugin.setup:main"
```

- [ ] **Step 2: Run uv sync to install**

Run: `cd ~/easyml && uv sync`
Expected: protomcp installs alongside existing packages

- [ ] **Step 3: Verify import**

Run: `cd ~/easyml && uv run python -c "from protomcp import tool_group, action, ToolResult, local_middleware, telemetry_sink, server_context, sidecar, configure; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/harness-plugin/pyproject.toml uv.lock
git commit -m "build: add protomcp dependency to harness-plugin"
```

---

### Task 2: Create server context for project_dir

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/pmcp_context.py`
- Test: Verified via E2E test later

- [ ] **Step 1: Create the context resolver**

```python
"""Server context: resolve project_dir for all tool handlers."""
from __future__ import annotations

import os
from pathlib import Path

from protomcp import server_context


@server_context("project_dir", expose=True)
def resolve_project_dir(args: dict) -> Path:
    """Resolve project_dir from explicit param, env var, or cwd."""
    explicit = args.pop("project_dir", None)
    if explicit:
        return Path(explicit).resolve()
    env = os.environ.get("HARNESS_PROJECT_DIR")
    if env:
        return Path(env).resolve()
    return Path.cwd()
```

- [ ] **Step 2: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/pmcp_context.py
git commit -m "feat: add protomcp server context for project_dir resolution"
```

---

### Task 3: Create local middleware (error formatting + auto-install)

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/pmcp_middleware.py`

- [ ] **Step 1: Create the middleware file**

```python
"""Local middleware for harness-ml: error formatting and auto-install."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import traceback

from protomcp import ToolResult, local_middleware

# Map of Python import names to pip package names (where they differ)
_IMPORT_TO_PACKAGE = {
    "pytorch_tabnet": "pytorch-tabnet",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "google": "google-api-python-client",
    "googleapiclient": "google-api-python-client",
    "google_auth_oauthlib": "google-auth-oauthlib",
}


def _auto_install(module_name: str) -> bool:
    """Auto-install a missing package."""
    package = _IMPORT_TO_PACKAGE.get(module_name, module_name)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True, timeout=120, check=True,
        )
        return True
    except Exception:
        return False


@local_middleware(priority=10)
def auto_install_middleware(ctx, tool_name, args, next_handler):
    """Catch ModuleNotFoundError, auto-install, retry once."""
    try:
        return next_handler(ctx, args)
    except ModuleNotFoundError as e:
        package = _IMPORT_TO_PACKAGE.get(e.name, e.name)
        if _auto_install(e.name):
            try:
                return next_handler(ctx, args)
            except ModuleNotFoundError as retry_err:
                return ToolResult(
                    result=f"**Error**: {retry_err} (after installing {package})",
                    is_error=True,
                    error_code="MISSING_PACKAGE",
                )
        return ToolResult(
            result=f"**Error**: Missing package `{package}`. Auto-install failed.",
            is_error=True,
            error_code="MISSING_PACKAGE",
        )


@local_middleware(priority=90)
def error_format_middleware(ctx, tool_name, args, next_handler):
    """Convert unhandled exceptions to markdown error strings."""
    try:
        return next_handler(ctx, args)
    except json.JSONDecodeError as e:
        return ToolResult(
            result=f"**Error**: Invalid JSON input: {e}",
            is_error=True,
            error_code="INVALID_JSON",
        )
    except ValueError as e:
        return ToolResult(
            result=f"**Error**: {e}",
            is_error=True,
            error_code="VALIDATION_ERROR",
        )
    except Exception as e:
        tb_lines = traceback.format_exception(e)
        tb_str = "".join(tb_lines)[-2000:]
        return ToolResult(
            result=f"**Error**: Unexpected error in `{tool_name}`: {e}\n\n```\n{tb_str}\n```",
            is_error=True,
            error_code="INTERNAL_ERROR",
        )
```

- [ ] **Step 2: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/pmcp_middleware.py
git commit -m "feat: add protomcp local middleware for error handling and auto-install"
```

---

### Task 4: Create telemetry sink for Studio events

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/pmcp_telemetry.py`

- [ ] **Step 1: Create the telemetry file**

```python
"""Telemetry sink: forward tool call events to Studio's SQLite event store."""
from __future__ import annotations

import os
import threading

from protomcp import ToolCallEvent, telemetry_sink

_emitter = None
_init_lock = threading.Lock()


def _get_emitter():
    global _emitter
    if _emitter is not None:
        return _emitter
    with _init_lock:
        if _emitter is not None:
            return _emitter
        from harnessml.plugin.event_emitter import create_emitter
        _emitter = create_emitter()
    return _emitter


@telemetry_sink
def studio_telemetry(event: ToolCallEvent):
    """Forward tool call events to Studio's SQLite event store."""
    emitter = _get_emitter()
    if not emitter.enabled:
        return

    if event.phase == "start":
        emitter.set_project(str(event.args.get("project_dir", os.getcwd())))
        emitter.set_current(event.tool_name, event.action)
        emitter.emit(
            tool=event.tool_name, action=event.action,
            params={k: v for k, v in event.args.items() if k != "ctx"},
            result="", duration_ms=0, status="running",
        )
    elif event.phase == "success":
        emitter.clear_current()
        emitter.emit(
            tool=event.tool_name, action=event.action,
            params={}, result=event.result[:20000],
            duration_ms=event.duration_ms, status="success",
        )
    elif event.phase == "error":
        emitter.clear_current()
        emitter.emit(
            tool=event.tool_name, action=event.action,
            params={}, result=str(event.error)[:20000],
            duration_ms=event.duration_ms, status="error",
        )
    elif event.phase == "progress":
        emitter.progress(
            current=event.progress, total=event.total,
            message=event.message,
            tool_override=event.tool_name, action_override=event.action,
        )
```

- [ ] **Step 2: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/pmcp_telemetry.py
git commit -m "feat: add protomcp telemetry sink for Studio event emission"
```

---

### Task 5: Create Studio sidecar

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/pmcp_sidecar.py`

- [ ] **Step 1: Create the sidecar file**

```python
"""Sidecar: auto-start Harness Studio alongside the MCP server."""
from __future__ import annotations

import os
import sys

from protomcp import sidecar

_STUDIO_PORT = int(os.environ.get("HARNESS_STUDIO_PORT", "8421"))


@sidecar(
    name="harness-studio",
    command=[sys.executable, "-m", "harnessml.studio.cli", "--port", str(_STUDIO_PORT)],
    health_check=f"http://localhost:{_STUDIO_PORT}/api/health",
    start_on="first_tool_call",
    health_timeout=5.0,
)
def studio_sidecar():
    """Harness Studio companion dashboard."""
    pass
```

- [ ] **Step 2: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/pmcp_sidecar.py
git commit -m "feat: add protomcp sidecar for Studio auto-start"
```

---

### Task 6: Create the new server entry point

**Files:**
- Create: `packages/harness-plugin/src/harnessml/plugin/server.py`

- [ ] **Step 1: Create the server file**

This file imports all the infrastructure (middleware, telemetry, sidecar, context) and all handler modules (which register tool groups at import time), then calls `protomcp.run()`.

```python
"""HarnessML MCP server — protomcp entry point.

Run via: pmcp dev packages/harness-plugin/src/harnessml/plugin/server.py
"""
from __future__ import annotations

from protomcp import configure, run

# Import infrastructure — registrations happen at import time
import harnessml.plugin.pmcp_context      # noqa: F401 — registers server_context
import harnessml.plugin.pmcp_middleware   # noqa: F401 — registers local_middleware
import harnessml.plugin.pmcp_telemetry   # noqa: F401 — registers telemetry_sink
import harnessml.plugin.pmcp_sidecar     # noqa: F401 — registers sidecar

# Import all handler modules — each registers a @tool_group
import harnessml.plugin.handlers.models        # noqa: F401
import harnessml.plugin.handlers.data          # noqa: F401
import harnessml.plugin.handlers.features      # noqa: F401
import harnessml.plugin.handlers.experiments   # noqa: F401
import harnessml.plugin.handlers.config        # noqa: F401
import harnessml.plugin.handlers.competitions  # noqa: F401
import harnessml.plugin.handlers.notebook      # noqa: F401
import harnessml.plugin.handlers.pipeline      # noqa: F401

# Run the protomcp event loop
run()
```

- [ ] **Step 2: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/server.py
git commit -m "feat: add protomcp server entry point"
```

---

## Chunk 2: Migrate handler files to @tool_group

The pattern for each handler file is the same:

1. Keep all existing `_handle_*` functions as-is (they contain the business logic)
2. Add a `@tool_group` class at the bottom that wraps each `_handle_*` as an `@action`
3. Remove the `ACTIONS` dict and `dispatch()` function
4. For async handlers: convert to sync (remove async/await, call sync functions directly, use sync `ctx.report_progress()`)
5. Move `requires` and `enum_fields` validation from manual `validate_required()`/`validate_enum()` calls into `@action` decorator
6. Remove `**_kwargs` from handler signatures — protomcp only passes declared params
7. `project_dir` parameter is removed from individual handlers — it's injected by server context

**Critical:** Each handler function currently receives `project_dir` as a string and calls `resolve_project_dir()`. After migration, `project_dir` arrives as an already-resolved `Path` from the server context. Handler functions that do `resolve_project_dir(project_dir)` should be changed to just use `project_dir` directly. However, `_common.resolve_project_dir()` also validates that the config dir exists — this validation should move into the handlers or server context.

### Task 7: Migrate handlers/notebook.py (simplest — 5 actions, no async)

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py`

- [ ] **Step 1: Read the current file**

Read `packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py` in full.

- [ ] **Step 2: Add tool_group class, keep existing _handle_* functions**

Add at the bottom of the file, replacing the `ACTIONS` dict and `dispatch` function:

```python
from protomcp import tool_group, action, ToolResult
from protomcp.context import ToolContext

_VALID_TYPES_LIST = sorted(_VALID_TYPES)


@tool_group("notebook", description="Project notebook for persistent learnings across sessions.")
class NotebookTools:

    @action("write", description="Add a notebook entry.",
            requires=["content"],
            enum_fields={"type": _VALID_TYPES_LIST})
    def write(self, type: str | None = None, content: str | None = None,
              tags: str | None = None, experiment_id: str | None = None,
              project_dir=None) -> str:
        return _handle_write(type=type, content=content, tags=tags,
                             experiment_id=experiment_id, project_dir=project_dir)

    @action("read", description="Read entries (newest first, excludes struck).")
    def read(self, type: str | None = None, tags: str | None = None,
             page: int | None = None, per_page: int | None = None,
             project_dir=None) -> str:
        return _handle_read(type=type, tags=tags, page=page,
                            per_page=per_page, project_dir=project_dir)

    @action("search", description="Full-text search notebook entries.",
            requires=["query"])
    def search(self, query: str | None = None, project_dir=None) -> str:
        return _handle_search(query=query, project_dir=project_dir)

    @action("strike", description="Hide an entry with a reason.",
            requires=["entry_id", "reason"])
    def strike(self, entry_id: str | None = None, reason: str | None = None,
               project_dir=None) -> str:
        return _handle_strike(entry_id=entry_id, reason=reason,
                              project_dir=project_dir)

    @action("summary", description="Get current theory, plan, recent findings.")
    def summary(self, project_dir=None) -> str:
        return _handle_summary(project_dir=project_dir)
```

Remove the old `ACTIONS` dict and `async def dispatch(...)` function.

- [ ] **Step 3: Verify notebook handlers still work standalone**

Run: `cd ~/easyml && uv run python -c "from harnessml.plugin.handlers.notebook import NotebookTools; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/handlers/notebook.py
git commit -m "feat: migrate notebook handler to protomcp tool_group"
```

---

### Task 8: Migrate handlers/models.py (10 actions, no async)

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/handlers/models.py`

- [ ] **Step 1: Read the current file in full**

- [ ] **Step 2: Add @tool_group class wrapping all 10 actions**

Same pattern as notebook. The actions are: `add`, `update`, `remove`, `list`, `show`, `presets`, `add_batch`, `update_batch`, `remove_batch`, `clone`. Each `@action` method delegates to the existing `_handle_*` function. Move `validate_required` checks to `requires=[]` on the decorator.

Key params: `name`, `model_type`, `preset`, `features` (list[str]), `params` (str | dict), `active` (bool), `mode`, `items` (str | list), `new_name`, etc.

- [ ] **Step 3: Remove old ACTIONS dict and dispatch function**

- [ ] **Step 4: Test import**

Run: `cd ~/easyml && uv run python -c "from harnessml.plugin.handlers.models import ModelsTools; print('OK')"`

- [ ] **Step 5: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/handlers/models.py
git commit -m "feat: migrate models handler to protomcp tool_group"
```

---

### Task 9: Migrate handlers/config.py (13 actions, no async)

Same pattern. Actions: `init`, `update_data`, `ensemble`, `backtest`, `show`, `check_guardrails`, `exclude_columns`, `set_denylist`, `add_target`, `list_targets`, `set_target`, `studio`, `suggest_cv`.

- [ ] **Step 1: Read current file**
- [ ] **Step 2: Add @tool_group class**
- [ ] **Step 3: Remove old dispatch**
- [ ] **Step 4: Test import**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: migrate config handler to protomcp tool_group"
```

---

### Task 10: Migrate handlers/features.py (7 actions, 1 async)

Actions: `add`, `add_batch`, `test_transformations`, `discover`, `diversity`, `auto_search`, `prune`.

`_handle_discover` is async — convert to sync:
- Remove `async` keyword
- The handler uses `await loop.run_in_executor(None, ...)` — replace with direct sync call
- Remove `ctx` parameter (discover doesn't report progress, it just uses ctx for the executor pattern)

- [ ] **Step 1: Read current file**
- [ ] **Step 2: Convert _handle_discover to sync**
- [ ] **Step 3: Add @tool_group class**
- [ ] **Step 4: Remove old dispatch**
- [ ] **Step 5: Test import**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat: migrate features handler to protomcp tool_group"
```

---

### Task 11: Migrate handlers/data.py (35 actions, no async)

Largest handler. Actions include: `add`, `validate`, `fill_nulls`, `drop_duplicates`, `detect_outliers`, `drop_rows`, `rename`, `derive_column`, `inspect`, `profile`, `list_features`, `status`, `list_sources`, `add_source`, `add_view`, `update_view`, `remove_view`, `list_views`, `preview_view`, `set_features_view`, `view_dag`, `add_sources_batch`, `fill_nulls_batch`, `add_views_batch`, `check_freshness`, `refresh`, `refresh_all`, `validate_source`, `sample`, `restore`, `fetch_url`, `upload_drive`, `upload_kaggle`, `snapshot`, `restore_snapshot`.

Same pattern — each action becomes an `@action` method delegating to `_handle_*`.

- [ ] **Step 1: Read current file**
- [ ] **Step 2: Add @tool_group class with all 35 actions**
- [ ] **Step 3: Remove old dispatch**
- [ ] **Step 4: Test import**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: migrate data handler to protomcp tool_group"
```

---

### Task 12: Migrate handlers/experiments.py (10 actions, 3 async)

Actions: `create`, `write_overlay`, `run`, `promote`, `quick_run`, `explore`, `promote_trial`, `compare`, `journal`, `log_result`.

3 async handlers need sync conversion:
- `_handle_run`: uses `loop.run_in_executor` → call sync `cw.experiment_run()` directly. Uses `ctx.report_progress()` → protomcp `ToolContext.report_progress()` is already sync. Replace `make_progress_callback(ctx, loop)` with a direct callback using `ctx.report_progress()`.
- `_handle_quick_run`: same pattern
- `_handle_explore`: same pattern

Each converted handler should accept `ctx: ToolContext` parameter (protomcp injects it if the handler signature includes `ctx`).

- [ ] **Step 1: Read current file**
- [ ] **Step 2: Convert 3 async handlers to sync**
- [ ] **Step 3: Add @tool_group class**
- [ ] **Step 4: Remove old dispatch**
- [ ] **Step 5: Test import**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat: migrate experiments handler to protomcp tool_group"
```

---

### Task 13: Migrate handlers/pipeline.py (15 actions, 2 async)

Actions: `progress`, `run_backtest`, `predict`, `diagnostics`, `list_runs`, `show_run`, `compare_runs`, `compare_latest`, `compare_targets`, `explain`, `inspect_predictions`, `export_notebook`, `clear_cache`, `model_correlation`, `residual_analysis`.

2 async handlers:
- `_handle_run_backtest`: main training pipeline — convert same way as experiments
- `_handle_compare_targets`: iterates over targets with progress — convert to sync

- [ ] **Step 1: Read current file**
- [ ] **Step 2: Convert 2 async handlers to sync**
- [ ] **Step 3: Add @tool_group class**
- [ ] **Step 4: Remove old dispatch**
- [ ] **Step 5: Test import**
- [ ] **Step 6: Commit**

```bash
git commit -m "feat: migrate pipeline handler to protomcp tool_group"
```

---

### Task 14: Migrate handlers/competitions.py (13 actions, no async)

Actions: `create`, `list_formats`, `simulate`, `standings`, `round_probs`, `generate_brackets`, `score_bracket`, `adjust`, `explain`, `profiles`, `confidence`, `export`, `list_strategies`.

- [ ] **Step 1: Read current file**
- [ ] **Step 2: Add @tool_group class**
- [ ] **Step 3: Remove old dispatch**
- [ ] **Step 4: Test import**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat: migrate competitions handler to protomcp tool_group"
```

---

## Chunk 3: Wire up, test, and clean up

### Task 15: Update pyproject.toml entry point

**Files:**
- Modify: `packages/harness-plugin/pyproject.toml`

- [ ] **Step 1: Remove old harness-ml script entry, add new one**

The old entry point was:
```toml
harness-ml = "harnessml.plugin.mcp_server:main"
```

For protomcp, the server is run via `pmcp dev server.py`. But for MCP client config, we need a command that Claude Code can run. Create a thin wrapper script entry point:

```toml
[project.scripts]
harness-setup = "harnessml.plugin.setup:main"
```

The `.mcp.json` will change from:
```json
{"command": "uv", "args": ["run", "--directory", "...", "harness-ml"]}
```
to:
```json
{"command": "/tmp/protomcp/bin/pmcp", "args": ["run", "packages/harness-plugin/src/harnessml/plugin/server.py"]}
```

- [ ] **Step 2: Commit**

```bash
git add packages/harness-plugin/pyproject.toml
git commit -m "build: update entry points for protomcp migration"
```

---

### Task 16: Update E2E test to use pmcp

**Files:**
- Modify: `tests/test_e2e_mcp.py`

- [ ] **Step 1: Update server startup in the MCP fixture**

Change the `Popen` command from:
```python
["uv", "run", "--directory", EASYML_DIR, "harness-ml"]
```
to:
```python
["/tmp/protomcp/bin/pmcp", "run", os.path.join(EASYML_DIR, "packages/harness-plugin/src/harnessml/plugin/server.py")]
```

The `HARNESS_PROJECT_DIR` env var stays the same.

- [ ] **Step 2: Run the E2E test**

Run: `cd ~/easyml && uv run pytest tests/test_e2e_mcp.py -v -s`
Expected: All 33 tests pass

- [ ] **Step 3: Fix any failures**

Common issues to expect:
- Parameter name mismatches (protomcp may filter unknown params differently)
- `ctx` injection differences
- Progress callback signature changes
- Return type differences (ToolResult vs string)

Debug by reading error messages, checking handler signatures match what the E2E test sends.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_mcp.py
git commit -m "test: update E2E test for protomcp migration"
```

---

### Task 17: Run full test suite

- [ ] **Step 1: Run all existing tests**

Run: `cd ~/easyml && uv run pytest packages/harness-core/tests/ packages/harness-plugin/tests/ packages/harness-sports/tests/ packages/harness-studio/tests/ -q --tb=short`
Expected: All 3099+ tests pass (harness-plugin tests may need updates if they import from mcp_server.py)

- [ ] **Step 2: Fix any harness-plugin test failures**

Plugin tests likely import `from harnessml.plugin.mcp_server import mcp`. These need updating to work with the new server structure.

- [ ] **Step 3: Commit fixes**

```bash
git commit -m "fix: update plugin tests for protomcp migration"
```

---

### Task 18: Delete old files

- [ ] **Step 1: Delete mcp_server.py**

```bash
git rm packages/harness-plugin/src/harnessml/plugin/mcp_server.py
```

- [ ] **Step 2: Delete _validation.py**

```bash
git rm packages/harness-plugin/src/harnessml/plugin/handlers/_validation.py
```

- [ ] **Step 3: Verify no remaining imports of deleted files**

Run: `cd ~/easyml && grep -rn "mcp_server\|_validation" packages/harness-plugin/src/ --include="*.py" | grep -v __pycache__`
Expected: No matches (or only the new server.py references)

- [ ] **Step 4: Run tests again**

Run: `cd ~/easyml && uv run pytest tests/test_e2e_mcp.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git commit -m "chore: remove FastMCP server and manual validation layer"
```

---

### Task 19: Update harness-setup to use pmcp

**Files:**
- Modify: `packages/harness-plugin/src/harnessml/plugin/setup.py`

- [ ] **Step 1: Update .mcp.json generation**

The setup command writes `.mcp.json`. Update it to use `pmcp run` instead of `uv run harness-ml`:

Find the section that writes the MCP config and change the command/args to point to `pmcp run server.py`.

- [ ] **Step 2: Test setup**

Run: `cd ~/easyml && rm -rf harness-demo && uv run harness-setup --no-claude --no-studio`
Expected: `.mcp.json` written with pmcp command

- [ ] **Step 3: Commit**

```bash
git add packages/harness-plugin/src/harnessml/plugin/setup.py
git commit -m "feat: update harness-setup to use pmcp for MCP server"
```

---

### Task 20: Final verification

- [ ] **Step 1: Full test suite**

Run: `cd ~/easyml && uv run pytest packages/harness-core/tests/ packages/harness-plugin/tests/ packages/harness-sports/tests/ packages/harness-studio/tests/ tests/test_e2e_mcp.py -q`
Expected: All tests pass

- [ ] **Step 2: Manual smoke test with Studio**

```bash
cd ~/easyml
rm -rf harness-demo
# Kill any stale Studio
lsof -ti:8421 | xargs kill 2>/dev/null
# Run setup
uv run harness-setup --no-claude --no-studio
# Start Studio
uv run harness-studio --project-dir harness-demo --port 8421 &
# Verify MCP server starts
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' | /tmp/protomcp/bin/pmcp run packages/harness-plugin/src/harnessml/plugin/server.py
```

Expected: JSON response with `serverInfo`

- [ ] **Step 3: Commit and push**

```bash
git add -A
git commit -m "feat: complete protomcp migration — all tests passing"
git push origin feat/sync-split-repo-changes
```
