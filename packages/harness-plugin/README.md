# Harness Plugin

MCP server for [HarnessML](https://github.com/msilverblatt/harness-ml), built on [protomcp](https://github.com/msilverblatt/protomcp). Provides AI-driven ML experimentation through `@tool_group` classes with per-action schemas.

## Architecture

```
server.py              # protomcp entry point (25 lines)
pmcp_middleware.py     # Error formatting + auto-install middleware
pmcp_telemetry.py     # Studio event emission telemetry sink
pmcp_sidecar.py       # Studio auto-start sidecar
handlers/
├── data.py            # 34 actions: add, validate, fill_nulls, inspect, profile, views...
├── features.py        #  7 actions: add, add_batch, test, discover, diversity, auto_search, prune
├── models.py          # 10 actions: add, update, remove, list, show, presets, batch ops, clone
├── config.py          # 13 actions: init, update_data, ensemble, backtest, show, targets...
├── pipeline.py        # 15 actions: run_backtest, predict, diagnostics, compare, explain...
├── experiments.py     # 10 actions: create, write_overlay, run, promote, quick_run, explore...
├── notebook.py        #  5 actions: write, read, search, strike, summary
├── competitions.py    # 13 actions: create, simulate, standings, brackets, score, adjust...
├── _validation.py     # Runtime validation helpers (fuzzy enum matching, required params)
└── _common.py         # Shared helpers (resolve_project_dir, parse_json_param)
```

## Key Design

**Tool groups with per-action schemas**: Each handler is a `@tool_group` class. Each `@action` method has its own typed signature, so the LLM sees clean per-action schemas instead of a monolithic parameter blob.

**Middleware**: Cross-cutting concerns (error formatting, auto-install of missing packages) are handled by protomcp local middleware — no manual wrappers.

**Telemetry**: Tool call events (start, success, error, progress) flow to Studio's SQLite event store via a protomcp telemetry sink.

**Sidecar**: Studio auto-starts as a companion process on first tool call via protomcp's sidecar system.

**Hot-reload**: `pmcp dev server.py` watches for file changes and reloads all handler modules automatically.

## Tools

8 MCP tools exposing 107 actions:

| Tool | Actions | Purpose |
|------|---------|---------|
| `data` | 34 | Data ingestion, validation, profiling, views, sources |
| `features` | 7 | Feature engineering: add, batch, discover, auto-search, prune |
| `models` | 10 | Model configuration: add, update, remove, presets, clone |
| `configure` | 13 | Project setup: init, ensemble, backtest, targets, guardrails |
| `pipeline` | 15 | Execution: backtest, predict, diagnostics, compare, explain |
| `experiments` | 10 | Experiment management: create, run, promote, compare, journal |
| `notebook` | 5 | Project notebook: theory, plan, finding, search, summary |
| `competitions` | 13 | Tournament simulation: brackets, scoring, strategies |

## Handler Pattern

Each handler module follows the same pattern:

```python
from protomcp import tool_group, action

@tool_group("models", description="Manage models in the project.")
class ModelsTools:

    @action("add", description="Add a model.", requires=["name"])
    def add(self, name, model_type=None, preset=None, features=None, ...):
        return _handle_add(name=name, model_type=model_type, ...)

    @action("list", description="List all models.")
    def list(self, project_dir=None):
        return _handle_list(project_dir=project_dir)
```

## Setup

```json
{
  "mcpServers": {
    "harness-ml": {
      "command": "pmcp",
      "args": ["run", "/path/to/harness-ml/packages/harness-plugin/src/harnessml/plugin/server.py"]
    }
  }
}
```

For dev mode with hot-reload:

```json
{
  "mcpServers": {
    "harness-ml": {
      "command": "pmcp",
      "args": ["dev", "/path/to/harness-ml/packages/harness-plugin/src/harnessml/plugin/server.py"]
    }
  }
}
```

## Extending

**Adding a new action** to an existing handler:

1. Add handler function `_handle_my_action(**kwargs) -> str` in the handler module
2. Add `@action("my_action")` method to the `@tool_group` class
3. Hot-reload picks it up automatically in dev mode

**Adding a new tool group**:

1. Create handler module in `handlers/` with `@tool_group` class
2. Import the module in `server.py`
3. Hot-reload picks it up in dev mode; `pmcp run` needs restart

When adding new model config fields, update in 2 places:
1. `harness-core` config_writer (`add_model`, `update_model`)
2. `harness-plugin` handler (`handlers/models.py` — both `_handle_*` and `@action` method)

## Testing

```bash
uv run pytest packages/harness-plugin/tests/ -v
```
