# Harness Plugin

MCP (Model Context Protocol) server for [HarnessML](https://github.com/msilverblatt/harness-ml). Provides AI-driven ML experimentation through a thin async dispatcher with hot-reloadable handlers.

## Architecture

```
mcp_server.py          # Tool signatures + docstrings (thin dispatcher)
handlers/
├── data.py            # 19 actions: add, validate, fill_nulls, inspect, profile, views...
├── features.py        #  6 actions: add, add_batch, test, discover, diversity, auto_search
├── models.py          # 10 actions: add, update, remove, list, show, presets, batch ops, clone
├── config.py          # 12 actions: init, update_data, ensemble, backtest, show, targets...
├── pipeline.py        # 13 actions: run_backtest, predict, diagnostics, compare, explain...
├── experiments.py     # 10 actions: create, write_overlay, run, promote, quick_run, explore...
├── competitions.py    # 13 actions: create, simulate, standings, brackets, score, adjust...
├── _validation.py     # Fuzzy enum matching with "Did you mean?" hints
└── _common.py         # Shared helpers (resolve_project_dir, parse_json_param)
```

## Key Design Principles

**Thin dispatcher**: `mcp_server.py` contains only tool signatures and docstrings. All business logic lives in `handlers/*.py`.

**Hot-reload in dev mode**: Set `HARNESS_DEV=1` to enable hot-reloading of handler code. Changes to handler files take effect immediately without restarting the server. Changes to tool signatures or docstrings in `mcp_server.py` still require a restart.

**Fuzzy enum matching**: Invalid action names get helpful "Did you mean?" suggestions using edit distance matching. Cross-parameter hints guide users toward correct tool usage.

**Event emission**: Every tool call emits a structured event to SQLite for Studio observability. Emission is fail-safe -- it never blocks or breaks tool execution.

## Tools

7 MCP tools exposing ~83 actions:

| Tool | Actions | Purpose |
|------|---------|---------|
| `data` | 19 | Data ingestion, validation, profiling, views, sources |
| `features` | 6 | Feature engineering: add, batch, discover, auto-search |
| `models` | 10 | Model configuration: add, update, remove, presets, clone |
| `configure` | 12 | Project setup: init, ensemble, backtest, targets, guardrails |
| `pipeline` | 13 | Execution: backtest, predict, diagnostics, compare, explain |
| `experiments` | 10 | Experiment management: create, run, promote, compare, journal |
| `competitions` | 13 | Tournament simulation: brackets, scoring, strategies |

## Handler Dispatch Pattern

Each handler module follows the same pattern:

```python
ACTIONS = {
    "add": _handle_add,
    "remove": _handle_remove,
    ...
}

async def dispatch(action: str, **kwargs) -> str:
    err = validate_enum(action, set(ACTIONS), "action")
    if err:
        return err  # "Did you mean 'add'?"
    result = ACTIONS[action](**kwargs)
    if asyncio.iscoroutine(result):
        result = await result
    return result
```

## Setup

Add to your `.mcp.json` (Claude Desktop, Claude Code, or any MCP host):

```json
{
  "mcpServers": {
    "harness-ml": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/harness-ml",
        "run", "harness-ml"
      ]
    }
  }
}
```

For dev mode with hot-reload:

```json
{
  "mcpServers": {
    "harness-ml": {
      "command": "uv",
      "args": [
        "--directory", "/path/to/harness-ml",
        "run", "harness-ml"
      ],
      "env": {
        "HARNESS_DEV": "1"
      }
    }
  }
}
```

## Event Emission

Tool calls emit events to SQLite for Harness Studio observability:

- Event type, tool name, action, parameters, result summary
- Timestamps and session tracking
- Fail-safe: exceptions in emission are swallowed, never breaking tool execution

## Extending

**Adding a new action** to an existing handler:

1. Add handler function `_handle_my_action(**kwargs) -> str` in the handler module
2. Add entry to the `ACTIONS` dict
3. No server restart needed in dev mode

**Adding a new tool**:

1. Add tool function with signature and docstring in `mcp_server.py`
2. Create handler module in `handlers/`
3. Server restart required (tool signatures changed)

When adding new model config fields, update in 3 places:
1. `harness-core` config_writer (`add_model`, `update_model`)
2. `harness-plugin` handler (`handlers/models.py`)
3. `harness-plugin` tool signature (`mcp_server.py`, restart required)

## Testing

```bash
uv run pytest packages/harness-plugin/tests/ -v
```
