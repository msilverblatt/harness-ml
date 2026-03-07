# Harness Studio

Companion web dashboard for [HarnessML](https://github.com/msilverblatt/harness-ml). Provides real-time observability while the agent works in Claude Code, Claude Desktop, or any MCP host.

## Views

| Tab | What it shows |
|-----|--------------|
| **Activity** | Live event log, stat boxes (project, experiments, model types, tool calls, errors, WebSocket status) |
| **DAG** | Interactive pipeline topology — data sources, feature store, models, ensemble, calibration, output |
| **Experiments** | Sortable experiment table, metric trend chart, side-by-side comparison with color-coded deltas |
| **Diagnostics** | All metrics grouped by category, calibration reliability diagram, model correlation heatmap, fold breakdown |

## Quick Start

```bash
# From the monorepo root
uv sync
uv run harness-studio --project-dir examples/titanic
# Open http://localhost:8421
```

## Architecture

```
Claude Code ──MCP──> harness-plugin ──events──> SQLite
                                                  |
                                    FastAPI <─────┘
                                      |
                              React Dashboard
                       (Activity / DAG / Experiments / Diagnostics)
```

- **Event layer**: The MCP server emits structured events to SQLite as a side effect of tool calls. Fail-safe — never blocks or breaks tool execution.
- **Backend**: FastAPI reads SQLite for events, project config YAMLs for DAG/status, experiment journal for history, run outputs for metrics/calibration/correlations.
- **Frontend**: React 19 + TypeScript + Vite. CSS Modules with design tokens. Dark mode. Dependencies: React Flow (DAG), Recharts (charts).
- **Zero core changes**: Studio is a reader of existing artifacts. The only integration point is event emission in harness-plugin.

## Development

```bash
# Frontend dev server (hot reload)
cd packages/harness-studio/frontend
bun install
bun run dev

# Build + copy to Python package
bash scripts/build_frontend.sh

# Run tests
uv run pytest packages/harness-studio/tests/ -v
```

## Tech Stack

- **Backend**: FastAPI, uvicorn, sqlite3 (stdlib)
- **Frontend**: React 19, Vite, TypeScript, React Flow, Recharts
- **Communication**: REST for historical data, WebSocket for live streaming
- **Package manager**: bun (frontend), uv (Python)
