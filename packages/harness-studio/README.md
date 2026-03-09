# Harness Studio

Companion web dashboard for [HarnessML](https://github.com/msilverblatt/harness-ml). Provides real-time observability while the agent works in Claude Code, Claude Desktop, or any MCP host.

## Views

| Tab | What it shows |
|-----|--------------|
| **Activity** | Live event log with WebSocket streaming, stat boxes (project, experiments, model types, tool calls, errors, connection status) |
| **DAG** | Interactive pipeline topology via React Flow -- data sources, feature store, models, ensemble, calibration, output nodes with edge connections |
| **Experiments** | Sortable experiment table with hypothesis/conclusion, metric trend chart, side-by-side comparison with color-coded deltas |
| **Diagnostics** | All 45 metrics grouped by category, calibration reliability diagram, model correlation heatmap, per-fold breakdown |

### Activity Tab

Live event feed updated via WebSocket. Each event shows tool name, action, timestamp, and result summary. Stat boxes at the top provide at-a-glance project status: total experiments, model count, tool call history, error count, and WebSocket connection state.

### DAG Tab

Interactive directed acyclic graph built from project configuration. Nodes represent pipeline stages (data sources, views, feature store, individual models, ensemble, calibration, output). Edges show data flow. Powered by React Flow with automatic layout.

### Experiments Tab

Sortable table of all experiments from the project journal (JSONL). Columns include experiment name, hypothesis, metric values, and conclusion. Metric trend chart shows progression over time. Side-by-side comparison highlights deltas with color coding (green for improvement, red for regression).

### Diagnostics Tab

Displays run output metrics organized by category. Calibration reliability diagram plots predicted vs. actual probabilities. Model correlation heatmap shows agreement between ensemble members. Fold breakdown table shows per-fold metric values for the selected run.

## Architecture

```
Claude Code ──MCP──> harness-plugin ──events──> SQLite
                                                  |
                                    FastAPI <─────┘
                                      |
                              React Dashboard
                       (Activity / DAG / Experiments / Diagnostics)
```

- **Event layer**: The MCP server emits structured events to SQLite as a side effect of tool calls. Fail-safe -- never blocks or breaks tool execution.
- **Backend**: FastAPI reads SQLite for events, project config YAMLs for DAG/status, experiment journal for history, run outputs for metrics/calibration/correlations.
- **Frontend**: React 19 + TypeScript + Vite. CSS Modules with design tokens. Dependencies: React Flow (DAG), Recharts (charts).
- **Zero core changes**: Studio is a reader of existing artifacts. The only integration point is event emission in harness-plugin.

### Backend Components

| Module | Purpose |
|--------|---------|
| `server.py` | FastAPI app, static file serving, startup/shutdown |
| `event_store.py` | SQLite WAL-mode event store (record, query, session_stats) |
| `broadcaster.py` | asyncio.Queue fan-out for WebSocket live streaming |
| `routes/events.py` | Event history REST + WebSocket endpoint |
| `routes/project.py` | Config reading, DAG generation, project status |
| `routes/experiments.py` | Journal parsing, experiment table data |
| `routes/runs.py` | Run metrics, calibration data, correlation matrices |

### WebSocket Streaming

The broadcaster module manages a fan-out pattern: when a new event is recorded in SQLite, it is pushed to all connected WebSocket clients via asyncio queues. Each client gets its own queue to avoid blocking. Disconnected clients are cleaned up automatically.

## Themes

12 built-in themes selectable from the Preferences tab:

| Theme | Style |
|-------|-------|
| Default (dark) | Built-in dark theme |
| Claude | Anthropic-inspired palette |
| Nord | Arctic, north-bluish color scheme |
| Catppuccin | Soothing pastel theme |
| Rose Pine | Natural, muted palette |
| Solarized Dark | Ethan Schoonover's dark variant |
| GitHub Light | Light theme inspired by GitHub |
| Matrix | Green-on-black terminal aesthetic |
| OpenAI | OpenAI-inspired palette |
| Neon Light | Vibrant neon accents on light background |
| Brutalist Dark | High-contrast brutalist dark |
| Brutalist Light | High-contrast brutalist light |

## Quick Start

```bash
# From the monorepo root
uv sync
uv run harness-studio --project-dir examples/titanic
# Open http://localhost:8421
```

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
- **Frontend**: React 19, Vite, TypeScript, CSS Modules, React Flow, Recharts
- **Communication**: REST for historical data, WebSocket for live streaming
- **Package manager**: bun (frontend), uv (Python)
