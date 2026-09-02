# harness-studio Implementation Plan (Package 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Studio dashboard — a FastAPI backend that reads workspace state directly, plus a React frontend with 8 views (Version Tree, Version Detail, Pipeline Explorer, Diagnostics, Predictions, Data Profile, MCP Monitor, Preferences).

**Architecture:** FastAPI backend serves JSON endpoints by reading workspace files (no event store for state). SQLite event log for MCP monitoring only. React 19 + Vite frontend. WebSocket for live progress during backtests.

**Tech Stack:** Python: FastAPI, uvicorn, aiosqlite. Frontend: React 19, TypeScript, Vite.

**Spec Reference:** [2026-03-23-harness2-design.md](../specs/2026-03-23-harness2-design.md) — Section 11 (Studio Dashboard)

---

## File Structure

```
packages/harness-studio/
├── pyproject.toml
├── src/harness/studio/
│   ├── __init__.py
│   ├── server.py              # FastAPI app + startup
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── versions.py        # Version tree + detail endpoints
│   │   ├── pipeline.py        # DAG + config endpoints
│   │   ├── diagnostics.py     # Metrics, calibration, per-fold
│   │   ├── predictions.py     # Prediction browsing + export
│   │   ├── data.py            # Data profile endpoints
│   │   └── monitor.py         # MCP event log stream
│   ├── event_log.py           # Append-only SQLite event log
│   └── websocket.py           # WebSocket for live progress
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── index.html
│   ├── tsconfig.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── api.ts             # API client
│       └── views/
│           ├── VersionTree.tsx
│           ├── VersionDetail.tsx
│           ├── PipelineExplorer.tsx
│           ├── Diagnostics.tsx
│           ├── Predictions.tsx
│           ├── DataProfile.tsx
│           ├── MCPMonitor.tsx
│           └── Preferences.tsx
└── tests/
    ├── conftest.py
    ├── test_routes/
    │   ├── test_versions.py
    │   ├── test_pipeline.py
    │   ├── test_diagnostics.py
    │   ├── test_predictions.py
    │   └── test_data.py
    ├── test_event_log.py
    └── test_e2e.py
```

---

### Task 1: Backend Scaffolding + Version Routes

**Files:**
- Create: `packages/harness-studio/pyproject.toml`
- Create: `src/harness/studio/server.py`
- Create: `src/harness/studio/routes/__init__.py`
- Create: `src/harness/studio/routes/versions.py`
- Create: `tests/conftest.py`
- Create: `tests/test_routes/test_versions.py`

**pyproject.toml:**
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "harness-studio"
version = "0.1.0"
description = "Dashboard for the Harness ML platform"
requires-python = ">=3.11"
dependencies = [
    "harness-app>=0.1.0",
    "fastapi>=0.100",
    "uvicorn>=0.20",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.24"]

[project.scripts]
harness-studio = "harness.studio.server:main"

[tool.hatch.build.targets.wheel]
packages = ["src/harness"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

**server.py:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

def create_app(workspace_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Harness Studio")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # Store workspace dir in app state
    app.state.workspace_dir = workspace_dir

    from harness.studio.routes import versions, pipeline, diagnostics, predictions, data
    app.include_router(versions.router, prefix="/api/versions", tags=["versions"])
    app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
    app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])
    app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
    app.include_router(data.router, prefix="/api/data", tags=["data"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "workspace": str(workspace_dir)}

    return app

def main():
    import uvicorn
    from harness.app.workspace.discovery import find_workspace
    ws = find_workspace()
    app = create_app(ws)
    uvicorn.run(app, host="0.0.0.0", port=8421)
```

**routes/versions.py:**
```python
from fastapi import APIRouter, Request, HTTPException
from harness.app.workspace.versions import VersionTree

router = APIRouter()

@router.get("/tree")
def get_version_tree(request: Request):
    ws_dir = request.app.state.workspace_dir
    tree = VersionTree(ws_dir)
    versions = tree.list_versions()
    current = tree.get_current()
    return {
        "current": current,
        "versions": [vars(v) for v in versions],
    }

@router.get("/{version_id}")
def get_version_detail(version_id: str, request: Request):
    ws_dir = request.app.state.workspace_dir
    tree = VersionTree(ws_dir)
    meta = tree.get_version(version_id)
    if meta is None:
        raise HTTPException(404, f"Version not found: {version_id}")
    return vars(meta)

@router.get("/{version_id}/ancestry")
def get_ancestry(version_id: str, request: Request):
    ws_dir = request.app.state.workspace_dir
    tree = VersionTree(ws_dir)
    chain = tree.ancestry(version_id)
    return [vars(v) for v in chain]

@router.get("/compare/{v1}/{v2}")
def compare_versions(v1: str, v2: str, request: Request):
    ws_dir = request.app.state.workspace_dir
    tree = VersionTree(ws_dir)
    return tree.compare(v1, v2)
```

**Tests using FastAPI TestClient (httpx):**
```python
# tests/conftest.py
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from harness.studio.server import create_app
from harness.app.workspace.manager import WorkspaceManager
import numpy as np, pandas as pd

@pytest.fixture
def workspace(tmp_path):
    ws = WorkspaceManager.init(tmp_path, task_type="binary", target_column="target")
    # Add data and run a baseline
    rng = np.random.RandomState(42)
    n = 60
    df = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n)})
    df["target"] = (df["a"] + rng.randn(n) * 0.5 > 0).astype(int)
    csv = tmp_path / "data" / "raw" / "data.csv"
    df.to_csv(csv, index=False)
    ws.data.add_source("main", str(csv))
    ws.data.run_pipeline()
    ws.run_experiment("baseline", "Baseline", {
        "models": {"lr": {"model_type": "logistic", "features": ["a", "b"]}},
    })
    return tmp_path

@pytest.fixture
def client(workspace):
    app = create_app(workspace)
    return TestClient(app)
```

Tests: GET /api/health, GET /api/versions/tree returns versions, GET /api/versions/v001 returns detail, GET /api/versions/v001/ancestry.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-studio): backend scaffolding + version routes"
```

---

### Task 2: Pipeline + Data + Diagnostics + Predictions Routes

**Files:**
- Create: `src/harness/studio/routes/pipeline.py`
- Create: `src/harness/studio/routes/diagnostics.py`
- Create: `src/harness/studio/routes/predictions.py`
- Create: `src/harness/studio/routes/data.py`
- Create: tests for each

**pipeline.py:**
```python
@router.get("/config")
def get_pipeline_config(request: Request):
    """Current pipeline config (models, features, ensemble)."""

@router.get("/dag")
def get_pipeline_dag(request: Request):
    """Model dependency DAG."""
```

**diagnostics.py:**
```python
@router.get("/{version_id}")
def get_diagnostics(version_id: str, request: Request):
    """Metrics, per-fold stats for a version."""
```

**predictions.py:**
```python
@router.get("/{version_id}")
def get_predictions(version_id: str, request: Request):
    """Prediction table for a version."""

@router.get("/{version_id}/distribution")
def get_distribution(version_id: str, request: Request):
    """Prediction distribution histogram."""
```

**data.py:**
```python
@router.get("/schema")
def get_data_schema(request: Request):
    """Clean dataset schema."""

@router.get("/profile")
def get_data_profile(request: Request):
    """Data profiling results."""
```

Tests with TestClient: each endpoint returns correct structure, 404 for missing versions.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-studio): pipeline, diagnostics, predictions, data routes"
```

---

### Task 3: MCP Event Log

**Files:**
- Create: `src/harness/studio/event_log.py`
- Create: `src/harness/studio/routes/monitor.py`
- Create: `tests/test_event_log.py`

**event_log.py:**
```python
import sqlite3
import json
from datetime import datetime
from pathlib import Path

class EventLog:
    def __init__(self, db_path: Path):
        self._path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    action TEXT,
                    params TEXT,
                    result TEXT,
                    duration_ms INTEGER,
                    status TEXT DEFAULT 'success'
                )
            """)

    def log(self, tool: str, action: str = "", params: dict = None, result: str = "", duration_ms: int = 0, status: str = "success"):
        with sqlite3.connect(self._path) as conn:
            conn.execute(
                "INSERT INTO events (timestamp, tool, action, params, result, duration_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), tool, action, json.dumps(params or {}), result, duration_ms, status),
            )

    def query(self, limit: int = 50, offset: int = 0, tool: str = None) -> list[dict]:
        with sqlite3.connect(self._path) as conn:
            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM events"
            params = []
            if tool:
                sql += " WHERE tool = ?"
                params.append(tool)
            sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def stats(self) -> dict:
        with sqlite3.connect(self._path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            errors = conn.execute("SELECT COUNT(*) FROM events WHERE status = 'error'").fetchone()[0]
            return {"total": total, "errors": errors}
```

**monitor.py:**
```python
@router.get("/events")
def get_events(request: Request, limit: int = 50, offset: int = 0):

@router.get("/stats")
def get_event_stats(request: Request):
```

Tests: log and query roundtrip, stats, filter by tool.

- [ ] **Steps: Write tests → implement → verify → commit**

```bash
git commit -m "feat(harness-studio): MCP event log + monitor routes"
```

---

### Task 4: Frontend Scaffolding

**Files:**
- Create: `packages/harness-studio/frontend/package.json`
- Create: `packages/harness-studio/frontend/vite.config.ts`
- Create: `packages/harness-studio/frontend/index.html`
- Create: `packages/harness-studio/frontend/tsconfig.json`
- Create: `packages/harness-studio/frontend/src/main.tsx`
- Create: `packages/harness-studio/frontend/src/App.tsx`
- Create: `packages/harness-studio/frontend/src/api.ts`
- Create: view stubs for all 8 views

The frontend is a React 19 + Vite app with tab-based navigation. Each view is a component that fetches data from the FastAPI backend.

**App.tsx** — tab navigation between 8 views:
```tsx
import { useState } from 'react';
import VersionTree from './views/VersionTree';
// ... other imports

const VIEWS = [
  { name: 'Version Tree', component: VersionTree },
  { name: 'Version Detail', component: VersionDetail },
  // ... etc
];

export default function App() {
  const [activeView, setActiveView] = useState(0);
  const View = VIEWS[activeView].component;
  return (
    <div>
      <nav>{VIEWS.map((v, i) => (
        <button key={i} onClick={() => setActiveView(i)}>{v.name}</button>
      ))}</nav>
      <main><View /></main>
    </div>
  );
}
```

**api.ts** — fetch wrapper for backend endpoints:
```typescript
const BASE = '/api';
export async function fetchVersionTree() { return fetch(`${BASE}/versions/tree`).then(r => r.json()); }
export async function fetchVersionDetail(id: string) { return fetch(`${BASE}/versions/${id}`).then(r => r.json()); }
// ... etc
```

Each view component is a stub that fetches and displays JSON from the API. The full UI polish comes later — the goal here is working components that render real data.

- [ ] **Step 1: Create package.json + vite.config.ts + tsconfig.json**
- [ ] **Step 2: Create index.html + main.tsx + App.tsx + api.ts**
- [ ] **Step 3: Create all 8 view stubs**
- [ ] **Step 4: Verify build: `cd frontend && npm install && npm run build`**
- [ ] **Step 5: Commit**

```bash
git commit -m "feat(harness-studio): React frontend scaffolding with 8 views"
```

---

### Task 5: Backend E2E Tests

**Files:**
- Create: `tests/test_e2e.py`

```python
class TestE2EStudio:
    def test_full_api_flow(self, client):
        """Health → versions → detail → diagnostics → predictions → data."""
        # Health
        r = client.get("/api/health")
        assert r.status_code == 200

        # Version tree
        r = client.get("/api/versions/tree")
        assert r.status_code == 200
        data = r.json()
        assert len(data["versions"]) >= 1

        # Version detail
        v_id = data["versions"][0]["id"]
        r = client.get(f"/api/versions/{v_id}")
        assert r.status_code == 200
        assert "metrics" in r.json()

        # Pipeline config
        r = client.get("/api/pipeline/config")
        assert r.status_code == 200

        # Diagnostics
        r = client.get(f"/api/diagnostics/{v_id}")
        assert r.status_code == 200

        # Predictions
        r = client.get(f"/api/predictions/{v_id}")
        assert r.status_code == 200

        # Data schema
        r = client.get("/api/data/schema")
        assert r.status_code == 200

    def test_404_for_missing_version(self, client):
        r = client.get("/api/versions/v999")
        assert r.status_code == 404
```

- [ ] **Steps: Write tests → verify → commit**

```bash
git commit -m "feat(harness-studio): backend e2e tests (Package 5 complete)"
```
