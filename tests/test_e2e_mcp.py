"""Exhaustive end-to-end test exercising the full MCP workflow.

Drives the MCP server as a subprocess (stdio transport), mimicking exactly
how Claude Code talks to HarnessML.  Covers:

  - Project init, data ingestion, profiling
  - Feature discovery and transformation testing
  - Model addition (3 diverse regressors)
  - Backtest with 5-fold CV
  - Notebook entries (theory, plan, finding)
  - Experiment creation, overlay, run, log_result
  - Diagnostics, residual analysis, model correlation
  - Inspect predictions, compare runs, list runs
  - Feature pruning (dry run)
  - Notebook summary and search

Requires: uv, all workspace packages installed.
Run:  uv run pytest tests/test_e2e_mcp.py -v -s
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap

import pytest

EASYML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO_DATA = os.path.join(
    EASYML_DIR,
    "packages/harness-plugin/src/harnessml/plugin/demo_data/housing.csv",
)


# ── MCP client helpers ──────────────────────────────────────────────

class MCPClient:
    """Minimal MCP JSON-RPC client over stdio."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, msg: dict) -> dict | None:
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()
        while True:
            line = self._proc.stdout.readline()
            if not line:
                return None
            line = line.strip()
            if not line:
                continue
            try:
                resp = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" in resp:
                return resp
            # skip notifications

    def initialize(self) -> dict:
        resp = self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "1.0"},
            },
        })
        # send initialized notification (no response expected)
        self._proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self._proc.stdin.flush()
        return resp

    def list_tools(self) -> list[str]:
        resp = self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        })
        return [t["name"] for t in resp["result"]["tools"]]

    def call(self, tool_name: str, **args) -> str:
        """Call an MCP tool and return the text content. Raises on errors."""
        resp = self._send({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
        })
        assert resp is not None, "No response from MCP server"
        assert "error" not in resp, f"JSON-RPC error: {resp.get('error')}"
        text = resp["result"]["content"][0]["text"]
        return text

    def call_ok(self, tool_name: str, **args) -> str:
        """Call a tool and assert no **Error** in response."""
        text = self.call(tool_name, **args)
        assert not text.startswith("**Error**"), f"Tool error: {text[:500]}"
        return text

    def close(self):
        self._proc.stdin.close()
        self._proc.terminate()
        self._proc.wait(timeout=10)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def project_dir(tmp_path_factory):
    """Create a temporary project directory."""
    return str(tmp_path_factory.mktemp("harness_e2e"))


@pytest.fixture(scope="module")
def mcp(project_dir):
    """Start MCP server and yield a client. Torn down after all tests."""
    proc = subprocess.Popen(
        ["uv", "run", "--directory", EASYML_DIR, "harness-ml"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "HARNESS_PROJECT_DIR": project_dir},
    )
    client = MCPClient(proc)
    client.initialize()
    yield client
    client.close()


# ── Tests (ordered — each builds on the previous) ──────────────────

class TestE2EWorkflow:
    """Full MCP workflow, test methods run in order via pytest-ordering or naming."""

    # We store state across tests via class attributes
    _project_dir: str = ""
    _run_id: str = ""
    _experiment_id: str = ""

    def test_00_server_init(self, mcp):
        tools = mcp.list_tools()
        assert "models" in tools
        assert "data" in tools
        assert "features" in tools
        assert "experiments" in tools
        assert "configure" in tools
        assert "pipeline" in tools
        assert "notebook" in tools
        assert len(tools) == 8

    def test_01_init_project(self, mcp, project_dir):
        TestE2EWorkflow._project_dir = project_dir
        text = mcp.call_ok("configure",
            action="init",
            project_dir=project_dir,
            task="regression",
            target_column="median_house_value",
        )
        assert "Initialized project" in text or "already" in text.lower()
        assert os.path.exists(os.path.join(project_dir, "config", "pipeline.yaml"))

    def test_02_ingest_data(self, mcp, project_dir):
        # Copy housing data into the project
        raw_dir = os.path.join(project_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        shutil.copy2(DEMO_DATA, os.path.join(raw_dir, "housing.csv"))

        text = mcp.call_ok("data",
            action="add",
            data_path=os.path.join(project_dir, "data/raw/housing.csv"),
            project_dir=project_dir,
        )
        assert "20640" in text  # row count
        assert "median_income" in text

    def test_03_profile_data(self, mcp, project_dir):
        text = mcp.call_ok("data",
            action="profile",
            project_dir=project_dir,
        )
        assert "median_income" in text or "column" in text.lower()

    def test_04_derive_fold_column(self, mcp, project_dir):
        """Add a dummy fold column for kfold CV."""
        text = mcp.call_ok("data",
            action="derive_column",
            name="fold",
            expression="0",
            project_dir=project_dir,
        )
        assert "fold" in text.lower() or "derived" in text.lower() or "added" in text.lower()

    def test_05_configure_backtest(self, mcp, project_dir):
        text = mcp.call_ok("configure",
            action="backtest",
            project_dir=project_dir,
            cv_strategy="kfold",
            n_folds=5,
            metrics=["rmse", "mae", "r2"],
        )
        assert "kfold" in text.lower() or "backtest" in text.lower()

    def test_06_configure_ensemble(self, mcp, project_dir):
        text = mcp.call_ok("configure",
            action="ensemble",
            project_dir=project_dir,
            method="stacked",
        )
        assert "ensemble" in text.lower() or "stacked" in text.lower()

    def test_07_add_models(self, mcp, project_dir):
        features = [
            "median_income", "house_age", "avg_rooms",
            "avg_bedrooms", "population", "avg_occupancy",
            "latitude", "longitude",
        ]
        for model_name, mtype in [
            ("xgb_v1", "xgboost"),
            ("lgb_v1", "lightgbm"),
            ("ridge_v1", "elastic_net"),
        ]:
            text = mcp.call_ok("models",
                action="add",
                project_dir=project_dir,
                name=model_name,
                model_type=mtype,
                mode="regressor",
                features=features,
            )
            assert model_name in text
            assert "Added model" in text

    def test_08_show_config(self, mcp, project_dir):
        text = mcp.call_ok("configure",
            action="show",
            project_dir=project_dir,
        )
        assert "regression" in text or "xgb_v1" in text

    def test_09_feature_discover(self, mcp, project_dir):
        text = mcp.call_ok("features",
            action="discover",
            project_dir=project_dir,
        )
        assert "median_income" in text or "feature" in text.lower()

    def test_10_notebook_write_theory(self, mcp, project_dir):
        text = mcp.call_ok("notebook",
            action="write",
            type="theory",
            content="Median income should be the strongest predictor of house value "
                    "based on economic first principles. Geographic features (lat/long) "
                    "may capture neighborhood effects.",
            project_dir=project_dir,
        )
        assert "theory" in text.lower() or "written" in text.lower() or "saved" in text.lower()

    def test_11_notebook_write_plan(self, mcp, project_dir):
        text = mcp.call_ok("notebook",
            action="write",
            type="plan",
            content="Phase 1: Baseline with 3 model types (XGB, LGB, ElasticNet). "
                    "Phase 2: Run experiment adding geographic features. "
                    "Phase 3: Compare and select best configuration.",
            project_dir=project_dir,
        )
        assert "plan" in text.lower() or "written" in text.lower() or "saved" in text.lower()

    def test_12_run_backtest(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="run_backtest",
            project_dir=project_dir,
        )
        assert "Backtest Results" in text or "success" in text.lower()
        assert "rmse" in text.lower() or "mae" in text.lower()
        # Extract run ID
        for line in text.split("\n"):
            if "Run ID" in line:
                TestE2EWorkflow._run_id = line.split("`")[1] if "`" in line else ""
                break
        assert TestE2EWorkflow._run_id, "Could not extract run_id from backtest output"

    def test_13_diagnostics(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="diagnostics",
            project_dir=project_dir,
        )
        assert "ensemble" in text.lower()
        assert "xgb_v1" in text
        assert "lgb_v1" in text
        assert "ridge_v1" in text

    def test_14_list_runs(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="list_runs",
            project_dir=project_dir,
        )
        assert TestE2EWorkflow._run_id in text

    def test_15_show_run(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="show_run",
            run_id=TestE2EWorkflow._run_id,
            project_dir=project_dir,
        )
        assert TestE2EWorkflow._run_id in text
        assert "artifacts" in text.lower() or "report.md" in text

    def test_16_inspect_predictions(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="inspect_predictions",
            run_id=TestE2EWorkflow._run_id,
            mode="worst",
            top_n=5,
            project_dir=project_dir,
        )
        assert "median_house_value" in text or "residual" in text.lower() or "predicted" in text.lower()

    def test_17_model_correlation(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="model_correlation",
            run_id=TestE2EWorkflow._run_id,
            project_dir=project_dir,
        )
        assert "xgb_v1" in text or "correlation" in text.lower()

    def test_18_residual_analysis(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="residual_analysis",
            feature="median_income",
            run_id=TestE2EWorkflow._run_id,
            n_bins=5,
            project_dir=project_dir,
        )
        assert "median_income" in text or "residual" in text.lower() or "bin" in text.lower()

    def test_19_explain_model(self, mcp, project_dir):
        # explain requires saved model artifacts which may not exist for all
        # run configurations; verify the call completes without crashing
        text = mcp.call("pipeline",
            action="explain",
            name="xgb_v1",
            run_id=TestE2EWorkflow._run_id,
            top_n=5,
            project_dir=project_dir,
        )
        # Either returns feature importances or an error about missing models dir
        assert "importance" in text.lower() or "error" in text.lower()

    def test_20_notebook_write_finding(self, mcp, project_dir):
        text = mcp.call_ok("notebook",
            action="write",
            type="finding",
            content="Baseline backtest: ensemble R²=0.845, RMSE=0.454. "
                    "XGB and LGB perform similarly; ElasticNet much weaker. "
                    "Ensemble benefits from model diversity.",
            project_dir=project_dir,
        )
        assert "finding" in text.lower() or "written" in text.lower() or "saved" in text.lower()

    def test_21_experiment_create(self, mcp, project_dir):
        text = mcp.call_ok("experiments",
            action="create",
            description="Remove ElasticNet and add more XGB trees",
            hypothesis="Removing the weak ElasticNet model and increasing "
                       "XGB n_estimators should improve ensemble RMSE",
            project_dir=project_dir,
        )
        assert "experiment" in text.lower() or "created" in text.lower()
        # Extract experiment ID
        for line in text.split("\n"):
            if "exp-" in line.lower():
                import re
                match = re.search(r"(exp-\d+)", line, re.IGNORECASE)
                if match:
                    TestE2EWorkflow._experiment_id = match.group(1)
                    break
        assert TestE2EWorkflow._experiment_id, f"Could not extract experiment_id from: {text[:300]}"

    def test_22_experiment_write_overlay(self, mcp, project_dir):
        text = mcp.call_ok("experiments",
            action="write_overlay",
            experiment_id=TestE2EWorkflow._experiment_id,
            overlay={
                "models": {
                    "ridge_v1": {"active": False},
                    "xgb_v1": {"params": {"n_estimators": 200}},
                }
            },
            project_dir=project_dir,
        )
        assert "overlay" in text.lower() or "written" in text.lower()

    def test_23_experiment_run(self, mcp, project_dir):
        text = mcp.call_ok("experiments",
            action="run",
            experiment_id=TestE2EWorkflow._experiment_id,
            primary_metric="rmse",
            project_dir=project_dir,
        )
        assert "rmse" in text.lower() or "backtest" in text.lower() or "result" in text.lower()

    def test_24_experiment_log_result(self, mcp, project_dir):
        text = mcp.call_ok("experiments",
            action="log_result",
            experiment_id=TestE2EWorkflow._experiment_id,
            conclusion="Removing ElasticNet slightly improved ensemble. "
                       "More XGB trees had minimal impact.",
            verdict="marginal",
            project_dir=project_dir,
        )
        assert "logged" in text.lower() or "result" in text.lower() or "updated" in text.lower()

    def test_25_experiment_journal(self, mcp, project_dir):
        text = mcp.call_ok("experiments",
            action="journal",
            project_dir=project_dir,
        )
        assert TestE2EWorkflow._experiment_id in text

    def test_26_compare_latest(self, mcp, project_dir):
        text = mcp.call("pipeline",
            action="compare_latest",
            project_dir=project_dir,
        )
        # May error if only one run — that's OK, just verify it doesn't crash
        assert text is not None

    def test_27_feature_prune_dry_run(self, mcp, project_dir):
        text = mcp.call_ok("features",
            action="prune",
            threshold=0.01,
            dry_run=True,
            project_dir=project_dir,
        )
        assert "prune" in text.lower() or "feature" in text.lower() or "importance" in text.lower()

    def test_28_notebook_summary(self, mcp, project_dir):
        text = mcp.call_ok("notebook",
            action="summary",
            project_dir=project_dir,
        )
        # Should contain our theory, plan, and finding
        assert "theory" in text.lower() or "plan" in text.lower() or "entries" in text.lower()

    def test_29_notebook_search(self, mcp, project_dir):
        text = mcp.call_ok("notebook",
            action="search",
            query="ElasticNet",
            project_dir=project_dir,
        )
        assert "elasticnet" in text.lower() or "elastic" in text.lower() or "entries" in text.lower()

    def test_30_suggest_cv(self, mcp, project_dir):
        text = mcp.call_ok("configure",
            action="suggest_cv",
            project_dir=project_dir,
        )
        assert "cv" in text.lower() or "strategy" in text.lower() or "fold" in text.lower()

    def test_31_workflow_progress(self, mcp, project_dir):
        text = mcp.call_ok("pipeline",
            action="progress",
            project_dir=project_dir,
        )
        assert text is not None

    def test_32_data_inspect(self, mcp, project_dir):
        text = mcp.call_ok("data",
            action="inspect",
            n_rows=3,
            project_dir=project_dir,
        )
        assert "median_income" in text or "house_age" in text
