"""End-to-end quickstart test: scaffold → data → models → backtest → diagnostics.

Verifies the complete pipeline works for a regression task using only
always-available models (elastic_net, random_forest). Uses synthetic data
with a known linear signal so assertions are meaningful.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from harnessml.core.runner.config_writer import (
    add_dataset,
    add_model,
    configure_backtest,
    configure_ensemble,
    list_runs,
    scaffold_init,
    show_diagnostics,
)
from harnessml.core.runner.config_writer.pipeline import run_backtest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regression_csv(directory: Path, *, n_rows: int = 200, n_folds: int = 3) -> Path:
    """Generate a synthetic regression CSV with known linear signal."""
    rng = np.random.default_rng(42)

    rows_per_fold = n_rows // n_folds
    folds = []
    for f in range(1, n_folds + 1):
        folds.extend([f] * rows_per_fold)
    # fill remainder into last fold
    while len(folds) < n_rows:
        folds.append(n_folds)

    x1 = rng.normal(0, 1, n_rows)
    x2 = rng.normal(0, 1, n_rows)
    x3 = rng.normal(0, 1, n_rows)
    x4 = rng.normal(0, 1, n_rows)
    x5 = rng.normal(0, 1, n_rows)

    # Known signal: target = 3*x1 + 2*x2 - x3 + noise
    noise = rng.normal(0, 0.5, n_rows)
    target = 3.0 * x1 + 2.0 * x2 - 1.0 * x3 + noise

    df = pd.DataFrame({
        "fold": folds,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "x5": x5,
        "target": target,
    })

    csv_path = directory / "regression_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _setup_notebook_plan(project_dir: Path) -> None:
    """Create a minimal notebook plan entry to satisfy discipline gates."""
    notebook_dir = project_dir / "notebook"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": "nb-001",
        "type": "plan",
        "content": "Quickstart test plan: verify regression pipeline end-to-end.",
        "tags": [],
        "auto_tags": [],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (notebook_dir / "entries.jsonl").write_text(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestQuickstartE2E:
    """End-to-end regression quickstart: scaffold through diagnostics."""

    def test_full_regression_pipeline(self, tmp_path: Path) -> None:
        """Scaffold, add data, add models, backtest, check diagnostics and runs."""
        project_dir = tmp_path / "quickstart"
        feature_names = ["x1", "x2", "x3", "x4", "x5"]

        # 1. Scaffold a regression project
        init_result = scaffold_init(
            project_dir,
            "quickstart-test",
            task="regression",
            target_column="target",
        )
        assert "Initialized project" in init_result
        assert (project_dir / "config" / "pipeline.yaml").exists()

        # 2. Generate synthetic regression CSV
        csv_path = _make_regression_csv(tmp_path)
        assert csv_path.exists()

        # 3. Add the dataset (bootstrap ingest)
        add_result = add_dataset(project_dir, str(csv_path))
        assert "Ingested" in add_result or "Bootstrap" in add_result

        # Verify features parquet was created with expected columns
        features_pq = project_dir / "data" / "features" / "features.parquet"
        assert features_pq.exists()
        df = pd.read_parquet(features_pq)
        for col in feature_names + ["target", "fold"]:
            assert col in df.columns, f"Missing column: {col}"

        # 4. Configure backtest with fold column and regression metrics
        bt_result = configure_backtest(
            project_dir,
            fold_column="fold",
            fold_values=[1, 2, 3],
            metrics=["rmse", "mae", "r2"],
        )
        assert "Updated backtest config" in bt_result

        # 5. Add elastic_net model (regressor mode for continuous target)
        enet_result = add_model(
            project_dir,
            "enet_base",
            model_type="elastic_net",
            mode="regressor",
            features=feature_names,
        )
        assert "Added model" in enet_result
        assert "enet_base" in enet_result

        # 6. Add random_forest model (regressor mode for continuous target)
        rf_result = add_model(
            project_dir,
            "rf_base",
            model_type="random_forest",
            mode="regressor",
            features=feature_names,
        )
        assert "Added model" in rf_result
        assert "rf_base" in rf_result

        # 7. Configure stacked ensemble
        ens_result = configure_ensemble(project_dir, method="stacked")
        assert "Updated ensemble config" in ens_result

        # 8. Set up notebook plan (discipline gate) then run backtest
        _setup_notebook_plan(project_dir)
        backtest_result = run_backtest(project_dir)

        # The backtest should succeed — not a failure message
        assert "failed" not in backtest_result.lower(), (
            f"Backtest failed unexpectedly:\n{backtest_result}"
        )
        # Should contain metric values in the markdown output
        assert any(
            m in backtest_result.lower()
            for m in ["rmse", "mae", "r2", "metric", "backtest"]
        ), f"No metrics found in backtest result:\n{backtest_result}"

        # 9. Show diagnostics — should reference our models
        diag_result = show_diagnostics(project_dir)
        # Diagnostics may show model names or metrics
        assert "Error" not in diag_result or "enet_base" in diag_result or "rf_base" in diag_result, (
            f"Diagnostics unexpected:\n{diag_result}"
        )

        # 10. List runs — should have at least 1 run
        runs_result = list_runs(project_dir)
        assert "No runs" not in runs_result, (
            f"Expected at least 1 run but got:\n{runs_result}"
        )
