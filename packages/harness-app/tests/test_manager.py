import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from harness.app.workspace.manager import WorkspaceManager
from harness.app.workspace.versions import VersionMeta
from harness.ml.runners.backtest import BacktestResult


class TestInit:
    def test_creates_structure(self, tmp_workspace):
        ws = WorkspaceManager.init(tmp_workspace)
        assert (tmp_workspace / "harness.yaml").exists()
        assert (tmp_workspace / "config" / "project.yaml").exists()
        assert (tmp_workspace / "config" / "models.yaml").exists()
        assert (tmp_workspace / "config" / "ensemble.yaml").exists()
        assert (tmp_workspace / "config" / "features.yaml").exists()
        assert (tmp_workspace / "config" / "evals.yaml").exists()
        assert (tmp_workspace / "data" / "raw").is_dir()
        assert (tmp_workspace / "data" / "clean").is_dir()
        assert (tmp_workspace / "versions").is_dir()
        assert (tmp_workspace / "artifacts").is_dir()
        assert (tmp_workspace / ".harness").is_dir()

    def test_custom_task_type(self, tmp_workspace):
        ws = WorkspaceManager.init(
            tmp_workspace, task_type="regression", target_column="score"
        )
        project = ws.config.read_project()
        assert project.task_type == "regression"
        assert project.target_column == "score"
        assert project.metrics == ["rmse", "mae", "r2"]


class TestRunExperiment:
    def test_run_creates_version(self, initialized_workspace, tmp_path):
        ws = initialized_workspace
        root = ws._root

        # Create sample clean data
        rng = np.random.RandomState(42)
        n = 100
        df = pd.DataFrame(
            {
                "feature_a": rng.randn(n),
                "feature_b": rng.randn(n),
                "target": rng.randint(0, 2, n),
            }
        )
        (root / "data" / "clean").mkdir(parents=True, exist_ok=True)
        df.to_parquet(root / "data" / "clean" / "dataset.parquet", index=False)

        # Mock run_backtest to avoid full pipeline
        mock_result = BacktestResult(
            metrics={"accuracy": 0.82, "brier": 0.19},
            per_fold_metrics=[{"accuracy": 0.80}, {"accuracy": 0.84}],
            predictions=pd.DataFrame({"y_true": [0, 1], "y_pred": [0.1, 0.9]}),
            models_trained=2,
            models_cached=0,
            models_failed=[],
            duration_s=1.5,
            meta_coefficients={"lr": 0.5, "xgb": 0.5},
        )

        with patch(
            "harness.app.workspace.manager.run_backtest", return_value=mock_result
        ):
            result = ws.run_experiment(
                experiment_type="baseline",
                hypothesis="Initial baseline",
                params={
                    "models": {
                        "lr": {"model_type": "logistic"},
                    }
                },
            )

        assert result.metrics["accuracy"] == 0.82
        # Version was created
        versions = ws.versions.list_versions()
        assert len(versions) == 1
        assert versions[0].experiment_type == "baseline"
        # Run results were written
        run_dir = root / "versions" / versions[0].id / "run"
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "predictions.parquet").exists()
        assert (run_dir / "diagnostics.json").exists()
        assert (run_dir / "eval_report.json").exists()


class TestConclude:
    def test_conclude_updates_meta(self, initialized_workspace):
        ws = initialized_workspace
        # Manually create a version
        meta = VersionMeta(id="v001", hypothesis="test")
        ws.versions.create_version(meta, ws.config)

        ws.conclude_experiment("v001", conclusion="Worked well", verdict="improved")
        updated = ws.versions.get_version("v001")
        assert updated.conclusion == "Worked well"
        assert updated.verdict == "improved"

    def test_rejects_invalid_verdict(self, initialized_workspace):
        ws = initialized_workspace
        ws.versions.create_version(VersionMeta(id="v001"), ws.config)
        with pytest.raises(ValueError, match="Invalid verdict"):
            ws.conclude_experiment("v001", conclusion="Nope", verdict="keep")


class TestRealExperimentWorkflow:
    def test_unmocked_baseline_and_child_experiment(self, initialized_workspace):
        ws = initialized_workspace
        root = ws._root
        rng = np.random.RandomState(123)
        n = 180
        feature = rng.randn(n)
        pd.DataFrame(
            {
                "feature": feature,
                "noise": rng.randn(n),
                "target": (feature + rng.randn(n) * 0.5 > 0).astype(int),
            }
        ).to_parquet(root / "data" / "clean" / "dataset.parquet", index=False)

        baseline = ws.run_experiment(
            "baseline",
            "Establish a leakage-free logistic baseline",
            {"models": {"lr": {"model_type": "logistic"}}},
        )
        assert baseline.metrics["accuracy"] > 0.6
        assert ws.versions.get_current() == "v001"

        child = ws.run_experiment(
            "hyperparameter",
            "Test stronger regularization",
            {"model_name": "lr", "params": {"C": 0.5}},
        )
        assert "accuracy" in child.metrics
        meta = ws.versions.get_version("v002")
        assert meta.parent == "v001"
        assert meta.data_hash.startswith("sha256:")
        assert (root / "versions" / "v002" / "diff.yaml").exists()
        assert (root / "versions" / "v002" / "run" / "model.bundle").exists()
        assert (root / "versions" / "v002" / "run" / "explainability.json").exists()
        eval_report = json.loads(
            (root / "versions" / "v002" / "run" / "eval_report.json").read_text()
        )
        assert "probability_accuracy" in eval_report["dimensions"]
        assert ws.config.read_models().models["lr"].params["C"] == 0.5

        with pytest.raises(ValueError, match="requires changed clean data"):
            ws.run_experiment("data_refresh", "No new data yet", {})

        # Metrics from a parent evaluated on old data are not a valid baseline.
        refreshed = pd.read_parquet(root / "data" / "clean" / "dataset.parquet")
        refreshed.loc[len(refreshed)] = [0.25, -0.5, 1]
        refreshed.to_parquet(
            root / "data" / "clean" / "dataset.parquet", index=False
        )
        refresh = ws.run_experiment(
            "data_refresh",
            "Re-establish the parent configuration on refreshed data",
            {},
        )
        refreshed_report = json.loads(
            (root / "versions" / "v003" / "run" / "eval_report.json").read_text()
        )
        assert refresh.metrics
        assert all(
            not dimension["comparisons"]
            for dimension in refreshed_report["dimensions"].values()
        )
        with pytest.raises(ValueError, match="different datasets"):
            ws.versions.compare("v002", "v003")

        ws.run_experiment(
            "hyperparameter",
            "Tune against the refreshed baseline",
            {"model_name": "lr", "params": {"C": 0.25}},
        )
        assert ws.versions.compare("v003", "v004")

    def test_all_advertised_experiment_types_change_config(self, initialized_workspace):
        ws = initialized_workspace
        root = ws._root
        pd.DataFrame(
            {
                "feature": np.arange(30, dtype=float),
                "other": np.arange(30, dtype=float) * 2,
                "target": np.tile([0, 1], 15),
            }
        ).to_parquet(root / "data" / "clean" / "dataset.parquet", index=False)
        result = BacktestResult(metrics={"accuracy": 0.5})

        experiments = [
            ("baseline", {"models": {"lr": {"model_type": "logistic"}}}),
            (
                "feature",
                {
                    "name": "sum_feature",
                    "type": "pairwise",
                    "formula": "feature + other",
                },
            ),
            ("model", {"name": "rf", "model_type": "random_forest"}),
            ("hyperparameter", {"model_name": "lr", "params": {"C": 0.5}}),
            ("ensemble", {"temperature": 1.2}),
            ("calibration", {"method": "platt"}),
            ("cv_strategy", {"n_folds": 3}),
            ("feature_selection", {"model_name": "lr", "features": ["feature"]}),
            ("data_refresh", {}),
        ]
        with patch("harness.app.workspace.manager.run_backtest", return_value=result):
            for index, (kind, params) in enumerate(experiments, start=1):
                if kind == "data_refresh":
                    refreshed = pd.read_parquet(
                        root / "data" / "clean" / "dataset.parquet"
                    )
                    refreshed.loc[len(refreshed)] = [100.0, 200.0, 1]
                    refreshed.to_parquet(
                        root / "data" / "clean" / "dataset.parquet", index=False
                    )
                ws.run_experiment(kind, f"Test {kind}", params)
                assert ws.versions.get_current() == f"v{index:03d}"

        assert len(ws.versions.list_versions()) == len(experiments)

    def test_failed_experiment_preserves_current_and_config(
        self, initialized_workspace
    ):
        ws = initialized_workspace
        root = ws._root
        pd.DataFrame({"feature": [0, 1], "target": [0, 1]}).to_parquet(
            root / "data" / "clean" / "dataset.parquet", index=False
        )
        result = BacktestResult(metrics={"accuracy": 0.5})
        with patch("harness.app.workspace.manager.run_backtest", return_value=result):
            ws.run_experiment(
                "baseline", "baseline", {"models": {"lr": {"model_type": "logistic"}}}
            )
        before = ws.config.read_models().model_dump()

        with patch(
            "harness.app.workspace.manager.run_backtest",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                ws.run_experiment(
                    "hyperparameter",
                    "failing child",
                    {"model_name": "lr", "params": {"C": 99}},
                )

        assert ws.versions.get_current() == "v001"
        assert ws.config.read_models().model_dump() == before
        assert len(ws.versions.list_versions()) == 1


class TestSwitchVersion:
    def test_switch_restores_config(self, initialized_workspace):
        ws = initialized_workspace
        from harness.ml.config.project import ProjectConfig

        # Create v001 with regression config
        ws.config.write_project(
            ProjectConfig(task_type="regression", target_column="score")
        )
        ws.versions.create_version(VersionMeta(id="v001"), ws.config)

        # Change config to binary
        ws.config.write_project(
            ProjectConfig(task_type="binary", target_column="label")
        )
        ws.versions.create_version(VersionMeta(id="v002", parent="v001"), ws.config)

        # Switch back to v001
        ws.switch_version("v001")
        project = ws.config.read_project()
        assert project.task_type == "regression"
        assert project.target_column == "score"


class TestStatus:
    def test_status_no_versions(self, initialized_workspace):
        ws = initialized_workspace
        s = ws.status()
        assert s["workspace"] == ws._root.name
        assert s["current_version"] is None
        assert s["metrics"] == {}
        assert s["version_count"] == 0

    def test_status_with_version(self, initialized_workspace):
        ws = initialized_workspace
        meta = VersionMeta(id="v001", metrics={"accuracy": 0.85})
        ws.versions.create_version(meta, ws.config)
        (ws._root / "current").write_text("v001")

        s = ws.status()
        assert s["current_version"] == "v001"
        assert s["metrics"]["accuracy"] == 0.85
        assert s["version_count"] == 1
