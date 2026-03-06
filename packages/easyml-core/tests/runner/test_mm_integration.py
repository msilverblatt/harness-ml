"""Integration tests validating mm pipeline config through easyml-runner.

Tests that the actual mm pipeline configuration validates correctly,
and that a simplified version can produce ensemble predictions with
synthetic data through the full PipelineRunner.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from easyml.core.runner.pipeline import PipelineRunner
from easyml.core.runner.validator import validate_project

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _copy_fixture_as(fixture_name: str, dest: Path) -> None:
    """Copy a fixture file from tests/fixtures/ to a destination path."""
    src = FIXTURES_DIR / fixture_name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _setup_mm_config_dir(tmp_path: Path) -> Path:
    """Set up a config directory with mm fixture files using standard names.

    The validator expects pipeline.yaml, models.yaml, ensemble.yaml.
    Our fixture files are named mm_pipeline.yaml, mm_models.yaml, mm_ensemble.yaml.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _copy_fixture_as("mm_pipeline.yaml", config_dir / "pipeline.yaml")
    _copy_fixture_as("mm_models.yaml", config_dir / "models.yaml")
    _copy_fixture_as("mm_ensemble.yaml", config_dir / "ensemble.yaml")
    return config_dir


def _make_synthetic_matchup_data(
    path: Path,
    feature_names: list[str],
    n_rows: int = 300,
    n_seasons: int = 3,
    season_start: int = 2022,
) -> None:
    """Create synthetic features parquet with specified features.

    Generates random data with result, season, margin, diff_prior,
    and all requested diff_* feature columns.
    """
    rng = np.random.default_rng(42)
    seasons = rng.choice(
        list(range(season_start, season_start + n_seasons)),
        size=n_rows,
    )
    result = rng.integers(0, 2, size=n_rows)
    margin = rng.standard_normal(n_rows) * 10

    data = {
        "season": seasons,
        "result": result,
        "margin": margin,
        "diff_prior": rng.integers(-15, 16, size=n_rows).astype(float),
    }

    for feat in feature_names:
        if feat not in data:
            data[feat] = rng.standard_normal(n_rows)

    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


# -----------------------------------------------------------------------
# Test: mm config validation
# -----------------------------------------------------------------------

class TestMmConfigValidates:
    """The mm pipeline's actual config must validate through easyml-runner."""

    def test_mm_config_validates(self, tmp_path):
        """Full mm config (pipeline, models, ensemble) validates without error."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        config = result.config
        assert len(config.models) >= 5
        assert config.ensemble.method == "stacked"
        assert config.ensemble.meta_learner.get("C") == 2.5

    def test_regression_model_type(self, tmp_path):
        """xgboost_regression type is recognized and parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        spread = result.config.models["xgb_spread_broad"]
        assert spread.type == "xgboost_regression"

    def test_mlp_margin_model(self, tmp_path):
        """MLP margin model has correct prediction_type and params."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        mlp = result.config.models["v2_mlp_margin"]
        assert mlp.prediction_type == "margin"
        assert mlp.type == "mlp"
        # n_seeds is in params for mm config (not top-level ModelDef field)
        assert mlp.params.get("n_seeds") == 5

    def test_ensemble_pre_calibration(self, tmp_path):
        """Ensemble pre-calibration config for MLP is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert "v2_mlp_margin" in result.config.ensemble.pre_calibration
        assert result.config.ensemble.pre_calibration["v2_mlp_margin"] == "spline"

    def test_ensemble_post_calibration(self, tmp_path):
        """Post-calibration is spline."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.ensemble.calibration == "spline"

    def test_all_model_types_recognized(self, tmp_path):
        """All model types in mm config are recognized by the schema."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        expected_types = {
            "logistic_regression",
            "xgboost",
            "xgboost_regression",
            "mlp",
            "tabnet",
            "catboost",
            "elastic_net",
            "gnn",
            "survival",
        }
        actual_types = {m.type for m in result.config.models.values()}
        assert expected_types == actual_types

    def test_bracket_key_silently_ignored(self, tmp_path):
        """Unknown top-level keys in pipeline.yaml (like 'bracket') are silently ignored."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"
        # bracket is in the fixture but should not appear in config
        assert not hasattr(result.config, "bracket")

    def test_feature_config_parsed(self, tmp_path):
        """pipeline.yaml 'features' section maps to feature_config."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.feature_config is not None
        assert result.config.feature_config.first_period == 2003
        assert result.config.feature_config.momentum_window == 10

    def test_backtest_fold_values(self, tmp_path):
        """Backtest fold values are correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        fold_values = result.config.backtest.fold_values
        assert len(fold_values) == 10
        assert 2015 in fold_values
        assert 2025 in fold_values
        # 2020 should not be present (no tournament that year)
        assert 2020 not in fold_values

    def test_exclude_models_list(self, tmp_path):
        """Exclude models list from ensemble config is properly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        exclude = result.config.ensemble.exclude_models
        assert "cat_broad" in exclude
        assert "gnn_hetero" in exclude
        assert "survival_hazard" in exclude
        assert "v2_tabtransformer_clf" in exclude

    def test_meta_learner_c(self, tmp_path):
        """Meta-learner C=2.5 is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.ensemble.meta_learner["C"] == 2.5

    def test_ensemble_temperature_and_clip(self, tmp_path):
        """Temperature and clip_floor are parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.ensemble.temperature == 1.0
        assert result.config.ensemble.clip_floor == 0.0

    def test_spline_params(self, tmp_path):
        """Spline calibration parameters are parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.ensemble.spline_prob_max == 0.985
        assert result.config.ensemble.spline_n_bins == 20

    def test_logreg_seed_model(self, tmp_path):
        """logreg_seed model is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        logreg = result.config.models["logreg_seed"]
        assert logreg.type == "logistic_regression"
        assert logreg.features == ["diff_prior"]
        assert logreg.params.get("C") == 1.0

    def test_catboost_model(self, tmp_path):
        """cat_roster model is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        cat = result.config.models["cat_roster"]
        assert cat.type == "catboost"
        assert len(cat.features) == 7
        assert cat.params.get("depth") == 4

    def test_elastic_net_model(self, tmp_path):
        """v2_elasticnet_clf model is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        en = result.config.models["v2_elasticnet_clf"]
        assert en.type == "elastic_net"
        assert en.params.get("C") == 0.01
        assert en.params.get("l1_ratio") == 1.0

    def test_tabnet_model(self, tmp_path):
        """v2_tabnet_clf model is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        tabnet = result.config.models["v2_tabnet_clf"]
        assert tabnet.type == "tabnet"
        assert tabnet.params.get("n_d") == 16

    def test_gnn_model(self, tmp_path):
        """gnn_hetero model validates (custom type, no standard features)."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        gnn = result.config.models["gnn_hetero"]
        assert gnn.type == "gnn"
        assert gnn.feature_sets == []
        assert gnn.params.get("embedding_dim") == 64

    def test_survival_model(self, tmp_path):
        """survival_hazard model validates."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        surv = result.config.models["survival_hazard"]
        assert surv.type == "survival"
        assert "seed_num" in surv.features

    def test_xgb_travel_model(self, tmp_path):
        """xgb_travel regression model validates with travel features."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        travel = result.config.models["xgb_travel"]
        assert travel.type == "xgboost_regression"
        assert "diff_travel_distance_mi" in travel.features
        assert "diff_travel_cbrt_distance" in travel.features

    def test_data_config(self, tmp_path):
        """Data config is correctly parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.data.task == "classification"
        assert result.config.data.raw_dir == "data/raw"
        assert result.config.data.processed_dir == "data/processed"
        assert result.config.data.features_dir == "data/features"

    def test_availability_adjustment(self, tmp_path):
        """Availability adjustment is parsed."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        assert result.config.ensemble.availability_adjustment == 0.1

    def test_model_count(self, tmp_path):
        """All 10 models in the fixture are loaded."""
        config_dir = _setup_mm_config_dir(tmp_path)
        result = validate_project(config_dir)
        assert result.valid, f"Validation failed:\n{result.format()}"

        expected_models = {
            "logreg_seed", "xgb_core", "xgb_spread_broad",
            "v2_mlp_margin", "v2_tabnet_clf", "cat_roster",
            "v2_elasticnet_clf", "xgb_travel", "gnn_hetero",
            "survival_hazard",
        }
        assert set(result.config.models.keys()) == expected_models


# -----------------------------------------------------------------------
# Test: backtest pipeline smoke test with synthetic data
# -----------------------------------------------------------------------

class TestMmBacktestSmoke:
    """Smoke test: PipelineRunner can run backtest with mm-style config
    and synthetic data using trainable model types."""

    def test_backtest_with_synthetic_data(self, tmp_path):
        """Backtest with xgboost + logistic_regression on synthetic data
        produces ensemble predictions and metrics."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Use a simplified config that only includes trainable models
        # with features that match the synthetic data
        smoke_features = [
            "diff_prior",
            "diff_feat_a",
            "diff_feat_b",
            "diff_feat_c",
        ]

        _write_yaml(
            config_dir / "pipeline.yaml",
            {
                "data": {
                    "raw_dir": str(tmp_path / "data" / "raw"),
                    "processed_dir": str(tmp_path / "data" / "processed"),
                    "features_dir": str(tmp_path / "data" / "features"),
                },
                "backtest": {
                    "cv_strategy": "leave_one_out",
                    "fold_column": "season",
                    "fold_values": [2022, 2023, 2024],
                    "metrics": ["brier", "accuracy"],
                },
            },
        )

        _write_yaml(
            config_dir / "models.yaml",
            {
                "models": {
                    "logreg_base": {
                        "type": "logistic_regression",
                        "features": ["diff_prior"],
                        "train_folds": "all",
                        "params": {"C": 1.0, "max_iter": 200},
                    },
                    "xgb_core": {
                        "type": "xgboost",
                        "features": smoke_features,
                        "train_folds": "all",
                        "params": {
                            "max_depth": 3,
                            "learning_rate": 0.1,
                            "n_estimators": 50,
                        },
                    },
                    "xgb_spread": {
                        "type": "xgboost_regression",
                        "features": smoke_features,
                        "train_folds": "all",
                        "params": {
                            "max_depth": 1,
                            "learning_rate": 0.1,
                            "n_estimators": 50,
                            "objective": "reg:squarederror",
                        },
                    },
                }
            },
        )

        _write_yaml(
            config_dir / "ensemble.yaml",
            {
                "ensemble": {
                    "method": "stacked",
                    "meta_learner": {"C": 2.5},
                    "calibration": "none",
                    "temperature": 1.0,
                    "clip_floor": 0.0,
                }
            },
        )

        # Create synthetic data with 5 seasons so temporal filtering
        # in train_single_model always finds prior data for each test fold
        features_dir = tmp_path / "data" / "features"
        _make_synthetic_matchup_data(
            features_dir / "features.parquet",
            feature_names=smoke_features,
            n_rows=500,
            n_seasons=5,
            season_start=2020,
        )

        # Run backtest
        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "metrics" in result
        assert "brier" in result["metrics"]
        assert "accuracy" in result["metrics"]
        assert result["metrics"]["brier"] >= 0
        assert result["metrics"]["brier"] <= 1.0
        assert result["metrics"]["accuracy"] >= 0
        assert result["metrics"]["accuracy"] <= 1.0
        # 3 test folds (2022, 2023, 2024), each with prior data available
        assert len(result["per_fold"]) == 3
        assert len(result["models_trained"]) == 3

    def test_backtest_with_stacked_calibration(self, tmp_path):
        """Stacked backtest with spline post-calibration produces results."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        smoke_features = [
            "diff_prior",
            "diff_feat_a",
            "diff_feat_b",
        ]

        _write_yaml(
            config_dir / "pipeline.yaml",
            {
                "data": {
                    "raw_dir": str(tmp_path / "data" / "raw"),
                    "processed_dir": str(tmp_path / "data" / "processed"),
                    "features_dir": str(tmp_path / "data" / "features"),
                },
                "backtest": {
                    "cv_strategy": "leave_one_out",
                    "fold_column": "season",
                    "fold_values": [2022, 2023, 2024],
                    "metrics": ["brier", "accuracy"],
                },
            },
        )

        _write_yaml(
            config_dir / "models.yaml",
            {
                "models": {
                    "logreg_a": {
                        "type": "logistic_regression",
                        "features": smoke_features,
                        "train_folds": "all",
                        "params": {"C": 1.0, "max_iter": 200},
                    },
                    "logreg_b": {
                        "type": "logistic_regression",
                        "features": smoke_features,
                        "train_folds": "all",
                        "params": {"C": 0.5, "max_iter": 200},
                    },
                }
            },
        )

        _write_yaml(
            config_dir / "ensemble.yaml",
            {
                "ensemble": {
                    "method": "stacked",
                    "meta_learner": {"C": 1.0},
                    "calibration": "spline",
                    "spline_prob_max": 0.985,
                    "spline_n_bins": 20,
                    "temperature": 1.0,
                    "clip_floor": 0.0,
                }
            },
        )

        features_dir = tmp_path / "data" / "features"
        _make_synthetic_matchup_data(
            features_dir / "features.parquet",
            feature_names=smoke_features,
            n_rows=500,
            n_seasons=5,
            season_start=2020,
        )

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "brier" in result["metrics"]
        assert result["metrics"]["brier"] >= 0

    def test_backtest_with_exclude_models(self, tmp_path):
        """Exclude models from ensemble works correctly."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        smoke_features = ["diff_prior", "diff_feat_a"]

        _write_yaml(
            config_dir / "pipeline.yaml",
            {
                "data": {
                    "raw_dir": str(tmp_path / "data" / "raw"),
                    "processed_dir": str(tmp_path / "data" / "processed"),
                    "features_dir": str(tmp_path / "data" / "features"),
                },
                "backtest": {
                    "cv_strategy": "leave_one_out",
                    "fold_column": "season",
                    "fold_values": [2022, 2023, 2024],
                    "metrics": ["brier", "accuracy"],
                },
            },
        )

        _write_yaml(
            config_dir / "models.yaml",
            {
                "models": {
                    "logreg_active": {
                        "type": "logistic_regression",
                        "features": smoke_features,
                        "train_folds": "all",
                        "params": {"C": 1.0, "max_iter": 200},
                    },
                    "logreg_excluded": {
                        "type": "logistic_regression",
                        "features": smoke_features,
                        "train_folds": "all",
                        "params": {"C": 0.1, "max_iter": 200},
                    },
                }
            },
        )

        _write_yaml(
            config_dir / "ensemble.yaml",
            {
                "ensemble": {
                    "method": "average",
                    "exclude_models": ["logreg_excluded"],
                }
            },
        )

        features_dir = tmp_path / "data" / "features"
        _make_synthetic_matchup_data(
            features_dir / "features.parquet",
            feature_names=smoke_features,
            n_rows=500,
            n_seasons=5,
            season_start=2020,
        )

        runner = PipelineRunner(
            project_dir=str(tmp_path),
            config_dir=str(config_dir),
        )
        runner.load()
        result = runner.backtest()

        assert result["status"] == "success"
        assert "logreg_excluded" not in result["models_trained"]
        assert "logreg_active" in result["models_trained"]


# -----------------------------------------------------------------------
# Test: all exports importable
# -----------------------------------------------------------------------

class TestAllExportsImportable:
    """Verify all public symbols are importable from easyml.core.runner."""

    def test_all_exports_importable(self):
        from easyml.core.runner import (
            SplineCalibrator,
            IsotonicCalibrator,
            PlattCalibrator,
            build_calibrator,
            temperature_scale,
            StackedEnsemble,
            train_meta_learner_loso,
            apply_ensemble_postprocessing,
            RunManager,
            compute_pooled_metrics,
            compute_brier_score,
            compute_ece,
            compute_calibration_curve,
            evaluate_fold_predictions,
            compute_fingerprint,
            is_cached,
            save_fingerprint,
            FeaturesConfig,
            generate_pairwise_matchups,
            predict_all_matchups,
        )

        # Verify they are the correct types
        assert callable(build_calibrator)
        assert callable(temperature_scale)
        assert callable(train_meta_learner_loso)
        assert callable(apply_ensemble_postprocessing)
        assert callable(compute_pooled_metrics)
        assert callable(compute_brier_score)
        assert callable(compute_ece)
        assert callable(compute_calibration_curve)
        assert callable(evaluate_fold_predictions)
        assert callable(compute_fingerprint)
        assert callable(is_cached)
        assert callable(save_fingerprint)
        assert callable(generate_pairwise_matchups)
        assert callable(predict_all_matchups)

    def test_calibrator_classes_importable(self):
        from easyml.core.runner import SplineCalibrator, IsotonicCalibrator, PlattCalibrator

        # Can instantiate
        s = SplineCalibrator()
        assert not s.is_fitted

        i = IsotonicCalibrator()
        assert not i.is_fitted

        p = PlattCalibrator()
        assert not p.is_fitted

    def test_stacked_ensemble_importable(self):
        from easyml.core.runner import StackedEnsemble

        ens = StackedEnsemble(model_names=["a", "b"])
        assert ens.model_names == ["a", "b"]

    def test_features_config_importable(self):
        from easyml.core.runner import FeaturesConfig

        fc = FeaturesConfig(first_period=2003, momentum_window=10)
        assert fc.first_period == 2003
        assert fc.momentum_window == 10

    def test_run_manager_importable(self):
        from easyml.core.runner import RunManager

        # Can instantiate with a path
        rm = RunManager(outputs_base=Path("/tmp/test_rm"))
        assert rm.outputs_base == Path("/tmp/test_rm")

    def test_pipeline_runner_importable(self):
        from easyml.core.runner import PipelineRunner

        # PipelineRunner is importable (already tested via other tests)
        assert PipelineRunner is not None

    def test_schema_classes_importable(self):
        from easyml.core.runner import (
            BacktestConfig,
            DataConfig,
            EnsembleDef,
            ExperimentDef,
            FeatureDecl,
            GuardrailDef,
            ModelDef,
            ProjectConfig,
            ServerDef,
            ServerToolDef,
            SourceDecl,
        )

        # All schema classes should be importable
        assert DataConfig is not None
        assert ModelDef is not None

    def test_validator_importable(self):
        from easyml.core.runner import ValidationResult, validate_project

        assert callable(validate_project)
        assert ValidationResult is not None
