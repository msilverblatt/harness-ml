"""Tests for ProjectConfig Pydantic models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyml.runner.schema import (
    BacktestConfig,
    DataConfig,
    EnsembleDef,
    ExperimentDef,
    FeatureDecl,
    FeaturesConfig,
    GuardrailDef,
    ModelDef,
    ProjectConfig,
    ServerDef,
    ServerToolDef,
    SourceDecl,
)


# -----------------------------------------------------------------------
# Helpers — minimal valid sub-configs
# -----------------------------------------------------------------------

def _minimal_data() -> dict:
    return {"raw_dir": "data/raw", "processed_dir": "data/processed", "features_dir": "data/features"}


def _minimal_model() -> dict:
    return {
        "type": "xgboost",
        "features": ["feat_a", "feat_b"],
        "params": {"max_depth": 3},
    }


def _minimal_ensemble() -> dict:
    return {"method": "stacked"}


def _minimal_backtest() -> dict:
    return {"cv_strategy": "leave_one_season_out", "seasons": [2023, 2024]}


def _minimal_project() -> dict:
    return {
        "data": _minimal_data(),
        "models": {"xgb_core": _minimal_model()},
        "ensemble": _minimal_ensemble(),
        "backtest": _minimal_backtest(),
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestMinimalConfig:
    """Minimal valid config: data + model + ensemble + backtest."""

    def test_minimal_valid(self):
        cfg = ProjectConfig(**_minimal_project())
        assert cfg.data.raw_dir == "data/raw"
        assert "xgb_core" in cfg.models
        assert cfg.models["xgb_core"].type == "xgboost"
        assert cfg.ensemble.method == "stacked"
        assert cfg.backtest.cv_strategy == "leave_one_season_out"

    def test_defaults_applied(self):
        cfg = ProjectConfig(**_minimal_project())
        assert cfg.models["xgb_core"].active is True
        assert cfg.models["xgb_core"].mode == "classifier"
        assert cfg.models["xgb_core"].n_seeds == 1
        assert cfg.ensemble.temperature == 1.0
        assert cfg.ensemble.clip_floor == 0.0
        assert cfg.features is None
        assert cfg.sources is None
        assert cfg.experiments is None
        assert cfg.guardrails is None
        assert cfg.server is None


class TestFeatureDeclarations:
    """Config with features section."""

    def test_with_features(self):
        proj = _minimal_project()
        proj["features"] = {
            "adj_efficiency": {
                "module": "my_project.features.efficiency",
                "function": "compute_adj_efficiency",
                "category": "efficiency",
                "level": "team",
                "columns": ["adj_oe", "adj_de", "adj_net"],
                "nan_strategy": "median",
            }
        }
        cfg = ProjectConfig(**proj)
        assert "adj_efficiency" in cfg.features
        feat = cfg.features["adj_efficiency"]
        assert feat.module == "my_project.features.efficiency"
        assert feat.function == "compute_adj_efficiency"
        assert feat.columns == ["adj_oe", "adj_de", "adj_net"]

    def test_feature_nan_strategy_default(self):
        proj = _minimal_project()
        proj["features"] = {
            "simple": {
                "module": "mod",
                "function": "fn",
                "category": "cat",
                "level": "team",
                "columns": ["col1"],
            }
        }
        cfg = ProjectConfig(**proj)
        assert cfg.features["simple"].nan_strategy == "median"


class TestSourceDeclarations:
    """Config with sources section."""

    def test_with_sources(self):
        proj = _minimal_project()
        proj["sources"] = {
            "kenpom": {
                "module": "my_project.sources.kenpom",
                "function": "scrape_kenpom",
                "category": "external",
                "temporal_safety": "pre_tournament",
                "outputs": ["data/raw/kenpom/"],
                "leakage_notes": "Uses pre-tournament snapshots only",
            }
        }
        cfg = ProjectConfig(**proj)
        assert "kenpom" in cfg.sources
        src = cfg.sources["kenpom"]
        assert src.temporal_safety == "pre_tournament"
        assert src.leakage_notes == "Uses pre-tournament snapshots only"


class TestInvalidModelType:
    """Invalid model type should be rejected."""

    def test_invalid_model_type(self):
        proj = _minimal_project()
        proj["models"]["bad"] = {
            "type": "totally_fake_model",
            "features": ["a"],
            "params": {},
        }
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(**proj)
        assert "totally_fake_model" in str(exc_info.value)

    def test_register_custom_type(self):
        """Extensible type registration."""
        ModelDef.register_type("my_custom_model")
        try:
            proj = _minimal_project()
            proj["models"]["custom"] = {
                "type": "my_custom_model",
                "features": ["a"],
                "params": {},
            }
            cfg = ProjectConfig(**proj)
            assert cfg.models["custom"].type == "my_custom_model"
        finally:
            ModelDef.unregister_type("my_custom_model")


class TestInvalidCVStrategy:
    """Invalid CV strategy should be rejected."""

    def test_invalid_cv_strategy(self):
        proj = _minimal_project()
        proj["backtest"]["cv_strategy"] = "random_shuffle"
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(**proj)
        assert "cv_strategy" in str(exc_info.value) or "random_shuffle" in str(exc_info.value)


class TestInvalidEnsembleMethod:
    """Invalid ensemble method should be rejected."""

    def test_invalid_ensemble_method(self):
        proj = _minimal_project()
        proj["ensemble"]["method"] = "magic_averaging"
        with pytest.raises(ValidationError) as exc_info:
            ProjectConfig(**proj)
        assert "magic_averaging" in str(exc_info.value)


class TestSerializationRoundtrip:
    """Serialization roundtrip test."""

    def test_roundtrip(self):
        proj = _minimal_project()
        cfg = ProjectConfig(**proj)
        dumped = cfg.model_dump()
        cfg2 = ProjectConfig(**dumped)
        assert cfg2.data.raw_dir == cfg.data.raw_dir
        assert cfg2.models["xgb_core"].type == cfg.models["xgb_core"].type
        assert cfg2.ensemble.method == cfg.ensemble.method
        assert cfg2.backtest.seasons == cfg.backtest.seasons


class TestGuardrailConfig:
    """Guardrail config section."""

    def test_guardrail(self):
        proj = _minimal_project()
        proj["guardrails"] = {
            "feature_leakage_denylist": ["kp_adj_o", "kp_adj_d"],
            "critical_paths": ["data/features/"],
            "naming_pattern": r"exp-\d{3}-\w+",
            "rate_limit_seconds": 30,
        }
        cfg = ProjectConfig(**proj)
        assert cfg.guardrails.feature_leakage_denylist == ["kp_adj_o", "kp_adj_d"]
        assert cfg.guardrails.rate_limit_seconds == 30

    def test_guardrail_defaults(self):
        gd = GuardrailDef()
        assert gd.feature_leakage_denylist == []
        assert gd.critical_paths == []
        assert gd.naming_pattern is None
        assert gd.rate_limit_seconds is None


class TestServerConfig:
    """Server config section."""

    def test_server(self):
        proj = _minimal_project()
        proj["server"] = {
            "name": "mm-pipeline",
            "tools": {
                "pipeline": {
                    "command": "pipelines/train.py",
                    "args": ["--mode", "train"],
                    "guardrails": ["sanity_check"],
                    "description": "Train all models",
                    "timeout": 600,
                }
            },
            "inspection": ["config", "models"],
        }
        cfg = ProjectConfig(**proj)
        assert cfg.server.name == "mm-pipeline"
        assert "pipeline" in cfg.server.tools
        tool = cfg.server.tools["pipeline"]
        assert tool.command == "pipelines/train.py"
        assert tool.timeout == 600

    def test_server_tool_defaults(self):
        tool = ServerToolDef(command="run.py")
        assert tool.args == []
        assert tool.guardrails == []
        assert tool.description is None
        assert tool.timeout is None


class TestFullConfig:
    """Full config with all sections populated."""

    def test_full_config(self):
        proj = _minimal_project()
        proj["features"] = {
            "eff": {
                "module": "m",
                "function": "f",
                "category": "c",
                "level": "team",
                "columns": ["x"],
            }
        }
        proj["sources"] = {
            "src1": {
                "module": "m",
                "function": "f",
                "category": "external",
                "temporal_safety": "pre_tournament",
                "outputs": ["data/"],
            }
        }
        proj["experiments"] = {
            "naming_pattern": r"exp-\d{3}-\w+",
            "log_path": "EXPERIMENT_LOG.md",
            "experiments_dir": "experiments/",
        }
        proj["guardrails"] = {
            "feature_leakage_denylist": ["bad_feat"],
        }
        proj["server"] = {
            "name": "my-server",
            "tools": {},
        }
        cfg = ProjectConfig(**proj)
        assert cfg.features is not None
        assert cfg.sources is not None
        assert cfg.experiments is not None
        assert cfg.guardrails is not None
        assert cfg.server is not None


class TestModelDefExtensions:
    """Extended ModelDef fields for regression, GNN, survival, etc."""

    def test_prediction_type_margin(self):
        m = ModelDef(
            type="mlp",
            features=["diff_adj_net"],
            prediction_type="margin",
        )
        assert m.prediction_type == "margin"

    def test_train_seasons_last_n(self):
        m = ModelDef(
            type="xgboost",
            features=["diff_seed_num"],
            train_seasons="last_5",
        )
        assert m.train_seasons == "last_5"

    def test_cdf_scale(self):
        m = ModelDef(
            type="xgboost_regression",
            features=["diff_scoring_margin"],
            cdf_scale=5.5,
        )
        assert m.cdf_scale == 5.5

    def test_feature_sets(self):
        m = ModelDef(
            type="gnn",
            feature_sets=["roster", "game_graph"],
        )
        assert m.feature_sets == ["roster", "game_graph"]
        assert m.features == []

    def test_pre_calibration_string(self):
        m = ModelDef(
            type="mlp",
            features=["diff_adj_net"],
            pre_calibration="spline",
        )
        assert m.pre_calibration == "spline"

    def test_xgboost_regression_type(self):
        m = ModelDef(type="xgboost_regression", features=["a"])
        assert m.type == "xgboost_regression"

    def test_gnn_type(self):
        m = ModelDef(type="gnn", features=[])
        assert m.type == "gnn"

    def test_survival_type(self):
        m = ModelDef(type="survival", features=["seed_num"])
        assert m.type == "survival"

    def test_defaults_backward_compatible(self):
        """Old config format (without new fields) still validates."""
        m = ModelDef(type="xgboost", features=["feat_a"])
        assert m.feature_sets == []
        assert m.prediction_type is None
        assert m.train_seasons == "all"
        assert m.pre_calibration is None
        assert m.cdf_scale is None


class TestEnsembleDefExtensions:
    """Extended EnsembleDef fields for pre-calibration, spline settings, etc."""

    def test_pre_calibration_model_map(self):
        e = EnsembleDef(
            method="stacked",
            pre_calibration={"v2_mlp_margin": "spline"},
        )
        assert e.pre_calibration == {"v2_mlp_margin": "spline"}

    def test_calibration_string(self):
        e = EnsembleDef(method="stacked", calibration="spline")
        assert e.calibration == "spline"

    def test_spline_settings(self):
        e = EnsembleDef(
            method="stacked",
            spline_prob_max=0.985,
            spline_n_bins=20,
        )
        assert e.spline_prob_max == 0.985
        assert e.spline_n_bins == 20

    def test_meta_features(self):
        e = EnsembleDef(
            method="stacked",
            meta_features=["diff_seed_num"],
        )
        assert e.meta_features == ["diff_seed_num"]

    def test_seed_compression(self):
        e = EnsembleDef(
            method="stacked",
            seed_compression=0.5,
            seed_compression_threshold=4,
        )
        assert e.seed_compression == 0.5
        assert e.seed_compression_threshold == 4

    def test_defaults_backward_compatible(self):
        """Old config format (without new fields) still validates."""
        e = EnsembleDef(method="stacked")
        assert e.pre_calibration == {}
        assert e.calibration == "spline"
        assert e.spline_prob_max == 0.985
        assert e.spline_n_bins == 20
        assert e.meta_features == []
        assert e.seed_compression == 0.0
        assert e.seed_compression_threshold == 4


class TestFeaturesConfig:
    """FeaturesConfig defaults and custom values."""

    def test_defaults(self):
        fc = FeaturesConfig()
        assert fc.first_season == 2003
        assert fc.momentum_window == 10

    def test_custom_values(self):
        fc = FeaturesConfig(first_season=2010, momentum_window=5)
        assert fc.first_season == 2010
        assert fc.momentum_window == 5


class TestDataConfigExtensions:
    """Extended DataConfig fields."""

    def test_new_dir_fields(self):
        d = DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
            predictions_dir="data/predictions",
            survival_dir="data/survival",
            outputs_dir="outputs",
        )
        assert d.predictions_dir == "data/predictions"
        assert d.survival_dir == "data/survival"
        assert d.outputs_dir == "outputs"

    def test_new_dir_fields_default_none(self):
        d = DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
        )
        assert d.predictions_dir is None
        assert d.survival_dir is None
        assert d.outputs_dir is None


class TestProjectConfigFeatureConfig:
    """ProjectConfig with feature_config section."""

    def test_with_feature_config(self):
        proj = _minimal_project()
        proj["feature_config"] = {"first_season": 2003, "momentum_window": 10}
        cfg = ProjectConfig(**proj)
        assert cfg.feature_config is not None
        assert cfg.feature_config.first_season == 2003
        assert cfg.feature_config.momentum_window == 10

    def test_without_feature_config(self):
        proj = _minimal_project()
        cfg = ProjectConfig(**proj)
        assert cfg.feature_config is None


class TestOldFormatStillValidates:
    """Ensure that configs using only old fields still validate."""

    def test_minimal_still_works(self):
        """The original _minimal_project() must still pass."""
        cfg = ProjectConfig(**_minimal_project())
        assert cfg.data.raw_dir == "data/raw"
        assert cfg.models["xgb_core"].type == "xgboost"
        assert cfg.ensemble.method == "stacked"
