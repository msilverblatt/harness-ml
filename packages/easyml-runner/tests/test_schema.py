"""Tests for ProjectConfig Pydantic models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyml.runner.schema import (
    BacktestConfig,
    ColumnCleaningRule,
    DataConfig,
    EnsembleDef,
    ExperimentDef,
    FeatureDecl,
    FeaturesConfig,
    GuardrailDef,
    InjectionDef,
    InteractionDef,
    ModelDef,
    ProjectConfig,
    ServerDef,
    ServerToolDef,
    SourceConfig,
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
    """DataConfig ML problem definition fields."""

    def test_ml_problem_fields(self):
        d = DataConfig(
            raw_dir="data/raw",
            processed_dir="data/processed",
            features_dir="data/features",
            features_file="my_data.parquet",
            task="regression",
            target_column="price",
            key_columns=["item_id", "date"],
            time_column="date",
            exclude_columns=["notes"],
            outputs_dir="outputs",
        )
        assert d.features_file == "my_data.parquet"
        assert d.task == "regression"
        assert d.target_column == "price"
        assert d.key_columns == ["item_id", "date"]
        assert d.time_column == "date"
        assert d.exclude_columns == ["notes"]
        assert d.outputs_dir == "outputs"

    def test_defaults(self):
        d = DataConfig()
        assert d.raw_dir == "data/raw"
        assert d.processed_dir == "data/processed"
        assert d.features_dir == "data/features"
        assert d.features_file == "features.parquet"
        assert d.task == "classification"
        assert d.target_column == "result"
        assert d.key_columns == []
        assert d.time_column is None
        assert d.exclude_columns == []
        assert d.outputs_dir is None

    def test_team_features_path_default(self):
        d = DataConfig()
        assert d.team_features_path is None

    def test_team_features_path_set(self):
        d = DataConfig(team_features_path="data/features/team_features.parquet")
        assert d.team_features_path == "data/features/team_features.parquet"


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


class TestInteractionDef:
    """InteractionDef parsing and validation."""

    def test_interaction_def_valid(self):
        for op in ("multiply", "add", "subtract", "divide", "abs_diff"):
            i = InteractionDef(left="feat_a", right="feat_b", op=op)
            assert i.left == "feat_a"
            assert i.right == "feat_b"
            assert i.op == op

    def test_interaction_def_invalid_op(self):
        with pytest.raises(ValidationError) as exc_info:
            InteractionDef(left="a", right="b", op="power")
        assert "power" in str(exc_info.value)


class TestInjectionDef:
    """InjectionDef parsing and validation."""

    def test_injection_def_parquet(self):
        inj = InjectionDef(
            source_type="parquet",
            path_pattern="data/external/{season}_travel.parquet",
            merge_keys=["team_id", "season"],
            columns=["travel_distance_mi", "travel_time_zones"],
        )
        assert inj.source_type == "parquet"
        assert inj.path_pattern == "data/external/{season}_travel.parquet"
        assert inj.merge_keys == ["team_id", "season"]
        assert inj.columns == ["travel_distance_mi", "travel_time_zones"]
        assert inj.fill_na == 0.0
        assert inj.callable_module is None
        assert inj.callable_function is None

    def test_injection_def_callable(self):
        inj = InjectionDef(
            source_type="callable",
            merge_keys=["team_id"],
            columns=["elo_rating"],
            callable_module="my_project.elo",
            callable_function="compute_elo",
            fill_na=-1.0,
        )
        assert inj.source_type == "callable"
        assert inj.path_pattern is None
        assert inj.callable_module == "my_project.elo"
        assert inj.callable_function == "compute_elo"
        assert inj.fill_na == -1.0


class TestProjectConfigInteractionsAndInjections:
    """ProjectConfig with interactions and injections sections."""

    def test_project_config_with_interactions(self):
        proj = _minimal_project()
        proj["interactions"] = {
            "seed_x_margin": {
                "left": "diff_seed_num",
                "right": "diff_scoring_margin",
                "op": "multiply",
            }
        }
        cfg = ProjectConfig(**proj)
        assert cfg.interactions is not None
        assert "seed_x_margin" in cfg.interactions
        inter = cfg.interactions["seed_x_margin"]
        assert inter.left == "diff_seed_num"
        assert inter.op == "multiply"

    def test_project_config_with_injections(self):
        proj = _minimal_project()
        proj["injections"] = {
            "travel": {
                "source_type": "csv",
                "path_pattern": "data/travel_{season}.csv",
                "merge_keys": ["team_id", "season"],
                "columns": ["travel_distance"],
                "fill_na": 0.0,
            }
        }
        cfg = ProjectConfig(**proj)
        assert cfg.injections is not None
        assert "travel" in cfg.injections
        inj = cfg.injections["travel"]
        assert inj.source_type == "csv"
        assert inj.columns == ["travel_distance"]


class TestBacktestConfigExtensions:
    """BacktestConfig with sliding_window and purged_kfold fields."""

    def test_backtest_config_sliding_window(self):
        bt = BacktestConfig(
            cv_strategy="sliding_window",
            seasons=[2020, 2021, 2022, 2023, 2024],
            window_size=5,
        )
        assert bt.cv_strategy == "sliding_window"
        assert bt.window_size == 5
        assert bt.n_folds is None
        assert bt.purge_gap == 1

    def test_backtest_config_purged_kfold(self):
        bt = BacktestConfig(
            cv_strategy="purged_kfold",
            seasons=[2015, 2016, 2017, 2018, 2019],
            n_folds=5,
            purge_gap=2,
        )
        assert bt.cv_strategy == "purged_kfold"
        assert bt.n_folds == 5
        assert bt.purge_gap == 2
        assert bt.window_size is None

    def test_existing_configs_still_valid(self):
        """Existing BacktestConfig without new fields still validates."""
        bt = BacktestConfig(
            cv_strategy="leave_one_season_out",
            seasons=[2023, 2024],
        )
        assert bt.window_size is None
        assert bt.n_folds is None
        assert bt.purge_gap == 1


class TestColumnCleaningRule:
    """ColumnCleaningRule schema and defaults."""

    def test_defaults(self):
        rule = ColumnCleaningRule()
        assert rule.null_strategy == "median"
        assert rule.null_fill_value is None
        assert rule.coerce_numeric is False
        assert rule.clip_outliers is None
        assert rule.log_transform is False
        assert rule.normalize == "none"

    def test_custom_values(self):
        rule = ColumnCleaningRule(
            null_strategy="constant",
            null_fill_value=0,
            coerce_numeric=True,
            clip_outliers=(1.0, 99.0),
            log_transform=True,
            normalize="zscore",
        )
        assert rule.null_strategy == "constant"
        assert rule.null_fill_value == 0
        assert rule.clip_outliers == (1.0, 99.0)

    def test_invalid_null_strategy(self):
        with pytest.raises(ValidationError):
            ColumnCleaningRule(null_strategy="invalid")

    def test_invalid_normalize(self):
        with pytest.raises(ValidationError):
            ColumnCleaningRule(normalize="invalid")


class TestSourceConfig:
    """SourceConfig schema and defaults."""

    def test_defaults(self):
        src = SourceConfig(name="test_source")
        assert src.path is None
        assert src.format == "auto"
        assert src.join_on is None
        assert src.columns is None
        assert src.temporal_safety == "unknown"
        assert src.enabled is True

    def test_with_columns(self):
        src = SourceConfig(
            name="kenpom",
            path="data/raw/kenpom.csv",
            format="csv",
            join_on=["team_id", "season"],
            columns={
                "adj_oe": ColumnCleaningRule(null_strategy="zero"),
                "adj_de": ColumnCleaningRule(coerce_numeric=True),
            },
            temporal_safety="pre_tournament",
        )
        assert src.columns["adj_oe"].null_strategy == "zero"
        assert src.columns["adj_de"].coerce_numeric is True

    def test_invalid_format(self):
        with pytest.raises(ValidationError):
            SourceConfig(name="bad", format="jsonl")

    def test_invalid_temporal_safety(self):
        with pytest.raises(ValidationError):
            SourceConfig(name="bad", temporal_safety="future")


class TestDataConfigSources:
    """DataConfig with sources section."""

    def test_data_config_with_sources(self):
        d = DataConfig(
            sources={
                "kenpom": SourceConfig(
                    name="kenpom",
                    path="data/raw/kenpom.csv",
                    temporal_safety="pre_tournament",
                ),
            },
            default_cleaning=ColumnCleaningRule(null_strategy="zero"),
        )
        assert "kenpom" in d.sources
        assert d.default_cleaning.null_strategy == "zero"

    def test_data_config_sources_default_empty(self):
        d = DataConfig()
        assert d.sources == {}
        assert d.default_cleaning.null_strategy == "median"

    def test_backward_compat_no_sources(self):
        """Existing configs without sources still validate."""
        d = DataConfig(raw_dir="data/raw", features_dir="data/features")
        assert d.sources == {}


class TestDeclarativeFeatureSchemas:
    """Test new declarative feature type schemas."""

    def test_feature_type_enum(self):
        from easyml.runner.schema import FeatureType
        assert FeatureType.TEAM == "team"
        assert FeatureType.PAIRWISE == "pairwise"
        assert FeatureType.MATCHUP == "matchup"
        assert FeatureType.REGIME == "regime"

    def test_pairwise_mode_enum(self):
        from easyml.runner.schema import PairwiseMode
        assert PairwiseMode.DIFF == "diff"
        assert PairwiseMode.RATIO == "ratio"
        assert PairwiseMode.BOTH == "both"
        assert PairwiseMode.NONE == "none"

    def test_feature_def_minimal(self):
        from easyml.runner.schema import FeatureDef, FeatureType
        fd = FeatureDef(name="adj_em", type=FeatureType.TEAM)
        assert fd.name == "adj_em"
        assert fd.type == FeatureType.TEAM
        assert fd.pairwise_mode.value == "diff"
        assert fd.enabled is True

    def test_feature_def_full(self):
        from easyml.runner.schema import FeatureDef, FeatureType, PairwiseMode
        fd = FeatureDef(
            name="adj_em", type=FeatureType.TEAM,
            source="kenpom", column="AdjEM",
            pairwise_mode=PairwiseMode.BOTH,
            category="efficiency",
            description="Adjusted efficiency margin",
            nan_strategy="zero",
        )
        assert fd.source == "kenpom"
        assert fd.column == "AdjEM"
        assert fd.pairwise_mode == PairwiseMode.BOTH

    def test_feature_def_regime(self):
        from easyml.runner.schema import FeatureDef, FeatureType
        fd = FeatureDef(name="late_season", type=FeatureType.REGIME, condition="day_num > 100")
        assert fd.condition == "day_num > 100"

    def test_feature_def_formula(self):
        from easyml.runner.schema import FeatureDef, FeatureType
        fd = FeatureDef(name="em_tempo", type=FeatureType.PAIRWISE, formula="diff_adj_em * diff_adj_tempo")
        assert fd.formula == "diff_adj_em * diff_adj_tempo"

    def test_feature_store_config_defaults(self):
        from easyml.runner.schema import FeatureStoreConfig
        fsc = FeatureStoreConfig()
        assert fsc.cache_dir == "data/features/cache"
        assert fsc.auto_pairwise is True
        assert fsc.default_pairwise_mode.value == "diff"
        assert fsc.entity_a_column == "entity_a_id"
        assert fsc.entity_b_column == "entity_b_id"
        assert fsc.entity_column == "entity_id"
        assert fsc.period_column == "period_id"

    def test_data_config_feature_store(self):
        from easyml.runner.schema import DataConfig, FeatureDef, FeatureType
        dc = DataConfig(
            feature_defs={
                "adj_em": FeatureDef(name="adj_em", type=FeatureType.TEAM, source="kenpom", column="AdjEM"),
            }
        )
        assert "adj_em" in dc.feature_defs
        assert dc.feature_store.auto_pairwise is True
