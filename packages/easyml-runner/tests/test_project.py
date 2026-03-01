"""Tests for the Project programmatic builder API."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyml.runner.project import Project


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def project_dir(tmp_path) -> Path:
    """Create a minimal project directory with sample data."""
    project = tmp_path / "test_project"
    features_dir = project / "data" / "features"
    features_dir.mkdir(parents=True)

    n = 300
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "Season": np.repeat([2015, 2016, 2017, 2018, 2019], 60),
        "TeamAWon": rng.integers(0, 2, size=n).astype(float),
        "TeamAMargin": rng.normal(0, 10, size=n),
        "diff_seed_num": rng.normal(0, 3, size=n),
        "diff_sr_srs": rng.normal(0, 5, size=n),
        "diff_win_pct": rng.uniform(-0.5, 0.5, size=n),
        "diff_ppg": rng.normal(0, 8, size=n),
        "diff_scoring_margin": rng.normal(0, 10, size=n),
        "diff_adj_net": rng.normal(0, 8, size=n),
        "diff_composite_net": rng.normal(0, 5, size=n),
        "diff_elo_rating": rng.normal(0, 100, size=n),
        "diff_prestige_wins_last5": rng.normal(0, 2, size=n),
        "diff_leaky_stat": rng.normal(0, 1, size=n),
    })

    df.to_parquet(features_dir / "matchup_features.parquet")

    # Create config dir (needed for PipelineRunner guards)
    (project / "config").mkdir(parents=True)
    (project / "models").mkdir(parents=True)

    return project


@pytest.fixture
def project(project_dir) -> Project:
    """Create a Project instance with default config."""
    p = Project(project_dir)
    p.set_data(features_dir="data/features")
    return p


# -----------------------------------------------------------------------
# Tests: data awareness
# -----------------------------------------------------------------------

class TestDataAwareness:
    """Project knows about available features in the dataset."""

    def test_available_features_returns_columns(self, project):
        feats = project.available_features()
        assert "diff_seed_num" in feats
        assert "diff_sr_srs" in feats

    def test_available_features_with_prefix(self, project):
        diff_feats = project.available_features(prefix="diff_")
        assert all(f.startswith("diff_") for f in diff_feats)
        assert "diff_seed_num" in diff_feats

    def test_available_features_excludes_non_matching_prefix(self, project):
        diff_feats = project.available_features(prefix="diff_")
        assert "Season" not in diff_feats
        assert "TeamAWon" not in diff_feats

    def test_profile_returns_profile(self, project):
        profile = project.profile()
        assert profile.n_rows == 300
        assert profile.n_cols > 0


# -----------------------------------------------------------------------
# Tests: add_model
# -----------------------------------------------------------------------

class TestAddModel:
    """add_model validates features against data."""

    def test_add_valid_model(self, project):
        project.add_model(
            "logreg_seed", "logistic_regression",
            features=["diff_seed_num"],
        )
        config = project.build()
        assert "logreg_seed" in config.models

    def test_add_model_with_params(self, project):
        project.add_model(
            "xgb_core", "xgboost",
            features=["diff_seed_num", "diff_sr_srs"],
            params={"max_depth": 2, "learning_rate": 0.01},
        )
        config = project.build()
        assert config.models["xgb_core"].params["max_depth"] == 2

    def test_build_rejects_missing_features(self, project):
        """Feature validation happens at build() time (deferred for provider support)."""
        project.add_model(
            "bad_model", "xgboost",
            features=["diff_nonexistent_feature"],
        )
        with pytest.raises(ValueError, match="not found in data"):
            project.build()

    def test_add_model_rejects_bad_type(self, project):
        with pytest.raises(ValueError, match="Unknown model type"):
            project.add_model(
                "bad_model", "definitely_not_a_model_type",
                features=["diff_seed_num"],
            )

    def test_add_regressor(self, project):
        project.add_model(
            "xgb_spread", "xgboost_regression",
            features=["diff_seed_num", "diff_scoring_margin"],
            mode="regressor",
        )
        config = project.build()
        assert config.models["xgb_spread"].mode == "regressor"

    def test_add_multiple_models(self, project):
        project.add_model(
            "model_a", "logistic_regression",
            features=["diff_seed_num"],
        )
        project.add_model(
            "model_b", "xgboost",
            features=["diff_seed_num", "diff_sr_srs"],
        )
        config = project.build()
        assert len(config.models) == 2


# -----------------------------------------------------------------------
# Tests: remove/exclude model
# -----------------------------------------------------------------------

class TestModelManagement:
    """Models can be removed or excluded."""

    def test_remove_model(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        project.add_model("m2", "xgboost", features=["diff_seed_num"])
        project.remove_model("m1")
        config = project.build()
        assert "m1" not in config.models
        assert "m2" in config.models

    def test_exclude_model(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        project.add_model("m2", "xgboost", features=["diff_seed_num"])
        project.exclude_model("m1")
        config = project.build()
        assert "m1" in config.ensemble.exclude_models


# -----------------------------------------------------------------------
# Tests: leakage detection
# -----------------------------------------------------------------------

class TestLeakageDetection:
    """Source temporal safety is validated against model features."""

    def test_no_leakage_with_safe_sources(self, project):
        project.add_source(
            "seeds",
            temporal_safety="pre_tournament",
            outputs=["seed_num"],
        )
        project.add_model(
            "m1", "logistic_regression",
            features=["diff_seed_num"],
        )
        warnings = project.check_leakage()
        assert len(warnings) == 0

    def test_detects_post_tournament_leakage(self, project):
        project.add_source(
            "leaky_source",
            temporal_safety="post_tournament",
            outputs=["leaky_stat"],
            leakage_notes="Includes tournament game results",
        )
        project.add_model(
            "m1", "logistic_regression",
            features=["diff_leaky_stat"],
        )
        warnings = project.check_leakage()
        assert len(warnings) == 1
        assert warnings[0].temporal_safety == "post_tournament"
        assert warnings[0].model_name == "m1"

    def test_build_raises_on_leakage(self, project):
        project.add_source(
            "leaky_source",
            temporal_safety="post_tournament",
            outputs=["leaky_stat"],
        )
        project.add_model(
            "m1", "logistic_regression",
            features=["diff_leaky_stat"],
        )
        with pytest.raises(ValueError, match="Leakage detected"):
            project.build()

    def test_detects_mixed_temporal_safety(self, project):
        project.add_source(
            "mixed_source",
            temporal_safety="mixed",
            outputs=["leaky_stat"],
        )
        project.add_model(
            "m1", "logistic_regression",
            features=["diff_leaky_stat"],
        )
        warnings = project.check_leakage()
        assert len(warnings) == 1
        assert warnings[0].temporal_safety == "mixed"

    def test_no_warning_for_undeclared_features(self, project):
        """Features without declared sources are not flagged (unknown provenance)."""
        project.add_model(
            "m1", "logistic_regression",
            features=["diff_seed_num"],
        )
        # No sources declared
        warnings = project.check_leakage()
        assert len(warnings) == 0


# -----------------------------------------------------------------------
# Tests: ensemble configuration
# -----------------------------------------------------------------------

class TestEnsembleConfig:
    """Ensemble can be configured programmatically."""

    def test_default_ensemble(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        config = project.build()
        assert config.ensemble.method == "stacked"
        assert config.ensemble.temperature == 1.0

    def test_custom_ensemble(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        project.configure_ensemble(
            method="stacked",
            C=2.5,
            temperature=1.1,
            calibration="spline",
            spline_n_bins=12,
        )
        config = project.build()
        assert config.ensemble.meta_learner["C"] == 2.5
        assert config.ensemble.temperature == 1.1
        assert config.ensemble.spline_n_bins == 12


# -----------------------------------------------------------------------
# Tests: backtest configuration
# -----------------------------------------------------------------------

class TestBacktestConfig:
    """Backtest seasons can be configured or auto-detected."""

    def test_explicit_seasons(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        project.configure_backtest(seasons=[2016, 2017, 2018])
        config = project.build()
        assert config.backtest.seasons == [2016, 2017, 2018]

    def test_auto_detect_seasons(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        project.configure_backtest()
        config = project.build()
        assert 2015 in config.backtest.seasons
        assert 2019 in config.backtest.seasons


# -----------------------------------------------------------------------
# Tests: build validation
# -----------------------------------------------------------------------

class TestBuild:
    """build() validates the full config."""

    def test_build_fails_with_no_models(self, project):
        with pytest.raises(ValueError, match="No models configured"):
            project.build()

    def test_build_succeeds_with_valid_config(self, project):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        config = project.build()
        assert config is not None
        assert len(config.models) == 1


# -----------------------------------------------------------------------
# Tests: YAML serialization
# -----------------------------------------------------------------------

class TestYamlSerialization:
    """Config can be saved to YAML files."""

    def test_save_to_yaml_creates_files(self, project, tmp_path):
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        config_dir = tmp_path / "saved_config"
        project.save_to_yaml(config_dir)

        assert (config_dir / "pipeline.yaml").exists()
        assert (config_dir / "models.yaml").exists()
        assert (config_dir / "ensemble.yaml").exists()

    def test_saved_yaml_roundtrips(self, project, tmp_path):
        """Config saved to YAML can be loaded back by the validator."""
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        project.configure_backtest(seasons=[2016, 2017])

        config_dir = tmp_path / "saved_config"
        project.save_to_yaml(config_dir)

        from easyml.runner.validator import validate_project

        result = validate_project(config_dir)
        assert result.valid, f"Roundtrip failed: {result.format()}"
        assert "m1" in result.config.models

    def test_sources_saved_when_present(self, project, tmp_path):
        project.add_source(
            "seeds", temporal_safety="pre_tournament", outputs=["seed_num"],
        )
        project.add_model("m1", "logistic_regression", features=["diff_seed_num"])
        config_dir = tmp_path / "saved_config"
        project.save_to_yaml(config_dir)

        assert (config_dir / "sources.yaml").exists()


# -----------------------------------------------------------------------
# Tests: method chaining
# -----------------------------------------------------------------------

class TestChaining:
    """Methods return self for fluent chaining."""

    def test_chaining(self, project_dir):
        project = Project(project_dir)
        config = (
            project
            .set_data(features_dir="data/features")
            .add_model("m1", "logistic_regression", features=["diff_seed_num"])
            .configure_ensemble(method="stacked", C=1.0)
            .configure_backtest(seasons=[2016, 2017])
            .build()
        )
        assert "m1" in config.models
        assert config.backtest.seasons == [2016, 2017]


# -----------------------------------------------------------------------
# Tests: provider models and dependencies
# -----------------------------------------------------------------------

class TestProviderModels:
    """Provider models declare features consumed by downstream models."""

    def test_provider_features_pass_validation(self, project):
        """Consumer features referencing provider outputs pass build()."""
        project.add_model(
            "provider", "xgboost",
            features=["diff_seed_num"],
            provides=["predicted_margin"],
            mode="regressor",
        )
        project.add_model(
            "consumer", "xgboost",
            features=["diff_seed_num", "diff_predicted_margin"],
        )
        config = project.build()
        assert "provider" in config.models
        assert "consumer" in config.models

    def test_provider_order_independent(self, project):
        """Consumer can be added before provider — validation at build()."""
        project.add_model(
            "consumer", "xgboost",
            features=["diff_seed_num", "diff_provided_feat"],
        )
        project.add_model(
            "provider", "xgboost",
            features=["diff_seed_num"],
            provides=["provided_feat"],
            mode="regressor",
        )
        config = project.build()
        assert len(config.models) == 2

    def test_missing_feature_still_rejected(self, project):
        """Features not in data AND not from a provider are still rejected."""
        project.add_model(
            "provider", "xgboost",
            features=["diff_seed_num"],
            provides=["some_feat"],
        )
        project.add_model(
            "consumer", "xgboost",
            features=["diff_totally_nonexistent"],
        )
        with pytest.raises(ValueError, match="not found in data"):
            project.build()

    def test_include_in_ensemble_false(self, project):
        """Provider with include_in_ensemble=False is stored in config."""
        project.add_model(
            "provider", "xgboost",
            features=["diff_seed_num"],
            provides=["margin"],
            include_in_ensemble=False,
        )
        project.add_model(
            "consumer", "xgboost",
            features=["diff_seed_num", "diff_margin"],
        )
        config = project.build()
        assert config.models["provider"].include_in_ensemble is False
        assert config.models["consumer"].include_in_ensemble is True

    def test_cycle_detected(self, project):
        """Circular dependencies raise ValueError."""
        project.add_model(
            "model_a", "xgboost",
            features=["diff_feat_b"],
            provides=["feat_a"],
        )
        project.add_model(
            "model_b", "xgboost",
            features=["diff_feat_a"],
            provides=["feat_b"],
        )
        with pytest.raises(ValueError, match="cycle"):
            project.build()

    def test_team_level_requires_team_features_path(self, project):
        """Team-level provider without team_features_path raises."""
        project.add_model(
            "survival", "survival",
            features=["diff_seed_num"],
            provides=["surv_e8"],
            provides_level="team",
            include_in_ensemble=False,
        )
        project.add_model(
            "consumer", "xgboost",
            features=["diff_seed_num", "diff_surv_e8"],
        )
        with pytest.raises(ValueError, match="team_features_path"):
            project.build()

    def test_provides_level_stored(self, project):
        """provides_level is correctly stored in config."""
        project.add_model(
            "provider", "xgboost",
            features=["diff_seed_num"],
            provides=["margin"],
            provides_level="matchup",
        )
        project.add_model(
            "consumer", "xgboost",
            features=["diff_seed_num", "diff_margin"],
        )
        config = project.build()
        assert config.models["provider"].provides_level == "matchup"

    def test_provider_isolation_stored(self, project_dir):
        """provider_isolation is correctly stored in config."""
        project = Project(project_dir)
        project.set_data(
            features_dir="data/features",
            team_features_path="data/features/team_features.parquet",
        )
        project.add_model(
            "survival", "survival",
            features=["diff_seed_num"],
            provides=["surv_e8"],
            provides_level="team",
            include_in_ensemble=False,
            provider_isolation="per_season",
        )
        project.add_model(
            "consumer", "xgboost",
            features=["diff_seed_num", "diff_surv_e8"],
        )
        config = project.build()
        assert config.models["survival"].provider_isolation == "per_season"
