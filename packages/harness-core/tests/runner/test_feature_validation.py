"""Tests for feature existence validation when adding models."""

from pathlib import Path

import pytest
from harnessml.core.runner.config_writer._helpers import _save_yaml
from harnessml.core.runner.config_writer.models import _check_feature_existence


@pytest.fixture
def project_with_features(tmp_path):
    """Create a minimal project with feature_defs in pipeline.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    pipeline_data = {
        "data": {
            "target_column": "result",
            "feature_defs": {
                "elo_diff": {"type": "pairwise", "formula": "home_elo - away_elo"},
                "win_pct_diff": {"type": "pairwise", "formula": "home_wp - away_wp"},
                "tempo": {"type": "entity", "source": "stats", "column": "pace"},
            },
        },
    }
    _save_yaml(config_dir / "pipeline.yaml", pipeline_data)
    return tmp_path


@pytest.fixture
def project_without_features(tmp_path):
    """Create a minimal project with no feature_defs."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    pipeline_data = {
        "data": {
            "target_column": "result",
        },
    }
    _save_yaml(config_dir / "pipeline.yaml", pipeline_data)
    return tmp_path


class TestCheckFeatureExistence:
    """Unit tests for _check_feature_existence helper."""

    def test_all_features_known(self, project_with_features):
        warnings = _check_feature_existence(
            project_with_features, ["elo_diff", "win_pct_diff"]
        )
        assert warnings == []

    def test_unknown_features_warned(self, project_with_features):
        warnings = _check_feature_existence(
            project_with_features, ["elo_diff", "nonexistent_feat", "bad_feat"]
        )
        assert len(warnings) == 1
        assert "nonexistent_feat" in warnings[0]
        assert "bad_feat" in warnings[0]
        assert "Warning" in warnings[0]

    def test_no_feature_defs_skips_validation(self, project_without_features):
        warnings = _check_feature_existence(
            project_without_features, ["anything", "goes"]
        )
        assert warnings == []

    def test_no_config_dir_skips_validation(self, tmp_path):
        # No config/ directory at all
        warnings = _check_feature_existence(tmp_path, ["anything"])
        assert warnings == []

    def test_empty_features_list(self, project_with_features):
        warnings = _check_feature_existence(project_with_features, [])
        assert warnings == []

    def test_mixed_known_and_unknown(self, project_with_features):
        warnings = _check_feature_existence(
            project_with_features, ["elo_diff", "unknown_x"]
        )
        assert len(warnings) == 1
        assert "unknown_x" in warnings[0]
        assert "elo_diff" not in warnings[0]


class TestAddModelWithFeatureWarnings:
    """Integration test: add_model includes warnings for unknown features."""

    def test_add_model_warns_on_unknown_features(self, project_with_features):
        from harnessml.core.runner.config_writer.models import add_model

        # Need models.yaml too
        _save_yaml(
            project_with_features / "config" / "models.yaml",
            {"models": {}},
        )
        result = add_model(
            project_with_features,
            "test_model",
            model_type="xgboost",
            features=["elo_diff", "nonexistent_feat"],
        )
        assert "Added model" in result
        assert "Warning" in result
        assert "nonexistent_feat" in result

    def test_add_model_no_warning_for_known_features(self, project_with_features):
        from harnessml.core.runner.config_writer.models import add_model

        _save_yaml(
            project_with_features / "config" / "models.yaml",
            {"models": {}},
        )
        result = add_model(
            project_with_features,
            "test_model",
            model_type="xgboost",
            features=["elo_diff", "win_pct_diff"],
        )
        assert "Added model" in result
        assert "Warning" not in result
