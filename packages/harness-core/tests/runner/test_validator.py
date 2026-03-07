"""Tests for YAML loading and validation."""
from __future__ import annotations

from pathlib import Path

import yaml
from harnessml.core.runner.validator import validate_project

# -----------------------------------------------------------------------
# Helpers — write YAML fixtures to tmp_path
# -----------------------------------------------------------------------

def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _minimal_pipeline() -> dict:
    return {
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "features_dir": "data/features",
        },
        "backtest": {
            "cv_strategy": "leave_one_out",
            "fold_values": [2023, 2024],
        },
    }


def _minimal_models() -> dict:
    return {
        "models": {
            "xgb_core": {
                "type": "xgboost",
                "features": ["feat_a"],
                "params": {"max_depth": 3},
            }
        }
    }


def _minimal_ensemble() -> dict:
    return {"ensemble": {"method": "stacked"}}


def _setup_minimal(tmp_path: Path) -> None:
    """Write pipeline.yaml, models.yaml, ensemble.yaml for a minimal valid config."""
    _write_yaml(tmp_path / "pipeline.yaml", _minimal_pipeline())
    _write_yaml(tmp_path / "models.yaml", _minimal_models())
    _write_yaml(tmp_path / "ensemble.yaml", _minimal_ensemble())


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestValidConfig:
    """Valid config passes validation."""

    def test_valid(self, tmp_path):
        _setup_minimal(tmp_path)
        result = validate_project(tmp_path)
        assert result.valid, f"Expected valid, got errors: {result.format()}"
        assert result.config is not None
        assert result.config.data.raw_dir == "data/raw"
        assert "xgb_core" in result.config.models

    def test_result_format_no_errors(self, tmp_path):
        _setup_minimal(tmp_path)
        result = validate_project(tmp_path)
        formatted = result.format()
        assert "valid" in formatted.lower() or "no errors" in formatted.lower() or formatted == ""


class TestMissingPipeline:
    """Missing pipeline.yaml fails validation."""

    def test_missing_pipeline(self, tmp_path):
        # No pipeline.yaml, only models
        _write_yaml(tmp_path / "models.yaml", _minimal_models())
        result = validate_project(tmp_path)
        assert not result.valid
        assert any("pipeline.yaml" in e for e in result.errors)


class TestInvalidModelType:
    """Invalid model type fails with clear error."""

    def test_invalid_type(self, tmp_path):
        pipeline = _minimal_pipeline()
        _write_yaml(tmp_path / "pipeline.yaml", pipeline)
        _write_yaml(
            tmp_path / "models.yaml",
            {
                "models": {
                    "bad_model": {
                        "type": "totally_fake",
                        "features": ["a"],
                        "params": {},
                    }
                }
            },
        )
        _write_yaml(tmp_path / "ensemble.yaml", _minimal_ensemble())
        result = validate_project(tmp_path)
        assert not result.valid
        assert any("totally_fake" in e for e in result.errors)


class TestOverlay:
    """Overlay merges and validates."""

    def test_overlay_merges(self, tmp_path):
        _setup_minimal(tmp_path)
        overlay = {
            "data": {"task": "regression"},
            "ensemble": {"temperature": 0.9},
        }
        result = validate_project(tmp_path, overlay=overlay)
        assert result.valid, f"Errors: {result.format()}"
        assert result.config.data.task == "regression"
        assert result.config.ensemble.temperature == 0.9

    def test_overlay_can_add_model(self, tmp_path):
        _setup_minimal(tmp_path)
        overlay = {
            "models": {
                "lgbm_new": {
                    "type": "lightgbm",
                    "features": ["x"],
                    "params": {},
                }
            }
        }
        result = validate_project(tmp_path, overlay=overlay)
        assert result.valid, f"Errors: {result.format()}"
        assert "lgbm_new" in result.config.models
        # Original model should still be there
        assert "xgb_core" in result.config.models


class TestFeaturesYaml:
    """Features YAML loaded."""

    def test_features_loaded(self, tmp_path):
        _setup_minimal(tmp_path)
        _write_yaml(
            tmp_path / "features.yaml",
            {
                "features": {
                    "eff": {
                        "module": "proj.feat",
                        "function": "compute_eff",
                        "category": "efficiency",
                        "level": "team",
                        "columns": ["adj_oe", "adj_de"],
                    }
                }
            },
        )
        result = validate_project(tmp_path)
        assert result.valid, f"Errors: {result.format()}"
        assert result.config.features is not None
        assert "eff" in result.config.features


class TestModelsSubdirectory:
    """Models from models/ subdirectory loaded."""

    def test_models_subdir(self, tmp_path):
        pipeline = _minimal_pipeline()
        _write_yaml(tmp_path / "pipeline.yaml", pipeline)
        _write_yaml(tmp_path / "ensemble.yaml", _minimal_ensemble())
        # No models.yaml at top level, but models/ subdirectory
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        _write_yaml(
            models_dir / "production.yaml",
            {
                "models": {
                    "xgb_core": {
                        "type": "xgboost",
                        "features": ["a"],
                        "params": {},
                    }
                }
            },
        )
        _write_yaml(
            models_dir / "experimental.yaml",
            {
                "models": {
                    "cat_new": {
                        "type": "catboost",
                        "features": ["b"],
                        "params": {},
                    }
                }
            },
        )
        result = validate_project(tmp_path)
        assert result.valid, f"Errors: {result.format()}"
        assert "xgb_core" in result.config.models
        assert "cat_new" in result.config.models


class TestVariantLoading:
    """Variant loading (e.g. pipeline_w.yaml)."""

    def test_variant_uses_variant_file(self, tmp_path):
        # Create both pipeline.yaml and pipeline_v2.yaml
        _write_yaml(tmp_path / "pipeline.yaml", _minimal_pipeline())
        variant_pipeline = _minimal_pipeline()
        variant_pipeline["data"]["task"] = "regression"
        _write_yaml(tmp_path / "pipeline_v2.yaml", variant_pipeline)
        _write_yaml(tmp_path / "models.yaml", _minimal_models())
        _write_yaml(tmp_path / "ensemble.yaml", _minimal_ensemble())

        result = validate_project(tmp_path, variant="v2")
        assert result.valid, f"Errors: {result.format()}"
        assert result.config.data.task == "regression"

    def test_variant_falls_back_to_base(self, tmp_path):
        """If variant file doesn't exist, fall back to base."""
        _setup_minimal(tmp_path)
        result = validate_project(tmp_path, variant="v2")
        assert result.valid, f"Errors: {result.format()}"
        # Should get the default task since no variant file exists
        assert result.config.data.task == "classification"


class TestMultiSectionPipeline:
    """pipeline.yaml with data, backtest, features, and unknown sections."""

    def test_mm_style_pipeline(self, tmp_path):
        """Loads a pipeline.yaml structured like mm's (data, backtest, features, bracket)."""
        pipeline = {
            "data": {
                "raw_dir": "data/raw",
                "processed_dir": "data/processed",
                "features_dir": "data/features",
                "task": "classification",
            },
            "features": {
                "first_period": 2003,
                "momentum_window": 10,
            },
            "backtest": {
                "cv_strategy": "leave_one_out",
                "fold_values": [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
            },
            # Unknown keys like "bracket" should be silently ignored
            "bracket": {
                "scoring": [10, 20, 40, 80, 160, 320],
                "simulations": 10000,
                "temperature": 1.75,
            },
        }
        _write_yaml(tmp_path / "pipeline.yaml", pipeline)
        _write_yaml(tmp_path / "models.yaml", _minimal_models())
        _write_yaml(
            tmp_path / "ensemble.yaml",
            {"ensemble": {"method": "stacked"}},
        )
        result = validate_project(tmp_path)
        assert result.valid, f"Expected valid, got errors: {result.format()}"
        cfg = result.config

        # data section parsed
        assert cfg.data.task == "classification"
        assert cfg.data.raw_dir == "data/raw"

        # features section mapped to feature_config
        assert cfg.feature_config is not None
        assert cfg.feature_config.first_period == 2003
        assert cfg.feature_config.momentum_window == 10

        # backtest section parsed
        assert cfg.backtest.cv_strategy == "leave_one_out"
        assert 2025 in cfg.backtest.fold_values

    def test_pipeline_without_features_section(self, tmp_path):
        """pipeline.yaml without a features key should still work."""
        _setup_minimal(tmp_path)
        result = validate_project(tmp_path)
        assert result.valid, f"Errors: {result.format()}"
        assert result.config.feature_config is None

    def test_bracket_key_ignored(self, tmp_path):
        """Unknown keys in pipeline.yaml (like bracket) don't cause errors."""
        pipeline = _minimal_pipeline()
        pipeline["bracket"] = {"scoring": [10, 20], "simulations": 5000}
        _write_yaml(tmp_path / "pipeline.yaml", pipeline)
        _write_yaml(tmp_path / "models.yaml", _minimal_models())
        _write_yaml(tmp_path / "ensemble.yaml", _minimal_ensemble())
        result = validate_project(tmp_path)
        assert result.valid, f"Errors: {result.format()}"


class TestFormatErrors:
    """format() produces readable output."""

    def test_format_with_errors(self, tmp_path):
        result = validate_project(tmp_path)  # Missing pipeline.yaml
        assert not result.valid
        formatted = result.format()
        assert len(formatted) > 0
        assert "pipeline.yaml" in formatted

    def test_format_multiple_errors(self, tmp_path):
        _write_yaml(
            tmp_path / "pipeline.yaml",
            {
                "data": {"raw_dir": "r", "processed_dir": "p", "features_dir": "f"},
                "backtest": {"cv_strategy": "bad_strat"},
                "models": {
                    "m": {"type": "fake_type", "features": [], "params": {}}
                },
                "ensemble": {"method": "stacked"},
            },
        )
        result = validate_project(tmp_path)
        assert not result.valid
        formatted = result.format()
        # Should contain info about both errors
        assert "bad_strat" in formatted or "fake_type" in formatted
