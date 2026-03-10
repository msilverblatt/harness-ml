from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from harnessml.core.runner.notebook.tagger import auto_tag


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_detects_model_names(tmp_path: Path) -> None:
    """Detect model names from config/models/*.yaml filenames."""
    models_dir = tmp_path / "config" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "xgb_main.yaml").write_text(yaml.dump({"type": "xgboost"}))
    (models_dir / "rf_core.yaml").write_text(yaml.dump({"type": "random_forest"}))

    tags = auto_tag("Trained xgb_main on new data", tmp_path)
    assert "model:xgb_main" in tags


def test_detects_model_names_from_single_models_yaml(tmp_path: Path) -> None:
    """Detect model names from top-level keys in config/models.yaml."""
    _write_yaml(
        tmp_path / "config" / "models.yaml",
        {
            "models": {
                "lgbm_v2": {"type": "lightgbm"},
                "catboost_base": {"type": "catboost"},
            }
        },
    )

    tags = auto_tag("Switched to lgbm_v2 for speed", tmp_path)
    assert "model:lgbm_v2" in tags
    assert "model:catboost_base" not in tags


def test_detects_feature_names(tmp_path: Path) -> None:
    """Detect feature names from features lists inside models.yaml."""
    _write_yaml(
        tmp_path / "config" / "models.yaml",
        {
            "models": {
                "xgb": {
                    "type": "xgboost",
                    "features": ["age_group", "income", "zip_code"],
                }
            }
        },
    )

    tags = auto_tag("Added age_group and income features", tmp_path)
    assert "feature:age_group" in tags
    assert "feature:income" in tags
    assert "feature:zip_code" not in tags


def test_detects_experiment_ids(tmp_path: Path) -> None:
    """Detect experiment IDs via regex."""
    tags = auto_tag("Results from exp-042 and exp-7 look promising", tmp_path)
    assert "experiment:exp-042" in tags
    assert "experiment:exp-7" in tags


def test_detects_source_names(tmp_path: Path) -> None:
    """Detect source names from data/source_registry.json."""
    _write_json(
        tmp_path / "data" / "source_registry.json",
        {
            "housing_csv": {"path": "data/raw/housing.csv"},
            "census_api": {"url": "https://api.census.gov"},
        },
    )

    tags = auto_tag("Refreshed housing_csv from disk", tmp_path)
    assert "source:housing_csv" in tags
    assert "source:census_api" not in tags


def test_case_insensitive(tmp_path: Path) -> None:
    """All entity detection should be case-insensitive."""
    _write_yaml(
        tmp_path / "config" / "models.yaml",
        {"models": {"XGB_Main": {"type": "xgboost"}}},
    )

    tags = auto_tag("Ran xgb_main with new params", tmp_path)
    assert "model:XGB_Main" in tags


def test_no_config_dir_returns_only_experiment_ids(tmp_path: Path) -> None:
    """When no config dir exists, still detect experiment IDs."""
    tags = auto_tag("Check exp-99 results", tmp_path)
    assert tags == ["experiment:exp-99"]


def test_no_duplicates(tmp_path: Path) -> None:
    """Duplicate mentions should not produce duplicate tags."""
    tags = auto_tag("exp-1 and exp-1 again", tmp_path)
    assert tags.count("experiment:exp-1") == 1


def test_multiple_entity_types(tmp_path: Path) -> None:
    """Content referencing model, feature, and experiment in one string."""
    _write_yaml(
        tmp_path / "config" / "models.yaml",
        {
            "models": {
                "rf_v1": {
                    "type": "random_forest",
                    "features": ["momentum"],
                }
            }
        },
    )

    tags = auto_tag("rf_v1 with momentum in exp-5", tmp_path)
    assert "model:rf_v1" in tags
    assert "feature:momentum" in tags
    assert "experiment:exp-5" in tags
