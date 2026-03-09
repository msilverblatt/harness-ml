"""Tests for pre-backtest validation."""

import numpy as np
import pandas as pd
import pytest
from harnessml.core.runner.config_writer._helpers import _save_yaml
from harnessml.core.runner.validation.validation import (
    Severity,
    ValidationCode,
    ValidationIssue,
    format_validation_issues,
    validate_project,
)


@pytest.fixture
def valid_project(tmp_path):
    """Create a valid project config for testing."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    _save_yaml(config_dir / "pipeline.yaml", {
        "data": {
            "target_column": "result",
        },
        "backtest": {
            "fold_column": "season",
        },
    })
    _save_yaml(config_dir / "models.yaml", {
        "models": {
            "xgb_core": {
                "type": "xgboost",
                "active": True,
                "features": ["feat_a", "feat_b"],
            },
        },
    })
    return tmp_path


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for data-dependent checks."""
    return pd.DataFrame({
        "season": [2020, 2021, 2022, 2023],
        "result": [1, 0, 1, 0],
        "feat_a": [0.5, 0.3, 0.7, 0.9],
        "feat_b": [1.0, 2.0, 3.0, 4.0],
    })


class TestValidateProject:
    """Tests for validate_project function."""

    def test_valid_project_passes(self, valid_project, sample_df):
        issues = validate_project(valid_project, data_df=sample_df)
        errors = [i for i in issues if i.severity == Severity.ERROR]
        assert len(errors) == 0

    def test_no_config_dir(self, tmp_path):
        issues = validate_project(tmp_path)
        assert len(issues) == 1
        assert issues[0].severity == Severity.ERROR

    def test_target_column_missing_in_data(self, tmp_path):
        """When target_column is set but not in data, produce error."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        _save_yaml(config_dir / "pipeline.yaml", {
            "data": {"target_column": "nonexistent_target"},
        })
        _save_yaml(config_dir / "models.yaml", {
            "models": {"m1": {"type": "xgboost", "active": True, "features": ["a"]}},
        })
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        issues = validate_project(tmp_path, data_df=df)
        codes = {i.code for i in issues}
        assert ValidationCode.TARGET_COLUMN_MISSING in codes

    def test_no_active_models(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        _save_yaml(config_dir / "pipeline.yaml", {
            "data": {"target_column": "result"},
        })
        _save_yaml(config_dir / "models.yaml", {
            "models": {
                "m1": {"type": "xgboost", "active": False},
            },
        })
        issues = validate_project(tmp_path)
        codes = {i.code for i in issues}
        assert ValidationCode.NO_ACTIVE_MODELS in codes

    def test_fold_column_missing_in_data(self, valid_project):
        df = pd.DataFrame({"result": [1, 0], "feat_a": [0.1, 0.2], "feat_b": [0.3, 0.4]})
        issues = validate_project(valid_project, data_df=df)
        codes = {i.code for i in issues}
        assert ValidationCode.FOLD_COLUMN_MISSING in codes

    def test_class_imbalance_warning(self, valid_project):
        # 2% minority class
        df = pd.DataFrame({
            "season": list(range(100)),
            "result": [1] * 2 + [0] * 98,
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100),
        })
        issues = validate_project(valid_project, data_df=df)
        imbalance = [i for i in issues if i.code == ValidationCode.CLASS_IMBALANCE]
        assert len(imbalance) == 1
        assert imbalance[0].severity == Severity.WARNING

    def test_high_missing_rate_warning(self, valid_project):
        df = pd.DataFrame({
            "season": [2020, 2021, 2022, 2023],
            "result": [1, 0, 1, 0],
            "feat_a": [0.5, np.nan, np.nan, np.nan],  # 75% missing
            "feat_b": [1.0, 2.0, 3.0, 4.0],
        })
        issues = validate_project(valid_project, data_df=df)
        high_missing = [i for i in issues if i.code == ValidationCode.HIGH_MISSING_RATE]
        assert len(high_missing) == 1
        assert "feat_a" in high_missing[0].message

    def test_feature_not_found_warning(self, valid_project):
        # feat_b is missing from data
        df = pd.DataFrame({
            "season": [2020, 2021, 2022, 2023],
            "result": [1, 0, 1, 0],
            "feat_a": [0.5, 0.3, 0.7, 0.9],
        })
        issues = validate_project(valid_project, data_df=df)
        not_found = [i for i in issues if i.code == ValidationCode.FEATURE_NOT_FOUND]
        assert len(not_found) == 1
        assert "feat_b" in not_found[0].message

    def test_no_data_df_skips_data_checks(self, valid_project):
        issues = validate_project(valid_project)
        # Should not produce data-dependent errors
        data_codes = {
            ValidationCode.FOLD_COLUMN_MISSING,
            ValidationCode.CLASS_IMBALANCE,
            ValidationCode.HIGH_MISSING_RATE,
            ValidationCode.FEATURE_NOT_FOUND,
            ValidationCode.TEMPORAL_UNSORTED,
        }
        found_codes = {i.code for i in issues}
        assert not (found_codes & data_codes)


class TestFormatValidationIssues:
    """Tests for format_validation_issues."""

    def test_empty_issues(self):
        result = format_validation_issues([])
        assert "passed" in result.lower()

    def test_errors_formatted(self):
        issues = [
            ValidationIssue(
                code=ValidationCode.NO_ACTIVE_MODELS,
                severity=Severity.ERROR,
                message="No active models.",
            ),
        ]
        result = format_validation_issues(issues)
        assert "NO_ACTIVE_MODELS" in result
        assert "Error" in result

    def test_warnings_formatted(self):
        issues = [
            ValidationIssue(
                code=ValidationCode.CLASS_IMBALANCE,
                severity=Severity.WARNING,
                message="Class imbalance detected.",
                detail="Consider resampling.",
            ),
        ]
        result = format_validation_issues(issues)
        assert "CLASS_IMBALANCE" in result
        assert "Warning" in result
        assert "resampling" in result
