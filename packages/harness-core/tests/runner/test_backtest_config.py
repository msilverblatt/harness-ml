"""Tests for BacktestConfig strategy validation."""
import pytest
from harnessml.core.runner.schema import BacktestConfig


class TestCVStrategyAliases:
    """Aliases should be normalized to canonical names."""

    def test_loso_alias(self):
        bc = BacktestConfig(cv_strategy="loso")
        assert bc.cv_strategy == "leave_one_out"

    def test_loo_alias(self):
        bc = BacktestConfig(cv_strategy="loo")
        assert bc.cv_strategy == "leave_one_out"

    def test_expanding_alias(self):
        bc = BacktestConfig(cv_strategy="expanding")
        assert bc.cv_strategy == "expanding_window"

    def test_sliding_alias(self):
        bc = BacktestConfig(cv_strategy="sliding", window_size=5)
        assert bc.cv_strategy == "sliding_window"

    def test_purged_alias(self):
        bc = BacktestConfig(cv_strategy="purged", n_folds=5)
        assert bc.cv_strategy == "purged_kfold"

    def test_canonical_name_passthrough(self):
        bc = BacktestConfig(cv_strategy="leave_one_out")
        assert bc.cv_strategy == "leave_one_out"

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Invalid cv_strategy"):
            BacktestConfig(cv_strategy="bogus")


class TestStrategySpecificValidation:
    """Strategy-specific params must be present."""

    def test_sliding_window_requires_window_size(self):
        with pytest.raises(ValueError, match="sliding_window strategy requires window_size"):
            BacktestConfig(cv_strategy="sliding_window")

    def test_sliding_window_with_window_size_ok(self):
        bc = BacktestConfig(cv_strategy="sliding_window", window_size=3)
        assert bc.window_size == 3

    def test_purged_kfold_requires_n_folds(self):
        with pytest.raises(ValueError, match="purged_kfold strategy requires n_folds"):
            BacktestConfig(cv_strategy="purged_kfold")

    def test_purged_kfold_with_n_folds_ok(self):
        bc = BacktestConfig(cv_strategy="purged_kfold", n_folds=10)
        assert bc.n_folds == 10

    def test_leave_one_out_no_extra_params_needed(self):
        bc = BacktestConfig(cv_strategy="leave_one_out")
        assert bc.cv_strategy == "leave_one_out"

    def test_expanding_window_no_extra_params_needed(self):
        bc = BacktestConfig(cv_strategy="expanding_window")
        assert bc.cv_strategy == "expanding_window"
