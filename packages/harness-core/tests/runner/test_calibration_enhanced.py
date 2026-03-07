"""Tests for BetaCalibrator and calibration diagnostics."""
from __future__ import annotations

import numpy as np
import pytest
from harnessml.core.runner.calibration import (
    BetaCalibrator,
    bootstrap_ci,
    build_calibrator,
    calibration_slope_intercept,
    hosmer_lemeshow_test,
    reliability_diagram_data,
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _synth_calibration_data(n: int = 200, seed: int = 42):
    """Generate synthetic calibration data: miscalibrated predictions + labels."""
    rng = np.random.RandomState(seed)
    true_p = rng.beta(2, 2, size=n)
    y_true = (rng.rand(n) < true_p).astype(float)
    y_prob = np.clip(true_p * 1.2 - 0.1, 0.05, 0.95)
    return y_true, y_prob


def _brier_score(y_true, y_prob):
    """Simple Brier score for bootstrap tests."""
    return float(np.mean((y_prob - y_true) ** 2))


# -----------------------------------------------------------------------
# BetaCalibrator
# -----------------------------------------------------------------------

class TestBetaCalibrator:
    def test_fit_transform_roundtrip(self):
        y_true, y_prob = _synth_calibration_data(200)
        cal = BetaCalibrator()
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        assert len(calibrated) == len(y_prob)
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_is_fitted_property(self):
        cal = BetaCalibrator()
        assert cal.is_fitted is False
        y_true, y_prob = _synth_calibration_data(50)
        cal.fit(y_true, y_prob)
        assert cal.is_fitted is True

    def test_transform_before_fit_raises(self):
        cal = BetaCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.transform(np.array([0.5]))

    def test_params_property(self):
        cal = BetaCalibrator()
        assert cal.params == (1.0, 1.0)
        y_true, y_prob = _synth_calibration_data(200)
        cal.fit(y_true, y_prob)
        a, b = cal.params
        assert a > 0
        assert b > 0

    def test_preserves_ordering(self):
        """Beta calibration (via CDF) should preserve ordering."""
        y_true, y_prob = _synth_calibration_data(300)
        cal = BetaCalibrator()
        cal.fit(y_true, y_prob)

        sorted_probs = np.sort(y_prob)
        calibrated = cal.transform(sorted_probs)

        # CDF is always monotonically non-decreasing
        diffs = np.diff(calibrated)
        assert np.all(diffs >= -1e-10)

    def test_output_valid_probabilities(self):
        """All outputs should be valid probabilities."""
        y_true, y_prob = _synth_calibration_data(200)
        cal = BetaCalibrator()
        cal.fit(y_true, y_prob)

        test_probs = np.linspace(0.01, 0.99, 100)
        calibrated = cal.transform(test_probs)
        assert np.all(calibrated >= 0)
        assert np.all(calibrated <= 1)
        assert np.all(np.isfinite(calibrated))

    def test_build_calibrator_beta(self):
        cal = build_calibrator("beta", {})
        assert isinstance(cal, BetaCalibrator)


# -----------------------------------------------------------------------
# reliability_diagram_data
# -----------------------------------------------------------------------

class TestReliabilityDiagramData:
    def test_returns_list_of_dicts(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        assert isinstance(result, list)
        assert len(result) > 0
        for entry in result:
            assert "bin_center" in entry
            assert "mean_predicted" in entry
            assert "fraction_positive" in entry
            assert "count" in entry

    def test_bin_centers_in_range(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        for entry in result:
            assert 0.0 <= entry["bin_center"] <= 1.0

    def test_counts_sum_to_total(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        total = sum(e["count"] for e in result)
        assert total == 200

    def test_fraction_positive_in_range(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        for entry in result:
            assert 0.0 <= entry["fraction_positive"] <= 1.0

    def test_empty_bins_excluded(self):
        """Bins with no samples are excluded."""
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.45, 0.48, 0.52, 0.55])
        result = reliability_diagram_data(y_true, y_prob, n_bins=10)
        assert len(result) < 10

    def test_custom_n_bins(self):
        y_true, y_prob = _synth_calibration_data(200)
        result_5 = reliability_diagram_data(y_true, y_prob, n_bins=5)
        result_20 = reliability_diagram_data(y_true, y_prob, n_bins=20)
        assert len(result_5) <= 5
        assert len(result_20) <= 20


# -----------------------------------------------------------------------
# hosmer_lemeshow_test
# -----------------------------------------------------------------------

class TestHosmerLemeshowTest:
    def test_returns_expected_keys(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = hosmer_lemeshow_test(y_true, y_prob)
        assert "statistic" in result
        assert "p_value" in result
        assert "n_bins" in result

    def test_well_calibrated_high_p_value(self):
        """Well-calibrated data should yield a high p-value (fail to reject)."""
        rng = np.random.RandomState(42)
        n = 500
        y_prob = rng.uniform(0.2, 0.8, size=n)
        y_true = (rng.rand(n) < y_prob).astype(float)

        result = hosmer_lemeshow_test(y_true, y_prob)
        # p-value should be reasonably high for well-calibrated data
        assert result["p_value"] > 0.01

    def test_poorly_calibrated_low_p_value(self):
        """Poorly calibrated data should yield a low p-value."""
        rng = np.random.RandomState(42)
        n = 500
        # Predict 0.9 but true rate is 0.5
        y_prob = np.full(n, 0.9)
        y_true = (rng.rand(n) < 0.5).astype(float)

        result = hosmer_lemeshow_test(y_true, y_prob)
        assert result["statistic"] > 0
        assert result["p_value"] < 0.05

    def test_statistic_nonnegative(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = hosmer_lemeshow_test(y_true, y_prob)
        assert result["statistic"] >= 0

    def test_p_value_in_range(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = hosmer_lemeshow_test(y_true, y_prob)
        assert 0.0 <= result["p_value"] <= 1.0


# -----------------------------------------------------------------------
# calibration_slope_intercept
# -----------------------------------------------------------------------

class TestCalibrationSlopeIntercept:
    def test_returns_slope_and_intercept(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = calibration_slope_intercept(y_true, y_prob)
        assert "slope" in result
        assert "intercept" in result

    def test_well_calibrated_slope_near_one(self):
        """Well-calibrated predictions should have slope near 1.0."""
        rng = np.random.RandomState(42)
        n = 1000
        y_prob = rng.uniform(0.1, 0.9, size=n)
        y_true = (rng.rand(n) < y_prob).astype(float)

        result = calibration_slope_intercept(y_true, y_prob)
        # Slope should be near 1.0 for well-calibrated data
        assert abs(result["slope"] - 1.0) < 0.5

    def test_values_are_finite(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = calibration_slope_intercept(y_true, y_prob)
        assert np.isfinite(result["slope"])
        assert np.isfinite(result["intercept"])


# -----------------------------------------------------------------------
# bootstrap_ci
# -----------------------------------------------------------------------

class TestBootstrapCI:
    def test_returns_expected_keys(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = bootstrap_ci(y_true, y_prob, _brier_score)
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert "std" in result

    def test_lower_le_mean_le_upper(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = bootstrap_ci(y_true, y_prob, _brier_score)
        assert result["lower"] <= result["mean"] <= result["upper"]

    def test_std_nonnegative(self):
        y_true, y_prob = _synth_calibration_data(200)
        result = bootstrap_ci(y_true, y_prob, _brier_score)
        assert result["std"] >= 0

    def test_ci_narrows_with_more_data(self):
        """CI width should narrow as sample size increases."""
        y_true_small, y_prob_small = _synth_calibration_data(50)
        y_true_large, y_prob_large = _synth_calibration_data(500)

        ci_small = bootstrap_ci(y_true_small, y_prob_small, _brier_score)
        ci_large = bootstrap_ci(y_true_large, y_prob_large, _brier_score)

        width_small = ci_small["upper"] - ci_small["lower"]
        width_large = ci_large["upper"] - ci_large["lower"]
        assert width_large < width_small

    def test_reproducible_with_seed(self):
        y_true, y_prob = _synth_calibration_data(200)
        r1 = bootstrap_ci(y_true, y_prob, _brier_score, seed=123)
        r2 = bootstrap_ci(y_true, y_prob, _brier_score, seed=123)
        assert r1["mean"] == r2["mean"]
        assert r1["lower"] == r2["lower"]
        assert r1["upper"] == r2["upper"]

    def test_custom_alpha(self):
        y_true, y_prob = _synth_calibration_data(200)
        ci_95 = bootstrap_ci(y_true, y_prob, _brier_score, alpha=0.05)
        ci_99 = bootstrap_ci(y_true, y_prob, _brier_score, alpha=0.01)
        # 99% CI should be wider than 95% CI
        width_95 = ci_95["upper"] - ci_95["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert width_99 >= width_95
