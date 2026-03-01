"""Tests for calibration utilities."""
from __future__ import annotations

import numpy as np
import pytest

from easyml.runner.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SplineCalibrator,
    build_calibrator,
    temperature_scale,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _synth_calibration_data(n: int = 200, seed: int = 42):
    """Generate synthetic calibration data: miscalibrated predictions + labels."""
    rng = np.random.RandomState(seed)
    # True probabilities
    true_p = rng.beta(2, 2, size=n)
    # Observed labels
    y_true = (rng.rand(n) < true_p).astype(float)
    # Miscalibrated predictions (shifted toward extremes)
    y_prob = np.clip(true_p * 1.2 - 0.1, 0.05, 0.95)
    return y_true, y_prob


# -----------------------------------------------------------------------
# SplineCalibrator
# -----------------------------------------------------------------------

class TestSplineCalibrator:
    def test_fit_transform_roundtrip(self):
        y_true, y_prob = _synth_calibration_data(200)
        cal = SplineCalibrator(prob_max=0.985, n_bins=20)
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        # Output should be clipped within bounds
        assert calibrated.min() >= 0.001
        assert calibrated.max() <= 0.985
        assert len(calibrated) == len(y_prob)

    def test_preserves_monotonicity(self):
        """Spline-calibrated output should roughly preserve ordering."""
        y_true, y_prob = _synth_calibration_data(300)
        cal = SplineCalibrator(prob_max=0.985, n_bins=20)
        cal.fit(y_true, y_prob)

        # Test on sorted input
        sorted_probs = np.sort(y_prob)
        calibrated = cal.transform(sorted_probs)

        # Allow a few violations due to spline smoothing, but mostly monotonic
        diffs = np.diff(calibrated)
        n_violations = (diffs < -0.01).sum()
        assert n_violations < len(diffs) * 0.1, (
            f"Too many monotonicity violations: {n_violations}/{len(diffs)}"
        )

    def test_requires_min_samples(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_prob = np.array([0.2, 0.8, 0.3, 0.7, 0.4])
        cal = SplineCalibrator()
        with pytest.raises(ValueError, match="at least 20 samples"):
            cal.fit(y_true, y_prob)

    def test_is_fitted_property(self):
        cal = SplineCalibrator()
        assert cal.is_fitted is False
        y_true, y_prob = _synth_calibration_data(50)
        cal.fit(y_true, y_prob)
        assert cal.is_fitted is True

    def test_transform_before_fit_raises(self):
        cal = SplineCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.transform(np.array([0.5]))

    def test_custom_params(self):
        cal = SplineCalibrator(prob_max=0.95, n_bins=10)
        assert cal.prob_max == 0.95
        assert cal.n_bins == 10

        y_true, y_prob = _synth_calibration_data(100)
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)
        assert calibrated.max() <= 0.95


# -----------------------------------------------------------------------
# IsotonicCalibrator
# -----------------------------------------------------------------------

class TestIsotonicCalibrator:
    def test_basic_fit_transform(self):
        y_true, y_prob = _synth_calibration_data(200)
        cal = IsotonicCalibrator()
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        assert len(calibrated) == len(y_prob)
        assert calibrated.min() >= 0.001
        assert calibrated.max() <= 0.999

    def test_is_fitted_property(self):
        cal = IsotonicCalibrator()
        assert cal.is_fitted is False
        y_true, y_prob = _synth_calibration_data(50)
        cal.fit(y_true, y_prob)
        assert cal.is_fitted is True

    def test_transform_before_fit_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.transform(np.array([0.5]))

    def test_monotonic_output(self):
        y_true, y_prob = _synth_calibration_data(200)
        cal = IsotonicCalibrator()
        cal.fit(y_true, y_prob)

        sorted_probs = np.sort(y_prob)
        calibrated = cal.transform(sorted_probs)
        # Isotonic regression guarantees monotonicity
        assert np.all(np.diff(calibrated) >= -1e-10)


# -----------------------------------------------------------------------
# PlattCalibrator
# -----------------------------------------------------------------------

class TestPlattCalibrator:
    def test_basic_fit_transform(self):
        y_true, y_prob = _synth_calibration_data(200)
        cal = PlattCalibrator()
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)

        assert len(calibrated) == len(y_prob)
        # Output should be valid probabilities
        assert calibrated.min() >= 0.0
        assert calibrated.max() <= 1.0

    def test_is_fitted_property(self):
        cal = PlattCalibrator()
        assert cal.is_fitted is False
        y_true, y_prob = _synth_calibration_data(50)
        cal.fit(y_true, y_prob)
        assert cal.is_fitted is True

    def test_transform_before_fit_raises(self):
        cal = PlattCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            cal.transform(np.array([0.5]))

    def test_preserves_ordering(self):
        """Platt scaling is a monotonic transform (sigmoid on log-odds)."""
        y_true, y_prob = _synth_calibration_data(200)
        cal = PlattCalibrator()
        cal.fit(y_true, y_prob)

        sorted_probs = np.sort(y_prob)
        calibrated = cal.transform(sorted_probs)
        # The fitted sigmoid should be monotonic
        assert np.all(np.diff(calibrated) >= -1e-10)


# -----------------------------------------------------------------------
# build_calibrator factory
# -----------------------------------------------------------------------

class TestBuildCalibrator:
    def test_spline(self):
        cal = build_calibrator("spline", {"spline_prob_max": 0.99, "spline_n_bins": 15})
        assert isinstance(cal, SplineCalibrator)
        assert cal.prob_max == 0.99
        assert cal.n_bins == 15

    def test_spline_defaults(self):
        cal = build_calibrator("spline", {})
        assert isinstance(cal, SplineCalibrator)
        assert cal.prob_max == 0.985
        assert cal.n_bins == 20

    def test_isotonic(self):
        cal = build_calibrator("isotonic", {})
        assert isinstance(cal, IsotonicCalibrator)

    def test_platt(self):
        cal = build_calibrator("platt", {})
        assert isinstance(cal, PlattCalibrator)

    def test_none(self):
        cal = build_calibrator("none", {})
        assert cal is None

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            build_calibrator("magic", {})


# -----------------------------------------------------------------------
# temperature_scale
# -----------------------------------------------------------------------

class TestTemperatureScale:
    def test_identity_at_T1(self):
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = temperature_scale(probs, T=1.0)
        np.testing.assert_allclose(result, probs, atol=1e-10)

    def test_softens_at_T2(self):
        """T>1 pushes probabilities toward 0.5."""
        probs = np.array([0.1, 0.3, 0.7, 0.9])
        result = temperature_scale(probs, T=2.0)
        # Values below 0.5 should increase, values above should decrease
        assert result[0] > probs[0]
        assert result[1] > probs[1]
        assert result[2] < probs[2]
        assert result[3] < probs[3]
        # All should be closer to 0.5
        assert abs(result[0] - 0.5) < abs(probs[0] - 0.5)
        assert abs(result[3] - 0.5) < abs(probs[3] - 0.5)

    def test_sharpens_at_T05(self):
        """T<1 pushes probabilities away from 0.5."""
        probs = np.array([0.2, 0.4, 0.6, 0.8])
        result = temperature_scale(probs, T=0.5)
        # Values below 0.5 should decrease, values above should increase
        assert result[0] < probs[0]
        assert result[1] < probs[1]
        assert result[2] > probs[2]
        assert result[3] > probs[3]

    def test_preserves_05(self):
        """0.5 should remain 0.5 at any temperature."""
        probs = np.array([0.5])
        for T in [0.5, 1.0, 2.0, 5.0]:
            result = temperature_scale(probs, T=T)
            np.testing.assert_allclose(result, [0.5], atol=1e-10)

    def test_negative_T_raises(self):
        with pytest.raises(ValueError, match="positive"):
            temperature_scale(np.array([0.5]), T=-1.0)

    def test_zero_T_raises(self):
        with pytest.raises(ValueError, match="positive"):
            temperature_scale(np.array([0.5]), T=0.0)

    def test_handles_extreme_probs(self):
        """Should not produce NaN/Inf for very extreme probabilities."""
        probs = np.array([0.001, 0.999])
        result = temperature_scale(probs, T=2.0)
        assert np.all(np.isfinite(result))
        assert np.all(result > 0)
        assert np.all(result < 1)
