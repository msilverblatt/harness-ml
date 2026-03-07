"""Tests for probability calibrators."""
import numpy as np
from harnessml.core.models.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    SplineCalibrator,
)


def test_spline_calibrator_roundtrip():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true * 0.7 + (1 - y_true) * 0.3 + np.random.randn(200) * 0.1, 0.01, 0.99)

    cal = SplineCalibrator(n_bins=10, prob_max=0.985)
    cal.fit(y_true, y_raw)
    y_cal = cal.transform(y_raw)

    assert y_cal.shape == y_raw.shape
    assert all(0 <= p <= 1 for p in y_cal)


def test_spline_calibrator_save_load(tmp_path):
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true * 0.7 + (1 - y_true) * 0.3 + np.random.randn(200) * 0.1, 0.01, 0.99)

    cal = SplineCalibrator(n_bins=10, prob_max=0.985)
    cal.fit(y_true, y_raw)
    cal.save(tmp_path / "cal.joblib")

    loaded = SplineCalibrator.load(tmp_path / "cal.joblib")
    np.testing.assert_array_almost_equal(cal.transform(y_raw), loaded.transform(y_raw))


def test_platt_calibrator():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true * 0.8 + np.random.randn(200) * 0.2, 0.01, 0.99)

    cal = PlattCalibrator()
    cal.fit(y_true, y_raw)
    y_cal = cal.transform(y_raw)
    assert all(0 <= p <= 1 for p in y_cal)


def test_isotonic_calibrator():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true * 0.8 + np.random.randn(200) * 0.2, 0.01, 0.99)

    cal = IsotonicCalibrator()
    cal.fit(y_true, y_raw)
    y_cal = cal.transform(y_raw)
    assert all(0 <= p <= 1 for p in y_cal)


def test_platt_save_load(tmp_path):
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true * 0.8 + np.random.randn(200) * 0.2, 0.01, 0.99)

    cal = PlattCalibrator()
    cal.fit(y_true, y_raw)
    cal.save(tmp_path / "platt.joblib")

    loaded = PlattCalibrator.load(tmp_path / "platt.joblib")
    np.testing.assert_array_almost_equal(cal.transform(y_raw), loaded.transform(y_raw))


def test_isotonic_save_load(tmp_path):
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 200)
    y_raw = np.clip(y_true * 0.8 + np.random.randn(200) * 0.2, 0.01, 0.99)

    cal = IsotonicCalibrator()
    cal.fit(y_true, y_raw)
    cal.save(tmp_path / "iso.joblib")

    loaded = IsotonicCalibrator.load(tmp_path / "iso.joblib")
    np.testing.assert_array_almost_equal(cal.transform(y_raw), loaded.transform(y_raw))


def test_unfitted_calibrator_raises():
    """Calling transform before fit should raise RuntimeError."""
    for cal_cls in [SplineCalibrator, PlattCalibrator, IsotonicCalibrator]:
        cal = cal_cls() if cal_cls != SplineCalibrator else cal_cls(n_bins=10)
        try:
            cal.transform(np.array([0.5]))
            assert False, f"{cal_cls.__name__} should have raised"
        except RuntimeError:
            pass


def test_spline_fallback_to_isotonic_with_tiny_data():
    """With very few samples, spline should fall back to isotonic."""
    np.random.seed(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_raw = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.15, 0.85, 0.25, 0.75])

    cal = SplineCalibrator(n_bins=20, prob_max=0.985)
    cal.fit(y_true, y_raw)
    y_cal = cal.transform(y_raw)
    assert y_cal.shape == y_raw.shape
    assert all(0 <= p <= 1 for p in y_cal)
