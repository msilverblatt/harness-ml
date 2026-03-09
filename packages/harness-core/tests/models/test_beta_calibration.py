"""Tests for BetaCalibrator (3-parameter beta calibration)."""
import numpy as np
import pytest
from harnessml.core.models.calibration import BetaCalibrator


def test_beta_calibration_basic():
    np.random.seed(42)
    raw_probs = np.random.beta(2, 5, 200)
    labels = (np.random.rand(200) < raw_probs).astype(float)

    cal = BetaCalibrator()
    cal.fit(labels, raw_probs)
    calibrated = cal.transform(raw_probs)
    assert calibrated.min() >= 0
    assert calibrated.max() <= 1
    assert len(calibrated) == len(raw_probs)


def test_beta_calibration_is_fitted():
    cal = BetaCalibrator()
    assert not cal.is_fitted
    with pytest.raises(RuntimeError):
        cal.transform(np.array([0.5]))


def test_beta_calibration_identity():
    """Well-calibrated probs should stay roughly the same."""
    np.random.seed(42)
    probs = np.linspace(0.1, 0.9, 100)
    labels = (np.random.rand(100) < probs).astype(float)

    cal = BetaCalibrator()
    cal.fit(labels, probs)
    calibrated = cal.transform(probs)
    # Should not dramatically change well-calibrated probs
    assert np.corrcoef(probs, calibrated)[0, 1] > 0.9


def test_beta_calibration_save_load(tmp_path):
    np.random.seed(42)
    probs = np.random.beta(2, 5, 100)
    labels = (np.random.rand(100) < probs).astype(float)

    cal = BetaCalibrator()
    cal.fit(labels, probs)

    path = tmp_path / "beta_cal.joblib"
    cal.save(path)
    loaded = BetaCalibrator.load(path)

    np.testing.assert_array_almost_equal(
        cal.transform(probs), loaded.transform(probs)
    )
