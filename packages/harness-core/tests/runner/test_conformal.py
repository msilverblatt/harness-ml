import numpy as np
import pytest
from harnessml.core.runner.conformal import ConformalPredictor


def test_conformal_prediction_coverage():
    np.random.seed(42)
    cal_probs = np.random.rand(500)
    cal_labels = (np.random.rand(500) < cal_probs).astype(float)
    test_probs = np.random.rand(200)
    test_labels = (np.random.rand(200) < test_probs).astype(float)

    cp = ConformalPredictor(alpha=0.1)  # 90% coverage
    cp.calibrate(cal_probs, cal_labels)
    intervals = cp.predict(test_probs)

    covered = sum(1 for i, label in enumerate(test_labels)
                  if intervals[i][0] <= label <= intervals[i][1])
    coverage = covered / len(test_labels)
    assert coverage >= 0.85  # Allow some slack


def test_conformal_intervals_valid():
    np.random.seed(42)
    cal_probs = np.random.rand(100)
    cal_labels = (np.random.rand(100) < cal_probs).astype(float)

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(cal_probs, cal_labels)
    intervals = cp.predict(np.array([0.5, 0.1, 0.9]))

    for lower, upper in intervals:
        assert 0.0 <= lower <= upper <= 1.0


def test_conformal_not_calibrated():
    cp = ConformalPredictor()
    with pytest.raises(RuntimeError, match="calibrate"):
        cp.predict(np.array([0.5]))


def test_prediction_sets():
    np.random.seed(42)
    cal_probs = np.random.rand(200)
    cal_labels = (np.random.rand(200) < cal_probs).astype(float)

    cp = ConformalPredictor(alpha=0.1)
    cp.calibrate(cal_probs, cal_labels)

    # Very confident prediction should have small set
    sets = cp.predict_sets(np.array([0.99]))
    assert 1 in sets[0]


def test_wider_intervals_with_lower_confidence():
    np.random.seed(42)
    cal_probs = np.random.rand(200)
    cal_labels = (np.random.rand(200) < cal_probs).astype(float)

    cp_90 = ConformalPredictor(alpha=0.1)  # 90%
    cp_95 = ConformalPredictor(alpha=0.05)  # 95%
    cp_90.calibrate(cal_probs, cal_labels)
    cp_95.calibrate(cal_probs, cal_labels)

    intervals_90 = cp_90.predict(np.array([0.5]))
    intervals_95 = cp_95.predict(np.array([0.5]))

    width_90 = intervals_90[0][1] - intervals_90[0][0]
    width_95 = intervals_95[0][1] - intervals_95[0][0]
    assert width_95 >= width_90
