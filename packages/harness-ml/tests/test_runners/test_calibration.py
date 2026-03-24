import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as LR

from harness.ml.runners.calibration import Calibrator


def make_calibration_data(seed=0, n=200):
    rng = np.random.RandomState(seed)
    y_pred = rng.uniform(0, 1, n)
    # Make labels loosely correlated with predictions
    y_true = (y_pred + rng.normal(0, 0.3, n) > 0.5).astype(int)
    return y_true, y_pred


class TestIsotonicCalibration:
    def test_fit_returns_isotonic_regression(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="isotonic")
        assert isinstance(cal, IsotonicRegression)

    def test_transform_roundtrip(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="isotonic")
        calibrated = Calibrator.transform(y_pred, cal)
        assert calibrated.shape == y_pred.shape

    def test_calibrated_predictions_in_unit_interval(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="isotonic")
        calibrated = Calibrator.transform(y_pred, cal)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_transform_on_new_data(self):
        y_true, y_pred = make_calibration_data(seed=0)
        cal = Calibrator.fit(y_true, y_pred, method="isotonic")
        rng = np.random.RandomState(99)
        new_preds = rng.uniform(0, 1, 50)
        calibrated = Calibrator.transform(new_preds, cal)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)


class TestPlattCalibration:
    def test_fit_returns_logistic_regression(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="platt")
        assert isinstance(cal, LR)

    def test_transform_roundtrip(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="platt")
        calibrated = Calibrator.transform(y_pred, cal)
        assert calibrated.shape == y_pred.shape

    def test_calibrated_predictions_in_unit_interval(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="platt")
        calibrated = Calibrator.transform(y_pred, cal)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)


class TestNoneCalibration:
    def test_fit_unknown_method_returns_none(self):
        y_true, y_pred = make_calibration_data()
        cal = Calibrator.fit(y_true, y_pred, method="unknown")
        assert cal is None

    def test_transform_none_calibrator_returns_original(self):
        _, y_pred = make_calibration_data()
        result = Calibrator.transform(y_pred, None)
        np.testing.assert_array_equal(result, y_pred)
