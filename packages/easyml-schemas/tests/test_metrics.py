import numpy as np
from easyml.schemas.metrics import brier_score, log_loss, ece, calibration_table, accuracy


def test_brier_score_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y_true, y_prob) == 0.0


def test_brier_score_worst():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_prob) == 1.0


def test_brier_score_mid():
    y_true = np.array([1, 0])
    y_prob = np.array([0.5, 0.5])
    assert abs(brier_score(y_true, y_prob) - 0.25) < 1e-10


def test_log_loss_basic():
    y_true = np.array([1, 0])
    y_prob = np.array([0.9, 0.1])
    result = log_loss(y_true, y_prob)
    assert 0 < result < 1


def test_accuracy_basic():
    y_true = np.array([1, 0, 1, 0])
    y_prob = np.array([0.8, 0.3, 0.6, 0.4])
    assert accuracy(y_true, y_prob, threshold=0.5) == 1.0


def test_accuracy_custom_threshold():
    y_true = np.array([1, 0])
    y_prob = np.array([0.6, 0.4])
    assert accuracy(y_true, y_prob, threshold=0.7) == 0.5


def test_ece_perfect_calibration():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])  # 30% positive
    y_prob = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])
    result = ece(y_true, y_prob, n_bins=5)
    assert result < 0.05


def test_calibration_table_structure():
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.6, 0.4, 0.85, 0.15])
    table = calibration_table(y_true, y_prob, n_bins=5)
    assert isinstance(table, list)
    assert all("bin_start" in row and "mean_predicted" in row and "actual_accuracy" in row for row in table)
