"""Conformal prediction for calibrated confidence intervals."""
from __future__ import annotations

import numpy as np


class ConformalPredictor:
    """Split-conformal prediction.

    Calibrates nonconformity scores on a held-out calibration set,
    then uses the quantile of scores to generate prediction intervals.

    Parameters
    ----------
    alpha : float
        Significance level. 1 - alpha = target coverage (e.g., alpha=0.1 -> 90% coverage).
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._quantile: float | None = None
        self._calibrated = False

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Calibrate using held-out data.

        Computes nonconformity scores and finds the (1-alpha) quantile.
        """
        probs = np.asarray(probs, dtype=float)
        labels = np.asarray(labels, dtype=float)

        # Nonconformity score: absolute difference between predicted prob and true label
        scores = np.abs(probs - labels)

        # Compute quantile with finite-sample correction
        n = len(scores)
        adjusted_alpha = np.ceil((n + 1) * (1 - self.alpha)) / n
        adjusted_alpha = min(adjusted_alpha, 1.0)
        self._quantile = float(np.quantile(scores, adjusted_alpha))
        self._calibrated = True

    def predict(self, probs: np.ndarray) -> list[tuple[float, float]]:
        """Generate prediction intervals.

        Returns list of (lower, upper) tuples for each prediction.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        probs = np.asarray(probs, dtype=float)
        intervals = []
        for p in probs:
            lower = max(0.0, p - self._quantile)
            upper = min(1.0, p + self._quantile)
            intervals.append((lower, upper))
        return intervals

    def predict_sets(self, probs: np.ndarray, classes: list | None = None) -> list[list]:
        """Generate prediction sets for classification.

        Returns list of sets of plausible classes for each prediction.
        For binary: returns which classes are plausible.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_sets()")

        if classes is None:
            classes = [0, 1]

        probs = np.asarray(probs, dtype=float)
        sets = []
        for p in probs:
            plausible = []
            # Class 1 is plausible if interval includes values > 0.5
            if p + self._quantile > 0.5:
                plausible.append(classes[1])
            # Class 0 is plausible if interval includes values < 0.5
            if p - self._quantile < 0.5:
                plausible.append(classes[0])
            sets.append(sorted(plausible))
        return sets
