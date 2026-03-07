"""Probability calibration — Spline, Platt scaling, and Isotonic regression.

All calibrators follow a fit/transform interface and support persistence
via joblib save/load.
"""
from __future__ import annotations

from pathlib import Path
from typing import Self

import joblib
import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Base Calibrator
# ---------------------------------------------------------------------------

class BaseCalibrator:
    """Shared save/load/is_fitted logic for all calibrators."""

    _is_fitted: bool = False

    def save(self, path: str | Path) -> None:
        """Persist the fitted calibrator to disk via joblib."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted calibrator")
        joblib.dump(self, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a previously saved calibrator."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        return obj

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# ---------------------------------------------------------------------------
# Spline Calibrator
# ---------------------------------------------------------------------------

class SplineCalibrator(BaseCalibrator):
    """Monotonic spline calibration using PCHIP interpolation.

    Uses quantile binning to compute empirical calibration points, then fits
    a PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) through them.
    Falls back to isotonic regression when there are too few bins.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for the empirical calibration curve.
    prob_max : float
        Maximum allowed output probability (clipped symmetrically).
    """

    def __init__(self, n_bins: int = 15, prob_max: float = 0.985) -> None:
        self._n_bins = n_bins
        self._prob_max = prob_max
        self._spline: PchipInterpolator | None = None
        self._isotonic: IsotonicRegression | None = None
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit the calibrator on held-out labels and raw predictions."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        # Step 1: quantile binning for empirical calibration curve
        n = len(y_pred)
        sort_idx = np.argsort(y_pred)
        sorted_pred = y_pred[sort_idx]
        sorted_true = y_true[sort_idx]

        bin_size = max(n // self._n_bins, 10)
        bin_centers: list[float] = []
        bin_rates: list[float] = []
        for i in range(0, n, bin_size):
            chunk_pred = sorted_pred[i : i + bin_size]
            chunk_true = sorted_true[i : i + bin_size]
            if len(chunk_pred) >= 5:
                bin_centers.append(float(chunk_pred.mean()))
                bin_rates.append(float(chunk_true.mean()))

        if len(bin_centers) < 4:
            # Fall back to isotonic
            self._isotonic = IsotonicRegression(
                y_min=1e-7, y_max=self._prob_max, out_of_bounds="clip",
            )
            self._isotonic.fit(y_pred, y_true)
            self._is_fitted = True
            return

        # Step 2: enforce monotonicity on bin rates
        bin_x = np.array(bin_centers)
        bin_y = np.array(bin_rates)
        iso_bin = IsotonicRegression(
            y_min=1e-7, y_max=self._prob_max, out_of_bounds="clip",
        )
        iso_bin.fit(bin_x, bin_y)
        bin_y_mono = iso_bin.predict(bin_x)

        # Remove duplicate x values (can happen with ties)
        unique_mask = np.concatenate([[True], np.diff(bin_x) > 1e-8])
        bin_x = bin_x[unique_mask]
        bin_y_mono = bin_y_mono[unique_mask]

        if len(bin_x) < 3:
            self._isotonic = IsotonicRegression(
                y_min=1e-7, y_max=self._prob_max, out_of_bounds="clip",
            )
            self._isotonic.fit(y_pred, y_true)
            self._is_fitted = True
            return

        # Step 3: fit PCHIP through monotonic bin centers
        self._spline = PchipInterpolator(bin_x, bin_y_mono, extrapolate=True)
        self._is_fitted = True

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply calibration to raw predictions."""
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted")
        y_pred = np.asarray(y_pred, dtype=float)
        if self._spline is None:
            return self._isotonic.predict(y_pred)
        out = self._spline(y_pred)
        return np.clip(out, 1 - self._prob_max, self._prob_max)


# ---------------------------------------------------------------------------
# Platt Calibrator
# ---------------------------------------------------------------------------

class PlattCalibrator(BaseCalibrator):
    """Platt scaling — logistic regression on raw probabilities.

    Parameters
    ----------
    C : float
        Regularisation strength for the logistic regression.
    """

    def __init__(self, C: float = 1.0) -> None:
        self._C = C
        self._lr: LogisticRegression | None = None
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit logistic regression on held-out labels and raw predictions."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        self._lr = LogisticRegression(C=self._C, solver="lbfgs", max_iter=1000)
        self._lr.fit(y_pred, y_true)
        self._is_fitted = True

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw predictions."""
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted")
        y_pred = np.asarray(y_pred).reshape(-1, 1)
        out = self._lr.predict_proba(y_pred)[:, 1]
        return np.clip(out, 1e-7, 1.0 - 1e-7)


# ---------------------------------------------------------------------------
# Isotonic Calibrator
# ---------------------------------------------------------------------------

class IsotonicCalibrator(BaseCalibrator):
    """Isotonic regression calibration — nonparametric, monotone."""

    def __init__(self) -> None:
        self._iso: IsotonicRegression | None = None
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Fit isotonic regression on held-out labels and raw predictions."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        self._iso = IsotonicRegression(
            y_min=1e-7, y_max=1.0 - 1e-7, out_of_bounds="clip",
        )
        self._iso.fit(y_pred, y_true)
        self._is_fitted = True

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to raw predictions."""
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted")
        y_pred = np.asarray(y_pred, dtype=float)
        return self._iso.predict(y_pred)
