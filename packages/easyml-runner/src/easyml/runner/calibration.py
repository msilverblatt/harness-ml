"""Probability calibration utilities for ensemble post-processing.

Provides SplineCalibrator, IsotonicCalibrator, PlattCalibrator,
a factory function build_calibrator(), and temperature_scale().
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class SplineCalibrator:
    """Spline-based probability calibration.

    Bins predictions into equal-frequency bins, computes mean predicted
    vs mean actual per bin, and fits a cubic spline through the mapping.
    """

    def __init__(self, prob_max: float = 0.985, n_bins: int = 20) -> None:
        self.prob_max = prob_max
        self.n_bins = n_bins
        self._spline: UnivariateSpline | None = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit spline calibrator on observed labels and predicted probabilities.

        Requires at least 20 samples. Bins predictions into n_bins
        equal-frequency bins, computes (mean_pred, mean_actual) per bin,
        and fits a cubic UnivariateSpline.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        if len(y_true) < 20:
            raise ValueError(
                f"SplineCalibrator requires at least 20 samples, got {len(y_true)}"
            )

        # Sort by predicted probability
        order = np.argsort(y_prob)
        y_prob_sorted = y_prob[order]
        y_true_sorted = y_true[order]

        # Equal-frequency bins
        bins = np.array_split(np.arange(len(y_prob_sorted)), self.n_bins)

        mean_preds = []
        mean_actuals = []
        for bin_idx in bins:
            if len(bin_idx) == 0:
                continue
            mean_preds.append(y_prob_sorted[bin_idx].mean())
            mean_actuals.append(y_true_sorted[bin_idx].mean())

        mean_preds = np.array(mean_preds)
        mean_actuals = np.array(mean_actuals)

        # Fit cubic spline; s controls smoothness — use len/4 for moderate smoothing
        s = max(len(mean_preds) / 4.0, 1.0)
        self._spline = UnivariateSpline(mean_preds, mean_actuals, k=3, s=s)

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply spline calibration, clipping output to (0.001, prob_max)."""
        if self._spline is None:
            raise RuntimeError("SplineCalibrator has not been fitted yet.")
        y_prob = np.asarray(y_prob, dtype=float)
        calibrated = self._spline(y_prob)
        return np.clip(calibrated, 0.001, self.prob_max)

    @property
    def is_fitted(self) -> bool:
        return self._spline is not None


class IsotonicCalibrator:
    """Isotonic regression calibration.

    Wraps sklearn.isotonic.IsotonicRegression with sensible defaults.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(
            y_min=0.001, y_max=0.999, out_of_bounds="clip"
        )
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit isotonic regression on observed labels and predicted probabilities."""
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        self._model.fit(y_prob, y_true)
        self._fitted = True

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration."""
        if not self._fitted:
            raise RuntimeError("IsotonicCalibrator has not been fitted yet.")
        y_prob = np.asarray(y_prob, dtype=float)
        return self._model.predict(y_prob)

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class PlattCalibrator:
    """Platt scaling (logistic regression on log-odds).

    Fits a logistic regression on log-odds of predictions to learn
    a sigmoid recalibration mapping.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit Platt scaling on observed labels and predicted probabilities."""
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        eps = 1e-7
        clipped = np.clip(y_prob, eps, 1.0 - eps)
        log_odds = logit(clipped).reshape(-1, 1)
        self._model.fit(log_odds, y_true)
        self._fitted = True

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if not self._fitted:
            raise RuntimeError("PlattCalibrator has not been fitted yet.")
        y_prob = np.asarray(y_prob, dtype=float)
        eps = 1e-7
        clipped = np.clip(y_prob, eps, 1.0 - eps)
        log_odds = logit(clipped).reshape(-1, 1)
        return self._model.predict_proba(log_odds)[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._fitted


def build_calibrator(
    method: str, ensemble_config: dict
) -> SplineCalibrator | IsotonicCalibrator | PlattCalibrator | None:
    """Factory: create calibrator from config string.

    Parameters
    ----------
    method : str
        One of 'spline', 'isotonic', 'platt', 'none'.
    ensemble_config : dict
        Config dict; for spline, reads 'spline_prob_max' and 'spline_n_bins'.

    Returns
    -------
    Calibrator instance or None (for 'none').
    """
    if method == "spline":
        prob_max = ensemble_config.get("spline_prob_max", 0.985)
        n_bins = ensemble_config.get("spline_n_bins", 20)
        return SplineCalibrator(prob_max=prob_max, n_bins=n_bins)
    elif method == "isotonic":
        return IsotonicCalibrator()
    elif method == "platt":
        return PlattCalibrator()
    elif method == "none":
        return None
    else:
        raise ValueError(
            f"Unknown calibration method {method!r}. "
            f"Must be one of: 'spline', 'isotonic', 'platt', 'none'."
        )


def temperature_scale(probs: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling: logit(p) / T -> sigmoid.

    T=1.0 is identity. T>1.0 softens (pushes toward 0.5).
    T<1.0 sharpens (pushes toward 0 and 1).

    Parameters
    ----------
    probs : array-like
        Probabilities in (0, 1).
    T : float
        Temperature. Must be positive.

    Returns
    -------
    np.ndarray
        Temperature-scaled probabilities.
    """
    if T <= 0:
        raise ValueError(f"Temperature must be positive, got {T}")
    probs = np.asarray(probs, dtype=float)
    if T == 1.0:
        return probs
    eps = 1e-7
    clipped = np.clip(probs, eps, 1.0 - eps)
    return expit(logit(clipped) / T)
