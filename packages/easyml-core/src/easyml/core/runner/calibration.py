"""Probability calibration utilities for ensemble post-processing.

Provides SplineCalibrator, IsotonicCalibrator, PlattCalibrator,
BetaCalibrator, a factory function build_calibrator(), temperature_scale(),
and calibration diagnostic functions.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class SplineCalibrator:
    """PCHIP-based probability calibration with isotonic pre-processing.

    Bins predictions into equal-frequency bins, computes mean predicted
    vs mean actual per bin, applies isotonic regression to enforce
    monotonicity, then fits a PCHIP interpolator through the mapping.
    """

    def __init__(self, prob_max: float = 0.985, n_bins: int = 20) -> None:
        self.prob_max = prob_max
        self.n_bins = n_bins
        self._interpolator: PchipInterpolator | None = None

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit spline calibrator on observed labels and predicted probabilities.

        Requires at least 20 samples. Bins predictions into n_bins
        equal-frequency bins, computes (mean_pred, mean_actual) per bin,
        applies isotonic regression to enforce monotonicity, then fits
        a PCHIP interpolator for smooth monotone mapping.
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

        # Isotonic regression to enforce monotonicity before PCHIP
        iso = IsotonicRegression(y_min=0.001, y_max=self.prob_max, out_of_bounds="clip")
        mean_actuals_mono = iso.fit_transform(mean_preds, mean_actuals)

        # PCHIP guarantees monotone interpolation between nodes
        self._interpolator = PchipInterpolator(mean_preds, mean_actuals_mono)

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply PCHIP calibration, clipping output to (0.001, prob_max)."""
        if self._interpolator is None:
            raise RuntimeError("SplineCalibrator has not been fitted yet.")
        y_prob = np.asarray(y_prob, dtype=float)
        calibrated = self._interpolator(y_prob)
        return np.clip(calibrated, 0.001, self.prob_max)

    @property
    def is_fitted(self) -> bool:
        return self._interpolator is not None


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


class BetaCalibrator:
    """Beta calibration: fits CDF of Beta distribution.

    Learns parameters (a, b) of a Beta distribution such that
    ``Beta.cdf(p, a, b)`` maps raw probabilities to calibrated ones.
    """

    def __init__(self) -> None:
        self._a = 1.0
        self._b = 1.0
        self._fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        """Fit Beta calibration by maximizing log-likelihood.

        Parameters
        ----------
        y_true : array-like
            Binary outcomes (0 or 1).
        y_prob : array-like
            Predicted probabilities in (0, 1).
        """
        from scipy.optimize import minimize
        from scipy.stats import beta as beta_dist

        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        # Clip to avoid edge-case numerical issues
        eps = 1e-7
        y_prob = np.clip(y_prob, eps, 1.0 - eps)

        def neg_log_lik(params: np.ndarray) -> float:
            a, b = params
            if a <= 0 or b <= 0:
                return 1e12
            calibrated = beta_dist.cdf(y_prob, a, b)
            calibrated = np.clip(calibrated, eps, 1.0 - eps)
            return -float(np.sum(
                y_true * np.log(calibrated)
                + (1 - y_true) * np.log(1 - calibrated)
            ))

        result = minimize(neg_log_lik, [1.0, 1.0], method="Nelder-Mead")
        self._a, self._b = result.x
        self._fitted = True

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply Beta calibration.

        Parameters
        ----------
        y_prob : array-like
            Predicted probabilities in (0, 1).

        Returns
        -------
        np.ndarray
            Calibrated probabilities clipped to [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("BetaCalibrator has not been fitted yet.")
        from scipy.stats import beta as beta_dist

        y_prob = np.asarray(y_prob, dtype=float)
        return np.clip(beta_dist.cdf(y_prob, self._a, self._b), 0, 1)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def params(self) -> tuple[float, float]:
        """Return the fitted (a, b) parameters."""
        return (self._a, self._b)


def build_calibrator(
    method: str, ensemble_config: dict
) -> SplineCalibrator | IsotonicCalibrator | PlattCalibrator | BetaCalibrator | None:
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
    elif method == "beta":
        return BetaCalibrator()
    elif method == "none":
        return None
    else:
        raise ValueError(
            f"Unknown calibration method {method!r}. "
            f"Must be one of: 'spline', 'isotonic', 'platt', 'beta', 'none'."
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


# ---------------------------------------------------------------------------
# Calibration diagnostics
# ---------------------------------------------------------------------------

def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Return data for plotting a reliability diagram.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    list of dict
        Each dict has keys: bin_center, mean_predicted, fraction_positive, count.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = int(mask.sum())
        if count == 0:
            continue

        bins.append({
            "bin_center": float((lo + hi) / 2),
            "mean_predicted": float(y_prob[mask].mean()),
            "fraction_positive": float(y_true[mask].mean()),
            "count": count,
        })

    return bins


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Hosmer-Lemeshow goodness-of-fit test for calibration.

    Groups predictions into n_bins equal-frequency bins and computes
    a chi-squared statistic comparing observed vs expected outcomes.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    n_bins : int
        Number of groups (equal-frequency bins).

    Returns
    -------
    dict
        Keys: statistic, p_value, n_bins.
    """
    from scipy.stats import chi2

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Sort by predicted probability and split into equal-frequency groups
    order = np.argsort(y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    groups = np.array_split(np.arange(len(y_true_sorted)), n_bins)

    hl_stat = 0.0
    actual_bins = 0
    for grp in groups:
        if len(grp) == 0:
            continue
        actual_bins += 1
        n_g = len(grp)
        observed_pos = y_true_sorted[grp].sum()
        observed_neg = n_g - observed_pos
        expected_pos = y_prob_sorted[grp].sum()
        expected_neg = n_g - expected_pos

        # Avoid division by zero
        if expected_pos > 0:
            hl_stat += (observed_pos - expected_pos) ** 2 / expected_pos
        if expected_neg > 0:
            hl_stat += (observed_neg - expected_neg) ** 2 / expected_neg

    # Degrees of freedom = n_groups - 2
    df = max(actual_bins - 2, 1)
    p_value = float(1.0 - chi2.cdf(hl_stat, df))

    return {
        "statistic": float(hl_stat),
        "p_value": p_value,
        "n_bins": actual_bins,
    }


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute calibration slope and intercept from logistic regression.

    Fits ``logit(y_true) ~ slope * logit(y_prob) + intercept``.
    A perfectly calibrated model has slope=1.0, intercept=0.0.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.

    Returns
    -------
    dict
        Keys: slope, intercept.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    eps = 1e-7
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    log_odds = logit(clipped).reshape(-1, 1)

    model = LogisticRegression(C=1e10, max_iter=1000, solver="lbfgs")
    model.fit(log_odds, y_true)

    return {
        "slope": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for a calibration/evaluation metric.

    Parameters
    ----------
    y_true : array-like
        Binary outcomes (0 or 1).
    y_prob : array-like
        Predicted probabilities.
    metric_fn : callable
        Function ``(y_true, y_prob) -> float``.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (e.g. 0.05 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: mean, lower, upper, std.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        scores[i] = metric_fn(y_true[idx], y_prob[idx])

    lower_pct = (alpha / 2) * 100
    upper_pct = (1.0 - alpha / 2) * 100

    return {
        "mean": float(np.mean(scores)),
        "lower": float(np.percentile(scores, lower_pct)),
        "upper": float(np.percentile(scores, upper_pct)),
        "std": float(np.std(scores)),
    }
