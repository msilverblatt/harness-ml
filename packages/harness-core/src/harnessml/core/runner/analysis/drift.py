"""Feature and prediction drift detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class DriftResult:
    is_drifted: bool
    statistic: float
    p_value: float | None = None
    psi: float | None = None
    method: str = ""


def detect_drift(
    reference: np.ndarray,
    current: np.ndarray,
    method: str = "ks",
    threshold: float = 0.05,
) -> DriftResult:
    """Detect distribution drift between reference and current data.

    Parameters
    ----------
    reference : array of reference distribution values
    current : array of current distribution values
    method : "ks" (Kolmogorov-Smirnov) or "psi" (Population Stability Index)
    threshold : significance level for KS, or PSI threshold (default 0.05 for KS)
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if method == "ks":
        stat, p_value = stats.ks_2samp(reference, current)
        return DriftResult(
            is_drifted=p_value < threshold,
            statistic=stat,
            p_value=p_value,
            method="ks",
        )
    elif method == "psi":
        psi_value = _compute_psi(reference, current)
        return DriftResult(
            is_drifted=psi_value > 0.2,
            statistic=psi_value,
            psi=psi_value,
            method="psi",
        )
    else:
        raise ValueError(f"Unknown drift method: {method}")


def detect_multi_feature_drift(
    reference_df,
    current_df,
    method: str = "ks",
    threshold: float = 0.05,
) -> dict[str, DriftResult]:
    """Detect drift for each numeric column in a DataFrame."""
    results = {}
    for col in reference_df.select_dtypes(include=[np.number]).columns:
        if col in current_df.columns:
            results[col] = detect_drift(
                reference_df[col].dropna().values,
                current_df[col].dropna().values,
                method=method,
                threshold=threshold,
            )
    return results


def _compute_psi(reference, current, n_bins=10):
    """Population Stability Index."""
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)
    ref_counts = np.clip(ref_counts, 1e-6, None)
    cur_counts = np.clip(cur_counts, 1e-6, None)
    return float(np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts)))
