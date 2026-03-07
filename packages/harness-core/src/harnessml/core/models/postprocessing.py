"""Ensemble post-processing — probability clipping, temperature scaling, and chaining.

Steps are applied sequentially through an ``EnsemblePostprocessor`` chain.
Each step exposes a single ``apply(probs) -> probs`` interface.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class PostprocessingStep(Protocol):
    """Minimal interface for a post-processing step."""

    def apply(self, probs: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Probability Clipping
# ---------------------------------------------------------------------------

class ProbabilityClipping:
    """Clip probabilities to ``[floor, ceiling]``.

    Parameters
    ----------
    floor : float
        Minimum allowed probability.
    ceiling : float
        Maximum allowed probability.
    """

    def __init__(self, floor: float = 0.0, ceiling: float = 1.0) -> None:
        if floor < 0 or ceiling > 1 or floor >= ceiling:
            raise ValueError(f"Invalid clip range [{floor}, {ceiling}]")
        self.floor = floor
        self.ceiling = ceiling

    def apply(self, probs: np.ndarray) -> np.ndarray:
        return np.clip(probs, self.floor, self.ceiling)


# ---------------------------------------------------------------------------
# Temperature Scaling
# ---------------------------------------------------------------------------

class TemperatureScaling:
    """Divide logits by temperature *T* before converting back to probabilities.

    ``T > 1`` pushes probabilities toward 0.5 (softening).
    ``T < 1`` pushes them toward the extremes (sharpening).

    Parameters
    ----------
    T : float
        Temperature (must be > 0).
    """

    def __init__(self, T: float = 1.0) -> None:
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}")
        self.T = T

    def apply(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        # Clamp away from 0/1 to avoid log(0)
        eps = 1e-15
        probs = np.clip(probs, eps, 1.0 - eps)

        logits = np.log(probs / (1.0 - probs))
        scaled = logits / self.T
        return 1.0 / (1.0 + np.exp(-scaled))


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

class EnsemblePostprocessor:
    """Sequential chain of named post-processing steps.

    Parameters
    ----------
    steps : list[tuple[str, PostprocessingStep]]
        Named steps applied in order.
    """

    def __init__(self, steps: list[tuple[str, PostprocessingStep]] | None = None) -> None:
        self.steps = steps or []

    def apply(self, probs: np.ndarray) -> np.ndarray:
        """Apply all steps in sequence."""
        result = np.asarray(probs, dtype=float).copy()
        for _name, step in self.steps:
            result = step.apply(result)
        return result
