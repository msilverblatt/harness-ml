"""Abstract base class for all model wrappers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseModel(ABC):
    """Abstract base for all model wrappers."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on data."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) for each row."""
        ...

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        """Return raw margin prediction (only for regressors)."""
        raise NotImplementedError("This model does not support margin prediction")

    @property
    @abstractmethod
    def is_regression(self) -> bool:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> BaseModel:
        ...
