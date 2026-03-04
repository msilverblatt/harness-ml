"""LightGBM classifier wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from easyml.core.models.base import BaseModel


class LightGBMModel(BaseModel):
    """Wrapper around LGBMClassifier.

    Parameters
    ----------
    params : dict | None
        Forwarded to LGBMClassifier (e.g. n_estimators, max_depth, verbose).
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        from lightgbm import LGBMClassifier

        self._model = LGBMClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def is_regression(self) -> bool:
        return False

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> LightGBMModel:
        import joblib

        path = Path(path)
        instance = cls.__new__(cls)
        instance.params = {}
        instance._model = joblib.load(path / "model.joblib")
        instance._fitted = True
        return instance
