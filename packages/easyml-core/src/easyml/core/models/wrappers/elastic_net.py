"""Elastic net classifier wrapper around sklearn LogisticRegression with elasticnet penalty."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from easyml.core.models.base import BaseModel


class ElasticNetModel(BaseModel):
    """Elastic net classifier via sklearn LogisticRegression with saga solver.

    Accepts ``C``, ``l1_ratio``, and ``max_iter`` in *params*; all other
    kwargs forwarded to LogisticRegression. Defaults to ``l1_ratio=0.5``
    (true elastic net mix) and ``solver='saga'``.
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        from sklearn.linear_model import LogisticRegression

        # Build sklearn kwargs — force saga solver for elasticnet support.
        # l1_ratio controls the penalty mix (0=l2, 1=l1, between=elasticnet).
        sklearn_params = dict(self.params)
        sklearn_params.setdefault("solver", "saga")
        sklearn_params.setdefault("l1_ratio", 0.5)
        self._model = LogisticRegression(**sklearn_params)

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

        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")

    @classmethod
    def load(cls, path: Path) -> ElasticNetModel:
        import joblib

        instance = cls.__new__(cls)
        instance.params = {}
        instance._model = joblib.load(path / "model.joblib")
        instance._fitted = True
        return instance
