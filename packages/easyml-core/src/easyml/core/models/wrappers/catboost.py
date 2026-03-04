"""CatBoost classifier wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from easyml.core.models.base import BaseModel


class CatBoostModel(BaseModel):
    """Wrapper around CatBoostClassifier.

    Parameters
    ----------
    params : dict | None
        Forwarded to CatBoostClassifier (e.g. iterations, depth, verbose).
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)[:, 1]

    @property
    def is_regression(self) -> bool:
        return False

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path / "model.cbm"))
        meta = {"params": self.params}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> CatBoostModel:
        from catboost import CatBoostClassifier

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance._fitted = True
        instance._model = CatBoostClassifier()
        instance._model.load_model(str(path / "model.cbm"))
        return instance
