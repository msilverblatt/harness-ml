"""CatBoost classifier wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harnessml.core.models.base import BaseModel


class CatBoostModel(BaseModel):
    """Wrapper around CatBoostClassifier.

    Parameters
    ----------
    params : dict | None
        Forwarded to CatBoostClassifier (e.g. iterations, depth, verbose).
    """

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        self._build_model()

    def _build_model(self) -> None:
        from catboost import CatBoostClassifier

        params = dict(self.params)
        params.setdefault("allow_writing_files", False)
        self._model = CatBoostClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray, eval_set=None) -> None:
        from catboost import CatBoostClassifier

        params = dict(self.params)
        early_stopping = params.pop("early_stopping_rounds", None)
        params.setdefault("verbose", 0)
        params.setdefault("allow_writing_files", False)
        self._model = CatBoostClassifier(**params)

        fit_kwargs: dict = {}
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            fit_kwargs["eval_set"] = (X_val, y_val)
            if early_stopping:
                fit_kwargs["early_stopping_rounds"] = early_stopping

        self._model.fit(X, y, **fit_kwargs)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs

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
