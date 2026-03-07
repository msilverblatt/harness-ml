"""LightGBM classifier wrapper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel


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

        # Strip early_stopping_rounds from constructor — handled via callback
        self._early_stopping = self.params.get("early_stopping_rounds")
        clean_params = {
            k: v for k, v in self.params.items()
            if k != "early_stopping_rounds"
        }
        clean_params.setdefault("verbosity", -1)
        self._model = LGBMClassifier(**clean_params)

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, **kwargs) -> None:
        import lightgbm as lgb

        if "eval_set" in kwargs and self._early_stopping:
            kwargs.setdefault("callbacks", [])
            kwargs["callbacks"].append(
                lgb.early_stopping(self._early_stopping, verbose=False)
            )
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self._model.fit(X, y, **kwargs)
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
