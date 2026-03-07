"""Elastic net classifier wrapper around sklearn LogisticRegression with elasticnet penalty."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from harnessml.core.models.base import BaseModel

# Keys managed by this wrapper, not passed to LogisticRegression
_WRAPPER_KEYS = frozenset({"normalize"})


class ElasticNetModel(BaseModel):
    """Elastic net classifier via sklearn LogisticRegression with saga solver.

    Accepts ``C``, ``l1_ratio``, and ``max_iter`` in *params*; all other
    kwargs forwarded to LogisticRegression. Defaults to ``penalty='elasticnet'``,
    ``l1_ratio=0.5`` (true elastic net mix), and ``solver='saga'``.

    Parameters
    ----------
    params : dict | None
        Forwarded to LogisticRegression (minus wrapper keys).
    normalize : bool
        If True, apply StandardScaler internally before fit/predict so that
        L1/L2 penalties are fair across features with different scales.
    """

    def __init__(self, params: dict | None = None, *, normalize: bool = False):
        super().__init__(params)
        # normalize can come from params dict (via YAML config) or as kwarg
        self._normalize = normalize or bool(self.params.get("normalize", False))
        self._scaler = None
        self._build_model()

    def _build_model(self) -> None:
        from sklearn.linear_model import LogisticRegression

        sklearn_params = {k: v for k, v in self.params.items() if k not in _WRAPPER_KEYS}
        sklearn_params.setdefault("penalty", "elasticnet")
        sklearn_params.setdefault("solver", "saga")
        sklearn_params.setdefault("l1_ratio", 0.5)
        self._model = LogisticRegression(**sklearn_params)

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, **kwargs) -> None:
        X = np.nan_to_num(X, nan=0.0)
        if self._normalize:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._model.fit(X, y, sample_weight=sample_weight)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X, nan=0.0)
        if self._normalize and self._scaler is not None:
            X = self._scaler.transform(X)
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    @property
    def is_regression(self) -> bool:
        return False

    def save(self, path: Path) -> None:
        import json

        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
        }, path / "model.joblib")
        meta = {"params": self.params, "normalize": self._normalize}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> ElasticNetModel:
        import json

        import joblib

        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        instance = cls.__new__(cls)
        instance.params = meta["params"]
        instance._normalize = meta.get("normalize", False)
        data = joblib.load(path / "model.joblib")
        if isinstance(data, dict):
            instance._model = data["model"]
            instance._scaler = data.get("scaler")
        else:
            # Backward compat: old saves stored model directly
            instance._model = data
            instance._scaler = None
        instance._fitted = True
        return instance
