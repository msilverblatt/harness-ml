"""Elastic net wrapper supporting both classifier and regressor modes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel

# Keys managed by this wrapper, not passed to sklearn estimators
_WRAPPER_KEYS = frozenset({"normalize"})


class ElasticNetModel(BaseModel):
    """Elastic net via sklearn LogisticRegression (classifier) or ElasticNet (regressor).

    Classifier mode: ``penalty='elasticnet'``, ``solver='saga'``,
    ``l1_ratio=0.5`` defaults.

    Regressor mode: ``alpha`` and ``l1_ratio`` forwarded to sklearn ElasticNet.

    Parameters
    ----------
    params : dict | None
        Forwarded to the underlying estimator (minus wrapper keys).
    mode : str
        ``"classifier"`` or ``"regressor"``.
    normalize : bool
        If True, apply StandardScaler internally before fit/predict.
    """

    def __init__(self, params: dict | None = None, *, mode: str = "classifier", normalize: bool = False):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        # normalize can come from params dict (via YAML config) or as kwarg
        self._normalize = normalize or bool(self.params.get("normalize", False))
        self._scaler = None
        self._build_model()

    def _build_model(self) -> None:
        sklearn_params = {k: v for k, v in self.params.items() if k not in _WRAPPER_KEYS}
        if self._mode == "classifier":
            from sklearn.linear_model import LogisticRegression
            sklearn_params.setdefault("penalty", "elasticnet")
            sklearn_params.setdefault("solver", "saga")
            sklearn_params.setdefault("l1_ratio", 0.5)
            self._model = LogisticRegression(**sklearn_params)
        else:
            from sklearn.linear_model import ElasticNet
            # Map C -> alpha if user passed C (classifier convention)
            if "C" in sklearn_params and "alpha" not in sklearn_params:
                sklearn_params["alpha"] = 1.0 / sklearn_params.pop("C")
            elif "C" in sklearn_params:
                sklearn_params.pop("C")
            # Remove classifier-only params
            sklearn_params.pop("penalty", None)
            sklearn_params.pop("solver", None)
            self._model = ElasticNet(**sklearn_params)

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, **kwargs) -> None:
        X = np.nan_to_num(X, nan=0.0)
        if self._normalize:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._model.fit(X, y, sample_weight=sample_weight)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "regressor":
            raise ValueError(
                "predict_proba not available in regressor mode. Use predict_margin."
            )
        X = np.nan_to_num(X, nan=0.0)
        if self._normalize and self._scaler is not None:
            X = self._scaler.transform(X)
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        if self._mode != "regressor":
            raise NotImplementedError("predict_margin is only available in regressor mode")
        X = np.nan_to_num(X, nan=0.0)
        if self._normalize and self._scaler is not None:
            X = self._scaler.transform(X)
        return self._model.predict(X)

    @property
    def is_regression(self) -> bool:
        return self._mode == "regressor"

    def save(self, path: Path) -> None:
        import json

        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self._model,
            "scaler": self._scaler,
        }, path / "model.joblib")
        meta = {"params": self.params, "normalize": self._normalize, "mode": self._mode}
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
        instance._mode = meta.get("mode", "classifier")
        data = joblib.load(path / "model.joblib")
        if isinstance(data, dict):
            instance._model = data["model"]
            instance._scaler = data.get("scaler")
        else:
            instance._model = data
            instance._scaler = None
        instance._fitted = True
        return instance
