"""HistGradientBoosting wrapper supporting both classifier and regressor modes."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel

logger = logging.getLogger(__name__)


class HistGradientBoostingModel(BaseModel):
    """Wrapper around sklearn HistGradientBoostingClassifier / Regressor.

    Parameters
    ----------
    params : dict | None
        Forwarded to the underlying estimator (e.g. max_iter, max_depth,
        learning_rate, l2_regularization).
    mode : str
        ``"classifier"`` or ``"regressor"``.
    """

    def __init__(self, params: dict | None = None, *, mode: str = "classifier"):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        self._build_model()

    def _build_model(self) -> None:
        if self._mode == "classifier":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._model = HistGradientBoostingClassifier(**self.params)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            self._model = HistGradientBoostingRegressor(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, **kwargs) -> None:
        self._model.fit(X, y, sample_weight=sample_weight)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "regressor":
            raise ValueError(
                "predict_proba not available in regressor mode. Use predict_margin."
            )
        probs = self._model.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs

    def predict_margin(self, X: np.ndarray) -> np.ndarray:
        if self._mode != "regressor":
            raise NotImplementedError("predict_margin is only available in regressor mode")
        return self._model.predict(X)

    @property
    def is_regression(self) -> bool:
        return self._mode == "regressor"

    @property
    def feature_importances(self) -> np.ndarray | None:
        """Return feature importances from the underlying model if fitted."""
        if not self._fitted:
            return None
        return None

    def save(self, path: Path) -> None:
        import json

        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        meta = {"mode": self._mode, "params": self.params}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> HistGradientBoostingModel:
        import json

        import joblib

        path = Path(path)
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            mode = meta.get("mode", "classifier")
            params = meta.get("params", {})
        else:
            mode = "classifier"
            params = {}
        instance = cls.__new__(cls)
        instance.params = params
        instance._mode = mode
        instance._model = joblib.load(path / "model.joblib")
        instance._fitted = True
        return instance
