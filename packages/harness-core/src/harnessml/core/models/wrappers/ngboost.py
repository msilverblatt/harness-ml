"""NGBoost wrapper (optional dependency)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel

logger = logging.getLogger(__name__)


class NGBoostModel(BaseModel):
    """Wrapper around NGBoost for probabilistic predictions.

    Requires the ``ngboost`` package. Raises ImportError at init if not installed.

    Parameters
    ----------
    params : dict | None
        Forwarded to NGBClassifier or NGBRegressor.
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
        try:
            if self._mode == "classifier":
                from ngboost import NGBClassifier
                self._model = NGBClassifier(**self.params)
            else:
                from ngboost import NGBRegressor
                self._model = NGBRegressor(**self.params)
        except ImportError:
            raise ImportError(
                "NGBoost is required for NGBoostModel. Install with: pip install ngboost"
            )

    def fit(self, X: np.ndarray, y: np.ndarray, *, sample_weight: np.ndarray | None = None, **kwargs) -> None:
        if sample_weight is not None:
            self._model.fit(X, y, sample_weight=sample_weight)
        else:
            self._model.fit(X, y)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._mode == "regressor":
            raise ValueError(
                "predict_proba not available in regressor mode. Use predict_margin."
            )
        probs = self._model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] == 2:
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
        """Return feature importances from the underlying base learner if available."""
        if not self._fitted:
            return None
        return getattr(self._model, "feature_importances_", None)

    def save(self, path: Path) -> None:
        import json
        import pickle

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self._model, f)
        meta = {"mode": self._mode, "params": self.params}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> NGBoostModel:
        import json
        import pickle

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
        with open(path / "model.pkl", "rb") as f:
            instance._model = pickle.load(f)
        instance._fitted = True
        return instance
