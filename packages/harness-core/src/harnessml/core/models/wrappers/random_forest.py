"""Random forest wrapper supporting both classifier and regressor modes."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from harnessml.core.models.base import BaseModel

logger = logging.getLogger(__name__)

# Params that belong to boosting models (XGBoost, LightGBM, CatBoost) and are
# not valid for sklearn RandomForest.  Kept as a module-level
# frozenset so it can be extended without touching __init__.
_INVALID_RF_PARAMS: frozenset[str] = frozenset({
    "learning_rate",
    "colsample_bytree",
    "subsample",
    "reg_alpha",
    "reg_lambda",
    "gamma",
    "min_child_weight",
    "scale_pos_weight",
    "early_stopping_rounds",
    "num_leaves",
    "bagging_fraction",
    "feature_fraction",
})


class RandomForestModel(BaseModel):
    """Wrapper around sklearn RandomForestClassifier / RandomForestRegressor.

    Parameters
    ----------
    params : dict | None
        Forwarded to the underlying estimator (e.g. n_estimators, max_depth).
        Params meant for boosting models are filtered out with a warning.
    mode : str
        ``"classifier"`` or ``"regressor"``.
    """

    def __init__(self, params: dict | None = None, *, mode: str = "classifier"):
        super().__init__(params)
        if mode not in ("classifier", "regressor"):
            raise ValueError(f"mode must be 'classifier' or 'regressor', got {mode!r}")
        self._mode = mode
        filtered = {}
        for key, value in self.params.items():
            if key in _INVALID_RF_PARAMS:
                logger.warning(
                    "Dropping boosting param '%s' (not valid for RandomForest)", key
                )
            else:
                filtered[key] = value
        self.params = filtered
        self._build_model()

    def _build_model(self) -> None:
        if self._mode == "classifier":
            from sklearn.ensemble import RandomForestClassifier
            self._model = RandomForestClassifier(**self.params)
        else:
            from sklearn.ensemble import RandomForestRegressor
            self._model = RandomForestRegressor(**self.params)

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

    def save(self, path: Path) -> None:
        import json

        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        meta = {"mode": self._mode, "params": self.params}
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path) -> RandomForestModel:
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
