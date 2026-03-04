"""Stacked and averaging ensemble methods with per-fold pre-calibration.

Supports:
- Stacked ensembles via a logistic regression meta-learner trained on
  out-of-fold predictions (LOSO-safe).
- Simple averaging ensembles.
- Per-fold pre-calibration inside the CV loop to prevent leakage.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class StackedEnsemble:
    """Ensemble that combines multiple model predictions.

    Parameters
    ----------
    method : str
        ``"stacked"`` fits a meta-learner on OOF predictions.
        ``"average"`` simply averages all model predictions.
    meta_learner_type : str
        Type of meta-learner (only used when ``method="stacked"``).
        Currently only ``"logistic"`` is supported.
    meta_learner_params : dict | None
        Parameters forwarded to the meta-learner constructor.
    pre_calibrate : dict | None
        Mapping of ``model_name -> calibrator_instance``. When set, the
        calibrator is fit per-fold on the training portion and applied to
        the test portion *inside* the CV loop — preventing calibration
        leakage.
    """

    def __init__(
        self,
        method: str = "stacked",
        meta_learner_type: str = "logistic",
        meta_learner_params: dict | None = None,
        pre_calibrate: dict | None = None,
    ) -> None:
        if method not in ("stacked", "average"):
            raise ValueError(f"Unknown method {method!r}; expected 'stacked' or 'average'")
        self.method = method
        self.meta_learner_type = meta_learner_type
        self.meta_learner_params = meta_learner_params or {}
        self.pre_calibrate = pre_calibrate or {}

        self._meta_learner: Any | None = None
        self._model_names: list[str] = []
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        predictions: dict[str, np.ndarray],
        y: np.ndarray,
        cv: Any | None = None,
        fold_ids: np.ndarray | None = None,
    ) -> None:
        """Fit the ensemble.

        Parameters
        ----------
        predictions : dict[str, np.ndarray]
            Per-model prediction arrays, all the same length as *y*.
        y : np.ndarray
            Ground-truth binary labels.
        cv : CVStrategy | None
            Cross-validation splitter (required for ``method="stacked"``).
        fold_ids : np.ndarray | None
            Fold identifiers aligned with *y* (required for ``method="stacked"``).
        """
        y = np.asarray(y)
        self._model_names = sorted(predictions.keys())

        if self.method == "average":
            self._is_fitted = True
            return

        # Stacked path — need CV
        if cv is None or fold_ids is None:
            raise ValueError("cv and fold_ids are required for method='stacked'")

        fold_ids = np.asarray(fold_ids)
        folds = cv.split(None, fold_ids=fold_ids)

        # Build OOF prediction matrix
        n = len(y)
        n_models = len(self._model_names)
        oof_matrix = np.full((n, n_models), np.nan)

        for fold in folds:
            train_idx = fold.train_idx
            test_idx = fold.test_idx

            for col_idx, model_name in enumerate(self._model_names):
                preds = predictions[model_name]

                # Apply per-fold pre-calibration if configured
                if model_name in self.pre_calibrate:
                    import copy
                    cal = copy.deepcopy(self.pre_calibrate[model_name])
                    cal.fit(y[train_idx], preds[train_idx])
                    calibrated = cal.transform(preds[test_idx])
                    oof_matrix[test_idx, col_idx] = calibrated
                else:
                    oof_matrix[test_idx, col_idx] = preds[test_idx]

        # Drop rows that were never in any test fold
        valid_mask = ~np.isnan(oof_matrix).any(axis=1)
        X_meta = oof_matrix[valid_mask]
        y_meta = y[valid_mask]

        if len(y_meta) == 0:
            raise ValueError("No valid OOF predictions — check CV splits")

        # Fit meta-learner
        self._meta_learner = self._create_meta_learner()
        self._meta_learner.fit(X_meta, y_meta)
        self._is_fitted = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Generate ensemble predictions.

        Parameters
        ----------
        predictions : dict[str, np.ndarray]
            Per-model prediction arrays — must contain the same models
            used in ``fit()``.

        Returns
        -------
        np.ndarray
            Ensemble probability predictions.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted")

        if self.method == "average":
            arrays = [predictions[name] for name in self._model_names]
            return np.mean(arrays, axis=0)

        # Stacked path
        X = np.column_stack([predictions[name] for name in self._model_names])
        probs = self._meta_learner.predict_proba(X)[:, 1]
        return probs

    # ------------------------------------------------------------------
    # Coefficients
    # ------------------------------------------------------------------

    def coefficients(self) -> dict[str, float]:
        """Return meta-learner coefficients (stacked only).

        Returns
        -------
        dict[str, float]
            Model name to meta-learner coefficient mapping.

        Raises
        ------
        ValueError
            If method is not ``"stacked"``.
        """
        if self.method != "stacked":
            raise ValueError("coefficients() is only available for method='stacked'")
        if not self._is_fitted:
            raise RuntimeError("Ensemble not fitted")

        coefs = self._meta_learner.coef_[0]
        return {name: float(coefs[i]) for i, name in enumerate(self._model_names)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save ensemble to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "meta_learner_type": self.meta_learner_type,
            "meta_learner_params": self.meta_learner_params,
            "model_names": self._model_names,
            "meta_learner": self._meta_learner,
        }
        joblib.dump(payload, path / "ensemble.joblib")

    @classmethod
    def load(cls, path: str | Path) -> StackedEnsemble:
        """Load a previously saved ensemble."""
        path = Path(path)
        payload = joblib.load(path / "ensemble.joblib")
        obj = cls(
            method=payload["method"],
            meta_learner_type=payload["meta_learner_type"],
            meta_learner_params=payload["meta_learner_params"],
        )
        obj._model_names = payload["model_names"]
        obj._meta_learner = payload["meta_learner"]
        obj._is_fitted = True
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_meta_learner(self) -> Any:
        """Instantiate the meta-learner from config."""
        if self.meta_learner_type == "logistic":
            params = {"solver": "lbfgs", "max_iter": 5000}
            params.update(self.meta_learner_params)
            return LogisticRegression(**params)
        raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type!r}")
