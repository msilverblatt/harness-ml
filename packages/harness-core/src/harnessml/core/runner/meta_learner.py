"""Stacked meta-learner for ensemble prediction.

Provides StackedEnsemble class and train_meta_learner_loso() for
leave-one-out training with nested calibrator CV.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from harnessml.core.runner.calibration import build_calibrator
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV

logger = logging.getLogger(__name__)

_META_LEARNER_TYPES = ("logistic", "ridge", "gbm")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class _RidgeProbaWrapper:
    """Wraps RidgeCV to provide predict_proba via sigmoid for binary classification."""

    def __init__(self, ridge: RidgeCV) -> None:
        self._ridge = ridge

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeProbaWrapper":
        self._ridge.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._ridge.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._ridge.predict(X)
        p1 = _sigmoid(raw)
        return np.column_stack([1.0 - p1, p1])

    # Expose attributes needed by save/load and get_coefficients
    @property
    def coef_(self) -> np.ndarray:
        c = self._ridge.coef_
        return c if c.ndim == 2 else c.reshape(1, -1)

    @coef_.setter
    def coef_(self, value: np.ndarray) -> None:
        self._ridge.coef_ = value.ravel() if value.ndim == 2 else value

    @property
    def intercept_(self) -> np.ndarray:
        v = self._ridge.intercept_
        return np.atleast_1d(v)

    @intercept_.setter
    def intercept_(self, value: np.ndarray) -> None:
        self._ridge.intercept_ = float(value.ravel()[0]) if np.ndim(value) > 0 else float(value)

    @property
    def classes_(self) -> np.ndarray:
        return np.array([0.0, 1.0])

    @classes_.setter
    def classes_(self, value: np.ndarray) -> None:
        pass  # Not used by Ridge, but needed for load compatibility


class StackedEnsemble:
    """Meta-learner over base model predictions.

    Supports multiple meta-learner types via the ``meta_learner_type`` parameter:

    - ``"logistic"`` (default): LogisticRegression (binary/multiclass) or
      LinearRegression (regression). This is the original behaviour.
    - ``"ridge"``: RidgeCV with sigmoid probability conversion for
      binary classification, or plain RidgeCV for regression.
    - ``"gbm"``: LightGBM classifier/regressor with conservative defaults
      (n_estimators=50, max_depth=3, learning_rate=0.1).

    For binary (n_classes <= 2): Feature matrix is
    [model_pred_1, ..., model_pred_N, prior_diff, extra_1, ...].

    For multiclass (n_classes > 2): Feature matrix flattens per-class
    probabilities from each model:
    [model_1_c0, model_1_c1, ..., model_N_cK, prior_diff, extra_1, ...].
    """

    def __init__(
        self,
        model_names: list[str],
        n_classes: int = 2,
        task_type: str = "binary",
        meta_learner_type: str = "logistic",
    ) -> None:
        if meta_learner_type not in _META_LEARNER_TYPES:
            raise ValueError(
                f"Unknown meta_learner_type '{meta_learner_type}'. "
                f"Must be one of {_META_LEARNER_TYPES}."
            )
        self.model_names = list(model_names)
        self.n_classes = n_classes
        self.task_type = task_type
        self.meta_learner_type = meta_learner_type
        self._model: Any = None

    @property
    def _is_regression(self) -> bool:
        return self.task_type == "regression"

    @property
    def _is_multiclass(self) -> bool:
        return self.n_classes > 2

    def _build_feature_matrix(
        self,
        model_preds: dict[str, np.ndarray],
        prior_diffs: np.ndarray,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Build the feature matrix for the meta-learner.

        Missing model predictions are filled with 0.5 (uninformative prior).
        """
        n = len(prior_diffs)
        cols = []
        for name in self.model_names:
            if name in model_preds:
                arr = np.asarray(model_preds[name], dtype=float)
            else:
                arr = np.full(n, 0.5)
            if self._is_multiclass and arr.ndim == 2:
                # Flatten per-class probabilities into separate columns
                for cls_idx in range(arr.shape[1]):
                    cols.append(arr[:, cls_idx])
            else:
                cols.append(arr)
        cols.append(np.asarray(prior_diffs, dtype=float))
        if extra_features:
            for feat_name in sorted(extra_features.keys()):
                cols.append(np.asarray(extra_features[feat_name], dtype=float))
        return np.column_stack(cols)

    def _feature_names(
        self, extra_features: dict[str, np.ndarray] | None = None
    ) -> list[str]:
        """Return ordered list of feature names matching the feature matrix columns."""
        names = []
        if self._is_multiclass:
            for model_name in self.model_names:
                for cls_idx in range(self.n_classes):
                    names.append(f"{model_name}_c{cls_idx}")
        else:
            names = list(self.model_names)
        names.append("prior_diff")
        if extra_features:
            names.extend(sorted(extra_features.keys()))
        return names

    def _create_model(self, C: float = 1.0) -> Any:
        """Instantiate the underlying sklearn/lgbm model based on meta_learner_type."""
        if self._is_regression:
            if self.meta_learner_type == "logistic":
                return LinearRegression()
            elif self.meta_learner_type == "ridge":
                return RidgeCV()
            elif self.meta_learner_type == "gbm":
                from lightgbm import LGBMRegressor
                return LGBMRegressor(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    verbosity=-1,
                )
        else:
            if self.meta_learner_type == "logistic":
                return LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
            elif self.meta_learner_type == "ridge":
                return _RidgeProbaWrapper(RidgeCV())
            elif self.meta_learner_type == "gbm":
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.1,
                    verbosity=-1,
                )
        # Should not reach here due to __init__ validation
        raise ValueError(f"Unknown meta_learner_type: {self.meta_learner_type}")

    def fit(
        self,
        model_preds: dict[str, np.ndarray],
        prior_diffs: np.ndarray,
        y_true: np.ndarray,
        C: float = 1.0,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Fit the meta-learner on base model predictions.

        Parameters
        ----------
        model_preds : dict mapping model_name -> prediction array
            For binary: 1D arrays. For multiclass: 2D (n_samples, n_classes).
        prior_diffs : prior difference array (e.g. higher_prior - lower_prior)
        y_true : outcome array (binary labels or integer class labels)
        C : regularization strength for LogisticRegression
        extra_features : optional dict of additional feature arrays
        """
        X = self._build_feature_matrix(model_preds, prior_diffs, extra_features)
        y = np.asarray(y_true, dtype=float)
        self._model = self._create_model(C=C)
        self._model.fit(X, y)

    def predict(
        self,
        model_preds: dict[str, np.ndarray],
        prior_diffs: np.ndarray,
        extra_features: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return probability array from meta-learner.

        For binary: returns 1D array of P(class=1).
        For multiclass: returns 2D array (n_samples, n_classes).
        """
        if self._model is None:
            raise RuntimeError("StackedEnsemble has not been fitted yet.")
        X = self._build_feature_matrix(model_preds, prior_diffs, extra_features)
        if self._is_regression:
            return self._model.predict(X)
        if self._is_multiclass:
            return self._model.predict_proba(X)
        return self._model.predict_proba(X)[:, 1]

    def get_coefficients(
        self, extra_features: dict[str, np.ndarray] | None = None
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """Return coefficient mapping.

        For binary: returns {feature_name: coefficient}.
        For multiclass: returns {class_i: {feature_name: coefficient}}.
        For GBM: returns feature importances instead of coefficients.
        """
        if self._model is None:
            raise RuntimeError("StackedEnsemble has not been fitted yet.")
        names = self._feature_names(extra_features)

        if self.meta_learner_type == "gbm":
            importances = self._model.feature_importances_
            return {name: float(imp) for name, imp in zip(names, importances)}

        if self._is_multiclass:
            result = {}
            for cls_idx in range(self._model.coef_.shape[0]):
                coeffs = self._model.coef_[cls_idx]
                result[f"class_{cls_idx}"] = {
                    name: float(c) for name, c in zip(names, coeffs)
                }
            return result
        coeffs = self._model.coef_[0]
        return {name: float(c) for name, c in zip(names, coeffs)}

    def save(self, path: Path) -> None:
        """Save meta-learner state to JSON file.

        For GBM meta-learners, uses LightGBM's native model serialization
        via ``model_to_string()`` stored under the ``"lgbm_model_str"`` key.
        For logistic/ridge, saves coefficients and intercepts as before.
        """
        if self._model is None:
            raise RuntimeError("StackedEnsemble has not been fitted yet.")
        path = Path(path)
        state: dict[str, Any] = {
            "model_names": self.model_names,
            "n_classes": self.n_classes,
            "task_type": self.task_type,
            "meta_learner_type": self.meta_learner_type,
        }
        if self.meta_learner_type == "gbm":
            state["lgbm_model_str"] = self._model.booster_.model_to_string()
        else:
            state["coef"] = self._model.coef_.tolist()
            state["intercept"] = self._model.intercept_.tolist()
            if not self._is_regression:
                state["classes"] = self._model.classes_.tolist()
                if self.meta_learner_type == "logistic":
                    state["C"] = self._model.C
        path.write_text(json.dumps(state, indent=2))

    def load(self, path: Path) -> None:
        """Load meta-learner state from JSON file."""
        path = Path(path)
        state = json.loads(path.read_text())
        self.model_names = state["model_names"]
        self.n_classes = state.get("n_classes", 2)
        self.task_type = state.get("task_type", "binary")
        self.meta_learner_type = state.get("meta_learner_type", "logistic")

        if self.meta_learner_type == "gbm":
            import lightgbm as lgb
            booster = lgb.Booster(model_str=state["lgbm_model_str"])
            if self._is_regression:
                from lightgbm import LGBMRegressor
                self._model = LGBMRegressor()
                self._model._Booster = booster
                self._model.fitted_ = True
                self._model._n_features = len(self.model_names) + 1  # +1 for prior_diff
            else:
                from lightgbm import LGBMClassifier
                self._model = LGBMClassifier()
                self._model._Booster = booster
                self._model.fitted_ = True
                self._model._n_features = len(self.model_names) + 1
                self._model._n_classes = self.n_classes
                self._model.classes_ = np.arange(self.n_classes, dtype=float)
        elif self.meta_learner_type == "ridge":
            if self._is_regression:
                self._model = RidgeCV()
            else:
                self._model = _RidgeProbaWrapper(RidgeCV())
            self._model.coef_ = np.array(state["coef"])
            self._model.intercept_ = np.array(state["intercept"])
            if not self._is_regression:
                self._model.classes_ = np.array(state["classes"])
        else:
            # logistic (default)
            if self._is_regression:
                self._model = LinearRegression()
            else:
                self._model = LogisticRegression(
                    C=state["C"], max_iter=1000, solver="lbfgs"
                )
                self._model.classes_ = np.array(state["classes"])
            self._model.coef_ = np.array(state["coef"])
            self._model.intercept_ = np.array(state["intercept"])


def train_meta_learner_loso(
    y_true: np.ndarray,
    model_preds: dict[str, np.ndarray],
    prior_diffs: np.ndarray,
    fold_labels: np.ndarray,
    model_names: list[str],
    ensemble_config: dict,
    extra_features: dict[str, np.ndarray] | None = None,
    n_classes: int = 2,
    task_type: str = "binary",
) -> tuple[StackedEnsemble, Any, dict]:
    """Train stacked meta-learner with LOSO + nested calibrator CV.

    Algorithm:
    1. For each training fold (nested CV):
       a. Hold out that fold
       b. Fit per-model pre-calibration on remaining folds ONLY (binary only)
       c. Pre-calibrate both train and val predictions (binary only)
       d. Train meta-learner on pre-calibrated train predictions
       e. Predict on held-out fold -> OOF meta-learner predictions
    2. Fit post-calibrator on nested OOF predictions (binary only, min 20 samples)
    3. Fit final pre-calibrators on ALL data (binary only)
    4. Fit final meta-learner on ALL pre-calibrated data

    Parameters
    ----------
    y_true : array of labels (binary or integer class labels)
    model_preds : dict mapping model_name -> prediction array
        For binary: 1D arrays. For multiclass: 2D (n_samples, n_classes).
    prior_diffs : prior difference array
    fold_labels : array of fold identifiers (same length as y_true)
    model_names : list of model names to include
    ensemble_config : dict with keys like meta_learner.C, calibration,
        pre_calibration, spline_prob_max, spline_n_bins
    extra_features : optional dict of extra feature arrays
    n_classes : number of classes (default 2 for binary)

    Returns
    -------
    tuple of (meta_learner, post_calibrator, pre_calibrators_dict)
    """
    is_multiclass = n_classes > 2
    is_regression = task_type == "regression"
    skip_calibration = is_multiclass or is_regression

    y_true = np.asarray(y_true, dtype=float)
    prior_diffs = np.asarray(prior_diffs, dtype=float)
    fold_labels = np.asarray(fold_labels)

    meta_config = ensemble_config.get("meta_learner", {})
    C = meta_config.get("C", 1.0)
    meta_learner_type = meta_config.get("type", "logistic")
    cal_method = ensemble_config.get("calibration", "spline")
    pre_cal_config = ensemble_config.get("pre_calibration", {})

    unique_folds = sorted(set(fold_labels))

    # Step 1: Nested LOSO for OOF predictions
    # For multiclass, OOF predictions are 2D; for binary, 1D
    if is_multiclass:
        oof_preds = np.full((len(y_true), n_classes), np.nan)
    else:
        oof_preds = np.full(len(y_true), np.nan)

    for held_out_fold in unique_folds:
        val_mask = fold_labels == held_out_fold
        train_mask = ~val_mask

        n_train = train_mask.sum()
        n_val = val_mask.sum()
        if n_train < 10 or n_val < 1:
            continue

        # Step 1b: Fit per-model pre-calibrators on train fold only
        # Skip pre-calibration for multiclass and regression
        fold_pre_cals = {}
        if not skip_calibration:
            for model_name in model_names:
                if model_name in pre_cal_config:
                    method = pre_cal_config[model_name]
                    cal = build_calibrator(method, ensemble_config)
                    if cal is not None and n_train >= 20:
                        cal.fit(y_true[train_mask], model_preds[model_name][train_mask])
                        fold_pre_cals[model_name] = cal

        # Step 1c: Pre-calibrate train and val predictions
        train_preds = {}
        val_preds = {}
        for model_name in model_names:
            train_vals = model_preds[model_name][train_mask].copy()
            val_vals = model_preds[model_name][val_mask].copy()
            if model_name in fold_pre_cals:
                train_vals = fold_pre_cals[model_name].transform(train_vals)
                val_vals = fold_pre_cals[model_name].transform(val_vals)
            train_preds[model_name] = train_vals
            val_preds[model_name] = val_vals

        # Extra features for this fold
        train_extra = None
        val_extra = None
        if extra_features:
            train_extra = {k: v[train_mask] for k, v in extra_features.items()}
            val_extra = {k: v[val_mask] for k, v in extra_features.items()}

        # Step 1d: Train meta-learner on train fold
        fold_meta = StackedEnsemble(model_names, n_classes=n_classes, task_type=task_type, meta_learner_type=meta_learner_type)
        fold_meta.fit(
            train_preds,
            prior_diffs[train_mask],
            y_true[train_mask],
            C=C,
            extra_features=train_extra,
        )

        # Step 1e: Predict on held-out fold
        oof_preds[val_mask] = fold_meta.predict(
            val_preds,
            prior_diffs[val_mask],
            extra_features=val_extra,
        )

    # Step 2: Fit post-calibrator on nested OOF predictions
    # Skip post-calibration for multiclass and regression
    post_calibrator = None
    if not skip_calibration:
        valid_mask = ~np.isnan(oof_preds)
        if valid_mask.sum() >= 20:
            post_calibrator = build_calibrator(cal_method, ensemble_config)
            if post_calibrator is not None:
                post_calibrator.fit(y_true[valid_mask], oof_preds[valid_mask])

    # Step 3: Fit final pre-calibrators on ALL data
    # Skip pre-calibration for multiclass and regression
    final_pre_cals: dict[str, Any] = {}
    if not skip_calibration:
        for model_name in model_names:
            if model_name in pre_cal_config:
                method = pre_cal_config[model_name]
                cal = build_calibrator(method, ensemble_config)
                if cal is not None and len(y_true) >= 20:
                    cal.fit(y_true, model_preds[model_name])
                    final_pre_cals[model_name] = cal

    # Step 4: Fit final meta-learner on ALL pre-calibrated data
    all_preds_cal = {}
    for model_name in model_names:
        preds = model_preds[model_name].copy()
        if model_name in final_pre_cals:
            preds = final_pre_cals[model_name].transform(preds)
        all_preds_cal[model_name] = preds

    final_meta = StackedEnsemble(model_names, n_classes=n_classes, task_type=task_type, meta_learner_type=meta_learner_type)
    final_meta.fit(
        all_preds_cal,
        prior_diffs,
        y_true,
        C=C,
        extra_features=extra_features,
    )

    return final_meta, post_calibrator, final_pre_cals
