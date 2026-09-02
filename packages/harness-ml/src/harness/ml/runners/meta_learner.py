from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV

from harness.ml.config.ensemble import EnsembleConfig


@dataclass
class MetaLearnerResult:
    fold_predictions: dict[str, np.ndarray]  # {fold_id: ensemble predictions}
    meta_model: Any = None  # Production meta-learner (fit on all data)
    meta_coefficients: dict[str, float] = field(default_factory=dict)
    model_columns: list[str] = field(default_factory=list)
    method: str = "stacked"


class MetaLearner:
    """Nested LOSO meta-learner for ensemble stacking."""

    def train(
        self,
        fold_predictions: dict[str, pd.DataFrame],
        ensemble_config: EnsembleConfig,
        target_col: str = "target",
        task_type: str = "binary",
    ) -> MetaLearnerResult:
        """Phase 2: For each holdout fold, train meta on others, predict holdout.

        fold_predictions: {fold_id: DataFrame with columns [prob_model1, prob_model2, ..., target]}
        """
        if ensemble_config.method == "average":
            return self._simple_average(fold_predictions, target_col)

        # Stacked ensemble
        fold_ids = sorted(fold_predictions.keys())
        model_columns = self._get_model_columns(fold_predictions, target_col)

        # Filter to include_in_ensemble models (exclude exclude_models)
        active_cols = [
            c
            for c in model_columns
            if not any(
                c == f"prob_{model}" or c.startswith(f"prob_{model}__class_")
                for model in ensemble_config.exclude_models
            )
        ]

        if not active_cols:
            return self._simple_average(fold_predictions, target_col)

        result_preds = {}

        # Nested LOSO: for each holdout fold, train on others
        for holdout_id in fold_ids:
            train_dfs = [fold_predictions[fid] for fid in fold_ids if fid != holdout_id]
            holdout_df = fold_predictions[holdout_id]

            if not train_dfs:
                result_preds[holdout_id] = holdout_df[active_cols].mean(axis=1).values
                continue

            train_data = pd.concat(train_dfs, ignore_index=True)

            X_train = train_data[active_cols].values
            y_train = train_data[target_col].values
            X_holdout = holdout_df[active_cols].values

            # Train meta-learner
            meta = self._create_meta_learner(ensemble_config, task_type)
            try:
                meta.fit(X_train, y_train)
                if hasattr(meta, "predict_proba"):
                    preds = meta.predict_proba(X_holdout)
                    preds = (
                        preds[:, 1] if preds.ndim == 2 and preds.shape[1] == 2 else preds
                    )
                else:
                    preds = meta.predict(X_holdout)
            except Exception:
                # Fallback to simple average
                preds = holdout_df[active_cols].mean(axis=1).values

            result_preds[holdout_id] = preds

        # Fit production meta-learner on all data
        all_data = pd.concat(list(fold_predictions.values()), ignore_index=True)
        X_all = all_data[active_cols].values
        y_all = all_data[target_col].values

        production_meta = self._create_meta_learner(ensemble_config, task_type)
        try:
            production_meta.fit(X_all, y_all)
            coeffs = {}
            if hasattr(production_meta, "coef_"):
                for i, col in enumerate(active_cols):
                    model_name = col.replace("prob_", "")
                    coeffs[model_name] = (
                        float(production_meta.coef_[0][i])
                        if production_meta.coef_.ndim > 1
                        else float(production_meta.coef_[i])
                    )
        except Exception:
            production_meta = None
            coeffs = {}

        return MetaLearnerResult(
            fold_predictions=result_preds,
            meta_model=production_meta,
            meta_coefficients=coeffs,
            model_columns=active_cols,
            method="stacked",
        )

    def _simple_average(self, fold_predictions, target_col):
        result_preds = {}
        model_cols = self._get_model_columns(fold_predictions, target_col)
        class_suffixes = sorted(
            {column.split("__class_", 1)[1] for column in model_cols if "__class_" in column},
            key=int,
        )
        for fold_id, df in fold_predictions.items():
            if class_suffixes:
                class_predictions = []
                for suffix in class_suffixes:
                    columns = [
                        column for column in model_cols if column.endswith(f"__class_{suffix}")
                    ]
                    class_predictions.append(df[columns].mean(axis=1).values)
                result_preds[fold_id] = np.column_stack(class_predictions)
            elif model_cols:
                result_preds[fold_id] = df[model_cols].mean(axis=1).values
            else:
                raise ValueError("No successful model predictions available for ensemble")
        return MetaLearnerResult(
            fold_predictions=result_preds,
            model_columns=model_cols,
            method="average",
        )

    def _get_model_columns(self, fold_predictions, target_col):
        sample = next(iter(fold_predictions.values()))
        return [c for c in sample.columns if c.startswith("prob_") and c != target_col]

    def _create_meta_learner(self, config: EnsembleConfig, task_type: str = "binary"):
        if task_type == "regression":
            return RidgeCV()
        if config.meta_learner_type == "logistic":
            params = {"C": 1.0, "max_iter": 1000, **config.meta_learner_params}
            return LogisticRegression(**params)
        elif config.meta_learner_type == "ridge":
            return RidgeCV()
        else:
            return LogisticRegression(C=1.0, max_iter=1000)
