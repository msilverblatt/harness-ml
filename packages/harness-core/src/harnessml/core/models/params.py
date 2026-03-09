"""Per-model Pydantic parameter schemas with validation.

Each schema captures the common hyperparameters for a model type with
appropriate constraints. All schemas use ``model_config = ConfigDict(extra="allow")``
so that unknown parameters are forwarded to the underlying library.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
class XGBoostParams(BaseModel):
    """Validated hyperparameters for XGBoost."""

    model_config = ConfigDict(extra="allow")

    n_estimators: int = Field(100, ge=1)
    max_depth: int = Field(6, ge=1, le=50)
    learning_rate: float = Field(0.1, gt=0, le=1)
    subsample: float = Field(1.0, gt=0, le=1)
    colsample_bytree: float = Field(1.0, gt=0, le=1)
    reg_alpha: float = Field(0.0, ge=0)
    reg_lambda: float = Field(1.0, ge=0)
    gamma: float = Field(0.0, ge=0)
    min_child_weight: float = Field(1.0, ge=0)
    scale_pos_weight: float = Field(1.0, gt=0)
    early_stopping_rounds: int | None = Field(None, ge=1)


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
class LightGBMParams(BaseModel):
    """Validated hyperparameters for LightGBM."""

    model_config = ConfigDict(extra="allow")

    n_estimators: int = Field(100, ge=1)
    max_depth: int = Field(-1)
    learning_rate: float = Field(0.1, gt=0, le=1)
    num_leaves: int = Field(31, ge=2)
    subsample: float = Field(1.0, gt=0, le=1)
    colsample_bytree: float = Field(1.0, gt=0, le=1)
    reg_alpha: float = Field(0.0, ge=0)
    reg_lambda: float = Field(0.0, ge=0)
    min_child_weight: float = Field(1e-3, ge=0)
    early_stopping_rounds: int | None = Field(None, ge=1)
    verbosity: int = Field(-1)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------
class CatBoostParams(BaseModel):
    """Validated hyperparameters for CatBoost."""

    model_config = ConfigDict(extra="allow")

    iterations: int = Field(1000, ge=1)
    depth: int = Field(6, ge=1, le=16)
    learning_rate: float = Field(0.03, gt=0, le=1)
    l2_leaf_reg: float = Field(3.0, ge=0)
    subsample: float | None = Field(None, gt=0, le=1)
    early_stopping_rounds: int | None = Field(None, ge=1)
    verbose: int = Field(0, ge=0)
    allow_writing_files: bool = Field(False)


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
class RandomForestParams(BaseModel):
    """Validated hyperparameters for sklearn RandomForest."""

    model_config = ConfigDict(extra="allow")

    n_estimators: int = Field(100, ge=1)
    max_depth: int | None = Field(None, ge=1)
    min_samples_split: int | float = Field(2, ge=1)
    min_samples_leaf: int | float = Field(1, ge=1)
    max_features: str | float | int | None = Field("sqrt")
    n_jobs: int | None = Field(None)
    random_state: int | None = Field(None)


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------
class LogisticParams(BaseModel):
    """Validated hyperparameters for sklearn LogisticRegression."""

    model_config = ConfigDict(extra="allow")

    C: float = Field(1.0, gt=0)
    penalty: str | None = Field("l2")
    solver: str = Field("lbfgs")
    max_iter: int = Field(100, ge=1)
    l1_ratio: float | None = Field(None, ge=0, le=1)
    random_state: int | None = Field(None)


# ---------------------------------------------------------------------------
# Elastic Net
# ---------------------------------------------------------------------------
class ElasticNetParams(BaseModel):
    """Validated hyperparameters for ElasticNet (classifier or regressor)."""

    model_config = ConfigDict(extra="allow")

    C: float = Field(1.0, gt=0)
    l1_ratio: float = Field(0.5, ge=0, le=1)
    penalty: str = Field("elasticnet")
    solver: str = Field("saga")
    max_iter: int = Field(1000, ge=1)
    alpha: float = Field(1.0, gt=0)
    normalize: bool = Field(False)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class MLPParams(BaseModel):
    """Validated hyperparameters for the PyTorch MLP wrapper."""

    model_config = ConfigDict(extra="allow")

    hidden_layers: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = Field(0.0, ge=0, le=1)
    learning_rate: float = Field(0.001, gt=0)
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(32, ge=1)
    n_seeds: int = Field(1, ge=1)
    normalize: bool = Field(False)
    batch_norm: bool = Field(False)
    weight_decay: float = Field(0.0, ge=0)
    early_stopping_rounds: int | None = Field(None, ge=1)
    seed_stride: int = Field(1, ge=1)
    seed: int = Field(0, ge=0)

    @field_validator("hidden_layers")
    @classmethod
    def _hidden_layers_non_empty(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("hidden_layers must not be empty")
        if any(h < 1 for h in v):
            raise ValueError("all hidden layer sizes must be >= 1")
        return v


# ---------------------------------------------------------------------------
# TabNet
# ---------------------------------------------------------------------------
class TabNetParams(BaseModel):
    """Validated hyperparameters for the TabNet wrapper."""

    model_config = ConfigDict(extra="allow")

    n_d: int = Field(8, ge=1)
    n_a: int = Field(8, ge=1)
    n_steps: int = Field(3, ge=1)
    relaxation_factor: float = Field(1.3, gt=0)
    max_epochs: int = Field(200, ge=1)
    patience: int = Field(15, ge=1)
    batch_size: int = Field(1024, ge=1)
    virtual_batch_size: int = Field(128, ge=1)
    n_seeds: int = Field(1, ge=1)
    normalize: bool = Field(False)
    val_fraction: float | None = Field(None, gt=0, lt=1)
    learning_rate: float | None = Field(None, gt=0)
    scheduler_step_size: int | None = Field(None, ge=1)
    scheduler_gamma: float | None = Field(None, gt=0, lt=1)
    seed_stride: int = Field(1, ge=1)
    seed: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_PARAMS_REGISTRY: dict[str, type[BaseModel]] = {
    "xgboost": XGBoostParams,
    "lightgbm": LightGBMParams,
    "catboost": CatBoostParams,
    "random_forest": RandomForestParams,
    "logistic_regression": LogisticParams,
    "elastic_net": ElasticNetParams,
    "mlp": MLPParams,
    "tabnet": TabNetParams,
}


def get_params_schema(model_type: str) -> type[BaseModel] | None:
    """Return the Pydantic params schema for *model_type*, or ``None``."""
    return _PARAMS_REGISTRY.get(model_type)
