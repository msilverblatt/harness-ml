from typing import Any

from pydantic import BaseModel, Field, model_validator

_DEFAULT_METRICS = {
    "binary": ["brier", "accuracy"],
    "multiclass": ["log_loss", "accuracy"],
    "regression": ["rmse", "mae", "r2"],
}


class CVConfig(BaseModel):
    strategy: str = "kfold"
    n_folds: int = 5
    fold_column: str | None = None
    fold_values: list | None = None
    min_train_folds: int = 2


class ProjectConfig(BaseModel):
    task_type: str = "binary"
    target_column: str = "target"
    cv: CVConfig = Field(default_factory=CVConfig)
    metrics: list[str] = Field(default_factory=lambda: ["brier", "accuracy"])
    eval_filter: str | None = None
    exclude_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def apply_task_specific_metric_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict) and "metrics" not in value:
            task_type = value.get("task_type", "binary")
            if task_type in _DEFAULT_METRICS:
                value = {**value, "metrics": list(_DEFAULT_METRICS[task_type])}
        return value
