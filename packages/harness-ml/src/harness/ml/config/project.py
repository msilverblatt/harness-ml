from pydantic import BaseModel, Field


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
