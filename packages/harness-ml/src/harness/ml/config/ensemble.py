from pydantic import BaseModel, Field


class EnsembleConfig(BaseModel):
    method: str = "stacked"
    meta_learner_type: str = "logistic"
    meta_learner_params: dict = Field(default_factory=dict)
    exclude_models: list[str] = Field(default_factory=list)
    calibration: str = "none"
    pre_calibration: dict[str, str] = Field(default_factory=dict)
    temperature: float = 1.0
    clip_floor: float | None = None
    meta_features: list[str] = Field(default_factory=list)
    prior_feature: str | None = None
    conformal_alpha: float | None = Field(default=None, gt=0, lt=1)
