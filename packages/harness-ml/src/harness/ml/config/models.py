from pydantic import BaseModel, Field
from typing import Any


class SingleModelConfig(BaseModel):
    name: str
    model_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    active: bool = True
    include_in_ensemble: bool = True
    n_seeds: int = 1
    depends_on: list[str] = Field(default_factory=list)
    provides: str | None = None
    provides_level: str = "instance"
    training_filter: str | None = None
    zero_fill_features: list[str] = Field(default_factory=list)
    class_weight: str | dict | None = None
    augment_symmetry: bool = False


class ModelsConfig(BaseModel):
    models: dict[str, SingleModelConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml_dict(cls, data: dict) -> "ModelsConfig":
        models = {}
        for name, defn in data.items():
            defn = dict(defn)
            defn["name"] = name
            models[name] = SingleModelConfig(**defn)
        return cls(models=models)
