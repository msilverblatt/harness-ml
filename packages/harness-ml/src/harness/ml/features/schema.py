from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FeatureType(StrEnum):
    ENTITY = "entity"
    PAIRWISE = "pairwise"
    INSTANCE = "instance"
    MODEL_OUTPUT = "model_output"


class FeatureDefinition(BaseModel):
    name: str
    feature_type: FeatureType
    source_column: str | None = None
    formula: str | None = None
    model: str | None = None  # For model_output type
    auto_pairwise: bool = True
    pairwise_methods: list[str] = Field(default_factory=lambda: ["diff", "ratio"])
    active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureSet(BaseModel):
    features: dict[str, FeatureDefinition] = Field(default_factory=dict)

    def active_features(self) -> dict[str, FeatureDefinition]:
        return {k: v for k, v in self.features.items() if v.active}

    def features_by_type(self, feature_type: FeatureType) -> dict[str, FeatureDefinition]:
        return {k: v for k, v in self.features.items() if v.feature_type == feature_type}

    @classmethod
    def from_yaml_dict(cls, data: dict) -> "FeatureSet":
        features = {}
        for name, defn in data.items():
            defn = dict(defn)
            defn["name"] = name
            if "type" in defn:
                defn["feature_type"] = defn.pop("type")
            features[name] = FeatureDefinition(**defn)
        return cls(features=features)
