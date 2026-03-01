"""Registry mapping model type strings to model classes."""
from __future__ import annotations

from easyml.models.base import BaseModel


class ModelRegistry:
    """Registry mapping model type strings to model classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseModel]] = {}

    def register(self, name: str, cls: type[BaseModel]) -> None:
        self._registry[name] = cls

    def create(self, name: str, params: dict | None = None) -> BaseModel:
        if name not in self._registry:
            raise KeyError(f"Unknown model type: {name!r}")
        return self._registry[name](params=params)

    def create_from_config(self, config) -> BaseModel:
        """Create from a ModelConfig schema."""
        return self.create(config.type, params=config.params)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    @classmethod
    def with_defaults(cls) -> ModelRegistry:
        """Create registry with all built-in models registered."""
        registry = cls()

        # Always available (sklearn)
        from easyml.models.wrappers.logistic import LogisticRegressionModel
        from easyml.models.wrappers.elastic_net import ElasticNetModel

        registry.register("logistic_regression", LogisticRegressionModel)
        registry.register("elastic_net", ElasticNetModel)

        # Optional extras — only register if the backing library is installed
        try:
            from easyml.models.wrappers.random_forest import RandomForestModel

            registry.register("random_forest", RandomForestModel)
        except ImportError:
            pass

        try:
            from easyml.models.wrappers.xgboost import XGBoostModel

            registry.register("xgboost", XGBoostModel)
        except ImportError:
            pass

        try:
            from easyml.models.wrappers.catboost import CatBoostModel

            registry.register("catboost", CatBoostModel)
        except ImportError:
            pass

        try:
            from easyml.models.wrappers.lightgbm import LightGBMModel

            registry.register("lightgbm", LightGBMModel)
        except ImportError:
            pass

        try:
            from easyml.models.wrappers.mlp import MLPModel

            registry.register("mlp", MLPModel)
        except ImportError:
            pass

        try:
            from easyml.models.wrappers.tabnet import TabNetModel

            registry.register("tabnet", TabNetModel)
        except ImportError:
            pass

        return registry
