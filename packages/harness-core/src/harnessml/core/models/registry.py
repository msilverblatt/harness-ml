"""Registry mapping model type strings to model classes."""
from __future__ import annotations

import inspect
import logging

from harnessml.core.models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry mapping model type strings to model classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseModel]] = {}

    def register(self, name: str, cls: type[BaseModel]) -> None:
        self._registry[name] = cls

    def create(self, name: str, params: dict | None = None, **kwargs) -> BaseModel:
        if name not in self._registry:
            raise KeyError(f"Unknown model type: {name!r}")
        model_cls = self._registry[name]
        if kwargs:
            sig = inspect.signature(model_cls)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if has_var_keyword:
                forwarded = kwargs
            else:
                accepted = set(sig.parameters.keys())
                forwarded = {k: v for k, v in kwargs.items() if k in accepted}
                dropped = set(kwargs) - set(forwarded)
                if dropped:
                    logger.debug(
                        "ModelRegistry.create(%r): dropping kwargs not accepted "
                        "by constructor: %s",
                        name,
                        dropped,
                    )
            return model_cls(params=params, **forwarded)
        return model_cls(params=params)

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
        from harnessml.core.models.wrappers.logistic import LogisticRegressionModel
        from harnessml.core.models.wrappers.elastic_net import ElasticNetModel

        registry.register("logistic_regression", LogisticRegressionModel)
        registry.register("elastic_net", ElasticNetModel)

        # Optional extras — only register if the backing library is installed
        try:
            from harnessml.core.models.wrappers.random_forest import RandomForestModel

            registry.register("random_forest", RandomForestModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "sklearn (random_forest)")

        try:
            from harnessml.core.models.wrappers.xgboost import XGBoostModel

            registry.register("xgboost", XGBoostModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "xgboost")

        try:
            from harnessml.core.models.wrappers.catboost import CatBoostModel

            registry.register("catboost", CatBoostModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "catboost")

        try:
            from harnessml.core.models.wrappers.lightgbm import LightGBMModel

            registry.register("lightgbm", LightGBMModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "lightgbm")

        try:
            from harnessml.core.models.wrappers.mlp import MLPModel

            registry.register("mlp", MLPModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "torch (mlp)")

        try:
            from harnessml.core.models.wrappers.tabnet import TabNetModel

            registry.register("tabnet", TabNetModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "pytorch-tabnet (tabnet)")

        return registry
