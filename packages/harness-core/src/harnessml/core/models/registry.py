"""Registry mapping model type strings to model classes."""
from __future__ import annotations

import inspect
import logging

from harnessml.core.models.base import BaseModel

logger = logging.getLogger(__name__)

# Model types that can be auto-installed: {model_type: (pip_package, wrapper_module, class_name)}
_INSTALLABLE_MODELS = {
    "xgboost": ("xgboost", "xgboost", "XGBoostModel"),
    "lightgbm": ("lightgbm", "lightgbm", "LightGBMModel"),
    "catboost": ("catboost", "catboost", "CatBoostModel"),
    "mlp": ("torch", "mlp", "MLPModel"),
    "tabnet": ("pytorch-tabnet", "tabnet", "TabNetModel"),
    "gam": ("pygam", "gam", "GAMModel"),
    "ngboost": ("ngboost", "ngboost", "NGBoostModel"),
}


class ModelRegistry:
    """Registry mapping model type strings to model classes."""

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseModel]] = {}

    def register(self, name: str, cls: type[BaseModel]) -> None:
        self._registry[name] = cls

    def create(self, name: str, params: dict | None = None, **kwargs) -> BaseModel:
        if name not in self._registry:
            # Try auto-installing the missing package
            if name in _INSTALLABLE_MODELS:
                pip_pkg, wrapper_mod, cls_name = _INSTALLABLE_MODELS[name]
                if self._try_install_and_register(name, pip_pkg, wrapper_mod, cls_name):
                    logger.info("Auto-installed %s for model type %r", pip_pkg, name)
                else:
                    raise KeyError(
                        f"Unknown model type: {name!r}. "
                        f"Package `{pip_pkg}` is required but could not be installed."
                    )
            else:
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

    def _try_install_and_register(
        self, name: str, pip_pkg: str, wrapper_mod: str, cls_name: str,
    ) -> bool:
        """Try to pip-install a package and register the model. Returns True on success."""
        import subprocess
        import sys

        # Ensure pip is available (bootstrap via ensurepip if needed)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--default-pip"],
                    capture_output=True, timeout=60, check=True,
                )
        except Exception:
            logger.debug("Failed to bootstrap pip", exc_info=True)
            return False

        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", pip_pkg],
                capture_output=True, timeout=120, check=True,
            )
        except Exception:
            logger.debug("Failed to install %s", pip_pkg, exc_info=True)
            return False

        try:
            import importlib
            mod = importlib.import_module(
                f"harnessml.core.models.wrappers.{wrapper_mod}"
            )
            model_cls = getattr(mod, cls_name)
            self.register(name, model_cls)
            return True
        except Exception:
            logger.debug("Failed to import %s after install", wrapper_mod, exc_info=True)
            return False

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    @classmethod
    def with_defaults(cls) -> ModelRegistry:
        """Create registry with all built-in models registered."""
        registry = cls()

        # Always available (sklearn)
        from harnessml.core.models.wrappers.elastic_net import ElasticNetModel
        from harnessml.core.models.wrappers.logistic import LogisticRegressionModel

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

        # SVM (sklearn — always available)
        try:
            from harnessml.core.models.wrappers.svm import SVMModel

            registry.register("svm", SVMModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "sklearn (svm)")

        # HistGradientBoosting (sklearn — always available)
        try:
            from harnessml.core.models.wrappers.hist_gbm import HistGradientBoostingModel

            registry.register("hist_gradient_boosting", HistGradientBoostingModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "sklearn (hist_gradient_boosting)")

        # GAM (optional — requires pygam)
        try:
            from harnessml.core.models.wrappers.gam import GAMModel

            registry.register("gam", GAMModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "pygam (gam)")

        # NGBoost (optional — requires ngboost)
        try:
            from harnessml.core.models.wrappers.ngboost import NGBoostModel

            registry.register("ngboost", NGBoostModel)
        except ImportError:
            logger.debug("Optional dependency not available: %s", "ngboost")

        return registry
