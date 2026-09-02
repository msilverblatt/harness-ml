"""Model registry with auto-discovery from families/ subpackages."""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from harness.ml.models.protocol import Model


class ModelRegistry:
    """Registry that auto-discovers model classes from families/ subpackages."""

    _models: dict[str, Model] = {}
    _loaded: bool = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return

        import harness.ml.models.families as families_pkg

        for _importer, family_name, is_pkg in pkgutil.iter_modules(
            families_pkg.__path__, families_pkg.__name__ + "."
        ):
            if not is_pkg:
                continue
            family_mod = importlib.import_module(family_name)
            for _imp2, mod_name, _is_pkg2 in pkgutil.iter_modules(
                family_mod.__path__, family_mod.__name__ + "."
            ):
                try:
                    mod = importlib.import_module(mod_name)
                except ImportError:
                    continue
                if hasattr(mod, "NAME"):
                    name = mod.NAME
                    # Find the model class — it's the one with a 'name' attribute
                    # matching NAME
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if (
                            isinstance(attr, type)
                            and hasattr(attr, "name")
                            and hasattr(attr, "supports_tasks")
                            and getattr(attr, "name", None) == name
                        ):
                            cls._models[name] = attr()
                            break

        cls._loaded = True

    @classmethod
    def get(cls, name: str) -> Model | None:
        """Get a model instance by name."""
        cls._ensure_loaded()
        return cls._models.get(name)

    @classmethod
    def list_registered(cls) -> list[str]:
        """List every discovered wrapper, including optional backends."""
        cls._ensure_loaded()
        return sorted(cls._models.keys())

    @classmethod
    def list_available(cls) -> list[str]:
        """List models whose required backend packages are installed."""
        cls._ensure_loaded()
        return sorted(
            name
            for name, model in cls._models.items()
            if all(importlib.util.find_spec(package) is not None for package in model.requires_packages)
        )
