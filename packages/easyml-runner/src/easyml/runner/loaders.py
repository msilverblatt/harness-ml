"""Dynamic feature and source auto-loaders.

Given YAML declarations (FeatureDecl / SourceDecl), dynamically import
the referenced Python module.function and register it in the
appropriate registry.
"""
from __future__ import annotations

import importlib
from typing import Any

from easyml.data.sources import SourceRegistry
from easyml.features.registry import FeatureRegistry
from easyml.runner.schema import FeatureDecl, SourceDecl


def load_features(
    declarations: dict[str, FeatureDecl],
    registry: FeatureRegistry,
) -> None:
    """Dynamically import and register features from declarations.

    Parameters
    ----------
    declarations:
        Mapping of feature name to :class:`FeatureDecl`.
    registry:
        The :class:`FeatureRegistry` to register features into.

    Raises
    ------
    ImportError
        If the declared module cannot be imported.
    AttributeError
        If the declared function does not exist in the module.
    """
    for name, decl in declarations.items():
        try:
            mod = importlib.import_module(decl.module)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Cannot import module {decl.module!r} for feature {name!r}: {exc}"
            ) from exc

        fn = getattr(mod, decl.function, None)
        if fn is None:
            raise AttributeError(
                f"Module {decl.module!r} has no function {decl.function!r} "
                f"(required by feature {name!r})"
            )

        registry.register(
            name=name,
            category=decl.category,
            level=decl.level,
            output_columns=decl.columns,
            nan_strategy=decl.nan_strategy,
        )(fn)


def load_sources(
    declarations: dict[str, SourceDecl],
    registry: SourceRegistry,
) -> None:
    """Dynamically import and register data sources from declarations.

    Parameters
    ----------
    declarations:
        Mapping of source name to :class:`SourceDecl`.
    registry:
        The :class:`SourceRegistry` to register sources into.

    Raises
    ------
    ImportError
        If the declared module cannot be imported.
    AttributeError
        If the declared function does not exist in the module.
    """
    for name, decl in declarations.items():
        try:
            mod = importlib.import_module(decl.module)
        except ModuleNotFoundError as exc:
            raise ImportError(
                f"Cannot import module {decl.module!r} for source {name!r}: {exc}"
            ) from exc

        fn = getattr(mod, decl.function, None)
        if fn is None:
            raise AttributeError(
                f"Module {decl.module!r} has no function {decl.function!r} "
                f"(required by source {name!r})"
            )

        registry.register(
            name=name,
            category=decl.category,
            outputs=decl.outputs,
            temporal_safety=decl.temporal_safety,
            leakage_notes=decl.leakage_notes,
        )(fn)
