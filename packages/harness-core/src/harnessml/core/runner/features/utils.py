"""Feature utilities -- injection, grouping, resolution, and validation."""
from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from harnessml.core.runner.schema import FeatureDecl, InjectionDef, ModelDef

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Injection
# -----------------------------------------------------------------------

def inject_features(
    df: pd.DataFrame,
    injection_def: InjectionDef,
    fold_value: int | None = None,
) -> pd.DataFrame:
    """Merge external features into a DataFrame.

    Loads data from the source (parquet/csv/callable), selects merge_keys + columns,
    left-merges onto *df*, and fills NaN in injected columns with *fill_na*.

    Parameters
    ----------
    df : pd.DataFrame
        Existing DataFrame.
    injection_def : InjectionDef
        Injection definition.
    fold_value : int | None
        Current fold value for ``{fold_value}`` placeholder in *path_pattern*.

    Returns
    -------
    pd.DataFrame
        *df* with injected columns added.
    """
    source_type = injection_def.source_type
    columns = injection_def.columns
    merge_keys = injection_def.merge_keys
    fill_na = injection_def.fill_na

    if source_type in ("parquet", "csv"):
        path_pattern = injection_def.path_pattern or ""
        if fold_value is not None:
            resolved_path = path_pattern.format(fold_value=fold_value)
        else:
            resolved_path = path_pattern
        path = Path(resolved_path)

        if not path.exists():
            logger.warning(
                "Injection source %s does not exist; filling columns %s with %s",
                path,
                columns,
                fill_na,
            )
            result = df.copy()
            for col in columns:
                result[col] = fill_na
            return result

        if source_type == "parquet":
            source_df = pd.read_parquet(path)
        else:
            source_df = pd.read_csv(path)

    elif source_type == "callable":
        mod = importlib.import_module(injection_def.callable_module)  # type: ignore[arg-type]
        func = getattr(mod, injection_def.callable_function)  # type: ignore[arg-type]
        source_df = func(fold_value=fold_value)

    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")

    # Select only the columns we need from the source
    keep_cols = list(merge_keys) + [c for c in columns if c not in merge_keys]
    source_df = source_df[keep_cols]

    # Left merge
    result = df.merge(source_df, on=merge_keys, how="left")

    # Fill NaN in injected columns
    for col in columns:
        if col in result.columns:
            result[col] = result[col].fillna(fill_na)

    return result


# -----------------------------------------------------------------------
# Grouping
# -----------------------------------------------------------------------

def group_features_by_category(
    feature_decls: dict[str, FeatureDecl],
) -> dict[str, list[str]]:
    """Group all declared feature columns by their category.

    Parameters
    ----------
    feature_decls : dict[str, FeatureDecl]
        Mapping of feature name to declaration.

    Returns
    -------
    dict[str, list[str]]
        Mapping of category name to list of column names.
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for decl in feature_decls.values():
        groups[decl.category].extend(decl.columns)
    return dict(groups)


# -----------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------

def resolve_model_features(
    model_def: ModelDef,
    feature_decls: dict[str, FeatureDecl],
) -> list[str]:
    """Resolve a model's features + feature_sets to concrete column names.

    *feature_sets* entries are treated as category names — each is expanded
    to the columns of every :class:`FeatureDecl` whose category matches.

    Returns *model_def.features* + expanded feature_sets, de-duplicated in
    insertion order.

    Raises
    ------
    ValueError
        If a feature_set name doesn't match any category.
    """
    by_category = group_features_by_category(feature_decls)

    seen: set[str] = set()
    result: list[str] = []

    for feat in model_def.features:
        if feat not in seen:
            seen.add(feat)
            result.append(feat)

    for set_name in model_def.feature_sets:
        if set_name not in by_category:
            raise ValueError(
                f"feature_set {set_name!r} does not match any declared "
                f"feature category. Known categories: {sorted(by_category)}"
            )
        for feat in by_category[set_name]:
            if feat not in seen:
                seen.add(feat)
                result.append(feat)

    return result


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

def validate_model_features(
    model_def: ModelDef,
    feature_decls: dict[str, FeatureDecl],
    model_name: str = "",
) -> list[str]:
    """Check that features exist in some FeatureDecl's columns.

    Parameters
    ----------
    model_def : ModelDef
        The model whose features to check.
    feature_decls : dict[str, FeatureDecl]
        All declared features.
    model_name : str
        Optional name used in warning messages.

    Returns
    -------
    list[str]
        Warning strings for undeclared features (empty if all valid).
    """
    all_declared: set[str] = set()
    for decl in feature_decls.values():
        all_declared.update(decl.columns)

    warnings: list[str] = []
    for feat in model_def.features:
        if feat not in all_declared:
            prefix = f"Model {model_name!r}: " if model_name else ""
            warnings.append(f"{prefix}feature {feat!r} is not declared in any FeatureDecl")

    return warnings


def validate_registry_coverage(
    config: Any,  # ProjectConfig
    registry: Any,  # ModelRegistry — supports ``in`` operator
) -> list[str]:
    """Check all model types in *config* are registered.

    Maps ``xgboost_regression`` -> ``xgboost`` before the lookup so that
    regression variants resolve correctly.

    Parameters
    ----------
    config
        A :class:`ProjectConfig` (or duck-type with a ``models`` dict of
        :class:`ModelDef`).
    registry
        An object that supports ``item in registry``.

    Returns
    -------
    list[str]
        Warning strings for unregistered model types (empty if all covered).
    """
    type_aliases: dict[str, str] = {
        "xgboost_regression": "xgboost",
    }

    warnings: list[str] = []
    for name, model_def in config.models.items():
        lookup_type = type_aliases.get(model_def.type, model_def.type)
        if lookup_type not in registry:
            warnings.append(
                f"Model {name!r} uses type {model_def.type!r} which is not "
                f"registered in the model registry"
            )

    return warnings
