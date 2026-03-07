"""Tests for harnessml.runner.feature_utils."""
from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest
from harnessml.core.runner.feature_utils import (
    group_features_by_category,
    inject_features,
    resolve_model_features,
    validate_model_features,
    validate_registry_coverage,
)
from harnessml.core.runner.schema import FeatureDecl, InjectionDef, ModelDef

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

class MockRegistry:
    """Minimal registry supporting ``in`` operator."""

    def __init__(self, types: set[str]):
        self._types = set(types)

    def __contains__(self, item: str) -> bool:
        return item in self._types


def _make_feature_decls(**kwargs: tuple[str, str, list[str]]) -> dict[str, FeatureDecl]:
    """Shorthand: name -> (module, category, columns)."""
    out: dict[str, FeatureDecl] = {}
    for name, (module, category, columns) in kwargs.items():
        out[name] = FeatureDecl(
            module=module,
            function="compute",
            category=category,
            level="entity",
            columns=columns,
        )
    return out


# -----------------------------------------------------------------------
# inject_features
# -----------------------------------------------------------------------

class TestInjectFromParquet:
    def test_inject_from_parquet(self, tmp_path: Path) -> None:
        # Source parquet with merge key + extra column
        source = pd.DataFrame({"team_id": [1, 2, 3], "travel_dist": [100.0, 200.0, 300.0]})
        parquet_path = tmp_path / "travel.parquet"
        source.to_parquet(parquet_path)

        df = pd.DataFrame({"team_id": [1, 2, 4], "seed": [1, 2, 3]})

        inj = InjectionDef(
            source_type="parquet",
            path_pattern=str(parquet_path),
            merge_keys=["team_id"],
            columns=["travel_dist"],
            fill_na=-1.0,
        )
        result = inject_features(df, inj)

        assert "travel_dist" in result.columns
        assert result.loc[result["team_id"] == 1, "travel_dist"].iloc[0] == 100.0
        assert result.loc[result["team_id"] == 2, "travel_dist"].iloc[0] == 200.0
        # team_id=4 not in source -> fill_na
        assert result.loc[result["team_id"] == 4, "travel_dist"].iloc[0] == -1.0


class TestInjectMissingFile:
    def test_inject_missing_file_fills_na(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"team_id": [1, 2], "seed": [1, 2]})
        inj = InjectionDef(
            source_type="parquet",
            path_pattern=str(tmp_path / "nonexistent.parquet"),
            merge_keys=["team_id"],
            columns=["extra_col"],
            fill_na=0.0,
        )
        result = inject_features(df, inj)
        assert "extra_col" in result.columns
        assert (result["extra_col"] == 0.0).all()


class TestInjectFoldPlaceholder:
    def test_inject_with_fold_placeholder(self, tmp_path: Path) -> None:
        source = pd.DataFrame({"team_id": [1], "stat": [42.0]})
        parquet_path = tmp_path / "data_2024.parquet"
        source.to_parquet(parquet_path)

        df = pd.DataFrame({"team_id": [1]})
        pattern = str(tmp_path / "data_{fold_value}.parquet")
        inj = InjectionDef(
            source_type="parquet",
            path_pattern=pattern,
            merge_keys=["team_id"],
            columns=["stat"],
            fill_na=0.0,
        )
        result = inject_features(df, inj, fold_value=2024)
        assert result.loc[0, "stat"] == 42.0


class TestInjectFromCallable:
    def test_inject_from_callable(self) -> None:
        # Create a temporary module with a callable
        mod = types.ModuleType("_test_inject_mod")
        mod.get_data = lambda fold_value=None: pd.DataFrame(  # type: ignore[attr-defined]
            {"key": [10, 20], "val": [1.5, 2.5]}
        )
        import sys

        sys.modules["_test_inject_mod"] = mod
        try:
            df = pd.DataFrame({"key": [10, 20, 30]})
            inj = InjectionDef(
                source_type="callable",
                merge_keys=["key"],
                columns=["val"],
                fill_na=-1.0,
                callable_module="_test_inject_mod",
                callable_function="get_data",
            )
            result = inject_features(df, inj, fold_value=2025)
            assert result.loc[result["key"] == 10, "val"].iloc[0] == 1.5
            assert result.loc[result["key"] == 30, "val"].iloc[0] == -1.0
        finally:
            del sys.modules["_test_inject_mod"]


# -----------------------------------------------------------------------
# group_features_by_category
# -----------------------------------------------------------------------

class TestGroupSingleCategory:
    def test_group_single_category(self) -> None:
        decls = _make_feature_decls(
            feat_a=("m", "efficiency", ["adj_oe", "adj_de"]),
            feat_b=("m", "efficiency", ["adj_net"]),
        )
        groups = group_features_by_category(decls)
        assert set(groups.keys()) == {"efficiency"}
        assert set(groups["efficiency"]) == {"adj_oe", "adj_de", "adj_net"}


class TestGroupMultipleCategories:
    def test_group_multiple_categories(self) -> None:
        decls = _make_feature_decls(
            feat_a=("m", "efficiency", ["adj_oe"]),
            feat_b=("m", "resume", ["wins", "losses"]),
            feat_c=("m", "efficiency", ["adj_de"]),
        )
        groups = group_features_by_category(decls)
        assert set(groups.keys()) == {"efficiency", "resume"}
        assert set(groups["efficiency"]) == {"adj_oe", "adj_de"}
        assert set(groups["resume"]) == {"wins", "losses"}


# -----------------------------------------------------------------------
# resolve_model_features
# -----------------------------------------------------------------------

class TestResolveFeaturesOnly:
    def test_resolve_features_only(self) -> None:
        model = ModelDef(type="xgboost", features=["a", "b", "c"])
        decls = _make_feature_decls(x=("m", "cat", ["a", "b", "c"]))
        result = resolve_model_features(model, decls)
        assert result == ["a", "b", "c"]


class TestResolveFeatureSets:
    def test_resolve_feature_sets(self) -> None:
        model = ModelDef(type="xgboost", feature_sets=["efficiency"])
        decls = _make_feature_decls(
            eff=("m", "efficiency", ["adj_oe", "adj_de"]),
        )
        result = resolve_model_features(model, decls)
        assert result == ["adj_oe", "adj_de"]


class TestResolveBoth:
    def test_resolve_both(self) -> None:
        model = ModelDef(type="xgboost", features=["seed", "adj_oe"], feature_sets=["efficiency"])
        decls = _make_feature_decls(
            eff=("m", "efficiency", ["adj_oe", "adj_de"]),
        )
        result = resolve_model_features(model, decls)
        # seed + adj_oe from features, then adj_de from set (adj_oe deduped)
        assert result == ["seed", "adj_oe", "adj_de"]


class TestResolveUnknownSetRaises:
    def test_resolve_unknown_set_raises(self) -> None:
        model = ModelDef(type="xgboost", feature_sets=["nonexistent"])
        decls = _make_feature_decls(eff=("m", "efficiency", ["adj_oe"]))
        with pytest.raises(ValueError, match="nonexistent"):
            resolve_model_features(model, decls)


# -----------------------------------------------------------------------
# validate_model_features
# -----------------------------------------------------------------------

class TestAllValidEmptyWarnings:
    def test_all_valid_empty_warnings(self) -> None:
        model = ModelDef(type="xgboost", features=["adj_oe", "adj_de"])
        decls = _make_feature_decls(eff=("m", "efficiency", ["adj_oe", "adj_de", "adj_net"]))
        warnings = validate_model_features(model, decls, model_name="xgb_core")
        assert warnings == []


class TestUndeclaredReturnsWarning:
    def test_undeclared_returns_warning(self) -> None:
        model = ModelDef(type="xgboost", features=["adj_oe", "mystery_col"])
        decls = _make_feature_decls(eff=("m", "efficiency", ["adj_oe"]))
        warnings = validate_model_features(model, decls, model_name="xgb_core")
        assert len(warnings) == 1
        assert "mystery_col" in warnings[0]
        assert "xgb_core" in warnings[0]


# -----------------------------------------------------------------------
# validate_registry_coverage
# -----------------------------------------------------------------------

class TestAllTypesRegistered:
    def test_all_types_registered(self) -> None:
        from harnessml.core.runner.schema import (
            BacktestConfig,
            DataConfig,
            EnsembleDef,
            ProjectConfig,
        )

        config = ProjectConfig(
            data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
            models={
                "m1": ModelDef(type="xgboost", features=["a"]),
                "m2": ModelDef(type="catboost", features=["b"]),
            },
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(cv_strategy="leave_one_out", fold_column="season"),
        )
        registry = MockRegistry({"xgboost", "catboost"})
        warnings = validate_registry_coverage(config, registry)
        assert warnings == []


class TestMissingTypeWarns:
    def test_missing_type_warns(self) -> None:
        from harnessml.core.runner.schema import (
            BacktestConfig,
            DataConfig,
            EnsembleDef,
            ProjectConfig,
        )

        config = ProjectConfig(
            data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
            models={
                "m1": ModelDef(type="xgboost", features=["a"]),
            },
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(cv_strategy="leave_one_out", fold_column="season"),
        )
        registry = MockRegistry({"catboost"})  # xgboost NOT registered
        warnings = validate_registry_coverage(config, registry)
        assert len(warnings) == 1
        assert "xgboost" in warnings[0]
        assert "m1" in warnings[0]


class TestXgboostRegressionResolves:
    def test_xgboost_regression_resolves(self) -> None:
        from harnessml.core.runner.schema import (
            BacktestConfig,
            DataConfig,
            EnsembleDef,
            ProjectConfig,
        )

        config = ProjectConfig(
            data=DataConfig(raw_dir="r", processed_dir="p", features_dir="f"),
            models={
                "spread": ModelDef(type="xgboost_regression", features=["a"]),
            },
            ensemble=EnsembleDef(method="stacked"),
            backtest=BacktestConfig(cv_strategy="leave_one_out", fold_column="season"),
        )
        # Only "xgboost" in registry — xgboost_regression should map to it
        registry = MockRegistry({"xgboost"})
        warnings = validate_registry_coverage(config, registry)
        assert warnings == []
