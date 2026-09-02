import pandas as pd

from harness.data.expressions.engine import ExpressionEngine
from harness.ml.features.pairwise import generate_pairwise_derivatives
from harness.ml.features.schema import FeatureDefinition, FeatureSet, FeatureType


class FeatureResolver:
    def __init__(self):
        self._expr_engine = ExpressionEngine()
        self._resolved_names: list[str] = []

    @property
    def resolved_feature_names(self) -> list[str]:
        return list(self._resolved_names)

    def resolve(self, df: pd.DataFrame, feature_set: FeatureSet) -> pd.DataFrame:
        result = df.copy()
        self._resolved_names = []

        for name, defn in feature_set.active_features().items():
            if defn.feature_type == FeatureType.INSTANCE:
                self._resolve_instance(result, defn)
            elif defn.feature_type == FeatureType.ENTITY:
                self._resolve_entity(result, defn)
            elif defn.feature_type == FeatureType.PAIRWISE:
                self._resolve_pairwise(result, defn)
            elif defn.feature_type == FeatureType.MODEL_OUTPUT:
                pass  # Handled by ProviderContext

        return result

    def _resolve_instance(self, df: pd.DataFrame, defn: FeatureDefinition) -> None:
        col = defn.source_column or defn.name
        if col not in df.columns:
            raise ValueError(f"Instance feature '{defn.name}': column '{col}' not found")
        if col != defn.name:
            df[defn.name] = df[col]
        self._resolved_names.append(defn.name)

    def _resolve_entity(self, df: pd.DataFrame, defn: FeatureDefinition) -> None:
        if defn.auto_pairwise:
            feature_name = defn.source_column or defn.name
            updated = generate_pairwise_derivatives(df, feature_name, defn.pairwise_methods)
            for method in defn.pairwise_methods:
                col_name = f"{method}_{feature_name}"
                df[col_name] = updated[col_name]
                self._resolved_names.append(col_name)

    def _resolve_pairwise(self, df: pd.DataFrame, defn: FeatureDefinition) -> None:
        if defn.formula:
            df[defn.name] = self._expr_engine.evaluate(df, defn.formula)
        elif defn.source_column and defn.source_column in df.columns:
            if defn.source_column != defn.name:
                df[defn.name] = df[defn.source_column]
        else:
            raise ValueError(f"Pairwise feature '{defn.name}': needs formula or source_column")
        self._resolved_names.append(defn.name)
