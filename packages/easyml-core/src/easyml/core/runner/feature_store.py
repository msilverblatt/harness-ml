"""Declarative feature store with support for entity, pairwise, instance, and regime features.

Manages the lifecycle of features from entity-level data sources through to
pairwise derivatives. Handles auto-pairwise generation (diff, ratio, both),
formula evaluation, regime flag creation, and caching via the FeatureCache.

Feature types:
- entity: Per-entity per-period metric loaded from a source. Auto-generates
  pairwise derivatives (diff, ratio) on the instance DataFrame.
- pairwise: Per-instance feature computed from a formula against instance data.
- instance: Per-instance context property (column or formula).
- regime: Temporal/contextual boolean flag (0/1) from a condition expression.

Typical usage via the MCP tool layer or Project API::

    store = FeatureStore(project_dir, data_config)
    result = store.add(FeatureDef(name="adj_em", type="entity", source="kenpom"))
    df = store.resolve(["diff_adj_em", "ratio_adj_em"])
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from easyml.core.runner.feature_cache import FeatureCache
from easyml.core.runner.feature_engine import (
    FeatureResult,
    _check_redundancy,
    _compute_feature_stats,
    _resolve_formula,
)
from easyml.core.runner.schema import (
    DataConfig,
    FeatureDef,
    FeatureStoreConfig,
    FeatureType,
    PairwiseMode,
    SourceConfig,
)

logger = logging.getLogger(__name__)

# Small constant to avoid division by zero in ratio computation
_EPSILON = 1e-9


class FeatureStore:
    """Declarative feature store with caching and auto-pairwise generation.

    Parameters
    ----------
    project_dir : str | Path
        Root project directory.
    config : DataConfig
        Data configuration containing feature_store settings, feature_defs,
        and source definitions.
    """

    def __init__(self, project_dir: str | Path, config: DataConfig) -> None:
        self.project_dir = Path(project_dir)
        self.config = config
        self.store_config: FeatureStoreConfig = config.feature_store

        # Registry of feature definitions keyed by name
        self._registry: dict[str, FeatureDef] = dict(config.feature_defs)

        # Cache for computed feature Series
        cache_dir = Path(self.store_config.cache_dir)
        if not cache_dir.is_absolute():
            cache_dir = self.project_dir / cache_dir
        self._cache = FeatureCache(cache_dir)

        # Lazy-loaded DataFrames
        self._matchup_df: pd.DataFrame | None = None
        self._source_dfs: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_matchup_data(self) -> pd.DataFrame:
        """Load and cache the matchup-level features DataFrame."""
        if self._matchup_df is not None:
            return self._matchup_df

        from easyml.core.runner.data_utils import get_features_df

        self._matchup_df = get_features_df(self.project_dir, self.config)
        return self._matchup_df

    def _load_source(self, source_name: str) -> pd.DataFrame:
        """Load entity-level data from a declared source.

        Resolves the source from config.sources, auto-detects file format,
        and caches the loaded DataFrame.

        Raises ValueError if the source is not declared or its file is missing.
        """
        if source_name in self._source_dfs:
            return self._source_dfs[source_name]

        source_cfg = self.config.sources.get(source_name)
        if source_cfg is None:
            raise ValueError(
                f"Source '{source_name}' not found in config.sources. "
                f"Available sources: {sorted(self.config.sources.keys())}"
            )

        path = self._resolve_source_path(source_cfg)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        fmt = source_cfg.format
        if fmt == "auto":
            fmt = self._detect_format(path)

        if fmt == "parquet":
            df = pd.read_parquet(path)
        elif fmt == "csv":
            df = pd.read_csv(path)
        elif fmt == "excel":
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported format '{fmt}' for source '{source_name}'")

        self._source_dfs[source_name] = df
        return df

    def _resolve_source_path(self, source_cfg: SourceConfig) -> Path:
        """Resolve source path, handling relative paths."""
        if source_cfg.path is None:
            raise ValueError(
                f"Source '{source_cfg.name}' has no path configured."
            )
        path = Path(source_cfg.path)
        if not path.is_absolute():
            path = self.project_dir / path
        return path

    @staticmethod
    def _detect_format(path: Path) -> str:
        """Auto-detect file format from extension."""
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return "parquet"
        elif suffix == ".csv":
            return "csv"
        elif suffix in (".xlsx", ".xls"):
            return "excel"
        else:
            raise ValueError(
                f"Cannot auto-detect format for '{path.name}'. "
                f"Set format explicitly in source config."
            )

    # ------------------------------------------------------------------
    # Public API: add features
    # ------------------------------------------------------------------

    def add(self, feature: FeatureDef) -> FeatureResult:
        """Add a feature to the store.

        Dispatches to the appropriate handler based on feature type,
        registers the feature definition, and returns a result summary.
        """
        handlers = {
            FeatureType.ENTITY: self._add_team_feature,
            FeatureType.PAIRWISE: self._add_pairwise_feature,
            FeatureType.INSTANCE: self._add_matchup_feature,
            FeatureType.REGIME: self._add_regime_feature,
        }

        handler = handlers.get(feature.type)
        if handler is None:
            raise ValueError(f"Unknown feature type: {feature.type}")

        self._registry[feature.name] = feature
        return handler(feature)

    # ------------------------------------------------------------------
    # Team features
    # ------------------------------------------------------------------

    def _add_team_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a team (entity-level) feature with optional auto-pairwise."""
        if feature.source is None:
            raise ValueError(
                f"Team feature '{feature.name}' requires a 'source' field."
            )

        source_df = self._load_source(feature.source)
        entity_col = feature.column or feature.name

        if entity_col not in source_df.columns:
            raise ValueError(
                f"Column '{entity_col}' not found in source '{feature.source}'. "
                f"Available: {sorted(source_df.columns.tolist())}"
            )

        # Cache entity-level data
        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="entity",
            source=feature.source,
            column=entity_col,
        )
        entity_series = source_df[entity_col]
        self._cache.put(
            feature.name,
            cache_key,
            entity_series,
            feature_type="entity",
            source=feature.source,
        )

        # Determine effective pairwise mode
        pairwise_mode = feature.pairwise_mode
        auto_pairwise = self.store_config.auto_pairwise

        pairwise_names: list[str] = []
        if auto_pairwise and pairwise_mode != PairwiseMode.NONE:
            matchup_df = self._load_matchup_data()
            pairwise_names = self._generate_pairwise(
                feature, source_df, matchup_df, entity_col
            )

        # Compute correlation of the first pairwise derivative (if any)
        correlation = 0.0
        if pairwise_names:
            first_pw = pairwise_names[0]
            pw_key = self._cache.compute_key(
                name=first_pw,
                feature_type="pairwise",
                source=feature.source,
                column=entity_col,
                pairwise_mode=pairwise_mode.value,
            )
            pw_series = self._cache.get(first_pw, pw_key)
            if pw_series is not None:
                matchup_df = self._load_matchup_data()
                target = self.config.target_column
                if target in matchup_df.columns and len(pw_series) == len(matchup_df):
                    try:
                        correlation = float(
                            pw_series.corr(matchup_df[target].astype(float))
                        )
                        if np.isnan(correlation):
                            correlation = 0.0
                    except (TypeError, ValueError):
                        pass

        stats = _compute_feature_stats(entity_series)

        return FeatureResult(
            name=feature.name,
            column_added=pairwise_names[0] if pairwise_names else feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=float(entity_series.isna().mean()),
            stats=stats,
            description=feature.description or f"Team feature from {feature.source}.{entity_col}",
        )

    def _generate_pairwise(
        self,
        feature: FeatureDef,
        source_df: pd.DataFrame,
        matchup_df: pd.DataFrame,
        entity_col: str,
    ) -> list[str]:
        """Generate pairwise derivatives from entity-level data.

        Merges entity data onto matchup rows for both entity_a and entity_b,
        then computes diff and/or ratio based on the pairwise_mode setting.

        Returns list of generated pairwise feature names.
        """
        sc = self.store_config
        period_col = sc.period_column
        entity_id_col = sc.entity_column
        entity_a_col = sc.entity_a_column
        entity_b_col = sc.entity_b_column

        # Build entity lookup: entity_id + period_id -> value
        merge_cols = [entity_id_col, period_col]
        available_merge_cols = [c for c in merge_cols if c in source_df.columns]
        if not available_merge_cols:
            raise ValueError(
                f"Source '{feature.source}' is missing entity/period columns. "
                f"Expected '{entity_id_col}' and/or '{period_col}' "
                f"but found: {sorted(source_df.columns.tolist())}"
            )

        entity_data = source_df[available_merge_cols + [entity_col]].copy()

        # Merge for entity_a
        rename_a = {entity_id_col: entity_a_col, entity_col: f"_a_{entity_col}"}
        merge_a = entity_data.rename(columns=rename_a)
        merge_keys_a = [entity_a_col]
        if period_col in merge_a.columns:
            merge_keys_a.append(period_col)
        merged = matchup_df.merge(merge_a, on=merge_keys_a, how="left")

        # Merge for entity_b
        rename_b = {entity_id_col: entity_b_col, entity_col: f"_b_{entity_col}"}
        merge_b = entity_data.rename(columns=rename_b)
        merge_keys_b = [entity_b_col]
        if period_col in merge_b.columns:
            merge_keys_b.append(period_col)
        merged = merged.merge(merge_b, on=merge_keys_b, how="left")

        val_a = merged[f"_a_{entity_col}"]
        val_b = merged[f"_b_{entity_col}"]

        mode = feature.pairwise_mode
        pairwise_names: list[str] = []

        # Generate diff
        if mode in (PairwiseMode.DIFF, PairwiseMode.BOTH):
            diff_name = f"diff_{feature.name}"
            diff_series = val_a - val_b
            diff_key = self._cache.compute_key(
                name=diff_name,
                feature_type="pairwise",
                source=feature.source,
                column=entity_col,
                pairwise_mode=mode.value,
            )
            self._cache.put(
                diff_name,
                diff_key,
                diff_series,
                feature_type="pairwise",
                source=feature.source,
                derived_from=[feature.name],
            )
            # Register auto-generated definition
            self._registry[diff_name] = FeatureDef(
                name=diff_name,
                type=FeatureType.PAIRWISE,
                source=feature.source,
                column=entity_col,
                pairwise_mode=mode,
                description=f"Auto-pairwise diff of {feature.name}",
                category=feature.category,
                enabled=feature.enabled,
            )
            pairwise_names.append(diff_name)

        # Generate ratio
        if mode in (PairwiseMode.RATIO, PairwiseMode.BOTH):
            ratio_name = f"ratio_{feature.name}"
            ratio_series = val_a / (val_b + _EPSILON)
            ratio_key = self._cache.compute_key(
                name=ratio_name,
                feature_type="pairwise",
                source=feature.source,
                column=entity_col,
                pairwise_mode=mode.value,
            )
            self._cache.put(
                ratio_name,
                ratio_key,
                ratio_series,
                feature_type="pairwise",
                source=feature.source,
                derived_from=[feature.name],
            )
            self._registry[ratio_name] = FeatureDef(
                name=ratio_name,
                type=FeatureType.PAIRWISE,
                source=feature.source,
                column=entity_col,
                pairwise_mode=mode,
                description=f"Auto-pairwise ratio of {feature.name}",
                category=feature.category,
                enabled=feature.enabled,
            )
            pairwise_names.append(ratio_name)

        # Update parent cache entry with derivative names
        parent_entry = self._cache._entries.get(feature.name)
        if parent_entry is not None:
            parent_entry.derivatives = pairwise_names
            self._cache._save_manifest()

        return pairwise_names

    # ------------------------------------------------------------------
    # @-reference support
    # ------------------------------------------------------------------

    def _get_created_features(self, matchup_df: pd.DataFrame) -> dict[str, pd.Series]:
        """Gather cached features for @-reference resolution in formulas."""
        created: dict[str, pd.Series] = {}
        for name, entry in self._cache._entries.items():
            cached = self._cache.get(name, entry.cache_key)
            if cached is not None and len(cached) == len(matchup_df):
                created[name] = cached
        return created

    # ------------------------------------------------------------------
    # Pairwise formula features
    # ------------------------------------------------------------------

    def _add_pairwise_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a pairwise feature computed from a formula."""
        if feature.formula is None:
            raise ValueError(
                f"Pairwise feature '{feature.name}' requires a 'formula' field."
            )

        matchup_df = self._load_matchup_data()
        created_features = self._get_created_features(matchup_df)
        series = _resolve_formula(feature.formula, matchup_df, created_features=created_features)

        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="pairwise",
            formula=feature.formula,
        )
        self._cache.put(
            feature.name,
            cache_key,
            series,
            feature_type="pairwise",
        )

        target = self.config.target_column
        correlation = 0.0
        if target in matchup_df.columns:
            try:
                correlation = float(series.corr(matchup_df[target].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        stats = _compute_feature_stats(series)
        redundant = _check_redundancy(series, matchup_df)

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=float(series.isna().mean()),
            stats=stats,
            redundant_with=redundant,
            description=feature.description or f"Pairwise formula: {feature.formula}",
        )

    # ------------------------------------------------------------------
    # Matchup features
    # ------------------------------------------------------------------

    def _add_matchup_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a matchup-level feature from a column or formula."""
        matchup_df = self._load_matchup_data()

        if feature.formula is not None:
            created_features = self._get_created_features(matchup_df)
            series = _resolve_formula(feature.formula, matchup_df, created_features=created_features)
        else:
            col_name = feature.column or feature.name
            if col_name not in matchup_df.columns:
                raise ValueError(
                    f"Column '{col_name}' not found in matchup data. "
                    f"Available: {sorted(matchup_df.columns.tolist())}"
                )
            series = matchup_df[col_name].astype(float)

        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="instance",
            column=feature.column,
            formula=feature.formula,
        )
        self._cache.put(
            feature.name,
            cache_key,
            series,
            feature_type="instance",
        )

        target = self.config.target_column
        correlation = 0.0
        if target in matchup_df.columns:
            try:
                correlation = float(series.corr(matchup_df[target].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        stats = _compute_feature_stats(series)

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=float(series.isna().mean()),
            stats=stats,
            description=feature.description or f"Matchup feature: {feature.column or feature.name}",
        )

    # ------------------------------------------------------------------
    # Regime features
    # ------------------------------------------------------------------

    def _add_regime_feature(self, feature: FeatureDef) -> FeatureResult:
        """Add a regime feature (boolean flag from a condition)."""
        condition = feature.condition or feature.formula
        if condition is None:
            raise ValueError(
                f"Regime feature '{feature.name}' requires a 'condition' or 'formula' field."
            )

        matchup_df = self._load_matchup_data()
        created_features = self._get_created_features(matchup_df)
        raw = _resolve_formula(condition, matchup_df, created_features=created_features)
        # Coerce to 0/1 integer
        series = raw.astype(bool).astype(int).astype(float)

        cache_key = self._cache.compute_key(
            name=feature.name,
            feature_type="regime",
            condition=condition,
        )
        self._cache.put(
            feature.name,
            cache_key,
            series,
            feature_type="regime",
        )

        coverage = float(series.mean()) if len(series) > 0 else 0.0

        target = self.config.target_column
        correlation = 0.0
        if target in matchup_df.columns:
            try:
                correlation = float(series.corr(matchup_df[target].astype(float)))
                if np.isnan(correlation):
                    correlation = 0.0
            except (TypeError, ValueError):
                pass

        return FeatureResult(
            name=feature.name,
            column_added=feature.name,
            correlation=correlation,
            abs_correlation=abs(correlation),
            null_rate=0.0,  # boolean flag never has nulls after coercion
            stats={
                "mean": coverage,
                "std": float(series.std()) if len(series) > 0 else 0.0,
                "min": 0.0,
                "max": 1.0,
                "coverage": coverage,
            },
            description=feature.description or f"Regime: {condition}",
        )

    # ------------------------------------------------------------------
    # Compute / resolve
    # ------------------------------------------------------------------

    def compute(self, name: str) -> pd.Series:
        """Compute a single feature, using cache if available.

        Returns the feature as a Series aligned to matchup data.
        """
        feat = self._registry.get(name)
        if feat is None:
            raise ValueError(
                f"Feature '{name}' not registered. "
                f"Available: {sorted(self._registry.keys())}"
            )

        # Try cache first -- look up by manifest entry directly so we
        # don't need to reproduce the exact same cache key the add method used.
        entry = self._cache._entries.get(name)
        if entry is not None:
            cached = self._cache.get(name, entry.cache_key)
            if cached is not None:
                return cached

        # Cache miss or stale -- recompute via add
        self.add(feat)
        entry = self._cache._entries.get(name)
        if entry is not None:
            cached = self._cache.get(name, entry.cache_key)
            if cached is not None:
                return cached

        raise RuntimeError(f"Failed to compute feature '{name}' after re-add.")

    def compute_all(self, names: list[str] | None = None) -> pd.DataFrame:
        """Compute multiple features and return as a DataFrame.

        If names is None, computes all registered features.
        """
        if names is None:
            names = sorted(self._registry.keys())

        result = {}
        for name in names:
            result[name] = self.compute(name)

        return pd.DataFrame(result)

    def resolve(self, names: list[str]) -> pd.DataFrame:
        """Return matchup data with requested feature columns added."""
        matchup_df = self._load_matchup_data().copy()

        for name in names:
            series = self.compute(name)
            if len(series) == len(matchup_df):
                matchup_df[name] = series.values
            else:
                logger.warning(
                    "Feature '%s' length (%d) differs from matchup data (%d), skipping.",
                    name, len(series), len(matchup_df),
                )

        return matchup_df

    def resolve_sets(self, set_names: list[str]) -> list[str]:
        """Map category names to feature column lists.

        For team features, returns derivative (pairwise) names rather
        than the entity-level name.
        """
        result: list[str] = []
        for cat_name in set_names:
            for name, feat in self._registry.items():
                if feat.category != cat_name:
                    continue
                if feat.type == FeatureType.ENTITY:
                    # Return derivatives instead of the team feature itself
                    entry = self._cache._entries.get(name)
                    if entry and entry.derivatives:
                        result.extend(entry.derivatives)
                    else:
                        # Fallback: look for registered pairwise derived from this
                        for rname, rfeat in self._registry.items():
                            if (
                                rfeat.type == FeatureType.PAIRWISE
                                and rfeat.source == feat.source
                                and rname.endswith(f"_{name}")
                            ):
                                result.append(rname)
                else:
                    result.append(name)

        return sorted(set(result))

    # ------------------------------------------------------------------
    # Query / management
    # ------------------------------------------------------------------

    def available(self, type_filter: FeatureType | None = None) -> list[FeatureDef]:
        """List registered features, optionally filtered by type."""
        result = []
        for feat in self._registry.values():
            if type_filter is not None and feat.type != type_filter:
                continue
            result.append(feat)
        return result

    def remove(self, name: str) -> None:
        """Remove a feature and its derivatives from registry and cache.

        For team features, also removes auto-generated pairwise derivatives.
        """
        feat = self._registry.get(name)
        if feat is None:
            return

        # Collect derivatives to remove
        derivatives_to_remove: list[str] = []
        entry = self._cache._entries.get(name)
        if entry is not None:
            derivatives_to_remove = list(entry.derivatives)

        # Remove derivatives from registry
        for deriv_name in derivatives_to_remove:
            self._registry.pop(deriv_name, None)

        # Invalidate from cache (cascades to derivatives)
        self._cache.invalidate(name)

        # Remove from registry
        self._registry.pop(name, None)

    def refresh(self, sources: list[str] | None = None) -> dict[str, str]:
        """Invalidate and recompute features from specified sources.

        If sources is None, refreshes all features.

        Returns dict mapping feature name -> status ("refreshed" or "error: ...").
        """
        # Clear source cache for specified sources
        if sources is not None:
            for src in sources:
                self._source_dfs.pop(src, None)
        else:
            self._source_dfs.clear()
            sources = list({
                f.source for f in self._registry.values() if f.source is not None
            })

        # Reset matchup cache
        self._matchup_df = None

        results: dict[str, str] = {}
        for name, feat in list(self._registry.items()):
            if feat.source is not None and feat.source not in sources:
                continue
            # Skip auto-generated pairwise derivatives — they'll be
            # regenerated when their parent team feature is re-added.
            entry = self._cache._entries.get(name)
            if entry and entry.derived_from:
                continue

            self._cache.invalidate(name)
            try:
                self.add(feat)
                results[name] = "refreshed"
            except Exception as exc:
                results[name] = f"error: {exc}"

        return results

    def save_registry(self) -> None:
        """Persist the current registry back to config.feature_defs."""
        self.config.feature_defs = dict(self._registry)
