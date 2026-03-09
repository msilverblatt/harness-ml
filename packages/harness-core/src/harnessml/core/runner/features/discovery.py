"""Feature discovery and analysis tools.

Provides quick feature importance, correlation analysis, redundancy
detection, and grouping — designed to be called by an AI agent via
MCP tools to make informed feature selection decisions without manual
exploration.

When ``feature_defs`` (a dict of :class:`FeatureDef` objects) is supplied,
these tools annotate results with feature type and group by type/category
instead of relying solely on column-name heuristics.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from harnessml.core.runner.schema import FeatureDef

logger = logging.getLogger(__name__)


def _is_continuous(y: pd.Series) -> bool:
    """Return True if the target looks continuous (>20 unique values)."""
    return int(y.nunique()) > 20


def _lookup_type(feature_name: str, feature_defs: dict[str, FeatureDef] | None) -> str:
    """Return the feature type string for *feature_name*, or '' if unknown."""
    if feature_defs is None:
        return ""
    feat = feature_defs.get(feature_name)
    if feat is not None:
        return feat.type.value if hasattr(feat.type, "value") else str(feat.type)
    return ""


# -----------------------------------------------------------------------
# Target-feature correlations
# -----------------------------------------------------------------------

def compute_feature_correlations(
    df: pd.DataFrame,
    target_col: str = "result",
    top_n: int = 20,
    feature_columns: list[str] | None = None,
    feature_defs: dict[str, FeatureDef] | None = None,
) -> pd.DataFrame:
    """Compute correlation of each feature with the target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with feature columns and a target column.
    target_col : str
        Name of the target column.
    top_n : int
        Return at most this many features (0 = all).
    feature_columns : list[str] | None
        Explicit list of feature columns. If None, uses all numeric
        columns except target.
    feature_defs : dict[str, FeatureDef] | None
        When provided, a ``type`` column is added to the result.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, correlation, abs_correlation] (plus ``type``
        when *feature_defs* is supplied), sorted by abs_correlation
        descending.
    """
    if feature_columns is not None:
        feature_cols = [c for c in feature_columns if c != target_col and c in df.columns]
    else:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
    if not feature_cols:
        cols = ["feature", "correlation", "abs_correlation"]
        if feature_defs is not None:
            cols.append("type")
        return pd.DataFrame(columns=cols)

    if df[target_col].nunique() > 2:
        # Multiclass target: use ANOVA-based eta-squared instead of
        # point-biserial correlation, which is only meaningful for binary.
        from scipy import stats

        corr_values = []
        for col in feature_cols:
            groups = [
                df.loc[df[target_col] == c, col].dropna()
                for c in df[target_col].unique()
            ]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) < 2:
                corr_values.append(0.0)
                continue
            f_stat, _ = stats.f_oneway(*groups)
            n = sum(len(g) for g in groups)
            k = len(groups)
            denom = f_stat * (k - 1) + (n - k)
            eta_sq = (f_stat * (k - 1)) / denom if denom > 0 else 0.0
            corr_values.append(float(eta_sq))

        correlations_series = pd.Series(corr_values, index=feature_cols)
    else:
        correlations_series = df[feature_cols].corrwith(df[target_col]).dropna()

    result = pd.DataFrame({
        "feature": correlations_series.index,
        "correlation": correlations_series.values,
        "abs_correlation": correlations_series.abs().values,
    }).sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    if feature_defs is not None:
        result["type"] = result["feature"].map(
            lambda f: _lookup_type(f, feature_defs)
        )

    if top_n > 0:
        result = result.head(top_n)

    return result


# -----------------------------------------------------------------------
# Feature importance via quick model
# -----------------------------------------------------------------------

def compute_feature_importance(
    df: pd.DataFrame,
    target_col: str = "result",
    method: str = "xgboost",
    feature_columns: list[str] | None = None,
    top_n: int = 20,
    feature_defs: dict[str, FeatureDef] | None = None,
) -> pd.DataFrame:
    """Quick feature importance via a lightweight model or statistical method.

    Parameters
    ----------
    method : str
        - ``"xgboost"``: Fits a small XGBoost classifier, returns gain-based
          importance.
        - ``"mutual_info"``: sklearn mutual_info_classif.
    feature_columns : list[str] | None
        Explicit list of feature columns. If None, uses all numeric
        columns except target.
    feature_defs : dict[str, FeatureDef] | None
        When provided, a ``type`` column is added to the result.

    Returns
    -------
    pd.DataFrame
        Columns: [feature, importance] (plus ``type`` when
        *feature_defs* is supplied), sorted descending.
    """
    if feature_columns is not None:
        feature_cols = [c for c in feature_columns if c != target_col and c in df.columns]
    else:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != target_col
        ]
    if not feature_cols:
        cols = ["feature", "importance"]
        if feature_defs is not None:
            cols.append("type")
        return pd.DataFrame(columns=cols)

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Drop rows with NaN in target
    mask = y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # Fill NaN features with median
    X = X.fillna(X.median())

    logger.info("Computing feature importance (%s) for %d features, %d rows", method, len(feature_cols), len(X))

    if method == "xgboost":
        importance = _importance_xgboost(X, y, feature_cols)
    elif method == "mutual_info":
        importance = _importance_mutual_info(X, y, feature_cols)
    elif method == "random_forest":
        importance = _importance_random_forest(X, y, feature_cols)
    else:
        raise ValueError(f"Unknown importance method: {method!r}. Use 'xgboost', 'mutual_info', or 'random_forest'.")

    result = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    if not result.empty:
        logger.info("Feature importance complete — top feature: %s (%.4f)", result.iloc[0]["feature"], result.iloc[0]["importance"])

    if feature_defs is not None:
        result["type"] = result["feature"].map(
            lambda f: _lookup_type(f, feature_defs)
        )

    if top_n > 0:
        result = result.head(top_n)

    return result


def _importance_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    """Gain-based importance from a quick XGBoost fit."""
    try:
        if _is_continuous(y):
            from xgboost import XGBRegressor as _XGB
        else:
            from xgboost import XGBClassifier as _XGB
    except ImportError:
        logger.warning("xgboost not installed, falling back to mutual_info")
        return _importance_mutual_info(X, y, feature_cols)

    model = _XGB(
        max_depth=3,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        verbosity=0,
        random_state=42,
    )
    model.fit(X, y)

    # Gain-based importance
    raw = model.feature_importances_
    total = raw.sum()
    if total > 0:
        return raw / total
    return raw


def _importance_mutual_info(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    """Mutual information importance."""
    if _is_continuous(y):
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X, y, random_state=42)
    else:
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X, y, random_state=42)
    total = mi.sum()
    if total > 0:
        return mi / total
    return mi


def _importance_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    """Feature importance from a quick Random Forest fit."""
    try:
        if _is_continuous(y):
            from sklearn.ensemble import RandomForestRegressor as _RF
        else:
            from sklearn.ensemble import RandomForestClassifier as _RF
    except ImportError:
        logger.warning("sklearn not installed, falling back to mutual_info")
        return _importance_mutual_info(X, y, feature_cols)

    model = _RF(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )
    model.fit(X, y)

    raw = model.feature_importances_
    total = raw.sum()
    if total > 0:
        return raw / total
    return raw


# -----------------------------------------------------------------------
# Redundancy detection
# -----------------------------------------------------------------------

def detect_redundant_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    feature_columns: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Find pairs of features with absolute correlation above *threshold*.

    Parameters
    ----------
    threshold : float
        Minimum absolute correlation to flag as redundant.
    feature_columns : list[str] | None
        Explicit list of feature columns. If None, uses all numeric columns.

    Returns
    -------
    list[tuple[str, str, float]]
        (feature_a, feature_b, correlation) tuples, sorted by
        correlation descending.  Only the upper triangle is returned
        (no duplicates).
    """
    if feature_columns is not None:
        feature_cols = [c for c in feature_columns if c in df.columns]
    else:
        feature_cols = list(df.select_dtypes(include=[np.number]).columns)
    if len(feature_cols) < 2:
        return []

    corr_matrix = df[feature_cols].corr()
    pairs: list[tuple[str, str, float]] = []

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                pairs.append((feature_cols[i], feature_cols[j], float(r)))

    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    return pairs


# -----------------------------------------------------------------------
# Feature grouping
# -----------------------------------------------------------------------

def suggest_feature_groups(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    feature_defs: dict[str, FeatureDef] | None = None,
) -> dict[str, list[str]]:
    """Group features by type+category (store-aware) or common prefix (fallback).

    When *feature_defs* is provided, features are grouped into keys like
    ``"team/efficiency"`` or ``"pairwise/general"``.  Features not in the
    registry fall back to prefix grouping under ``"other/<prefix>"``.

    Without *feature_defs*, groups by the first underscore-delimited segment
    of the column name (e.g. ``user_age`` → group ``user``).

    Returns
    -------
    dict[str, list[str]]
        Group name → sorted list of feature names.
    """
    if feature_columns is not None:
        feature_cols = sorted(c for c in feature_columns if c in df.columns)
    else:
        feature_cols = sorted(df.select_dtypes(include=[np.number]).columns)

    groups: dict[str, list[str]] = {}

    for col in feature_cols:
        if feature_defs is not None:
            feat = feature_defs.get(col)
            if feat is not None:
                ftype = feat.type.value if hasattr(feat.type, "value") else str(feat.type)
                group_name = f"{ftype}/{feat.category}"
            else:
                # Unregistered column — fall back to prefix
                parts = col.split("_")
                group_name = f"other/{parts[0]}" if parts else "other/other"
        else:
            parts = col.split("_")
            group_name = parts[0] if parts else "other"
        groups.setdefault(group_name, []).append(col)

    return groups


# -----------------------------------------------------------------------
# Suggestion engine
# -----------------------------------------------------------------------

def suggest_features(
    df: pd.DataFrame,
    count: int = 10,
    target_col: str = "result",
    method: str = "xgboost",
    feature_columns: list[str] | None = None,
    exclude: list[str] | None = None,
    feature_defs: dict[str, FeatureDef] | None = None,
) -> list[str]:
    """Suggest the top *count* features for a model.

    Combines importance ranking with redundancy filtering — if two
    features are highly correlated (>0.95), only the more important
    one is kept.

    Parameters
    ----------
    count : int
        Number of features to suggest.
    feature_columns : list[str] | None
        Explicit list of feature columns. If None, uses all numeric
        columns except target.
    exclude : list[str] | None
        Features to exclude from suggestions.

    Returns
    -------
    list[str]
        Ordered list of recommended feature names.
    """
    importance = compute_feature_importance(
        df, target_col=target_col, method=method,
        feature_columns=feature_columns, top_n=0,
        feature_defs=feature_defs,
    )
    if importance.empty:
        return []

    exclude_set = set(exclude or [])
    redundant_pairs = detect_redundant_features(
        df, threshold=0.95, feature_columns=feature_columns,
    )

    # Build a set of features to skip (less important member of redundant pairs)
    importance_rank = {
        row.feature: i for i, row in importance.iterrows()
    }
    skip: set[str] = set()
    for a, b, _ in redundant_pairs:
        rank_a = importance_rank.get(a, float("inf"))
        rank_b = importance_rank.get(b, float("inf"))
        # Skip the less important one (higher rank = less important)
        skip.add(b if rank_a <= rank_b else a)

    selected: list[str] = []
    for _, row in importance.iterrows():
        feat = row["feature"]
        if feat in exclude_set or feat in skip:
            continue
        selected.append(feat)
        if len(selected) >= count:
            break

    return selected


# -----------------------------------------------------------------------
# Formatted report
# -----------------------------------------------------------------------

def format_discovery_report(
    correlations: pd.DataFrame,
    importance: pd.DataFrame,
    redundant: list[tuple[str, str, float]],
    groups: dict[str, list[str]],
) -> str:
    """Format feature discovery results as markdown.

    When the DataFrames contain a ``type`` column (produced by passing
    *feature_defs* to the correlation/importance functions), that column
    is included in the output tables.

    Parameters
    ----------
    correlations : pd.DataFrame
        Output of compute_feature_correlations.
    importance : pd.DataFrame
        Output of compute_feature_importance.
    redundant : list
        Output of detect_redundant_features.
    groups : dict
        Output of suggest_feature_groups.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    has_types = "type" in correlations.columns

    lines: list[str] = ["## Feature Discovery Report", ""]

    # Top correlations
    lines.append("### Target Correlations (top 15)")
    lines.append("")
    if has_types:
        lines.append("| Feature | Type | Correlation |")
        lines.append("|---------|------|------------|")
    else:
        lines.append("| Feature | Correlation |")
        lines.append("|---------|------------|")
    for _, row in correlations.head(15).iterrows():
        if has_types:
            lines.append(f"| {row['feature']} | {row['type']} | {row['correlation']:+.4f} |")
        else:
            lines.append(f"| {row['feature']} | {row['correlation']:+.4f} |")
    lines.append("")

    # Top importance
    has_imp_types = "type" in importance.columns
    lines.append("### Feature Importance (top 15)")
    lines.append("")
    if has_imp_types:
        lines.append("| Feature | Type | Importance |")
        lines.append("|---------|------|-----------|")
    else:
        lines.append("| Feature | Importance |")
        lines.append("|---------|-----------|")
    for _, row in importance.head(15).iterrows():
        if has_imp_types:
            lines.append(f"| {row['feature']} | {row['type']} | {row['importance']:.4f} |")
        else:
            lines.append(f"| {row['feature']} | {row['importance']:.4f} |")
    lines.append("")

    # Redundancy
    if redundant:
        lines.append(f"### Redundant Pairs ({len(redundant)} found, r > 0.95)")
        lines.append("")
        for a, b, r in redundant[:10]:
            lines.append(f"- {a} <-> {b} (r={r:.3f})")
        if len(redundant) > 10:
            lines.append(f"- ... and {len(redundant) - 10} more")
        lines.append("")

    # Groups
    lines.append(f"### Feature Groups ({len(groups)} groups)")
    lines.append("")
    for group_name, feats in sorted(groups.items()):
        lines.append(f"- **{group_name}** ({len(feats)}): {', '.join(feats[:5])}")
        if len(feats) > 5:
            lines[-1] += f", ... +{len(feats) - 5} more"
    lines.append("")

    return "\n".join(lines)
