"""Automated feature selection methods."""
from __future__ import annotations

import pandas as pd


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "k_best",
    k: int = 10,
    threshold: float = 0.9,
) -> list[str]:
    """Select top features using the specified method."""
    if method == "k_best":
        return _select_k_best(X, y, k)
    elif method == "rfe":
        return _rfe(X, y, k)
    elif method == "correlation_cluster":
        return _correlation_cluster(X, threshold)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")


def _select_k_best(X, y, k):
    from sklearn.feature_selection import SelectKBest, mutual_info_classif

    selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
    selector.fit(X, y)
    mask = selector.get_support()
    return list(X.columns[mask])


def _rfe(X, y, k):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import RFE

    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
    selector.fit(X, y)
    mask = selector.get_support()
    return list(X.columns[mask])


def _correlation_cluster(X, threshold):
    corr = X.corr().abs()
    selected = list(X.columns)
    to_drop = set()
    for i in range(len(corr)):
        if corr.columns[i] in to_drop:
            continue
        for j in range(i + 1, len(corr)):
            if corr.iloc[i, j] > threshold:
                to_drop.add(corr.columns[j])
    return [c for c in selected if c not in to_drop]
