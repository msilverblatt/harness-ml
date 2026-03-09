"""Text feature extraction for ML pipelines."""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def extract_text_features(
    df: pd.DataFrame,
    column: str,
    method: str = "tfidf",
    max_features: int = 100,
) -> pd.DataFrame:
    """Extract text features from a DataFrame column.

    Parameters
    ----------
    df : DataFrame with text column
    column : name of the text column
    method : "tfidf" or "count"
    max_features : max number of features to extract

    Returns
    -------
    DataFrame with text features
    """
    if method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=max_features)
    elif method == "count":
        vectorizer = CountVectorizer(max_features=max_features)
    else:
        raise ValueError(f"Unknown text method: {method}")

    matrix = vectorizer.fit_transform(df[column].fillna(""))
    feature_names = [f"{column}_text_{name}" for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)
