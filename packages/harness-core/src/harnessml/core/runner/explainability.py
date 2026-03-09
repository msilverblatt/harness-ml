"""Model explainability via SHAP values, PDP, and feature interactions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_shap_summary(model, X, feature_names, *, top_n=10):
    """Compute mean |SHAP| values and return sorted (feature, importance) pairs.

    Parameters
    ----------
    model : fitted tree model (XGBoost, LightGBM, CatBoost, etc.)
    X : array-like of shape (n_samples, n_features)
    feature_names : list of str
    top_n : int
        Number of top features to return.

    Returns
    -------
    list of (feature_name, mean_abs_shap) tuples, sorted descending.
    """
    import numpy as np
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1

    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1][:top_n]
    return [(feature_names[i], float(mean_abs[i])) for i in indices]


def format_shap_report(shap_results: list[tuple[str, float]], model_name: str = "") -> str:
    """Format SHAP results as markdown."""
    header = "## SHAP Feature Importance"
    if model_name:
        header += f" (`{model_name}`)"
    lines = [header, ""]
    lines.append("| Rank | Feature | Mean |SHAP| |")
    lines.append("|------|---------|------------|")
    for i, (name, val) in enumerate(shap_results, 1):
        lines.append(f"| {i} | {name} | {val:.4f} |")
    return "\n".join(lines)


def compute_shap_values(
    model, X: pd.DataFrame | np.ndarray, max_samples: int = 500
) -> np.ndarray:
    """Compute SHAP values for a fitted model.

    Parameters
    ----------
    model : fitted sklearn-compatible model
    X : feature matrix
    max_samples : max samples to explain

    Returns
    -------
    np.ndarray of SHAP values with shape (n_samples, n_features)
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap is required for explainability. Install with: pip install shap"
        )

    sample = X[:max_samples] if len(X) > max_samples else X
    background = X[: min(100, len(X))]

    if hasattr(model, "predict_proba"):
        explainer = shap.Explainer(model, background)
    else:
        explainer = shap.Explainer(model.predict, background)

    shap_values = explainer(sample)
    values = shap_values.values

    # For binary classification, SHAP returns (n_samples, n_features, 2);
    # select positive class (index 1) to return (n_samples, n_features).
    if values.ndim == 3 and values.shape[2] == 2:
        values = values[:, :, 1]

    return values


def compute_pdp(
    model,
    X: pd.DataFrame | np.ndarray,
    feature_idx: int | str,
    grid_resolution: int = 50,
) -> dict:
    """Compute Partial Dependence Plot data.

    Returns dict with 'values' (grid points) and 'avg_predictions' (mean predictions).
    """
    from sklearn.inspection import partial_dependence

    features = [feature_idx]
    result = partial_dependence(
        model, X, features=features, grid_resolution=grid_resolution
    )
    return {
        "values": result["grid_values"][0].tolist(),
        "avg_predictions": result["average"][0].tolist(),
    }


def compute_feature_interactions(
    model,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Compute top feature interaction pairs based on SHAP interaction approximation.

    Returns list of dicts with feature pair names and interaction strength.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "shap is required. Install with: pip install shap"
        )

    sample = X[: min(200, len(X))]
    background = X[: min(100, len(X))]
    explainer = shap.Explainer(model, background)
    shap_values = explainer(sample).values

    # For binary classification, select positive class
    if shap_values.ndim == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # Approximate interactions via correlation of absolute SHAP values
    abs_shap = np.abs(shap_values)
    n_features = abs_shap.shape[1]
    interactions = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr = np.corrcoef(abs_shap[:, i], abs_shap[:, j])[0, 1]
            if not np.isnan(corr):
                interactions.append(
                    {
                        "feature_1": feature_names[i],
                        "feature_2": feature_names[j],
                        "interaction_strength": abs(corr),
                    }
                )
    interactions.sort(key=lambda x: x["interaction_strength"], reverse=True)
    return interactions[:top_k]
