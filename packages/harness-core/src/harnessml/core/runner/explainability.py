"""Model explainability via SHAP values."""
from __future__ import annotations


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
