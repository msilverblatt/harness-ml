"""Optional visualization module -- requires matplotlib.

All rendering functions accept structured data (dicts/lists from diagnostics
or calibration modules) and an output path, returning the path on success or
an error message if matplotlib is not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def _check_matplotlib():
    """Import matplotlib with non-interactive backend; return plt or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def render_roc_curve(
    roc_data: dict,
    output_path: str | Path,
) -> str | None:
    """Render ROC curve to an image file.

    Parameters
    ----------
    roc_data : dict
        Output of :func:`diagnostics.roc_curve_data` with keys
        ``fpr``, ``tpr``, ``auc``.
    output_path : str or Path
        Destination file path (e.g. ``"roc.png"``).

    Returns
    -------
    str or None
        The file path on success, or an error message.
    """
    plt = _check_matplotlib()
    if plt is None:
        return "matplotlib not installed -- returning data only"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        label=f'ROC (AUC={roc_data["auc"]:.3f})',
        linewidth=2,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def render_pr_curve(
    pr_data: dict,
    output_path: str | Path,
) -> str | None:
    """Render precision-recall curve to an image file.

    Parameters
    ----------
    pr_data : dict
        Output of :func:`diagnostics.pr_curve_data` with keys
        ``precision``, ``recall``, ``average_precision``.
    output_path : str or Path
        Destination file path.

    Returns
    -------
    str or None
        The file path on success, or an error message.
    """
    plt = _check_matplotlib()
    if plt is None:
        return "matplotlib not installed -- returning data only"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        pr_data["recall"],
        pr_data["precision"],
        label=f'AP={pr_data["average_precision"]:.3f}',
        linewidth=2,
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def render_calibration(
    reliability_data: list[dict],
    output_path: str | Path,
) -> str | None:
    """Render a reliability (calibration) diagram.

    Parameters
    ----------
    reliability_data : list of dict
        Output of :func:`calibration.reliability_diagram_data`.
        Each dict has ``bin_center``, ``mean_predicted``,
        ``fraction_positive``, ``count``.
    output_path : str or Path
        Destination file path.

    Returns
    -------
    str or None
        The file path on success, or an error message.
    """
    plt = _check_matplotlib()
    if plt is None:
        return "matplotlib not installed -- returning data only"

    mean_pred = [d["mean_predicted"] for d in reliability_data]
    frac_pos = [d["fraction_positive"] for d in reliability_data]
    counts = [d["count"] for d in reliability_data]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]},
    )

    # Calibration plot
    ax1.plot(mean_pred, frac_pos, "s-", label="Model", markersize=6)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title("Reliability Diagram")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Histogram of bin counts
    bin_centers = [d["bin_center"] for d in reliability_data]
    bar_width = 1.0 / (len(bin_centers) + 1) if bin_centers else 0.1
    ax2.bar(bin_centers, counts, width=bar_width, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Predicted Probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim([0, 1])

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def render_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    labels: list[str] | None = None,
) -> str | None:
    """Render a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels (binary, not probabilities).
    output_path : str or Path
        Destination file path.
    labels : list of str, optional
        Class labels for the axes. Defaults to ``["0", "1"]``.

    Returns
    -------
    str or None
        The file path on success, or an error message.
    """
    plt = _check_matplotlib()
    if plt is None:
        return "matplotlib not installed -- returning data only"

    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def render_feature_importance(
    importance_data: dict,
    output_path: str | Path,
    top_n: int = 20,
) -> str | None:
    """Render a horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_data : dict
        Output of :func:`diagnostics.permutation_importance_data` with
        keys ``feature_names``, ``importances_mean``, ``importances_std``.
    output_path : str or Path
        Destination file path.
    top_n : int
        Show only the top N features.

    Returns
    -------
    str or None
        The file path on success, or an error message.
    """
    plt = _check_matplotlib()
    if plt is None:
        return "matplotlib not installed -- returning data only"

    names = importance_data["feature_names"]
    means = np.array(importance_data["importances_mean"])
    stds = np.array(importance_data["importances_std"])

    # Sort by importance descending and take top_n
    order = np.argsort(means)[::-1][:top_n]
    names = [names[i] for i in order]
    means = means[order]
    stds = stds[order]

    # Reverse for horizontal bar chart (top at top)
    names = names[::-1]
    means = means[::-1]
    stds = stds[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, means, xerr=stds, align="center", alpha=0.8, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def render_shap_summary(
    shap_data: dict,
    X,
    output_path: str | Path,
) -> str | None:
    """Render a SHAP summary (beeswarm) plot.

    Uses the ``shap`` library's own plotting if available, otherwise
    falls back to a simple bar chart of mean absolute SHAP values.

    Parameters
    ----------
    shap_data : dict
        Output of :func:`diagnostics.compute_shap_values` with keys
        ``values``, ``feature_names``, ``base_value``.
    X : DataFrame or array
        Feature data used for coloring the beeswarm plot.
    output_path : str or Path
        Destination file path.

    Returns
    -------
    str or None
        The file path on success, or an error message.
    """
    plt = _check_matplotlib()
    if plt is None:
        return "matplotlib not installed -- returning data only"

    if "error" in shap_data:
        return shap_data["error"]

    shap_values = np.asarray(shap_data["values"])
    feature_names = shap_data["feature_names"]

    # Handle multi-output: take the first output's values
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    # Try to use shap's native beeswarm plot
    try:
        import shap as shap_lib

        fig, ax = plt.subplots(figsize=(10, max(4, len(feature_names) * 0.35)))
        shap_lib.summary_plot(
            shap_values, X, feature_names=feature_names,
            show=False, plot_type="bar",
        )
        plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close("all")
        return str(output_path)
    except ImportError:
        pass

    # Fallback: bar chart of mean |SHAP|
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    sorted_names = [feature_names[i] for i in order]
    sorted_vals = mean_abs[order]

    # Reverse for horizontal bar chart
    sorted_names = sorted_names[::-1]
    sorted_vals = sorted_vals[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(sorted_names) * 0.35)))
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_vals, align="center", alpha=0.8, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)
