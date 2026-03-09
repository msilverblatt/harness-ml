"""Backtest reporting -- pick logs, diagnostics tables, and markdown reports."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from harnessml.core.runner.analysis.diagnostics import (
    compute_model_agreement,
    evaluate_fold_predictions,
    evaluate_fold_predictions_multiclass,
)
from harnessml.core.runner.hooks import get_entity_column_candidates

logger = logging.getLogger(__name__)


def build_pick_log(
    preds_df: pd.DataFrame,
    fold_id: int,
    fold_column: str = "fold",
    target_column: str = "result",
) -> pd.DataFrame:
    """Build per-prediction pick log from predictions DataFrame.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Predictions with prob_ensemble, result, and optionally team/seed columns.
    fold_id : int
        Fold identifier.
    fold_column : str
        Name of the fold column for the output DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, {fold_column}, prob_a, prob_b, predicted_winner,
        actual_winner, correct, confidence, model_agreement_pct.
        Plus entity_a, entity_b, prior_a, prior_b, round if available.
    """
    prob_a = preds_df["prob_ensemble"].values
    prob_b = 1.0 - prob_a

    predicted_winner = np.where(prob_a > 0.5, "A", "B")
    actual_winner = np.where(preds_df[target_column].values == 1, "A", "B")
    correct = predicted_winner == actual_winner
    confidence = np.abs(prob_a - 0.5)
    model_agreement = compute_model_agreement(preds_df)

    log = pd.DataFrame({
        "game_id": np.arange(len(preds_df)),
        fold_column: fold_id,
        "prob_a": prob_a,
        "prob_b": prob_b,
        "predicted_winner": predicted_winner,
        "actual_winner": actual_winner,
        "correct": correct,
        "confidence": confidence,
        "model_agreement_pct": model_agreement,
    })

    # Add optional entity/prior/round columns if they exist
    a_candidates, b_candidates = get_entity_column_candidates()
    _optional_cols = {
        "entity_a": a_candidates,
        "entity_b": b_candidates,
        "prior_a": ["prior_a", "seed_a"],
        "prior_b": ["prior_b", "seed_b"],
        "round": ["Round", "round"],
    }
    for target_name, source_candidates in _optional_cols.items():
        for src in source_candidates:
            if src in preds_df.columns:
                log[target_name] = preds_df[src].values
                break

    return log


def build_diagnostics_report(
    fold_data: dict[int, pd.DataFrame],
    fold_column: str = "fold",
    target_column: str = "result",
    task: str = "binary",
) -> pd.DataFrame:
    """Build per-fold metrics table.

    Parameters
    ----------
    fold_data : dict[int, pd.DataFrame]
        Mapping of fold_id -> predictions DataFrame with prob_ensemble and result.
    fold_column : str
        Name of the fold column for the output DataFrame.
    target_column : str
        Name of the target column.
    task : str
        Task type: ``"binary"`` or ``"multiclass"``. When multiclass, uses
        multiclass-specific metrics (accuracy, log_loss, f1_macro) instead
        of binary metrics (brier, ece).

    Returns
    -------
    pd.DataFrame
        Columns: {fold_column}, metric columns, n_samples.
    """
    _META_KEYS = {"model", "fold_id", "fold"}

    rows = []
    for fold_id in sorted(fold_data.keys()):
        df = fold_data[fold_id]

        if task == "multiclass":
            results = evaluate_fold_predictions_multiclass(
                df, fold_id=fold_id, target_column=target_column,
            )
        elif task == "regression":
            from harnessml.core.runner.analysis.diagnostics import evaluate_fold_predictions_regression
            results = evaluate_fold_predictions_regression(
                df, fold_id=fold_id, target_column=target_column,
            )
        else:
            results = evaluate_fold_predictions(
                df, {}, fold_id=fold_id, target_column=target_column,
            )

        # Find the ensemble entry
        ensemble_metrics = None
        for r in results:
            if r["model"] == "ensemble":
                ensemble_metrics = r
                break

        if ensemble_metrics is None:
            continue

        row = {fold_column: fold_id, "n_samples": len(df)}
        for k, v in ensemble_metrics.items():
            if k not in _META_KEYS:
                row[k] = v
        rows.append(row)

    return pd.DataFrame(rows)


def generate_markdown_report(
    pooled_metrics: dict,
    diagnostics_df: pd.DataFrame | None = None,
    pick_log: pd.DataFrame | None = None,
    meta_coefficients: dict | None = None,
    fold_column: str = "fold",
) -> str:
    """Generate human-readable markdown backtest report.

    Sections:
    - Top-Line Metrics table
    - Per-Fold Breakdown (if diagnostics_df provided)
    - Meta-Learner Coefficients (if provided)
    - Pick Analysis Summary (if pick_log provided)

    Parameters
    ----------
    pooled_metrics : dict
        Pooled metrics dict (model_name -> metric dict).
    diagnostics_df : pd.DataFrame | None
        Per-fold diagnostics from build_diagnostics_report.
    pick_log : pd.DataFrame | None
        Pick log from build_pick_log.
    meta_coefficients : dict | None
        Meta-learner coefficients (model_name -> coefficient).
    fold_column : str
        Name of the fold column in diagnostics_df.

    Returns
    -------
    str
        Markdown-formatted report string.
    """
    sections = []

    # Header
    sections.append("# Backtest Report\n")

    # Top-Line Metrics
    sections.append("## Top-Line Metrics\n")
    def _fmt(val: object, fmt: str = ".4f") -> str:
        try:
            return f"{val:{fmt}}"
        except (TypeError, ValueError):
            return str(val)

    if "ensemble" in pooled_metrics:
        m = pooled_metrics["ensemble"]
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        for key, val in m.items():
            label = key.replace("_", " ").title()
            sections.append(f"| {label} | {_fmt(val)} |")
    else:
        sections.append("No ensemble metrics available.")
    sections.append("")

    # Per-Fold Breakdown
    if diagnostics_df is not None and len(diagnostics_df) > 0:
        sections.append("## Per-Fold Breakdown\n")
        # Determine metric columns dynamically (exclude fold identifier and n_samples)
        meta_cols = {fold_column, "fold", "n_samples"}
        metric_cols = [c for c in diagnostics_df.columns if c not in meta_cols]
        header_labels = ["Fold"] + [c.replace("_", " ").title() for c in metric_cols] + ["Games"]
        sections.append("| " + " | ".join(header_labels) + " |")
        sections.append("|" + "|".join(["------"] * len(header_labels)) + "|")
        for _, row in diagnostics_df.iterrows():
            fold_val = row.get(fold_column, row.get("fold", "?"))
            cells = [str(int(fold_val))]
            for col in metric_cols:
                cells.append(f"{row[col]:.4f}")
            cells.append(str(int(row["n_samples"])))
            sections.append("| " + " | ".join(cells) + " |")
        sections.append("")

    # Meta-Learner Coefficients
    if meta_coefficients:
        sections.append("## Meta-Learner Coefficients\n")
        # Check if coefficients are nested (multiclass: {class_label: {model: coeff}})
        first_val = next(iter(meta_coefficients.values()), None)
        if isinstance(first_val, dict):
            # Multiclass: one table per class
            for class_label, class_coeffs in sorted(meta_coefficients.items()):
                sections.append(f"### {class_label}\n")
                sections.append("| Model | Coefficient |")
                sections.append("|-------|-------------|")
                for model_name, coeff in sorted(
                    class_coeffs.items(), key=lambda x: abs(x[1]), reverse=True
                ):
                    sections.append(f"| {model_name} | {coeff:+.4f} |")
                sections.append("")
        else:
            # Binary: flat {model: coeff}
            sections.append("| Model | Coefficient |")
            sections.append("|-------|-------------|")
            for model_name, coeff in sorted(
                meta_coefficients.items(), key=lambda x: abs(x[1]), reverse=True
            ):
                sections.append(f"| {model_name} | {coeff:+.4f} |")
            sections.append("")

    # Pick Analysis Summary
    if pick_log is not None and len(pick_log) > 0:
        sections.append("## Pick Analysis\n")
        n_total = len(pick_log)
        n_correct = int(pick_log["correct"].sum())
        accuracy = n_correct / n_total if n_total > 0 else 0.0
        avg_confidence = float(pick_log["confidence"].mean())
        avg_agreement = float(pick_log["model_agreement_pct"].mean())

        sections.append(f"- **Total Picks**: {n_total}")
        sections.append(f"- **Correct**: {n_correct} ({accuracy:.1%})")
        sections.append(f"- **Avg Confidence**: {avg_confidence:.4f}")
        sections.append(f"- **Avg Model Agreement**: {avg_agreement:.1%}")

        # Confidence breakdown: correct vs incorrect
        correct_mask = pick_log["correct"]
        if correct_mask.any():
            avg_conf_correct = float(
                pick_log.loc[correct_mask, "confidence"].mean()
            )
            sections.append(
                f"- **Avg Confidence (correct)**: {avg_conf_correct:.4f}"
            )
        if (~correct_mask).any():
            avg_conf_wrong = float(
                pick_log.loc[~correct_mask, "confidence"].mean()
            )
            sections.append(
                f"- **Avg Confidence (incorrect)**: {avg_conf_wrong:.4f}"
            )
        sections.append("")

    return "\n".join(sections)


def export_backtest_artifacts(
    run_dir: Path,
    fold_data: dict[int, pd.DataFrame],
    pooled_metrics: dict,
    diagnostics_df: pd.DataFrame,
    pick_log: pd.DataFrame,
    report_md: str,
) -> None:
    """Save all backtest artifacts to run directory.

    Creates:
    - run_dir/predictions/{fold_id}_probabilities.parquet
    - run_dir/diagnostics/diagnostics.parquet
    - run_dir/diagnostics/pooled_metrics.json
    - run_dir/diagnostics/pick_log.parquet
    - run_dir/diagnostics/report.md

    Parameters
    ----------
    run_dir : Path
        Root run directory (must exist or will be created).
    fold_data : dict[int, pd.DataFrame]
        Per-fold predictions DataFrames.
    pooled_metrics : dict
        Pooled metrics dict.
    diagnostics_df : pd.DataFrame
        Per-fold diagnostics table.
    pick_log : pd.DataFrame
        Pick log DataFrame.
    report_md : str
        Markdown report string.
    """
    run_dir = Path(run_dir)
    predictions_dir = run_dir / "predictions"
    diagnostics_dir = run_dir / "diagnostics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Save per-fold predictions
    for fold_id, df in sorted(fold_data.items()):
        path = predictions_dir / f"{fold_id}_probabilities.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved predictions: %s", path)

    # Save diagnostics
    diagnostics_df.to_parquet(
        diagnostics_dir / "diagnostics.parquet", index=False
    )
    logger.info("Saved diagnostics.parquet")

    # Save pooled metrics as JSON
    with open(diagnostics_dir / "pooled_metrics.json", "w") as f:
        json.dump(pooled_metrics, f, indent=2, default=_json_default)
    logger.info("Saved pooled_metrics.json")

    # Save pick log
    pick_log.to_parquet(diagnostics_dir / "pick_log.parquet", index=False)
    logger.info("Saved pick_log.parquet")

    # Save markdown report
    with open(diagnostics_dir / "report.md", "w") as f:
        f.write(report_md)
    logger.info("Saved report.md")


def _json_default(obj: object) -> object:
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
