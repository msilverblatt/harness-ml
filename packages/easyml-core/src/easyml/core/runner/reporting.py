"""Backtest reporting -- pick logs, diagnostics tables, and markdown reports."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from easyml.core.runner.diagnostics import (
    compute_model_agreement,
    evaluate_season_predictions,
)
from easyml.core.runner.hooks import get_entity_column_candidates

logger = logging.getLogger(__name__)


def build_pick_log(preds_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build per-game pick log from predictions DataFrame.

    Parameters
    ----------
    preds_df : pd.DataFrame
        Predictions with prob_ensemble, result, and optionally team/seed columns.
    season : int
        Season identifier.

    Returns
    -------
    pd.DataFrame
        Columns: game_id, season, prob_a, prob_b, predicted_winner,
        actual_winner, correct, confidence, model_agreement_pct.
        Plus team_a, team_b, seed_a, seed_b, round if available.
    """
    prob_a = preds_df["prob_ensemble"].values
    prob_b = 1.0 - prob_a

    predicted_winner = np.where(prob_a > 0.5, "A", "B")
    actual_winner = np.where(preds_df["result"].values == 1, "A", "B")
    correct = predicted_winner == actual_winner
    confidence = np.abs(prob_a - 0.5)
    model_agreement = compute_model_agreement(preds_df)

    log = pd.DataFrame({
        "game_id": np.arange(len(preds_df)),
        "season": season,
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
    season_data: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Build per-season metrics table.

    Parameters
    ----------
    season_data : dict[int, pd.DataFrame]
        Mapping of season -> predictions DataFrame with prob_ensemble and result.

    Returns
    -------
    pd.DataFrame
        Columns: season, brier_score, accuracy, ece, log_loss, n_games.
    """
    rows = []
    for season in sorted(season_data.keys()):
        df = season_data[season]
        results = evaluate_season_predictions(df, {}, season=season)

        # Find the ensemble entry
        ensemble_metrics = None
        for r in results:
            if r["model"] == "ensemble":
                ensemble_metrics = r
                break

        if ensemble_metrics is None:
            continue

        rows.append({
            "season": season,
            "brier_score": ensemble_metrics["brier_score"],
            "accuracy": ensemble_metrics["accuracy"],
            "ece": ensemble_metrics["ece"],
            "log_loss": ensemble_metrics["log_loss"],
            "n_games": len(df),
        })

    return pd.DataFrame(rows)


def generate_markdown_report(
    pooled_metrics: dict,
    diagnostics_df: pd.DataFrame | None = None,
    pick_log: pd.DataFrame | None = None,
    meta_coefficients: dict | None = None,
) -> str:
    """Generate human-readable markdown backtest report.

    Sections:
    - Top-Line Metrics table
    - Per-Season Breakdown (if diagnostics_df provided)
    - Meta-Learner Coefficients (if provided)
    - Pick Analysis Summary (if pick_log provided)

    Parameters
    ----------
    pooled_metrics : dict
        Pooled metrics dict (model_name -> metric dict).
    diagnostics_df : pd.DataFrame | None
        Per-season diagnostics from build_diagnostics_report.
    pick_log : pd.DataFrame | None
        Pick log from build_pick_log.
    meta_coefficients : dict | None
        Meta-learner coefficients (model_name -> coefficient).

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
    if "ensemble" in pooled_metrics:
        m = pooled_metrics["ensemble"]
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        def _fmt(val: object, fmt: str = ".4f") -> str:
            try:
                return f"{val:{fmt}}"
            except (TypeError, ValueError):
                return str(val)

        sections.append(f"| Brier Score | {_fmt(m.get('brier_score', 'N/A'))} |")
        sections.append(f"| Accuracy | {_fmt(m.get('accuracy', 'N/A'))} |")
        sections.append(f"| ECE | {_fmt(m.get('ece', 'N/A'))} |")
        sections.append(f"| Log Loss | {_fmt(m.get('log_loss', 'N/A'))} |")
        sections.append(f"| N Samples | {m.get('n_samples', 'N/A')} |")
    else:
        sections.append("No ensemble metrics available.")
    sections.append("")

    # Per-Season Breakdown
    if diagnostics_df is not None and len(diagnostics_df) > 0:
        sections.append("## Per-Season Breakdown\n")
        sections.append(
            "| Season | Brier | Accuracy | ECE | Log Loss | Games |"
        )
        sections.append(
            "|--------|-------|----------|-----|----------|-------|"
        )
        for _, row in diagnostics_df.iterrows():
            sections.append(
                f"| {int(row['season'])} "
                f"| {row['brier_score']:.4f} "
                f"| {row['accuracy']:.4f} "
                f"| {row['ece']:.4f} "
                f"| {row['log_loss']:.4f} "
                f"| {int(row['n_games'])} |"
            )
        sections.append("")

    # Meta-Learner Coefficients
    if meta_coefficients:
        sections.append("## Meta-Learner Coefficients\n")
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
    season_data: dict[int, pd.DataFrame],
    pooled_metrics: dict,
    diagnostics_df: pd.DataFrame,
    pick_log: pd.DataFrame,
    report_md: str,
) -> None:
    """Save all backtest artifacts to run directory.

    Creates:
    - run_dir/predictions/{season}_probabilities.parquet
    - run_dir/diagnostics/diagnostics.parquet
    - run_dir/diagnostics/pooled_metrics.json
    - run_dir/diagnostics/pick_log.parquet
    - run_dir/diagnostics/report.md

    Parameters
    ----------
    run_dir : Path
        Root run directory (must exist or will be created).
    season_data : dict[int, pd.DataFrame]
        Per-season predictions DataFrames.
    pooled_metrics : dict
        Pooled metrics dict.
    diagnostics_df : pd.DataFrame
        Per-season diagnostics table.
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

    # Save per-season predictions
    for season, df in sorted(season_data.items()):
        path = predictions_dir / f"{season}_probabilities.parquet"
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
