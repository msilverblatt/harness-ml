"""Run output endpoints."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request
from harnessml.studio.routes.project import resolve_project_dir_from_request

router = APIRouter(tags=["runs"])


def _load_predictions(run_dir: Path) -> pd.DataFrame | None:
    """Load predictions from single file or per-fold files."""
    pred_path = run_dir / "predictions" / "predictions.parquet"
    if pred_path.exists():
        return pd.read_parquet(pred_path)

    pred_dir = run_dir / "predictions"
    if not pred_dir.exists():
        return None

    frames = []
    for f in sorted(pred_dir.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    if frames:
        return pd.concat(frames, ignore_index=True)
    return None


def _detect_target_col(df: pd.DataFrame) -> str | None:
    """Detect target column from predictions dataframe."""
    prob_cols = {c for c in df.columns if c.startswith("prob_")}
    meta_cols = {"fold", "diff_prior"}
    for col in df.columns:
        if col not in prob_cols and col not in meta_cols:
            return col
    return None


def _load_experiments(project_dir: Path) -> list[dict]:
    """Load experiment entries with metrics for linking to runs.

    Reads from journal.jsonl first; falls back to scanning experiment
    directories for results.json files.
    """
    experiments_dir = project_dir / "experiments"
    if not experiments_dir.exists():
        return []

    experiments = []
    seen_ids: set[str] = set()

    journal_path = experiments_dir / "journal.jsonl"
    if journal_path.exists():
        text = journal_path.read_text().strip()
        if text:
            for line in text.split("\n"):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                exp_id = entry.get("experiment_id", "")
                seen_ids.add(exp_id)
                results_path = experiments_dir / exp_id / "results.json"
                if results_path.exists():
                    try:
                        results = json.loads(results_path.read_text())
                        entry.update({
                            k: results[k]
                            for k in ("metrics", "primary_metric", "primary_delta", "verdict")
                            if k in results
                        })
                    except json.JSONDecodeError:
                        pass
                experiments.append(entry)

    for d in sorted(experiments_dir.iterdir()):
        if not d.is_dir() or d.name in seen_ids:
            continue
        results_path = d / "results.json"
        if not results_path.exists():
            continue
        try:
            results = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            continue
        exp_id = results.get("experiment_id", d.name)
        if exp_id in seen_ids:
            continue
        seen_ids.add(exp_id)
        experiments.append({
            "experiment_id": exp_id,
            **{
                k: results[k]
                for k in ("metrics", "primary_metric", "primary_delta", "verdict")
                if k in results
            },
        })

    return experiments


def _match_experiment(run_metrics: dict, experiments: list[dict]) -> str | None:
    """Match a run to an experiment by comparing metric values."""
    if not run_metrics or not experiments:
        return None
    for exp in experiments:
        exp_metrics = exp.get("metrics")
        if not exp_metrics:
            continue
        shared_keys = set(run_metrics.keys()) & set(exp_metrics.keys())
        if not shared_keys:
            continue
        if all(
            abs(run_metrics[k] - exp_metrics[k]) < 1e-9
            for k in shared_keys
            if isinstance(run_metrics.get(k), (int, float))
            and isinstance(exp_metrics.get(k), (int, float))
        ):
            return exp.get("experiment_id")
    return None


def _compute_fold_std(run_dir: Path) -> dict[str, float]:
    """Compute per-metric standard deviation from per-fold diagnostics."""
    diag_path = run_dir / "diagnostics" / "diagnostics.parquet"
    if not diag_path.exists():
        return {}
    try:
        df = pd.read_parquet(diag_path)
        if df.empty or len(df) < 2:
            return {}
        metric_cols = [c for c in df.columns if c not in ("fold", "n_samples")]
        return {col: float(df[col].std()) for col in metric_cols if df[col].dtype.kind == "f"}
    except Exception:
        return {}


@router.get("/runs")
async def list_runs(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return []
    experiments = _load_experiments(project_dir)
    runs = []
    for d in sorted(outputs_dir.iterdir(), reverse=True):
        if d.is_dir():
            metrics_path = d / "diagnostics" / "pooled_metrics.json"
            metrics = {}
            meta_coefficients = {}
            if metrics_path.exists():
                try:
                    raw = json.loads(metrics_path.read_text())
                    if "ensemble" in raw and isinstance(raw["ensemble"], dict):
                        metrics = raw["ensemble"]
                    else:
                        metrics = raw
                    if "meta_coefficients" in raw:
                        meta_coefficients = raw["meta_coefficients"]
                except json.JSONDecodeError:
                    pass
            has_report = (d / "diagnostics" / "report.md").exists()
            experiment_id = _match_experiment(metrics, experiments)
            metric_std = _compute_fold_std(d)
            runs.append({
                "id": d.name,
                "metrics": metrics,
                "metric_std": metric_std,
                "meta_coefficients": meta_coefficients,
                "has_report": has_report,
                "experiment_id": experiment_id,
            })
    return runs


@router.get("/runs/{run_id}/metrics")
async def run_metrics(request: Request, run_id: str, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    metrics_path = project_dir / "outputs" / run_id / "diagnostics" / "pooled_metrics.json"
    if not metrics_path.exists():
        return {"error": f"Metrics not found for run {run_id}"}
    raw = json.loads(metrics_path.read_text())
    # Flatten ensemble metrics + include meta_coefficients separately
    result = {}
    if "ensemble" in raw and isinstance(raw["ensemble"], dict):
        result["metrics"] = raw["ensemble"]
    else:
        result["metrics"] = raw
    if "meta_coefficients" in raw:
        result["meta_coefficients"] = raw["meta_coefficients"]
    return result


@router.get("/runs/{run_id}/folds")
async def run_folds(request: Request, run_id: str, project: str | None = None):
    """Return per-fold breakdown from diagnostics.parquet."""
    project_dir = resolve_project_dir_from_request(request, project)
    diag_path = project_dir / "outputs" / run_id / "diagnostics" / "diagnostics.parquet"
    if not diag_path.exists():
        return {"error": "No fold diagnostics found"}
    df = pd.read_parquet(diag_path)
    if df.empty:
        return {"error": "Empty diagnostics"}
    metric_cols = [c for c in df.columns if c not in ("fold", "n_samples")]
    return {
        "folds": df.to_dict(orient="records"),
        "metric_names": metric_cols,
    }


@router.get("/runs/{run_id}/correlations")
async def run_correlations(request: Request, run_id: str, project: str | None = None):
    """Compute model prediction correlation matrix from predictions parquet."""
    project_dir = resolve_project_dir_from_request(request, project)
    run_dir = project_dir / "outputs" / run_id

    df = _load_predictions(run_dir)
    if df is None:
        return {"error": "No prediction files found"}

    prob_cols = [c for c in df.columns if c.startswith("prob_") and c != "prob_ensemble"]
    if len(prob_cols) < 2:
        return {"models": [], "matrix": []}

    corr = df[prob_cols].corr()
    model_names = [c.replace("prob_", "") for c in prob_cols]
    return {
        "models": model_names,
        "matrix": corr.values.tolist(),
    }


@router.get("/runs/{run_id}/calibration")
async def run_calibration(request: Request, run_id: str, bins: int = 10, project: str | None = None):
    """Compute calibration curve data from predictions."""
    import numpy as np

    project_dir = resolve_project_dir_from_request(request, project)
    run_dir = project_dir / "outputs" / run_id

    df = _load_predictions(run_dir)
    if df is None:
        return {"error": "No predictions found"}

    target_col = _detect_target_col(df)
    if target_col is None:
        return {"error": "Could not identify target column"}

    prob_col = "prob_ensemble"
    if prob_col not in df.columns:
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        if prob_cols:
            prob_col = prob_cols[0]
        else:
            return {"error": "No probability columns found"}

    y_true = df[target_col].values.astype(float)
    y_prob = df[prob_col].values.astype(float)

    # For regression tasks, calibration doesn't apply
    unique_targets = set(y_true)
    if len(unique_targets) > 2 or not unique_targets.issubset({0.0, 1.0}):
        return {"error": "Calibration only applies to binary classification tasks"}

    bin_edges = np.linspace(0, 1, bins + 1)
    curve = []
    for i in range(bins):
        if i == bins - 1:
            mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            curve.append({
                "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                "predicted": float(y_prob[mask].mean()),
                "actual": float(y_true[mask].mean()),
                "count": int(mask.sum()),
            })

    return {"calibration": curve, "prob_column": prob_col}


@router.get("/runs/{run_id}/residuals")
async def run_residuals(request: Request, run_id: str, project: str | None = None):
    """Compute residual analysis for regression tasks."""
    import numpy as np

    project_dir = resolve_project_dir_from_request(request, project)
    run_dir = project_dir / "outputs" / run_id

    df = _load_predictions(run_dir)
    if df is None:
        return {"error": "No predictions found"}

    target_col = _detect_target_col(df)
    if target_col is None:
        return {"error": "Could not identify target column"}

    y_true = df[target_col].values.astype(float)
    y_pred = df["prob_ensemble"].values.astype(float) if "prob_ensemble" in df.columns else None
    if y_pred is None:
        return {"error": "No ensemble predictions found"}

    residuals = y_true - y_pred

    # Scatter data (predicted vs actual)
    scatter = [
        {"predicted": float(p), "actual": float(a), "residual": float(r)}
        for p, a, r in zip(y_pred, y_true, residuals)
    ]

    # Residual distribution (histogram)
    hist_counts, hist_edges = np.histogram(residuals, bins=30)
    histogram = [
        {"bin_center": float((hist_edges[i] + hist_edges[i + 1]) / 2), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    return {
        "scatter": scatter,
        "histogram": histogram,
        "stats": {
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "median_residual": float(np.median(residuals)),
            "max_overpredict": float(np.min(residuals)),
            "max_underpredict": float(np.max(residuals)),
        },
    }


@router.get("/runs/{run_id}/report")
async def run_report(request: Request, run_id: str, project: str | None = None):
    """Return the markdown report."""
    project_dir = resolve_project_dir_from_request(request, project)
    report_path = project_dir / "outputs" / run_id / "diagnostics" / "report.md"
    if not report_path.exists():
        return {"error": "No report found"}
    return {"markdown": report_path.read_text()}


@router.get("/runs/{run_id}/picks")
async def run_picks(request: Request, run_id: str, project: str | None = None):
    """Return pick analysis stats from pick_log.parquet."""

    project_dir = resolve_project_dir_from_request(request, project)
    pick_path = project_dir / "outputs" / run_id / "diagnostics" / "pick_log.parquet"
    if not pick_path.exists():
        return {"error": "No pick log found"}

    df = pd.read_parquet(pick_path)
    if df.empty:
        return {"error": "Empty pick log"}

    n_total = len(df)
    n_correct = int(df["correct"].sum())
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    avg_confidence = float(df["confidence"].mean())
    avg_agreement = float(df["model_agreement_pct"].mean()) if "model_agreement_pct" in df.columns else None

    result: dict = {
        "total": n_total,
        "correct": n_correct,
        "accuracy": accuracy,
        "avg_confidence": avg_confidence,
        "avg_agreement": avg_agreement,
    }

    correct_mask = df["correct"].astype(bool)
    if correct_mask.any():
        result["avg_confidence_correct"] = float(df.loc[correct_mask, "confidence"].mean())
    if (~correct_mask).any():
        result["avg_confidence_incorrect"] = float(df.loc[~correct_mask, "confidence"].mean())

    return result
