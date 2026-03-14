"""Static HTML dashboard export for HarnessML projects."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from harnessml.core.logging import get_logger
from jinja2 import Environment, FileSystemLoader

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"

_LOWER_IS_BETTER = {"brier", "ece", "log_loss", "mae", "mse", "rmse"}


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _load_json(path: Path) -> dict | list:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError, OSError):
        return {}


def _gather_project_info(project_dir: Path) -> dict:
    """Gather project name, task type, and config summary."""
    config_dir = project_dir / "config"
    pipeline = _load_yaml(config_dir / "pipeline.yaml")
    models_cfg = _load_yaml(config_dir / "models.yaml")
    data_cfg = pipeline.get("data", {})

    project_name = (
        pipeline.get("name")
        or data_cfg.get("name")
        or project_dir.name
    )
    task = data_cfg.get("task", "unknown")
    target_column = data_cfg.get("target_column")

    all_models = models_cfg.get("models", {})
    active_models = [
        {"name": name, "type": mdef.get("type", "unknown"), "features": mdef.get("features", [])}
        for name, mdef in all_models.items()
        if isinstance(mdef, dict) and mdef.get("active", True)
    ]

    return {
        "project_name": project_name,
        "task": task,
        "target_column": target_column,
        "active_models": active_models,
        "pipeline": pipeline,
        "models_cfg": models_cfg,
    }


def _gather_runs(project_dir: Path, run_id: str | None = None) -> list[dict]:
    """Gather run data from outputs directory."""
    outputs_dir = project_dir / "outputs"
    if not outputs_dir.exists():
        return []

    runs = []
    run_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir()],
        reverse=True,
    )

    for d in run_dirs:
        if run_id is not None and d.name != run_id:
            continue
        metrics_path = d / "diagnostics" / "pooled_metrics.json"
        if not metrics_path.exists():
            continue
        raw = _load_json(metrics_path)
        if not raw:
            continue

        metrics = raw.get("ensemble", raw) if isinstance(raw.get("ensemble"), dict) else raw
        meta_coefficients = raw.get("meta_coefficients", {})

        # Per-fold data
        folds = []
        diag_path = d / "diagnostics" / "diagnostics.parquet"
        if diag_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(diag_path)
                if not df.empty:
                    folds = df.to_dict(orient="records")
            except Exception:
                pass

        # Calibration data
        calibration = []
        try:
            import numpy as np
            import pandas as pd

            pred_path = d / "predictions" / "predictions.parquet"
            pred_frames = []
            if pred_path.exists():
                pred_frames.append(pd.read_parquet(pred_path))
            else:
                pred_dir = d / "predictions"
                if pred_dir.exists():
                    for f in sorted(pred_dir.glob("*.parquet")):
                        pred_frames.append(pd.read_parquet(f))

            if pred_frames:
                pred_df = pd.concat(pred_frames, ignore_index=True)
                prob_col = "prob_ensemble"
                if prob_col not in pred_df.columns:
                    prob_cols = [c for c in pred_df.columns if c.startswith("prob_")]
                    prob_col = prob_cols[0] if prob_cols else None

                if prob_col:
                    # Detect target column
                    prob_cols_set = {c for c in pred_df.columns if c.startswith("prob_")}
                    meta_cols = {"fold", "diff_prior"}
                    target_col = None
                    for col in pred_df.columns:
                        if col not in prob_cols_set and col not in meta_cols:
                            target_col = col
                            break

                    if target_col:
                        y_true = pred_df[target_col].values.astype(float)
                        y_prob = pred_df[prob_col].values.astype(float)
                        unique_targets = set(y_true)
                        if unique_targets.issubset({0.0, 1.0}) and len(unique_targets) <= 2:
                            bins = 10
                            bin_edges = np.linspace(0, 1, bins + 1)
                            for i in range(bins):
                                if i == bins - 1:
                                    mask = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
                                else:
                                    mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
                                if mask.sum() > 0:
                                    calibration.append({
                                        "bin_center": float((bin_edges[i] + bin_edges[i + 1]) / 2),
                                        "predicted": float(y_prob[mask].mean()),
                                        "actual": float(y_true[mask].mean()),
                                        "count": int(mask.sum()),
                                    })
        except Exception:
            pass

        runs.append({
            "id": d.name,
            "metrics": metrics,
            "meta_coefficients": meta_coefficients,
            "folds": folds,
            "calibration": calibration,
        })

    return runs


def _gather_experiments(project_dir: Path) -> list[dict]:
    """Gather experiment data from journal and result files."""
    experiments_dir = project_dir / "experiments"
    journal_path = experiments_dir / "journal.jsonl"
    if not journal_path.exists():
        return []

    experiments = []
    text = journal_path.read_text().strip()
    if not text:
        return []

    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        exp_id = entry.get("experiment_id", "")

        # Enrich with results
        results_path = experiments_dir / exp_id / "results.json"
        if results_path.exists():
            try:
                results = json.loads(results_path.read_text())
                for key in ("metrics", "baseline_metrics", "primary_delta", "primary_metric", "verdict"):
                    if key in results:
                        entry[key] = results[key]
            except json.JSONDecodeError:
                pass

        # Read hypothesis/conclusion from files
        if not entry.get("hypothesis"):
            hyp_path = experiments_dir / exp_id / "hypothesis.txt"
            if hyp_path.exists():
                entry["hypothesis"] = hyp_path.read_text().strip()

        if not entry.get("conclusion"):
            conc_path = experiments_dir / exp_id / "conclusion.txt"
            if conc_path.exists():
                entry["conclusion"] = conc_path.read_text().strip()

        experiments.append(entry)

    experiments.reverse()  # newest first
    return experiments


def _determine_primary_metric(metrics: dict) -> tuple[str | None, float | None]:
    """Determine the primary metric name and value from a metrics dict."""
    if not metrics:
        return None, None
    for candidate in ("brier", "accuracy", "rmse", "mae", "auc"):
        if candidate in metrics:
            return candidate, metrics[candidate]
    # Fallback to first metric
    first_key = next(iter(metrics), None)
    return first_key, metrics.get(first_key) if first_key else None


def gather_export_data(
    project_dir: Path,
    run_id: str | None = None,
) -> dict:
    """Gather all data needed for the static HTML export.

    Parameters
    ----------
    project_dir : Path
        Root directory of the HarnessML project.
    run_id : str | None
        If provided, only include data for this specific run.

    Returns
    -------
    dict
        All data needed to render the report template.
    """
    project_info = _gather_project_info(project_dir)
    runs = _gather_runs(project_dir, run_id=run_id)
    experiments = _gather_experiments(project_dir)

    # Latest run metrics
    latest_metrics: dict = {}
    latest_meta_coefficients: dict = {}
    latest_folds: list[dict] = []
    latest_calibration: list[dict] = []
    latest_run_id: str | None = None
    if runs:
        latest = runs[0]
        latest_metrics = latest.get("metrics", {})
        latest_meta_coefficients = latest.get("meta_coefficients", {})
        latest_folds = latest.get("folds", [])
        latest_calibration = latest.get("calibration", [])
        latest_run_id = latest.get("id")

    primary_name, primary_value = _determine_primary_metric(latest_metrics)

    # Metric trend from experiments
    metric_trend: list[dict] = []
    if experiments and primary_name:
        for exp in reversed(experiments):  # chronological order
            exp_metrics = exp.get("metrics", {})
            if primary_name in exp_metrics:
                metric_trend.append({
                    "experiment_id": exp.get("experiment_id", ""),
                    "value": exp_metrics[primary_name],
                })

    return {
        "project_name": project_info["project_name"],
        "task": project_info["task"],
        "target_column": project_info["target_column"],
        "active_models": project_info["active_models"],
        "export_date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "latest_run_id": latest_run_id,
        "latest_metrics": latest_metrics,
        "primary_metric_name": primary_name,
        "primary_metric_value": primary_value,
        "meta_coefficients": latest_meta_coefficients,
        "folds": latest_folds,
        "calibration": latest_calibration,
        "experiments": experiments,
        "metric_trend": metric_trend,
        "lower_is_better": list(_LOWER_IS_BETTER),
    }


def export_html(
    project_dir: Path,
    output_path: Path,
    run_id: str | None = None,
) -> Path:
    """Export a self-contained HTML dashboard report.

    Parameters
    ----------
    project_dir : Path
        Root directory of the HarnessML project.
    output_path : Path
        Where to write the HTML file.
    run_id : str | None
        If provided, scope the report to a specific run.

    Returns
    -------
    Path
        The path to the written HTML file.
    """
    data = gather_export_data(project_dir, run_id=run_id)

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")
    html = template.render(data=data, data_json=json.dumps(data, default=str))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logger.info("exported static dashboard", path=str(output_path))
    return output_path
