"""Experiment history endpoints."""
from __future__ import annotations

import json

from fastapi import APIRouter, Request
from harnessml.core.logging import get_logger
from harnessml.studio.routes.project import resolve_project_dir_from_request
from harnessml.studio.routes.runs import _compute_fold_std

logger = get_logger(__name__)

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
async def list_experiments(request: Request, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    experiments_dir = project_dir / "experiments"
    journal_path = experiments_dir / "journal.jsonl"
    if not journal_path.exists():
        return []

    experiments = []
    for line in journal_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        if "created_at" in entry and "timestamp" not in entry:
            entry["timestamp"] = entry["created_at"]

        exp_id = entry.get("experiment_id", "")

        # Enrich with results.json if available
        results_path = experiments_dir / exp_id / "results.json"
        if results_path.exists():
            try:
                results = json.loads(results_path.read_text())
                # Merge results into entry (results has metrics, baseline_metrics, etc.)
                if "metrics" in results and results["metrics"]:
                    entry["metrics"] = results["metrics"]
                if "baseline_metrics" in results:
                    entry["baseline_metrics"] = results["baseline_metrics"]
                if "primary_delta" in results:
                    entry["primary_delta"] = results["primary_delta"]
                if "primary_metric" in results:
                    entry["primary_metric"] = results["primary_metric"]
                if "verdict" not in entry and "verdict" in results:
                    entry["verdict"] = results["verdict"]
            except json.JSONDecodeError:
                pass

        # Read hypothesis/conclusion from files if not in journal
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

    # Enrich experiments with fold-level std from matched runs
    outputs_dir = project_dir / "outputs"
    if outputs_dir.exists():
        run_dirs = sorted(outputs_dir.iterdir(), reverse=True)
        for exp in experiments:
            exp_metrics = exp.get("metrics")
            if not exp_metrics:
                continue
            for d in run_dirs:
                if not d.is_dir():
                    continue
                metrics_path = d / "diagnostics" / "pooled_metrics.json"
                if not metrics_path.exists():
                    continue
                try:
                    raw = json.loads(metrics_path.read_text())
                    run_metrics = raw.get("ensemble", raw) if isinstance(raw.get("ensemble"), dict) else raw
                except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
                    logger.warning("failed to parse run metrics", path=str(metrics_path), error=str(e))
                    continue
                shared = set(run_metrics.keys()) & set(exp_metrics.keys())
                if shared and all(
                    abs(run_metrics[k] - exp_metrics[k]) < 1e-9
                    for k in shared
                    if isinstance(run_metrics.get(k), (int, float))
                    and isinstance(exp_metrics.get(k), (int, float))
                ):
                    exp["metric_std"] = _compute_fold_std(d)
                    break

    return experiments


@router.get("/experiments/{experiment_id}")
async def get_experiment(request: Request, experiment_id: str, project: str | None = None):
    project_dir = resolve_project_dir_from_request(request, project)
    exp_dir = project_dir / "experiments" / experiment_id
    if not exp_dir.exists():
        return {"error": f"Experiment {experiment_id} not found"}

    result: dict = {"id": experiment_id}

    overlay_path = exp_dir / "overlay.yaml"
    if overlay_path.exists():
        import yaml
        result["overlay"] = yaml.safe_load(overlay_path.read_text()) or {}

    for txt_file in ["hypothesis.txt", "conclusion.txt"]:
        path = exp_dir / txt_file
        if path.exists():
            result[txt_file.replace(".txt", "")] = path.read_text().strip()

    results_path = exp_dir / "results.json"
    if results_path.exists():
        result["results"] = json.loads(results_path.read_text())

    return result
