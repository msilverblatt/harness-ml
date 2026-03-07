"""Experiment history endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
async def list_experiments(request: Request):
    project_dir = Path(request.app.state.project_dir)
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
    return experiments


@router.get("/experiments/{experiment_id}")
async def get_experiment(request: Request, experiment_id: str):
    project_dir = Path(request.app.state.project_dir)
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
