"""Experiment history endpoints."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Request

router = APIRouter(tags=["experiments"])


@router.get("/experiments")
async def list_experiments(request: Request):
    project_dir = Path(request.app.state.project_dir)
    journal_path = project_dir / "experiments" / "journal.jsonl"
    if not journal_path.exists():
        return []
    experiments = []
    for line in journal_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            experiments.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    experiments.reverse()  # newest first
    return experiments


@router.get("/experiments/{experiment_id}")
async def get_experiment(request: Request, experiment_id: str):
    project_dir = Path(request.app.state.project_dir)
    exp_dir = project_dir / "experiments" / experiment_id
    if not exp_dir.exists():
        return {"error": f"Experiment {experiment_id} not found"}

    result = {"id": experiment_id}

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
