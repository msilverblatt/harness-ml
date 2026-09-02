from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
from protomcp import action, tool_group

from harness.server.context import require_workspace


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


@tool_group("analyze", description="Analyze and compare experiment results.")
class AnalyzeTools:
    @action("diagnostics", description="Read metrics, diagnostics, and evals for a version.")
    def diagnostics(self, version: str | None = None) -> dict:
        workspace = require_workspace()
        version_id = version or workspace.versions.get_current()
        if not version_id:
            raise ValueError("No current version")
        run_dir = workspace._root / "versions" / version_id / "run"
        if not run_dir.exists():
            raise ValueError(f"Version run not found: {version_id}")
        return {
            "version": version_id,
            "metrics": _read_json(run_dir / "metrics.json"),
            "diagnostics": _read_json(run_dir / "diagnostics.json"),
            "eval_report": _read_json(run_dir / "eval_report.json"),
        }

    @action("compare", description="Compare metrics for two or more versions.")
    def compare(self, versions: list[str]) -> dict:
        if len(versions) < 2:
            raise ValueError("Provide at least two versions")
        workspace = require_workspace()
        baseline = versions[0]
        comparisons = {}
        for candidate in versions[1:]:
            comparisons[f"{baseline}..{candidate}"] = workspace.versions.compare(
                baseline, candidate
            )
        return {"baseline": baseline, "comparisons": comparisons}

    @action("explain", description="Return available ensemble/model attribution data.")
    def explain(self, version: str | None = None) -> dict:
        diagnostics = self.diagnostics(version)
        coefficients = diagnostics["diagnostics"].get("meta_coefficients", {})
        return {
            "version": diagnostics["version"],
            "method": "ensemble_coefficients",
            "coefficients": coefficients,
            "notice": (
                "Model-level SHAP artifacts are not available for this version."
                if not coefficients
                else "Coefficients describe the fitted ensemble meta-learner."
            ),
        }

    @action("discover", description="Suggest candidate numeric features using target correlation.")
    def discover(self, limit: int = 20) -> dict:
        workspace = require_workspace()
        frame = workspace.data.load_clean_data()
        project = workspace.config.read_project()
        target = project.target_column
        if target not in frame:
            raise ValueError(f"Target column not found: {target}")
        numeric = frame.select_dtypes(include="number")
        if target not in numeric:
            return {"target": target, "suggestions": []}
        correlations = numeric.corr(numeric_only=True)[target].drop(labels=[target])
        suggestions = [
            {"feature": name, "absolute_correlation": float(value)}
            for name, value in correlations.abs().sort_values(ascending=False).head(limit).items()
            if pd.notna(value)
        ]
        return {"target": target, "suggestions": suggestions}
