from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
from harness.server.context import require_workspace
from protomcp import action, tool_group


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


@tool_group("analyze", description="Analyze and compare experiment results.")
class AnalyzeTools:
    @action(
        "diagnostics", description="Read metrics, diagnostics, and evals for a version."
    )
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
        workspace = require_workspace()
        run_dir = workspace._root / "versions" / diagnostics["version"] / "run"
        native = _read_json(run_dir / "explainability.json")
        return {
            "version": diagnostics["version"],
            "native_feature_importance": native,
            "ensemble_coefficients": diagnostics["diagnostics"].get(
                "meta_coefficients", {}
            ),
        }

    @action(
        "discover",
        description=(
            "Rank existing numeric features and automatically search pairwise "
            "product/difference feature candidates."
        ),
    )
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
            for name, value in correlations.abs()
            .sort_values(ascending=False)
            .head(limit)
            .items()
            if pd.notna(value)
        ]
        search_columns = [
            item["feature"]
            for item in suggestions[:8]
            if item["feature"].isidentifier()
        ]
        generated = []
        target_values = numeric[target]
        for left, right in itertools.combinations(search_columns, 2):
            candidates = {
                f"{left}_x_{right}": (
                    numeric[left] * numeric[right],
                    f"{left} * {right}",
                ),
                f"{left}_minus_{right}": (
                    numeric[left] - numeric[right],
                    f"{left} - {right}",
                ),
            }
            for name, (values, expression) in candidates.items():
                score = values.replace([np.inf, -np.inf], np.nan).corr(target_values)
                if pd.notna(score):
                    generated.append(
                        {
                            "name": name,
                            "expression": expression,
                            "absolute_correlation": float(abs(score)),
                        }
                    )
        generated.sort(key=lambda item: item["absolute_correlation"], reverse=True)
        return {
            "target": target,
            "suggestions": suggestions,
            "generated_candidates": generated[:limit],
            "method": "univariate_and_pairwise_correlation_search",
        }
