from harness.server.context import require_workspace
from protomcp import action, tool_group


@tool_group("experiment", description="Run hypothesis-driven typed experiments.")
class ExperimentTools:
    @action("propose", description="Apply, train, evaluate, and version an experiment.")
    def propose(
        self,
        experiment_type: str,
        hypothesis: str,
        params: dict,
        parent: str | None = None,
    ) -> dict:
        workspace = require_workspace()
        result = workspace.run_experiment(
            experiment_type, hypothesis, params, parent=parent
        )
        version = workspace.versions.get_current()
        meta = workspace.versions.get_version(version) if version else None
        parent_meta = (
            workspace.versions.get_version(meta.parent)
            if meta and meta.parent
            else None
        )
        parent_metrics = parent_meta.metrics if parent_meta else {}
        return {
            "version": version,
            "parent": meta.parent if meta else None,
            "metrics": result.metrics,
            "parent_metrics": parent_metrics,
            "deltas": {
                key: value - parent_metrics[key]
                for key, value in result.metrics.items()
                if key in parent_metrics
            },
            "per_fold": result.per_fold_metrics,
            "models_trained": result.models_trained,
            "models_cached": result.models_cached,
            "models_failed": result.models_failed,
            "duration_s": result.duration_s,
        }

    @action("conclude", description="Record the conclusion and verdict for a version.")
    def conclude(
        self, conclusion: str, verdict: str, version: str | None = None
    ) -> dict:
        workspace = require_workspace()
        version_id = version or workspace.versions.get_current()
        if not version_id:
            raise ValueError("No current version to conclude")
        workspace.conclude_experiment(version_id, conclusion, verdict)
        return {"version": version_id, "conclusion": conclusion, "verdict": verdict}
