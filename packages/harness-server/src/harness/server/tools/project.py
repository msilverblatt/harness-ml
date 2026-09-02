from pathlib import Path

from protomcp import action, tool_group

from harness.server.context import initialize_workspace


@tool_group("project", description="Initialize a Harness ML project.")
class ProjectTools:
    @action("init", description="Initialize a new Harness workspace.")
    def init(
        self,
        task_type: str = "binary",
        target_column: str = "target",
        project_name: str | None = None,
        path: str | None = None,
    ) -> dict:
        if task_type not in {"binary", "multiclass", "regression"}:
            raise ValueError("task_type must be binary, multiclass, or regression")
        destination = Path(path).expanduser() if path else Path.cwd()
        if project_name:
            destination = destination / project_name
        workspace = initialize_workspace(destination, task_type, target_column)
        return {
            "workspace": str(workspace._root),
            "task_type": task_type,
            "target_column": target_column,
        }
