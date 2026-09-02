from harness.server.context import set_workspace
from protomcp import action, tool_group


@tool_group("workspace", description="Open and inspect Harness workspaces.")
class WorkspaceTools:
    @action("open", description="Open an existing Harness workspace.")
    def open(self, path: str) -> dict:
        workspace = set_workspace(path)
        return workspace.status()
