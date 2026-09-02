from dataclasses import asdict

from harness.server.context import require_workspace
from protomcp import action, tool_group


@tool_group("versions", description="Inspect and navigate the experiment version tree.")
class VersionTools:
    @action("list", description="List versions and the current pointer.")
    def list_versions(self) -> dict:
        workspace = require_workspace()
        return {
            "current": workspace.versions.get_current(),
            "versions": [asdict(item) for item in workspace.versions.list_versions()],
        }

    @action("show", description="Show one version and its config diff.")
    def show(self, version: str) -> dict:
        workspace = require_workspace()
        meta = workspace.versions.get_version(version)
        if meta is None:
            raise ValueError(f"Version not found: {version}")
        diff_path = workspace._root / "versions" / version / "diff.yaml"
        return {"meta": asdict(meta), "diff": diff_path.read_text() if diff_path.exists() else ""}

    @action("switch", description="Switch the working config to a version.")
    def switch(self, version: str) -> dict:
        workspace = require_workspace()
        workspace.switch_version(version)
        return workspace.status()

    @action("ancestry", description="Show ancestry from baseline to a version.")
    def ancestry(self, version: str | None = None) -> dict:
        workspace = require_workspace()
        version_id = version or workspace.versions.get_current()
        if not version_id:
            raise ValueError("No current version")
        return {
            "version": version_id,
            "ancestry": [
                asdict(item) for item in workspace.versions.ancestry(version_id)
            ],
        }
