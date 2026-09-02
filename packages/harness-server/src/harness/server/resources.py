from dataclasses import asdict

from harness.ml.models.registry import ModelRegistry
from harness.ml.tasks.registry import TaskRegistry
from harness.server.context import require_workspace
from protomcp import resource


@resource("harness://data/schema", description="Current clean dataset schema")
def data_schema() -> dict:
    return require_workspace().data.load_schema()


@resource("harness://versions/tree", description="Full experiment version tree")
def versions_tree() -> dict:
    workspace = require_workspace()
    return {
        "current": workspace.versions.get_current(),
        "versions": [asdict(item) for item in workspace.versions.list_versions()],
    }


@resource("harness://versions/current", description="Current version config and metrics")
def current_version() -> dict:
    workspace = require_workspace()
    version = workspace.versions.get_current()
    if not version:
        return {"current": None}
    meta = workspace.versions.get_version(version)
    return {"current": version, "meta": asdict(meta) if meta else None}


@resource("harness://models/available", description="Available model implementations")
def available_models() -> dict:
    return {"models": ModelRegistry.list_available()}


@resource("harness://tasks/supported", description="Supported task types and metrics")
def supported_tasks() -> dict:
    tasks = {}
    for name in TaskRegistry.list_available():
        task = TaskRegistry.get(name)
        tasks[name] = {"metrics": [metric.name for metric in task.metrics()]}
    return {"tasks": tasks}
