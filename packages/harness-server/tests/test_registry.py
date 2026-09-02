import harness.server.main  # noqa: F401
from protomcp import get_registered_tools
from protomcp.resource import get_registered_resources


def test_documented_tools_are_registered():
    names = {tool.name for tool in get_registered_tools()}
    assert names == {
        "project.init",
        "data.add_source",
        "data.transform",
        "data.run",
        "data.profile",
        "data.inspect",
        "experiment.propose",
        "experiment.conclude",
        "analyze.diagnostics",
        "analyze.explain",
        "analyze.compare",
        "analyze.discover",
        "versions.list",
        "versions.show",
        "versions.switch",
        "versions.ancestry",
        "workspace.open",
    }


def test_documented_resources_are_registered():
    uris = {resource.uri for resource in get_registered_resources()}
    assert uris == {
        "harness://data/schema",
        "harness://versions/tree",
        "harness://versions/current",
        "harness://models/available",
        "harness://tasks/supported",
    }
