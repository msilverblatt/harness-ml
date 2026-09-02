import json

import numpy as np
import pandas as pd

import harness.server.main  # noqa: F401
from protomcp import get_registered_tools
from protomcp.resource import get_registered_resources


def _tools():
    return {tool.name: tool.handler for tool in get_registered_tools()}


def _resources():
    return {resource.uri: resource.handler for resource in get_registered_resources()}


def test_full_registered_agent_workflow(tmp_path):
    """Drive the registered MCP handlers through a real training workflow."""
    tools = _tools()
    workspace = tmp_path / "project"
    source = tmp_path / "dataset.csv"
    rng = np.random.RandomState(42)
    n = 160
    signal = rng.randn(n)
    pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.randn(n),
            "target": (signal + rng.randn(n) * 0.7 > 0).astype(int),
        }
    ).to_csv(source, index=False)

    initialized = tools["project.init"](
        path=str(workspace), task_type="binary", target_column="target"
    )
    assert initialized["workspace"] == str(workspace)

    tools["data.add_source"](name="training", path=str(source))
    pipeline = tools["data.run"]()
    assert pipeline["row_count"] == n
    assert tools["data.profile"]()["row_count"] == n
    assert len(tools["data.inspect"](rows=3)["rows"]) == 3

    baseline = tools["experiment.propose"](
        experiment_type="baseline",
        hypothesis="Establish a logistic baseline",
        params={"models": {"lr": {"model_type": "logistic"}}},
    )
    assert baseline["version"] == "v001"
    assert baseline["metrics"]["accuracy"] > 0.6
    tools["experiment.conclude"](
        conclusion="The baseline learns the synthetic signal.", verdict="improved"
    )

    child = tools["experiment.propose"](
        experiment_type="hyperparameter",
        hypothesis="Stronger regularization may improve generalization",
        params={"model_name": "lr", "params": {"C": 0.5}},
    )
    assert child["version"] == "v002"
    assert child["parent"] == "v001"

    versions = tools["versions.list"]()
    assert versions["current"] == "v002"
    assert len(versions["versions"]) == 2
    assert len(tools["versions.ancestry"]()["ancestry"]) == 2
    assert tools["analyze.compare"](versions=["v001", "v002"])["comparisons"]
    diagnostics = tools["analyze.diagnostics"]()
    assert diagnostics["eval_report"]["dimensions"]
    discovery = tools["analyze.discover"]()
    assert discovery["suggestions"]
    assert discovery["generated_candidates"]
    explanation = tools["analyze.explain"]()
    assert explanation["native_feature_importance"]["aggregate"]

    tools["versions.switch"](version="v001")
    assert tools["versions.list"]()["current"] == "v001"

    resources = _resources()
    assert resources["harness://versions/tree"]()["current"] == "v001"
    assert "binary" in resources["harness://tasks/supported"]()["tasks"]
    assert resources["harness://data/schema"]()["row_count"] == n

    # Stored outputs must remain ordinary JSON-compatible protocol payloads.
    json.dumps(diagnostics)
