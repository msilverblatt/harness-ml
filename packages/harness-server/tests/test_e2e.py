import json

import harness.server.main  # noqa: F401
import numpy as np
import pandas as pd
import pytest
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

    # A source refresh must not compare new metrics to a stale parent baseline.
    refreshed = pd.read_csv(source)
    refreshed.loc[len(refreshed)] = [0.25, -0.5, 1]
    refreshed.to_csv(source, index=False)
    tools["data.run"]()
    post_refresh = tools["experiment.propose"](
        experiment_type="data_refresh",
        hypothesis="Re-establish the selected configuration after source refresh",
        params={},
    )
    assert post_refresh["parent_metrics"] == {}
    assert post_refresh["deltas"] == {}
    with pytest.raises(ValueError, match="different datasets"):
        tools["analyze.compare"](versions=["v001", "v003"])

    refreshed_child = tools["experiment.propose"](
        experiment_type="hyperparameter",
        hypothesis="Tune against the refreshed baseline",
        params={"model_name": "lr", "params": {"C": 0.25}},
    )
    assert refreshed_child["parent"] == "v003"
    assert refreshed_child["parent_metrics"]
    assert refreshed_child["deltas"]

    # Stored outputs must remain ordinary JSON-compatible protocol payloads.
    json.dumps(diagnostics)
