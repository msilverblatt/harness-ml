def test_pipeline_config(client):
    resp = client.get("/api/pipeline/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "project" in data
    assert "models" in data
    assert "ensemble" in data
    assert "features" in data
    assert data["project"]["task_type"] == "binary"
    assert data["project"]["target_column"] == "target"
    # Should have the lr model from baseline experiment
    assert "lr" in data["models"]


def test_pipeline_dag(client):
    resp = client.get("/api/pipeline/dag")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "waves" in data
    assert "errors" in data
    assert len(data["nodes"]) >= 1
    # lr should be in the nodes
    node_names = [n["name"] for n in data["nodes"]]
    assert "lr" in node_names
