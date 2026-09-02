def test_diagnostics(client):
    tree = client.get("/api/versions/tree").json()
    version_id = tree["versions"][0]["id"]

    resp = client.get(f"/api/diagnostics/{version_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version_id"] == version_id
    assert "metrics" in data
    assert "diagnostics" in data
    # metrics should have actual values from the experiment
    assert len(data["metrics"]) > 0


def test_diagnostics_404(client):
    resp = client.get("/api/diagnostics/v999")
    assert resp.status_code == 404
