def test_version_tree(client):
    resp = client.get("/api/versions/tree")
    assert resp.status_code == 200
    data = resp.json()
    assert "current" in data
    assert "versions" in data
    assert len(data["versions"]) >= 1
    assert data["current"] is not None


def test_version_detail(client):
    # Get the tree first to find a valid version id
    tree = client.get("/api/versions/tree").json()
    version_id = tree["versions"][0]["id"]

    resp = client.get(f"/api/versions/{version_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == version_id
    assert "metrics" in data
    assert "experiment_type" in data


def test_version_detail_404(client):
    resp = client.get("/api/versions/v999")
    assert resp.status_code == 404


def test_version_ancestry(client):
    tree = client.get("/api/versions/tree").json()
    version_id = tree["versions"][0]["id"]

    resp = client.get(f"/api/versions/{version_id}/ancestry")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version_id"] == version_id
    assert "ancestry" in data
    assert len(data["ancestry"]) >= 1


def test_version_ancestry_404(client):
    resp = client.get("/api/versions/v999/ancestry")
    assert resp.status_code == 404


def test_version_compare_404(client):
    resp = client.get("/api/versions/compare/v999/v998")
    assert resp.status_code == 404
