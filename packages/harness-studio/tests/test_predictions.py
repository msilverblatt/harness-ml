def test_predictions(client):
    tree = client.get("/api/versions/tree").json()
    version_id = tree["versions"][0]["id"]

    resp = client.get(f"/api/predictions/{version_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version_id"] == version_id
    assert "total" in data
    assert "rows" in data
    assert "columns" in data
    assert data["total"] > 0
    assert data["page"] == 1


def test_predictions_pagination(client):
    tree = client.get("/api/versions/tree").json()
    version_id = tree["versions"][0]["id"]

    resp = client.get(f"/api/predictions/{version_id}?page=1&page_size=5")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["rows"]) <= 5
    assert data["page_size"] == 5


def test_production_inference(client):
    version_id = client.get("/api/versions/tree").json()["current"]
    response = client.post(
        f"/api/predictions/{version_id}/infer",
        json=[{"a": 0.2, "b": -0.1}, {"a": 1.0, "b": 0.5}],
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 2
    assert all(0 <= row["prediction"] <= 1 for row in predictions)


def test_predictions_404(client):
    resp = client.get("/api/predictions/v999")
    assert resp.status_code == 404


def test_predictions_distribution(client):
    tree = client.get("/api/versions/tree").json()
    version_id = tree["versions"][0]["id"]

    resp = client.get(f"/api/predictions/{version_id}/distribution")
    assert resp.status_code == 200
    data = resp.json()
    assert data["version_id"] == version_id
    assert "distributions" in data


def test_predictions_distribution_404(client):
    resp = client.get("/api/predictions/v999/distribution")
    assert resp.status_code == 404
