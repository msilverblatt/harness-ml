def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "workspace" in data


def test_frontend_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<div id=\"root\"></div>" in response.text
