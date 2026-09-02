def test_data_schema(client):
    resp = client.get("/api/data/schema")
    assert resp.status_code == 200
    data = resp.json()
    # schema.json should be a dict with column info
    assert isinstance(data, dict)


def test_data_profile(client):
    resp = client.get("/api/data/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "row_count" in data
    assert "column_count" in data
    assert "columns" in data
    assert data["row_count"] > 0
    assert data["column_count"] > 0
