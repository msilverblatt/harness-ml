def test_monitor_events(client):
    resp = client.get("/api/monitor/events")
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data
    assert isinstance(data["events"], list)


def test_monitor_stats(client):
    resp = client.get("/api/monitor/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "errors" in data
    assert isinstance(data["total"], int)
    assert isinstance(data["errors"], int)
