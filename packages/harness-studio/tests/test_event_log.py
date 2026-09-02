import pytest
from harness.studio.event_log import EventLog


@pytest.fixture
def event_log(tmp_path):
    return EventLog(tmp_path / "test_events.db")


def test_log_and_query(event_log):
    event_id = event_log.log(
        tool="test_tool",
        action="test_action",
        params={"key": "value"},
        result={"ok": True},
        duration_ms=42.0,
        status="ok",
    )
    assert event_id is not None

    events = event_log.query(limit=10)
    assert len(events) == 1
    assert events[0]["tool"] == "test_tool"
    assert events[0]["action"] == "test_action"
    assert events[0]["status"] == "ok"


def test_query_filter_by_tool(event_log):
    event_log.log(tool="tool_a", action="act1")
    event_log.log(tool="tool_b", action="act2")

    events = event_log.query(tool="tool_a")
    assert len(events) == 1
    assert events[0]["tool"] == "tool_a"


def test_stats(event_log):
    event_log.log(tool="t", action="a", status="ok")
    event_log.log(tool="t", action="a", status="error")
    event_log.log(tool="t", action="a", status="ok")

    stats = event_log.stats()
    assert stats["total"] == 3
    assert stats["errors"] == 1


def test_pagination(event_log):
    for i in range(10):
        event_log.log(tool="t", action=f"act_{i}")

    page1 = event_log.query(limit=3, offset=0)
    page2 = event_log.query(limit=3, offset=3)
    assert len(page1) == 3
    assert len(page2) == 3
    # No overlap
    ids1 = {e["id"] for e in page1}
    ids2 = {e["id"] for e in page2}
    assert ids1.isdisjoint(ids2)
