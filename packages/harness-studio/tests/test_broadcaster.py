"""Tests for event broadcaster."""
from __future__ import annotations

import asyncio

from harnessml.studio.broadcaster import EventBroadcaster


class TestEventBroadcaster:
    def test_subscribe_creates_queue(self):
        b = EventBroadcaster()
        q = b.subscribe()
        assert isinstance(q, asyncio.Queue)
        assert b.subscriber_count == 1

    def test_unsubscribe_removes_queue(self):
        b = EventBroadcaster()
        q = b.subscribe()
        b.unsubscribe(q)
        assert b.subscriber_count == 0

    def test_unsubscribe_nonexistent_is_safe(self):
        b = EventBroadcaster()
        q: asyncio.Queue = asyncio.Queue()
        b.unsubscribe(q)  # should not raise

    def test_notify_sends_to_all_subscribers(self):
        b = EventBroadcaster()
        q1 = b.subscribe()
        q2 = b.subscribe()
        b.notify({"tool": "models", "action": "add"})
        assert q1.get_nowait() == {"tool": "models", "action": "add"}
        assert q2.get_nowait() == {"tool": "models", "action": "add"}

    def test_notify_no_subscribers_is_safe(self):
        b = EventBroadcaster()
        b.notify({"tool": "test"})  # should not raise

    def test_bounded_queue_drops_oldest(self):
        b = EventBroadcaster(maxsize=3)
        q = b.subscribe()
        # Fill beyond capacity
        for i in range(5):
            b.notify({"event": i})
        # Should have most recent 3
        events = []
        while not q.empty():
            events.append(q.get_nowait())
        assert len(events) == 3
        assert events[0]["event"] == 2  # oldest dropped
        assert events[1]["event"] == 3
        assert events[2]["event"] == 4
