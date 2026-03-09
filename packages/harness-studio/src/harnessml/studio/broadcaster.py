"""Event broadcaster for WebSocket fan-out."""
from __future__ import annotations

import asyncio


class EventBroadcaster:
    """Fan-out new events to connected WebSocket clients."""

    def __init__(self, maxsize: int = 100):
        self._maxsize = maxsize
        self._subscribers: list[asyncio.Queue] = []

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def notify(self, event: dict) -> None:
        for q in self._subscribers:
            if q.full():
                try:
                    q.get_nowait()  # Drop oldest
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
