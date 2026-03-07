"""WebSocket endpoint for live event streaming."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket):
    await websocket.accept()
    broadcaster = websocket.app.state.broadcaster
    queue = broadcaster.subscribe()
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        broadcaster.unsubscribe(queue)
