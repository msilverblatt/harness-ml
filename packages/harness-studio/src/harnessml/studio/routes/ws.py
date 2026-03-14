"""WebSocket endpoint for live event streaming."""
from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket):
    await websocket.accept()
    # Optional project_dir filter from query param
    filter_project_dir = websocket.query_params.get("project_dir", "")
    broadcaster = websocket.app.state.broadcaster
    queue = broadcaster.subscribe()
    try:
        while True:
            event = await queue.get()
            # Skip events that don't match the requested project_dir
            if filter_project_dir:
                event_dir = event.get("project_dir", "")
                if event_dir and event_dir != filter_project_dir:
                    continue
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        broadcaster.unsubscribe(queue)
