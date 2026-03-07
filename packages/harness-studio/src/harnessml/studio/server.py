"""Harness Studio — companion dashboard server."""
from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Harness Studio")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
