"""Sidecar: auto-start Harness Studio alongside the MCP server."""
from __future__ import annotations

import os
import sys

from protomcp import sidecar

_STUDIO_PORT = int(os.environ.get("HARNESS_STUDIO_PORT", "8421"))


@sidecar(
    name="harness-studio",
    command=[sys.executable, "-m", "harnessml.studio.cli", "--port", str(_STUDIO_PORT)],
    health_check=f"http://localhost:{_STUDIO_PORT}/api/health",
    start_on="first_tool_call",
    health_timeout=5.0,
)
def studio_sidecar():
    """Harness Studio companion dashboard."""
    pass
