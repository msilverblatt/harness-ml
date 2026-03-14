"""HarnessML MCP server — protomcp entry point.

Run via: pmcp dev server.py
"""
from __future__ import annotations

import harnessml.plugin.handlers.competitions  # noqa: F401
import harnessml.plugin.handlers.config  # noqa: F401
import harnessml.plugin.handlers.data  # noqa: F401
import harnessml.plugin.handlers.experiment_workflow  # noqa: F401
import harnessml.plugin.handlers.experiments  # noqa: F401
import harnessml.plugin.handlers.features  # noqa: F401

# Import all handler modules — each registers a @tool_group
import harnessml.plugin.handlers.models  # noqa: F401
import harnessml.plugin.handlers.notebook  # noqa: F401
import harnessml.plugin.handlers.pipeline  # noqa: F401

# Import infrastructure — registrations happen at import time
import harnessml.plugin.pmcp_context  # noqa: F401
import harnessml.plugin.pmcp_middleware  # noqa: F401
import harnessml.plugin.pmcp_sidecar  # noqa: F401
import harnessml.plugin.pmcp_telemetry  # noqa: F401
from protomcp.runner import run

run()
