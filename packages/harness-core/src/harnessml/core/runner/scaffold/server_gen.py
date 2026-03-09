"""MCP server generator from YAML config.

Reads ServerDef from project config and generates a GeneratedServer
with execution tools (subprocess wrappers) and inspection tools
(config introspection via validate_project).
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harnessml.core.runner.schema import GuardrailDef, ServerDef, ServerToolDef
from harnessml.core.runner.validation.validator import validate_project


@dataclass
class ToolSpec:
    """Specification for a single MCP tool."""

    name: str
    description: str
    fn: Callable
    args: list[str] = field(default_factory=list)
    guardrails: list[str] = field(default_factory=list)


@dataclass
class GeneratedServer:
    """A generated MCP server with named tools."""

    name: str
    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def to_fastmcp(self):
        """Convert to FastMCP server instance. Lazily imports fastmcp."""
        from fastmcp import FastMCP

        mcp = FastMCP(self.name)
        for name, spec in self.tools.items():
            mcp.tool(name=name, description=spec.description)(spec.fn)
        return mcp


def _check_guardrails(
    guardrails: list[str],
    guardrail_config: GuardrailDef | None,
    kwargs: dict[str, Any],
    tool_name: str,
) -> str | None:
    """Check guardrails before executing a tool command.

    Parameters
    ----------
    guardrails : list[str]
        List of guardrail names configured on the tool.
    guardrail_config : GuardrailDef | None
        Project-level guardrail configuration.
    kwargs : dict
        Keyword arguments passed to the tool.
    tool_name : str
        Name of the tool being executed.

    Returns
    -------
    str | None
        Error message if a guardrail fails, or None if all pass.
    """
    if not guardrails or guardrail_config is None:
        return None

    for guard in guardrails:
        if guard == "naming_check":
            # Validate naming patterns for arguments that look like identifiers
            pattern = guardrail_config.naming_pattern
            if pattern is not None:
                import re
                for key, value in kwargs.items():
                    if key in ("experiment_id", "name", "id") and isinstance(value, str):
                        if not re.match(pattern, value):
                            return (
                                f"Guardrail '{guard}' failed: "
                                f"'{value}' does not match pattern '{pattern}'"
                            )

        elif guard == "leakage_check":
            # Check for feature leakage denylist
            denylist = guardrail_config.feature_leakage_denylist
            if denylist:
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        for denied in denylist:
                            if denied in value:
                                return (
                                    f"Guardrail '{guard}' failed: "
                                    f"'{denied}' is in the feature leakage denylist"
                                )

        elif guard == "rate_limit":
            # Rate limiting — lightweight check (actual enforcement would
            # require persistent state; this is a placeholder for the pattern)
            pass

        # Unknown guardrails are silently skipped (forward-compatible)

    return None


def _make_execution_tool(
    tool_name: str,
    tool_def: ServerToolDef,
    config_dir: Path,
    guardrail_config: GuardrailDef | None = None,
) -> ToolSpec:
    """Create an execution tool that runs a subprocess command.

    The generated async function builds a command from ServerToolDef.command
    plus any kwargs, runs it with the configured timeout, and returns a
    JSON result dict with status/stdout/stderr.

    If the tool has guardrails configured, they are checked before
    executing the command. If any guardrail fails, the command is not
    executed and the guardrail error is returned.
    """
    command_template = tool_def.command
    tool_args = list(tool_def.args)
    guardrails = list(tool_def.guardrails)
    timeout = tool_def.timeout
    description = tool_def.description or f"Run {tool_name}"
    _guardrail_config = guardrail_config

    async def _execute(**kwargs: Any) -> str:
        """Execute the tool command via subprocess."""
        # Check guardrails before executing
        guardrail_error = _check_guardrails(
            guardrails, _guardrail_config, kwargs, tool_name
        )
        if guardrail_error is not None:
            result = {
                "status": "guardrail_failed",
                "returncode": -1,
                "stdout": "",
                "stderr": guardrail_error,
            }
            return json.dumps(result, indent=2)

        # Build command parts
        parts = command_template.split()

        # Add configured args
        for arg in tool_args:
            parts.append(arg)

        # Add kwargs as --key=value flags
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    parts.append(f"--{key}")
            else:
                parts.append(f"--{key}={value}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(config_dir.parent),
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
            result = {
                "status": "success" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            }
        except asyncio.TimeoutError:
            result = {
                "status": "timeout",
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
            }
        except Exception as exc:
            result = {
                "status": "error",
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
            }

        return json.dumps(result, indent=2)

    return ToolSpec(
        name=tool_name,
        description=description,
        fn=_execute,
        args=tool_args,
        guardrails=guardrails,
    )


# -----------------------------------------------------------------------
# Inspection tool builders
# -----------------------------------------------------------------------

_INSPECTION_TOOL_NAMES = {
    "show_config",
    "list_models",
    "list_features",
    "list_experiments",
}


def _make_show_config_tool(config_dir: Path) -> ToolSpec:
    """Inspection tool: dump resolved config as JSON."""

    async def _show_config(section: str | None = None) -> str:
        result = validate_project(config_dir)
        if not result.valid:
            return f"Config validation failed:\n{result.format()}"

        data = result.config.model_dump()
        if section is not None:
            if section in data:
                data = data[section]
            else:
                return f"Unknown section: {section!r}. Available: {sorted(data.keys())}"

        return json.dumps(data, indent=2, default=str)

    return ToolSpec(
        name="show_config",
        description="Show the resolved project configuration.",
        fn=_show_config,
    )


def _make_list_models_tool(config_dir: Path) -> ToolSpec:
    """Inspection tool: list models with type, status, and feature count."""

    async def _list_models() -> str:
        result = validate_project(config_dir)
        if not result.valid:
            return f"Config validation failed:\n{result.format()}"

        lines: list[str] = []
        for name, model_def in sorted(result.config.models.items()):
            status = "active" if model_def.active else "inactive"
            n_features = len(model_def.features)
            lines.append(
                f"  {name}: type={model_def.type}, {status}, {n_features} features"
            )
        return "\n".join(lines) if lines else "No models configured."

    return ToolSpec(
        name="list_models",
        description="List all models with type, active status, and feature count.",
        fn=_list_models,
    )


def _make_list_features_tool(config_dir: Path) -> ToolSpec:
    """Inspection tool: list declared features."""

    async def _list_features() -> str:
        result = validate_project(config_dir)
        if not result.valid:
            return f"Config validation failed:\n{result.format()}"

        features = result.config.features
        if not features:
            return "No features declared in config."

        lines: list[str] = []
        for name, feat in sorted(features.items()):
            lines.append(
                f"  {name}: category={feat.category}, level={feat.level}, "
                f"columns={feat.columns}"
            )
        return "\n".join(lines)

    return ToolSpec(
        name="list_features",
        description="List all declared features with category and level.",
        fn=_list_features,
    )


def _make_list_experiments_tool(config_dir: Path) -> ToolSpec:
    """Inspection tool: list experiment directories."""

    async def _list_experiments() -> str:
        result = validate_project(config_dir)
        if not result.valid:
            return f"Config validation failed:\n{result.format()}"

        exp_cfg = result.config.experiments
        if exp_cfg is None:
            return "No experiments section in config."

        exp_dir = Path(exp_cfg.experiments_dir or "experiments")
        if not exp_dir.exists():
            return "No experiments directory found."

        experiments = sorted(
            p.name
            for p in exp_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        if not experiments:
            return "No experiments found."

        return "\n".join(f"  {name}" for name in experiments)

    return ToolSpec(
        name="list_experiments",
        description="List all experiment directories.",
        fn=_list_experiments,
    )


_INSPECTION_BUILDERS: dict[str, Callable[[Path], ToolSpec]] = {
    "show_config": _make_show_config_tool,
    "list_models": _make_list_models_tool,
    "list_features": _make_list_features_tool,
    "list_experiments": _make_list_experiments_tool,
}


# -----------------------------------------------------------------------
# Main generator
# -----------------------------------------------------------------------

def generate_server(
    config: ServerDef,
    config_dir: Path,
    guardrails: GuardrailDef | None = None,
) -> GeneratedServer:
    """Generate an MCP server from a ServerDef configuration.

    Parameters
    ----------
    config:
        The server configuration from project YAML.
    config_dir:
        Path to the config directory (used for subprocess cwd and
        inspection tool validation).
    guardrails:
        Optional project-level guardrail configuration. If provided,
        execution tools with guardrail references will enforce them
        before running commands.

    Returns
    -------
    GeneratedServer
        A server with execution and inspection tools registered.
    """
    config_dir = Path(config_dir)
    server = GeneratedServer(name=config.name)

    # Register execution tools
    for tool_name, tool_def in config.tools.items():
        spec = _make_execution_tool(
            tool_name, tool_def, config_dir, guardrail_config=guardrails,
        )
        server.tools[tool_name] = spec

    # Register inspection tools
    for inspection_name in config.inspection:
        builder = _INSPECTION_BUILDERS.get(inspection_name)
        if builder is not None:
            spec = builder(config_dir)
            server.tools[inspection_name] = spec

    return server
