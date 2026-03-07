"""PipelineServer base class with auto-generated inspection tools.

Provides:
1. A tool registry with guardrail associations
2. Guardrail checking before tool execution
3. Auto-generated inspection tools from registry objects
4. Optional ``fastmcp`` integration (gracefully handles its absence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from harnessml.core.guardrails.base import Guardrail, GuardrailError

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

@dataclass
class ToolDef:
    """One registered tool with its metadata and guardrail bindings."""

    name: str
    fn: Callable
    description: str = ""
    guardrail_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PipelineServer
# ---------------------------------------------------------------------------

class PipelineServer:
    """Base class for building guardrail-protected pipeline servers.

    Parameters
    ----------
    name:
        Server name (used for logging and as the MCP server ID).
    guardrails:
        List of :class:`Guardrail` instances available to all tools.
    """

    def __init__(
        self,
        name: str,
        guardrails: list[Guardrail] | None = None,
    ) -> None:
        self.name = name
        self.guardrails = guardrails or []
        self._tools: dict[str, ToolDef] = {}

        # Index guardrails by name for fast lookup
        self._guardrail_map: dict[str, Guardrail] = {
            g.name: g for g in self.guardrails
        }

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def register_tool(
        self,
        name: str,
        fn: Callable,
        guardrails: list[str] | None = None,
        description: str = "",
    ) -> None:
        """Register a tool with optional guardrail names.

        Parameters
        ----------
        name:
            Unique tool name.
        fn:
            Callable implementing the tool.
        guardrails:
            List of guardrail names (matching :attr:`Guardrail.name`) that
            must pass before this tool executes.
        description:
            Human-readable description.
        """
        self._tools[name] = ToolDef(
            name=name,
            fn=fn,
            description=description,
            guardrail_names=guardrails or [],
        )

    # ------------------------------------------------------------------
    # Guardrail execution
    # ------------------------------------------------------------------

    def run_guardrails(
        self,
        tool_name: str,
        context: dict,
        human_override: bool = False,
    ) -> list[dict]:
        """Run all guardrails associated with a tool.

        Returns a list of check result dicts. Raises :class:`GuardrailError`
        on the first failing non-overridden guardrail.

        Parameters
        ----------
        tool_name:
            Name of the tool being invoked.
        context:
            Context dict passed to each guardrail's ``check()`` method.
        human_override:
            Whether to allow overridable guardrails to pass on failure.

        Returns
        -------
        list[dict]
            One entry per guardrail with keys: ``guardrail``, ``passed``.
        """
        tool_def = self._tools.get(tool_name)
        if tool_def is None:
            return []

        results: list[dict] = []
        for gname in tool_def.guardrail_names:
            guardrail = self._guardrail_map.get(gname)
            if guardrail is None:
                continue
            try:
                guardrail.check(context, human_override=human_override)
                results.append({"guardrail": gname, "passed": True})
            except GuardrailError:
                results.append({"guardrail": gname, "passed": False})
                raise

        return results

    # ------------------------------------------------------------------
    # Auto-generated inspection tools
    # ------------------------------------------------------------------

    def add_inspection_tools(
        self,
        config_resolver: Callable | None = None,
        feature_registry: Any | None = None,
        model_registry: Any | None = None,
        experiment_manager: Any | None = None,
    ) -> None:
        """Auto-generate read-only inspection tools from registries.

        Each non-None argument generates a corresponding ``show_*`` /
        ``list_*`` tool.

        Parameters
        ----------
        config_resolver:
            Callable returning a config dict.  Generates ``show_config``.
        feature_registry:
            Object with a ``list()`` method.  Generates ``list_features``.
        model_registry:
            Object with a ``list()`` method.  Generates ``list_models``.
        experiment_manager:
            Object with a ``list()``-like method.  Generates ``list_experiments``.
        """
        if config_resolver is not None:
            self.register_tool(
                name="show_config",
                fn=config_resolver,
                description="Show resolved pipeline configuration.",
            )

        if feature_registry is not None:
            def _list_features() -> Any:
                return feature_registry.list()
            self.register_tool(
                name="list_features",
                fn=_list_features,
                description="List registered features.",
            )

        if model_registry is not None:
            def _list_models() -> Any:
                return model_registry.list()
            self.register_tool(
                name="list_models",
                fn=_list_models,
                description="List registered models.",
            )

        if experiment_manager is not None:
            def _list_experiments() -> Any:
                return experiment_manager.list()
            self.register_tool(
                name="list_experiments",
                fn=_list_experiments,
                description="List experiments.",
            )

    # ------------------------------------------------------------------
    # FastMCP integration (optional)
    # ------------------------------------------------------------------

    def to_fastmcp(self) -> Any:
        """Create a ``FastMCP`` server from registered tools.

        Raises :class:`ImportError` if ``fastmcp`` is not installed.

        Returns
        -------
        fastmcp.FastMCP
            A server instance with all tools registered.
        """
        from fastmcp import FastMCP  # lazy import

        mcp = FastMCP(self.name)
        for tool_def in self._tools.values():
            mcp.tool(name=tool_def.name, description=tool_def.description)(
                tool_def.fn
            )
        return mcp
