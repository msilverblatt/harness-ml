"""Harness MCP server entry point."""

# Imports register tools and resources with protomcp.
import harness.server.resources  # noqa: F401
import harness.server.tools.analyze  # noqa: F401
import harness.server.tools.data  # noqa: F401
import harness.server.tools.experiment  # noqa: F401
import harness.server.tools.project  # noqa: F401
import harness.server.tools.versions  # noqa: F401
import harness.server.tools.workspace  # noqa: F401


def main() -> None:
    from protomcp.runner import run

    run()


if __name__ == "__main__":
    main()
