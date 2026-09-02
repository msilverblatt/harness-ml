"""Harness MCP server entry point."""

# Imports register tools and resources with protomcp.
import harness.server.resources
import harness.server.tools.analyze
import harness.server.tools.data
import harness.server.tools.experiment
import harness.server.tools.project
import harness.server.tools.versions
import harness.server.tools.workspace  # noqa: F401


def main() -> None:
    from protomcp.runner import run

    run()


if __name__ == "__main__":
    main()
