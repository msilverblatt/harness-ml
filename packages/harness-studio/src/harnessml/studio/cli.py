"""CLI entry point for harness-studio."""
from __future__ import annotations

import click


@click.command()
@click.option("--port", default=8421, help="Port to serve on")
@click.option("--project-dir", default=None, help="HarnessML project directory (optional, Studio is project-agnostic)")
@click.option("--db", default=None, help="Path to events.db (overrides ~/.harnessml/events.db)")
def main(port: int, project_dir: str | None, db: str | None):
    """Launch Harness Studio dashboard.

    Usually you don't need to run this directly — the MCP server
    auto-starts Studio when the first tool call is made.
    """
    import os
    from pathlib import Path

    import uvicorn
    from harnessml.studio.server import app

    if project_dir:
        app.state.project_dir = str(Path(project_dir).resolve())
    if db:
        app.state.db_path = str(Path(db).resolve())
    elif os.environ.get("HARNESS_STUDIO_DB"):
        app.state.db_path = os.environ["HARNESS_STUDIO_DB"]
    # else: server.py defaults to ~/.harnessml/events.db

    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
