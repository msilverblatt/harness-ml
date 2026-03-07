"""CLI entry point for harness-studio."""
from __future__ import annotations

import click


@click.command()
@click.option("--port", default=8421, help="Port to serve on")
@click.option("--project-dir", default=".", help="HarnessML project directory")
@click.option("--db", default=None, help="Path to events.db (overrides per-project default)")
def main(port: int, project_dir: str, db: str | None):
    """Launch Harness Studio dashboard.

    To connect MCP events from other projects, set HARNESS_STUDIO_DB
    in your .mcp.json env block to the same --db path.
    """
    import os
    from pathlib import Path

    import uvicorn
    from harnessml.studio.server import app

    resolved_project = str(Path(project_dir).resolve())
    app.state.project_dir = resolved_project

    if db:
        resolved_db = str(Path(db).resolve())
    else:
        resolved_db = os.environ.get("HARNESS_STUDIO_DB") or str(
            Path(resolved_project) / ".studio" / "events.db"
        )
    app.state.db_path = resolved_db

    click.echo(f"Studio  → http://127.0.0.1:{port}")
    click.echo(f"Project → {resolved_project}")
    click.echo(f"Events  → {resolved_db}")
    click.echo()
    click.echo("To send MCP events here, add to .mcp.json:")
    click.echo(f'  "env": {{"HARNESS_STUDIO_DB": "{resolved_db}"}}')
    click.echo()

    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
