"""CLI entry point for harness-studio."""
from __future__ import annotations

import click


@click.group(invoke_without_command=True)
@click.option("--port", default=8421, help="Port to serve on")
@click.option("--project-dir", default=None, help="HarnessML project directory (optional, Studio is project-agnostic)")
@click.option("--db", default=None, help="Path to events.db (overrides ~/.harnessml/events.db)")
@click.pass_context
def main(ctx: click.Context, port: int, project_dir: str | None, db: str | None):
    """Launch Harness Studio dashboard.

    Usually you don't need to run this directly — the MCP server
    auto-starts Studio when the first tool call is made.
    """
    ctx.ensure_object(dict)
    ctx.obj["port"] = port
    ctx.obj["project_dir"] = project_dir
    ctx.obj["db"] = db

    # If no subcommand given, run the server (backwards-compatible)
    if ctx.invoked_subcommand is None:
        _run_server(port, project_dir, db)


def _run_server(port: int, project_dir: str | None, db: str | None):
    """Start the uvicorn server."""
    import os
    from pathlib import Path

    import uvicorn
    from harnessml.studio.server import app

    if project_dir:
        resolved = str(Path(project_dir).resolve())
        app.state.project_dir = resolved
        app.state.single_project = True
    else:
        app.state.single_project = False
    if db:
        app.state.db_path = str(Path(db).resolve())
    elif os.environ.get("HARNESS_STUDIO_DB"):
        app.state.db_path = os.environ["HARNESS_STUDIO_DB"]
    # else: server.py defaults to ~/.harnessml/events.db

    uvicorn.run(app, host="127.0.0.1", port=port)


@main.command()
@click.option("--project-dir", required=True, help="HarnessML project directory")
@click.option("--output", "-o", default=None, help="Output HTML path (default: <project_name>-report.html)")
@click.option("--run-id", default=None, help="Scope report to a specific run ID")
def export(project_dir: str, output: str | None, run_id: str | None):
    """Export a self-contained static HTML dashboard report."""
    from pathlib import Path

    from harnessml.studio.export import export_html

    project_path = Path(project_dir).resolve()
    if not project_path.exists():
        raise click.ClickException(f"Project directory not found: {project_path}")

    if output is None:
        output = f"{project_path.name}-report.html"

    output_path = Path(output).resolve()
    result = export_html(project_path, output_path, run_id=run_id)
    click.echo(f"Exported dashboard to {result}")


if __name__ == "__main__":
    main()
