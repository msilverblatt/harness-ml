"""CLI entry point for harness-studio."""
from __future__ import annotations

import click


@click.command()
@click.option("--port", default=8421, help="Port to serve on")
@click.option("--project-dir", default=".", help="HarnessML project directory")
def main(port: int, project_dir: str):
    """Launch Harness Studio dashboard."""
    from pathlib import Path

    import uvicorn
    from harnessml.studio.server import app
    app.state.project_dir = str(Path(project_dir).resolve())
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
