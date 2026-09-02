import importlib
import sys
from pathlib import Path

import click


@click.group()
def cli():
    """Harness — Agent-first ML platform."""


@cli.command()
@click.argument("project_name", required=False)
@click.option(
    "--task-type",
    type=click.Choice(["binary", "multiclass", "regression"]),
    default="binary",
)
@click.option("--target", default="target", help="Target column name")
def init(project_name, task_type, target):
    """Initialize a new Harness workspace."""
    from harness.app.workspace.manager import WorkspaceManager

    workspace_dir = Path.cwd() / project_name if project_name else Path.cwd()
    WorkspaceManager.init(workspace_dir, task_type=task_type, target_column=target)
    click.echo(f"Initialized workspace at {workspace_dir}")
    click.echo(f"  Task type: {task_type}")
    click.echo(f"  Target: {target}")


@cli.command()
def status():
    """Show workspace status."""
    from harness.app.workspace.discovery import find_workspace
    from harness.app.workspace.manager import WorkspaceManager

    ws_dir = find_workspace()
    if ws_dir is None:
        click.echo(
            "No harness workspace found. Run 'harness init' to create one.", err=True
        )
        sys.exit(1)

    ws = WorkspaceManager(ws_dir)
    info = ws.status()
    click.echo(f"Workspace: {info['workspace']}")
    click.echo(f"Current version: {info['current_version'] or 'none'}")
    click.echo(f"Models: {info['model_count']}")
    click.echo(f"Versions: {info['version_count']}")
    if info["metrics"]:
        click.echo("Metrics:")
        for k, v in info["metrics"].items():
            click.echo(f"  {k}: {v:.4f}")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--version", default=None, help="Version to use; defaults to current")
def predict(input_path, output_path, version):
    """Generate predictions from a CSV or Parquet file."""
    import pandas as pd
    from harness.app.workspace.discovery import find_workspace
    from harness.ml.runners.production import ProductionBundle

    workspace = find_workspace()
    if workspace is None:
        raise click.ClickException("No Harness workspace found")
    version_id = version or (workspace / "current").read_text().strip()
    bundle_path = workspace / "versions" / version_id / "run" / "model.bundle"
    if not bundle_path.exists():
        raise click.ClickException(f"Production bundle not found for {version_id}")
    frame = (
        pd.read_parquet(input_path)
        if input_path.suffix.lower() in {".parquet", ".pq"}
        else pd.read_csv(input_path)
    )
    bundle = ProductionBundle.load(bundle_path)
    predictions = bundle.predict(frame)
    if predictions.ndim == 1:
        output = pd.DataFrame({"prediction": predictions}, index=frame.index)
    else:
        output = pd.DataFrame(
            predictions,
            columns=[
                f"prediction_class_{index}" for index in range(predictions.shape[1])
            ],
            index=frame.index,
        )
    if bundle.conformal_radius is not None:
        output = bundle.predict_interval(frame)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".parquet", ".pq"}:
        output.to_parquet(output_path, index=False)
    else:
        output.to_csv(output_path, index=False)
    click.echo(f"Wrote {len(output)} predictions to {output_path}")


@cli.command("export")
@click.argument("destination", type=click.Path(path_type=Path))
@click.option("--version", default=None, help="Version to export; defaults to current")
def export_bundle(destination, version):
    """Export a self-contained fitted production bundle."""
    import shutil

    from harness.app.workspace.discovery import find_workspace

    workspace = find_workspace()
    if workspace is None:
        raise click.ClickException("No Harness workspace found")
    version_id = version or (workspace / "current").read_text().strip()
    source = workspace / "versions" / version_id / "run" / "model.bundle"
    if not source.exists():
        raise click.ClickException(f"Production bundle not found for {version_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    click.echo(f"Exported {version_id} to {destination}")


@cli.command()
@click.option(
    "--studio", is_flag=True, help="Serve the Studio web application instead of MCP"
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, type=int, show_default=True)
def serve(studio, host, port):
    """Start the MCP server or Studio for the current workspace."""
    if studio:
        from harness.app.workspace.discovery import find_workspace

        workspace = find_workspace()
        if workspace is None:
            raise click.ClickException("No Harness workspace found")
        try:
            import uvicorn
            from harness.studio.server import create_app
        except ImportError as exc:
            raise click.ClickException(
                "Studio is not installed. Install harness-studio."
            ) from exc
        uvicorn.run(create_app(workspace), host=host, port=port)
        return

    try:
        from harness.server.main import main as run_server
    except ImportError as exc:
        raise click.ClickException(
            "MCP server is not installed. Install harness-server."
        ) from exc
    run_server()


@cli.command()
def doctor():
    """Check system dependencies and configuration."""
    checks = [
        ("Python 3.11+", sys.version_info >= (3, 11)),
        ("pandas", _check_import("pandas")),
        ("numpy", _check_import("numpy")),
        ("scikit-learn", _check_import("sklearn")),
        ("pydantic", _check_import("pydantic")),
        ("harness-data", _check_import("harness.data")),
        ("harness-ml", _check_import("harness.ml")),
        ("harness-server", _check_import("harness.server")),
        ("protomcp", _check_import("protomcp")),
        ("harness-studio", _check_import("harness.studio")),
        ("xgboost", _check_import("xgboost")),
        ("lightgbm", _check_import("lightgbm")),
        ("catboost", _check_import("catboost")),
        ("torch", _check_import("torch")),
    ]

    all_ok = True
    for name, ok in checks:
        status = "OK" if ok else "MISSING"
        symbol = "+" if ok else "-"
        click.echo(f"  [{symbol}] {name}: {status}")
        if not ok and name in (
            "Python 3.11+",
            "pandas",
            "numpy",
            "scikit-learn",
            "harness-data",
            "harness-ml",
        ):
            all_ok = False

    if all_ok:
        click.echo("\nAll required dependencies present.")
    else:
        click.echo("\nSome required dependencies are missing.", err=True)
        sys.exit(1)


def _check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
