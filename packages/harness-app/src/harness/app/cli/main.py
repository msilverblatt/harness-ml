import click
import importlib
import sys
from pathlib import Path


@click.group()
def cli():
    """Harness — Agent-first ML platform."""
    pass


@cli.command()
@click.argument("project_name", required=False)
@click.option("--task-type", type=click.Choice(["binary", "multiclass", "regression"]), default="binary")
@click.option("--target", default="target", help="Target column name")
def init(project_name, task_type, target):
    """Initialize a new Harness workspace."""
    from harness.app.workspace.manager import WorkspaceManager

    workspace_dir = Path.cwd() / project_name if project_name else Path.cwd()
    ws = WorkspaceManager.init(workspace_dir, task_type=task_type, target_column=target)
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
        click.echo("No harness workspace found. Run 'harness init' to create one.", err=True)
        sys.exit(1)

    ws = WorkspaceManager(ws_dir)
    info = ws.status()
    click.echo(f"Workspace: {info['workspace']}")
    click.echo(f"Current version: {info['current_version'] or 'none'}")
    click.echo(f"Models: {info['model_count']}")
    click.echo(f"Versions: {info['version_count']}")
    if info['metrics']:
        click.echo("Metrics:")
        for k, v in info['metrics'].items():
            click.echo(f"  {k}: {v:.4f}")


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
        if not ok and name in ("Python 3.11+", "pandas", "numpy", "scikit-learn"):
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
