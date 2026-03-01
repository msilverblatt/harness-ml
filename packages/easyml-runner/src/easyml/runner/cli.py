"""Click CLI for easyml-runner.

Provides validate, inspect, experiment, run, serve, and init commands.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from easyml.runner.validator import validate_project


# -----------------------------------------------------------------------
# Helper: load and validate config, exit on failure
# -----------------------------------------------------------------------

def _load_config(config_dir: str, gender: str | None = None):
    """Load and validate project config, exiting on failure."""
    variant = None
    if gender and gender.upper() == "W":
        variant = "w"
    result = validate_project(config_dir, variant=variant)
    if not result.valid:
        click.echo(result.format(), err=True)
        raise SystemExit(1)
    return result


# -----------------------------------------------------------------------
# Main group
# -----------------------------------------------------------------------

@click.group()
@click.option(
    "--config-dir",
    default="config",
    help="Path to the project config directory.",
    type=click.Path(),
)
@click.option(
    "--gender",
    default="M",
    help="Gender variant (M or W).",
)
@click.pass_context
def main(ctx: click.Context, config_dir: str, gender: str) -> None:
    """easyml — YAML-driven ML pipeline runner."""
    ctx.ensure_object(dict)
    ctx.obj["config_dir"] = config_dir
    ctx.obj["gender"] = gender


# -----------------------------------------------------------------------
# validate
# -----------------------------------------------------------------------

@main.command()
@click.pass_context
def validate(ctx: click.Context) -> None:
    """Validate the project configuration."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    variant = "w" if gender.upper() == "W" else None

    result = validate_project(config_dir, variant=variant)
    if result.valid:
        click.echo("Config is valid. OK")
    else:
        click.echo(result.format(), err=True)
        ctx.exit(1)


# -----------------------------------------------------------------------
# inspect subgroup
# -----------------------------------------------------------------------

@main.group()
@click.pass_context
def inspect(ctx: click.Context) -> None:
    """Inspect project configuration."""
    pass


@inspect.command("config")
@click.option("--section", default=None, help="Show only this section of the config.")
@click.pass_context
def inspect_config(ctx: click.Context, section: str | None) -> None:
    """Dump resolved config as JSON."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    data = result.config.model_dump()
    if section is not None:
        if section in data:
            data = data[section]
        else:
            click.echo(f"Unknown section: {section!r}", err=True)
            ctx.exit(1)
            return

    click.echo(json.dumps(data, indent=2, default=str))


@inspect.command("models")
@click.pass_context
def inspect_models(ctx: click.Context) -> None:
    """List models with type, active status, and feature count."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    models = result.config.models
    for name, model_def in sorted(models.items()):
        status = "active" if model_def.active else "inactive"
        n_features = len(model_def.features)
        click.echo(f"  {name}: type={model_def.type}, {status}, {n_features} features")


@inspect.command("features")
@click.pass_context
def inspect_features(ctx: click.Context) -> None:
    """List declared features."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    features = result.config.features
    if not features:
        click.echo("No features declared in config.")
        return

    for name, feat in sorted(features.items()):
        click.echo(
            f"  {name}: category={feat.category}, level={feat.level}, "
            f"columns={feat.columns}"
        )


# -----------------------------------------------------------------------
# experiment subgroup
# -----------------------------------------------------------------------

@main.group()
@click.pass_context
def experiment(ctx: click.Context) -> None:
    """Experiment management."""
    pass


def _get_experiment_manager(config):
    """Build ExperimentManager from ProjectConfig."""
    from easyml.experiments.manager import ExperimentManager

    exp_cfg = config.experiments
    if exp_cfg is None:
        click.echo("No experiments section in config.", err=True)
        raise SystemExit(1)

    return ExperimentManager(
        experiments_dir=exp_cfg.experiments_dir or "experiments",
        naming_pattern=exp_cfg.naming_pattern,
        log_path=exp_cfg.log_path,
        do_not_retry_path=exp_cfg.do_not_retry_path,
    )


@experiment.command("create")
@click.argument("experiment_id")
@click.pass_context
def experiment_create(ctx: click.Context, experiment_id: str) -> None:
    """Create a new experiment directory with an empty overlay."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    manager = _get_experiment_manager(result.config)
    try:
        exp_dir = manager.create(experiment_id)
        click.echo(f"Created experiment: {exp_dir}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@experiment.command("log")
@click.argument("experiment_id")
@click.option("--hypothesis", required=True, help="What the experiment tests.")
@click.option("--changes", required=True, help="What was changed.")
@click.option("--verdict", required=True, help="keep, revert, or partial.")
@click.option("--notes", default="", help="Additional notes.")
@click.pass_context
def experiment_log(
    ctx: click.Context,
    experiment_id: str,
    hypothesis: str,
    changes: str,
    verdict: str,
    notes: str,
) -> None:
    """Log an experiment with hypothesis, changes, and verdict."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    manager = _get_experiment_manager(result.config)
    try:
        manager.log(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            changes=changes,
            verdict=verdict,
            notes=notes,
        )
        click.echo(f"Logged experiment: {experiment_id}")
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)


@experiment.command("list")
@click.pass_context
def experiment_list(ctx: click.Context) -> None:
    """List experiment directories."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    exp_cfg = result.config.experiments
    if exp_cfg is None:
        click.echo("No experiments section in config.")
        return

    exp_dir = Path(exp_cfg.experiments_dir or "experiments")
    if not exp_dir.exists():
        click.echo("No experiments directory found.")
        return

    experiments = sorted(
        p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not experiments:
        click.echo("No experiments found.")
        return

    for name in experiments:
        click.echo(f"  {name}")


# -----------------------------------------------------------------------
# run subgroup (stubs that will use PipelineRunner)
# -----------------------------------------------------------------------

@main.group()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Run pipeline stages."""
    pass


@run.command("train")
@click.option("--run-id", default=None, help="Optional run identifier.")
@click.pass_context
def run_train(ctx: click.Context, run_id: str | None) -> None:
    """Train all active models."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
    )
    runner.load()
    result = runner.train(run_id=run_id)
    click.echo(json.dumps(result, indent=2, default=str))


@run.command("backtest")
@click.pass_context
def run_backtest(ctx: click.Context) -> None:
    """Run backtesting across all configured seasons."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
    )
    runner.load()
    result = runner.backtest()
    click.echo(json.dumps(result, indent=2, default=str))


@run.command("pipeline")
@click.pass_context
def run_pipeline(ctx: click.Context) -> None:
    """Run the full pipeline: load, train, backtest."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]

    from easyml.runner.pipeline import PipelineRunner

    variant = "w" if gender.upper() == "W" else None
    runner = PipelineRunner(
        project_dir=".",
        config_dir=config_dir,
        variant=variant,
    )
    result = runner.run_full()
    click.echo(json.dumps(result, indent=2, default=str))


# -----------------------------------------------------------------------
# serve (stub)
# -----------------------------------------------------------------------

@main.command()
@click.pass_context
def serve(ctx: click.Context) -> None:
    """Start the MCP server (if configured)."""
    config_dir = ctx.obj["config_dir"]
    gender = ctx.obj["gender"]
    result = _load_config(config_dir, gender)

    if result.config.server is None:
        click.echo("No server configuration found in config.")
        return

    click.echo(f"Server: {result.config.server.name}")
    click.echo("Server startup not yet implemented.")


# -----------------------------------------------------------------------
# init (stub)
# -----------------------------------------------------------------------

@main.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize a new easyml project (stub)."""
    click.echo("Project initialization not yet implemented.")
